"""Reduced-order model training routine."""
import functools
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterator, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from alive_progress import alive_bar
from jaxtyping import ArrayLike, PyTree
from loguru import logger
from orbax.checkpoint import v1 as ocp
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    SkipValidation,
    field_validator,
    model_validator,
)

from romjax.data_gen import DataLoader, LoadDataConfig
from romjax.graph import FunctionGraph
from romjax.model import ImplicitSampleable, SourceSampleable
from romjax.operators import BinaryOp, UnaryOp
from romjax.plotting import GridplotConfig, PlotSpec, gridplot
from romjax.profiling import profile_annotation, profile_step, profile_trace
from romjax.routine import Routine, RoutineError
from romjax.tree import (
    TreePath,
    coerce_tree_paths,
    get_subtree,
    is_shape_dtype_template_leaf,
    pytree_norm,
    pytree_path_iter,
    pytree_resolve_refs,
    pytree_square_norm,
    set_subtree,
    shape_dtype_template_like,
)
from romjax.typing import CallableModel, ThirdPartyType, from_registry, from_yaml, require_type, resolve_graph_refs
from romjax.utils import _NullProgress

__all__ = ["Train", "GraphLoss", "GraphTest", "BatchLoader", "OrbaxParams"]


def _prettify_timedelta(delta: float) -> str:
    total_seconds = int(delta)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    if days > 0:
        return f"{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if minutes > 0:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{delta:.3f} s"


class BatchLoader[T: Any](BaseModel, Iterator):
    """
    Helper for basic mini-batch data loading.

    !!! Example
        ```python
        data = list(range(10))
        
        for batch in BatchLoader(data=data, batch_size=2):
            print(batch)  # [0, 1],  [2, 3],  [4, 5], ...
        ```

    :ivar data: the sequence of data to load batches from (i.e. a list, array, etc.), if none then will just load empty
                tuples infinitely (for use with the `Train` routine as default). Will try to access items simply by
                integer index. If this is an ndarray, then it will take along the first axis per usual. If this is a
                tuple of equal-length sequences, each item will be batched independently and the loader will yield a
                tuple of mini-batches.
    :ivar batch_size: the number of items per batch, if none then loads the entire dataset at each iteration (default)
    :ivar shuffle_seed: the random seed for shuffling data at each epoch, if none then does not shuffle (default)
    :ivar max_epochs: maximum number of iterations through full dataset, if none then continues indefinitely (default)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Annotated[Sequence[T] | None, SkipValidation] = None  # default to empty loader
    batch_size: PositiveInt | None = None    # default to all
    shuffle_seed: int | None = None          # default no shuffle
    max_epochs: PositiveInt | None = None    # defaults to infinite

    _size: PositiveInt | None = PrivateAttr(default=None)
    _iterator: Iterator[Sequence[T]] | None = PrivateAttr(default=None)

    def _get_size(self):
        if self._size is None:
            if self.data is not None:
                if isinstance(self.data, tuple):
                    if len(self.data) == 0:
                        raise ValueError("Tuple data for batch loader must contain at least one sequence")

                    length = len(self.data[0])
                    for item in self.data[1:]:
                        if len(item) != length:
                            raise ValueError("All tuple items for batch loader must have equal length")

                    self._size = length
                else:
                    self._size = len(self.data)
        return self._size

    @model_validator(mode="after")
    def _validate_model(self):
        if self.data is not None:
            if not hasattr(self.data, "__len__"):
                raise ValueError("Data for batch loader must have a finite length")
            if isinstance(self.data, tuple) and len(self.data) > 0:
                size = len(self.data[0])
                for item in self.data[1:]:
                    if len(item) != size:
                        raise ValueError("All tuple items for batch loader must have equal length")
            self._get_size()
        
        return self

    @staticmethod
    def _batch_item(item: Sequence[T], window: np.ndarray) -> Sequence[T]:
        try:
            return item[window]
        except Exception:
            if isinstance(item, list):
                return [item[idx] for idx in window]
            if isinstance(item, tuple):
                return tuple(item[idx] for idx in window)
            return itemgetter(*window)(item)

    def _generator(self, start: int = 0) -> Generator[Sequence[T], None, None]:
        """Main mini-batch loading routine. Loads a single mini-batch from original data at a time."""
        size = self._get_size()

        if size is None or self.data is None:
            while True:
                yield ()  # empty dataloader
        
        batch_size = self.batch_size or size
        cursor = 0
        epoch = 0

        def _shuffle_indices(epoch: int):
            seed = np.random.SeedSequence([self.shuffle_seed, epoch])
            return np.random.default_rng(seed).permutation(size)
        
        indices = np.arange(size) if self.shuffle_seed is None else _shuffle_indices(epoch)
        
        def _advance_cursor(cursor, epoch, batch_size, size):
            next_cursor = cursor + batch_size
            
            if next_cursor >= size:
                cursor = 0
                epoch += 1
            else:
                cursor = next_cursor
            
            return cursor, epoch

        # Move the cursor up based on the starting index
        for _ in range(start):
            cursor, epoch = _advance_cursor(cursor, epoch, batch_size, size)

            if self.max_epochs is not None and epoch >= self.max_epochs:
                return
        
        while True:
            if self.shuffle_seed is not None:
                indices = _shuffle_indices(epoch)
            
            window = indices[cursor: cursor + batch_size]

            if isinstance(self.data, tuple):
                yield tuple(self._batch_item(item, window) for item in self.data)
            else:
                try:
                    yield self.data[window]  # try numpy fancy-indexing first
                except Exception:
                    yield itemgetter(*window)(self.data)

            cursor, epoch = _advance_cursor(cursor, epoch, batch_size, size)

            if self.max_epochs is not None and epoch >= self.max_epochs:
                return
    
    def set_iterator(self, start: int = 0) -> None:
        """Start the iterator at a given index."""
        self._iterator = self._generator(start)
    
    def __next__(self) -> Sequence[T]:
        if self._iterator is None:
            self.set_iterator()
        return next(self._iterator)
    
    def __iter__(self):
        self.set_iterator()
        return self


def reconstruction_loss(
    params: PyTree, 
    single_data: PyTree, 
    graph: FunctionGraph, 
    path: list[str] | None = None,
    error_op: BinaryOp | None = None,
    ignore: set | None = None,
):
    """State reconstruction objective. Minimize reconstruction error along a given path."""
    return graph.reconstruction_error(single_data, path, edge_payload_patches=params, error_op=error_op, ignore=ignore)


def residual_loss(
    params: PyTree,
    single_data: PyTree,
    graph: FunctionGraph,
    path: list[str] = None,
    error_op: BinaryOp | None = None,
    ignore: set | None = None,
):
    """Residual minimization objective. Minimize the result of a single forward path."""
    return graph.path_error(
        single_data,
        path_a=path,
        path_b=None,
        edge_payload_patches=params,
        error_op=error_op,
        ignore=ignore
    )


def solution_loss(
    params: PyTree,
    single_data: PyTree,
    graph: FunctionGraph,
    path: list[str] = None,
    template_paths: Sequence[TreePath] | None = None,
    aux_paths: Sequence[TreePath] | None = None,
    error_op: BinaryOp | None = None,
    ignore: set | None = None,
):
    """
    Solution error objective. Minimize difference between a single solution path and the data. Assumes solution data
    is precomputed and passed in. 
    """
    template_paths = coerce_tree_paths(template_paths)
    aux_paths = coerce_tree_paths(aux_paths)
    aux = None

    if aux_paths:
        if len(template_paths) == 0:
            selected = single_data
        else:
            selected = None
            for p in template_paths:
                selected = set_subtree(selected, p, get_subtree(single_data, p))
        
        template = shape_dtype_template_like(selected)
        
        for p in aux_paths:
            aux = set_subtree(aux, p, template)
    
    sol = graph.push_path(single_data, path, edge_payload_patches=params, aux=aux)

    end_node = graph._path_end_node(path)
    op = end_node.error_op if error_op is None else BinaryOp(error_op)
    ignore = ignore or end_node.ignore

    return op(single_data, sol, ignore=ignore)


def tikhonov_regularization(params: PyTree, single_data: PyTree, graph: FunctionGraph):
    del single_data, graph
    return pytree_square_norm(params)


def orthogonal_regularization(params: PyTree, single_data: PyTree, graph: FunctionGraph, ref: list[str] = None):
    del single_data, graph
    matrix = get_subtree(params, ref)  # expected Array[r x N] projection matrix
    if matrix is None:
        raise ValueError("Can't locate matrix for orthogonal regularization via ref: '{ref}'")
    
    gram = matrix @ matrix.T
    return pytree_square_norm(gram - jnp.eye(gram.shape[0]))


_LOSS_REGISTRY = {
    "reconstruction": reconstruction_loss,
    "residual": residual_loss,
    "solution": solution_loss,
    "tikhonov": tikhonov_regularization,
    "orthogonal": orthogonal_regularization,
}


class GraphLossCallable(CallableModel):
    """Loss function for a single data sample."""

    callable: Callable[[PyTree, PyTree, FunctionGraph], ArrayLike]


class GraphLossBalancing(BaseModel):
    """
    Optional adaptive scaling policy for :class:`GraphLoss` terms.

    Recommended: 
    - If loss varies on log-scale, then use ema_log
    - Set update_interval = 1-2 epochs
    - Decrease decay~0.9 for faster control
    - Don't use normalize

    :param kind: balancing strategy; ``"none"`` preserves static term weights, ``"ema"`` tracks raw magnitudes,
        and ``"ema_log"`` tracks log magnitudes for multiplicative scale adaptation
    :param decay: exponential decay factor for term magnitudes; for ``"ema_log"`` this is applied to log magnitudes
    :param eps: positive denominator floor for scale computation
    :param target: target scaled magnitude for each term before clipping (default is 1)
    :param min_scale: lower bound for the adaptive scale
    :param max_scale: upper bound for the adaptive scale
    :param bootstrap: initialize EMA scales from the first batch before the first optimizer update
    :param normalize: normalize adaptive scales to control the global learning-rate scale. For ``"ema"``, uses
        arithmetic mean. For ``"ema_log"``, uses geometric mean
    :param update_interval: number of optimizer steps between EMA updates
    """

    kind: Literal["none", "ema", "ema_log"] = "none"
    decay: float = 0.99
    eps: float = 1e-8
    target: float = 1.0
    min_scale: float = 1e-8
    max_scale: float = 1e8
    bootstrap: bool = True
    normalize: bool = False
    update_interval: PositiveInt = 1

    @model_validator(mode="before")
    @classmethod
    def _from_bool_or_str(cls, value):
        if isinstance(value, str):
            return {"kind": value}
        elif isinstance(value, bool):
            return {"kind": "ema" if value else "none"}
        return value

    @model_validator(mode="after")
    def _check_values(self):
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("decay must satisfy 0 <= decay < 1")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.target <= 0.0:
            raise ValueError("target must be positive")
        if self.min_scale <= 0.0 or self.max_scale <= 0.0:
            raise ValueError("scale bounds must be positive")
        if self.min_scale > self.max_scale:
            raise ValueError("min_scale must be less than or equal to max_scale")
        return self


class _GraphLossEmaState(BaseModel):
    """Mutable Python-side EMA state for adaptive :class:`GraphLoss` scaling.
    
    :ivar step: current optimizer step
    :ivar ema: current exponential moving average of each loss term, either in raw space or log space
    :ivar scales: clip(1/ema), this is what actually multiples each loss, e.g. w(t)
    :ivar initialized: whether each term's ema has been started
    """

    step: int = 0
    ema: dict[str, float]
    scales: dict[str, float]
    initialized: dict[str, bool]

    @classmethod
    def initialize(cls, names: Sequence[str]) -> "_GraphLossEmaState":
        """Create an all-ones EMA state for a fixed list of term names."""
        return cls(
            ema={name: 0.0 for name in names},
            scales={name: 1.0 for name in names},
            initialized={name: False for name in names},
        )

    @classmethod
    def from_checkpoint(cls, tree: Mapping[str, Any]) -> "_GraphLossEmaState":
        """Restore EMA state from a scalar-array checkpoint tree."""
        return cls(
            step=int(np.asarray(tree["step"])),
            ema={name: float(np.asarray(value)) for name, value in tree["ema"].items()},
            scales={name: float(np.asarray(value)) for name, value in tree["scales"].items()},
            initialized={name: bool(np.asarray(value)) for name, value in tree["initialized"].items()},
        )

    def checkpoint_tree(self) -> dict[str, PyTree]:
        """Return a checkpointable scalar-array representation."""
        return {
            "step": jnp.asarray(self.step, dtype=jnp.int32),
            "ema": {name: jnp.asarray(value, dtype=jnp.float32) for name, value in self.ema.items()},
            "scales": {name: jnp.asarray(value, dtype=jnp.float32) for name, value in self.scales.items()},
            "initialized": {name: jnp.asarray(value) for name, value in self.initialized.items()},
        }


class GraphLossTerm(BaseModel):
    """
    One weighted term in a :class:`GraphLoss`. Aggregates a loss function over batch data.

    :param name: stable term name used for adaptive balancing, diagnostics, and plotting
    :param term: callable to apply to a single sample of data
    :param dataset: which dataset name to read data from
    :param weight: scalar term weight
    :param batch_reduce: reduce the loss over batch data; skip batch reduce if none
    :param batch_size: optional, passed to jax.lax.map for balancing memory with vmap; use batch_size=0 for vmap-like
        behavior
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    term: Annotated[
        GraphLossCallable,
        BeforeValidator(lambda v: {"callable": v} if isinstance(v, str) else v),
        BeforeValidator(functools.partial(from_registry, _LOSS_REGISTRY))
    ]
    name: str | None = None
    dataset: str | None = None
    weight: float = 1.0
    batch_reduce: UnaryOp | None = "mean"
    batch_size: int | None = None

    @field_validator("name", mode="after")
    @classmethod
    def _check_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value:
            raise ValueError("GraphLossTerm name must not be empty")
        if "," in value:
            raise ValueError("GraphLossTerm name must not contain ','")
        return value

    @field_validator("batch_reduce", mode="before")
    @classmethod
    def _get_unary_op(cls, value):
        if value is not None:
            return UnaryOp(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def _from_plain_function(cls, value):
        if callable(value) or isinstance(value, str) or (isinstance(value, Mapping) and "callable" in value):
            return {"term": value}
        return value

    @staticmethod
    def _stack_sequence_batch(term_batch: Sequence[PyTree]) -> PyTree:
        """Convert a sequence of single-sample PyTrees into one batched PyTree."""
        def stack_leaf(*leaves: PyTree) -> PyTree:
            try:
                return jnp.stack(leaves)
            except (TypeError, ValueError):
                return leaves[0]

        return jax.tree.map(stack_leaf, *term_batch)

    def raw_value(
        self, 
        params: Mapping[str, PyTree], 
        batch_data: Mapping[str, PyTree], 
        graph: FunctionGraph
    ) -> jax.Array:
        """Evaluate the unweighted, batch-reduced term value."""
        if self.batch_reduce is not None:
            if self.dataset is not None and self.dataset not in batch_data:
                return jnp.asarray(0.0)  # if a dataset runs out during iteration

            term_batch = batch_data[self.dataset] if self.dataset is not None else batch_data
            if isinstance(term_batch, (list, tuple)):
                if len(term_batch) == 0:
                    return jnp.asarray(0.0)
                term_batch = self._stack_sequence_batch(term_batch)

            losses = jax.lax.map(
                lambda single_data: self.term(params, single_data, graph), term_batch, batch_size=self.batch_size
            )
            return self.batch_reduce(losses)
    
        else:
            return self.term(params, batch_data, graph)

    def __call__(
        self, 
        params: Mapping[str, PyTree], 
        batch_data: Mapping[str, PyTree], 
        graph: FunctionGraph
    ) -> jax.Array:
        """Evaluate the weighted term value."""
        return jnp.asarray(self.weight) * self.raw_value(params, batch_data, graph)


class GraphLoss(BaseModel):
    """
    Loss function for a `FunctionGraph`.

    :param terms: loss terms combined by weighted summation
    :param balancing: optional adaptive term scaling policy
    :param graph: the FunctionGraph, leave as None to defer to `Train.graph`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    terms: Sequence[GraphLossTerm]
    balancing: GraphLossBalancing = Field(default_factory=GraphLossBalancing)
    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None
    _term_names: tuple[str, ...] = PrivateAttr(default=())
    _term_weights: jax.Array = PrivateAttr(default_factory=lambda: jnp.asarray([], dtype=jnp.float32))
    _term_value_branches: tuple[Callable[[Mapping[str, PyTree], Mapping[str, PyTree]], jax.Array], ...] = PrivateAttr(
        default=()
    )

    __hash__ = object.__hash__

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value):
        if isinstance(value, list):
            return {"terms": value}
        return value

    @model_validator(mode="after")
    def _bind_default_datasets(self):
        self._set_default_term_names()
        if self.graph is not None:
            self._set_default_datasets()
        self._refresh_term_cache()
        return self

    def _set_default_term_names(self) -> None:
        """Assign deterministic names to unnamed loss terms and require uniqueness."""
        names = []
        for idx, term in enumerate(self.terms):
            if term.name is None:
                term.name = f"term_{idx}"
            if term.name in names:
                raise ValueError(f"Duplicate GraphLoss term name: {term.name}")
            names.append(term.name)
    
    def _set_default_datasets(self):
        """Grab the first sampleable edge as the default dataset, i.e. typically there is only one."""
        if self.graph is not None:
            _default_edge = None
            for edge_name, edge in self.graph.edges.items():
                if isinstance(edge, ImplicitSampleable | SourceSampleable):
                    _default_edge = edge_name
                break

            for term in self.terms:
                if term.dataset is None:
                    term.dataset = _default_edge
            self._refresh_term_cache()

    def _refresh_term_cache(self) -> None:
        """Cache term metadata and JAX branch callables used during loss evaluation."""
        self._term_names = tuple(term.name or f"term_{idx}" for idx, term in enumerate(self.terms))
        self._term_weights = jnp.asarray([term.weight for term in self.terms])
        self._term_value_branches = tuple(
            (lambda params, batch, term=term: term.raw_value(params, batch, self.graph))
            for term in self.terms
        )

    @property
    def term_names(self) -> tuple[str, ...]:
        """Stable names for all terms in order."""
        return self._term_names

    def _raw_term_array(self, params: Mapping[str, PyTree], batch: Mapping[str, PyTree]) -> jax.Array:
        """Evaluate raw term values as an ordered JAX array."""
        if len(self._term_value_branches) == 0:
            return jnp.asarray([])

        def body(carry: None, idx: jax.Array) -> tuple[None, jax.Array]:
            value = jax.lax.switch(idx, self._term_value_branches, params, batch)
            return carry, value

        _, values = jax.lax.scan(body, None, jnp.arange(len(self._term_value_branches)))
        return values

    def _scale_array(self, scales: Mapping[str, ArrayLike] | None = None) -> jax.Array:
        """Return ordered adaptive scale values."""
        scales = scales or {}
        return jnp.asarray([scales.get(name, 1.0) for name in self._term_names])

    def _array_to_term_dict(self, values: jax.Array) -> dict[str, jax.Array]:
        """Convert ordered term values into the public term-name mapping."""
        return {name: values[idx] for idx, name in enumerate(self._term_names)}

    def __call__(
        self,
        params: Mapping[str, PyTree],
        batch: Mapping[str, PyTree],
        scales: Mapping[str, ArrayLike] | None = None,
        return_aux: bool = False,
    ) -> jax.Array | tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
        """Evaluate the total graph loss.

        :param params: edge-payload parameter patches
        :param batch: batch data passed to graph loss terms
        :param scales: optional adaptive scale for each term; missing scales default to one
        :param return_aux: if True, return ``(total, (raw_terms, scaled_terms))``
        :return: scalar total loss, optionally with raw and scaled term values keyed by term name
        """
        params = pytree_resolve_refs(params)
        raw_values = self._raw_term_array(params, batch)
        scaled_values = self._term_weights * self._scale_array(scales) * raw_values
        total = jnp.sum(scaled_values)

        if return_aux:
            return total, (self._array_to_term_dict(raw_values), self._array_to_term_dict(scaled_values))
        return total


class GraphTest(GraphLoss):
    """
    Just compute a graph loss function over a set of validation data.
    """

    loader: DataLoader
    reduce: UnaryOp | None = "mean"
    _batch_loss: Callable[[PyTree, PyTree], ArrayLike] = PrivateAttr()
    
    @field_validator("reduce", mode="before")
    @classmethod
    def _get_unary_op(cls, value):
        if value is not None:
            return UnaryOp(value)
        return value
    
    @model_validator(mode="after")
    def _validate_loader_and_loss(self):
        for _, ds_cfg in pytree_path_iter(self.loader.datasets, is_leaf=lambda leaf: isinstance(leaf, LoadDataConfig)):
            ds_cfg.max_epochs = 1   # Only load data once

        self._batch_loss = eqx.filter_jit(lambda batch, params: super(GraphTest, self).__call__(batch, params))
        return self

    def __call__(self, params: Mapping[str, PyTree]) -> jax.Array:
        values = jnp.asarray([self._batch_loss(params, batch) for batch in self.loader])
        return self.reduce(values)
    

type SaveDecisionPolicy = Annotated[
    ThirdPartyType(default_modules=ocp.training.save_decision_policies.__name__),
    AfterValidator(partial(require_type, ocp.training.save_decision_policies.SaveDecisionPolicy)),
]
type PreservationPolicy = Annotated[
    ThirdPartyType(default_modules=ocp.training.preservation_policies.__name__),
    AfterValidator(partial(require_type, ocp.training.preservation_policies.PreservationPolicy)),
]
type GradientTransformation = Annotated[
    ThirdPartyType(default_modules="optax"),
    AfterValidator(partial(require_type, optax.GradientTransformation)),
]


class OrbaxParams(BaseModel):
    """
    Utility for loading parameter PyTrees from Orbax checkpoints.

    :param params: direct parameter PyTree or path to a checkpoint directory containing a ``params`` checkpointable
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    params: PyTree | str | Path
    _resolved_params: PyTree | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _from_plain_params(cls, value: PyTree | str | Path | Mapping[str, PyTree]) -> dict[str, PyTree] | PyTree:
        if not isinstance(value, OrbaxParams):
            if isinstance(value, Mapping) and "params" in value:
                return value
            return {"params": value}
        return value

    def resolve_params(self, template: PyTree | None = None) -> PyTree | None:
        """
        Load parameters from Orbax using a template for static leaves.

        :param template: optional full parameter PyTree template
        :returns: resolved parameters, or ``None`` if a checkpoint path has no latest checkpoint
        """
        if self._resolved_params is not None:
            return self._resolved_params

        if isinstance(self.params, str | Path):
            with ocp.training.Checkpointer(Path(self.params).absolute()) as ckptr:
                if ckptr.latest is not None:
                    if template is not None:
                        dynamic_params, static_params = eqx.partition(
                            template,
                            lambda leaf: eqx.is_array(leaf) or is_shape_dtype_template_leaf(leaf),
                        )
                        loaded = ckptr.load_checkpointables(abstract_checkpointables={"params": dynamic_params})
                        params = eqx.combine(loaded["params"], static_params)
                    else:
                        params = ckptr.load_checkpointables()["params"]

                    self._resolved_params = params
                    return params

            return None

        self._resolved_params = self.params
        return self._resolved_params


class CheckpointerConfig(BaseModel):
    """
    Orbax-policy checkpoint configuration for :class:`GraphTrain`.

    Note that the default save_decision_policy=None is different from Orbax. In Orbax, this means save every step.
    Here, it means don't save.

    :param save_decision_policy: Orbax save policy; short names resolve from
        ``ocp.training.save_decision_policies``, default=None, which will not save any (for performance)
    :param preservation_policy: Orbax preservation policy; short names resolve from
        ``ocp.training.preservation_policies``
    :param step_name_format: Orbax name format for saving training steps
    :param custom_metadata: see `Checkpointer`
    :param cleanup_tmp_directories: see `Checkpointer`
    :param lightweight_initialize: see `Checkpointer`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True, extra="allow")

    save_decision_policy: SaveDecisionPolicy | None = None
    preservation_policy: PreservationPolicy | None = None
    step_name_format: Any | None = None
    custom_metadata: dict | None = None
    cleanup_tmp_directories: bool = False
    lightweight_initialize: bool = False

    @field_validator("save_decision_policy", mode="before")
    @classmethod
    def _simple_fixed_interval_save(cls, value):
        if isinstance(value, int):
            return ocp.training.save_decision_policies.FixedIntervalPolicy(value)
        return value
    
    @field_validator("preservation_policy", mode="before")
    @classmethod
    def _simple_fixed_interval_preservation(cls, value):
        if isinstance(value, int):
            return ocp.training.preservation_policies.EveryNSteps(value)
        return value

    @field_validator("step_name_format", mode="before")
    @classmethod
    def _standard_name_format(cls, value: str | int | Mapping | ocp.path.step.NameFormat | None):
        """Build a standard name format via {step_prefix=..., step_format_fixed_length=...}."""
        if isinstance(value, Mapping):
            return ocp.path.step.standard_name_format(**value)
        elif isinstance(value, str):
            return ocp.path.step.standard_name_format(step_prefix=value)
        elif isinstance(value, int):
            return ocp.path.step.standard_name_format(step_format_fixed_length=value)
        
        return value


def _merge_plot_spec(base: PlotSpec, override: Mapping | PlotSpec | None) -> PlotSpec:
    """Merge a partial plot spec override into a default spec."""
    if override is None:
        return base
    if isinstance(override, PlotSpec):
        return override
    if not isinstance(override, Mapping):
        raise ValueError("plot spec override must be a Mapping or PlotSpec")

    spec = {
        "kind": override.get("kind", base.kind),
        "data": override.get("data", base.data),
        "name": override.get("name", base.name),
        "opts": GridplotConfig.merge(base.opts, override.get("opts", {})),
        "kwargs": GridplotConfig.merge(base.kwargs, override.get("kwargs", {})),
    }
    return PlotSpec(**spec)


def _default_plot_spec(name: str = "Loss", color: str | None = None) -> PlotSpec:
    """Default plot spec for training."""
    return PlotSpec(
        kind="line",
        data=([], []),
        name=name,
        opts={"xlabel": "Iteration", "ylabel": name, "yscale": "log", "grid": True},
        kwargs={"color": color} if color is not None else {}
    )


class TermPlotConfig(BaseModel):
    """
    Online plotting options for per-term :class:`GraphLoss` diagnostics.

    :param enabled: whether to include this term subplot in the training figure
    :param include: term names to plot; ``None`` means all terms not excluded
    :param exclude: term names to hide
    :param spec: base subplot style, matching :class:`romjax.plotting.PlotSpec`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    enabled: bool = False
    include: Sequence[str] | None = None
    exclude: Sequence[str] = ()
    spec: PlotSpec = Field(default_factory=_default_plot_spec)

    def selected_terms(self, names: Sequence[str]) -> tuple[str, ...]:
        """Return visible term names in their original order."""
        included = set(names) if self.include is None else set(self.include)
        excluded = set(self.exclude)
        return tuple(name for name in names if name in included and name not in excluded)


class DiagnosticsConfig(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    log_interval: PositiveInt | None = None
    plot_interval: PositiveInt | None = None
    test_interval: PositiveInt | None = None
    show_progress: bool = True  # progress bar
    callback_interval: PositiveInt | None = None
    progress_callback: Callable[[PyTree, FunctionGraph, Path], None] | None = None
    live_plot: bool = False
    save_plot: dict = Field(default_factory=lambda: dict(fname="loss.pdf", bbox_inches="tight"))
    raw_terms_plot: TermPlotConfig = Field(default_factory=lambda: TermPlotConfig(spec=_default_plot_spec("Raw terms")))
    scaled_terms_plot: TermPlotConfig = Field(
        default_factory=lambda: TermPlotConfig(spec=_default_plot_spec("Scaled terms"))
    )
    loss_plot: PlotSpec = Field(default_factory=lambda: _default_plot_spec("Loss", "g"))
    test_plot: PlotSpec = Field(default_factory=lambda: _default_plot_spec("Test", "orange"))

    @field_validator("save_plot", mode="before")
    @classmethod
    def _from_fname(cls, value):
        """Allow just specifying a filename for save plot."""
        if isinstance(value, str | Path):
            return {"fname": value}
        return value

    @field_validator("raw_terms_plot", "scaled_terms_plot", mode="before")
    @classmethod
    def _fill_term_plot_config(cls, value, info):
        """Fill term plot configs from partial mappings while preserving subplot-specific defaults."""
        default_spec = (
            _default_plot_spec("Raw terms") if info.field_name == "raw_terms_plot" else 
            _default_plot_spec("Scaled terms")
        )

        if value is None:
            return TermPlotConfig(spec=default_spec)
        if isinstance(value, bool):
            return TermPlotConfig(enabled=value, spec=default_spec)
        if isinstance(value, TermPlotConfig):
            value.spec = _merge_plot_spec(default_spec, value.spec)
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"{info.field_name} must be a Mapping, bool, or TermPlotConfig")
        
        cfg = dict(value)
        cfg["spec"] = _merge_plot_spec(default_spec, cfg.get("spec"))
        return TermPlotConfig(**cfg)
    
    @field_validator("loss_plot", "test_plot", mode="before")
    @classmethod
    def _fill_plot_spec_data(cls, value, info):
        """Ensure we are only doing line plots, and initialize empty data param."""
        spec = (
            _default_plot_spec("Loss", "g") if info.field_name == "loss_plot" else 
            _default_plot_spec("Test", "orange")
        )
        return _merge_plot_spec(spec, value)


class TerminationConfig(BaseModel):
    """
    Training termination criteria.

    :param max_steps: maximum optimizer steps
    :param loss_tol: rolling relative loss tolerance; disabled when non-positive
    :param test_tol: validation/test tolerance; disabled when non-positive
    :param grad_tol: gradient norm tolerance; disabled when non-positive
    :param max_runtime: runtime limit in seconds or a ``datetime.timedelta` supported string
    """

    @model_validator(mode="before")
    @classmethod
    def _from_plain_max_steps(cls, value):
        if isinstance(value, int):
            return {"max_steps": value}
        return value

    max_steps: PositiveInt = 200
    loss_tol: PositiveFloat | None = None
    test_tol: PositiveFloat | None = None
    grad_tol: PositiveFloat | None  = None
    max_runtime: timedelta = timedelta(seconds=300.0)


class _NullCheckpointer:

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @property
    def latest(self):
        return None

    @property
    def checkpoints(self):
        return []
    
    def save_checkpointables(*args, **kwargs):
        pass


class _ScalarHistory:
    """Scalar metric history with O(1) step replacement."""

    def __init__(self, steps: Sequence[int] | None = None, values: Sequence[float] | None = None):
        self.steps = list(steps or [])
        self.values = list(values or [])
        self._step_index = {step: idx for idx, step in enumerate(self.steps)}

    @classmethod
    def load(cls, path: Path) -> "_ScalarHistory":
        """Load a scalar CSV history."""
        if not path.exists():
            return cls()

        arr = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
        if arr.size == 0:
            return cls()
        return cls(arr[:, 0].astype(int).tolist(), arr[:, 1].tolist())

    def has_step(self, step: int) -> bool:
        """Return whether a row exists for ``step``."""
        return step in self._step_index

    def record(self, step: int, value: float) -> None:
        """Append or replace a scalar value for an optimizer step."""
        idx = self._step_index.get(step)
        if idx is None:
            self._step_index[step] = len(self.steps)
            self.steps.append(step)
            self.values.append(value)
        else:
            self.values[idx] = value

    def series(self) -> tuple[list[int], list[float]]:
        """Return data in ``matplotlib`` line format."""
        return self.steps, self.values

    def save(self, path: Path) -> None:
        """Write this scalar history as CSV."""
        if self.steps:
            arr = np.column_stack((
                np.asarray(self.steps, dtype=int),
                np.asarray(self.values, dtype=float),
            ))
        else:
            arr = np.empty((0, 2))
        np.savetxt(
            path,
            arr,
            fmt="%d,%.6e",
            header="Iteration,Value",
            comments="",
        )


class _TermHistory:
    """Multi-term metric history with O(1) step replacement."""

    def __init__(
        self,
        term_names: Sequence[str],
        steps: Sequence[int] | None = None,
        values: Mapping[str, Sequence[float]] | None = None,
    ):
        self.term_names = tuple(term_names)
        self.steps = list(steps or [])
        values = values or {}
        self.values = {name: list(values.get(name, [])) for name in self.term_names}
        for name in self.term_names:
            missing = len(self.steps) - len(self.values[name])
            if missing > 0:
                self.values[name].extend([np.nan] * missing)
        self._step_index = {step: idx for idx, step in enumerate(self.steps)}

    @classmethod
    def load(cls, path: Path, term_names: Sequence[str]) -> "_TermHistory":
        """Load a multi-term CSV history."""
        if not path.exists():
            return cls(term_names)

        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        arr = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
        if arr.size == 0:
            return cls(term_names)

        steps = arr[:, 0].astype(int).tolist()
        values = {name: [] for name in term_names}
        for col_idx, name in enumerate(header[1:], start=1):
            if name in values:
                values[name] = arr[:, col_idx].tolist()
        return cls(term_names, steps, values)

    def record(self, step: int, values: Mapping[str, float]) -> None:
        """Append or replace a row in a multi-term history."""
        idx = self._step_index.get(step)
        if idx is None:
            idx = len(self.steps)
            self._step_index[step] = idx
            self.steps.append(step)
            for name in self.term_names:
                self.values[name].append(np.nan)

        for name in self.term_names:
            value = values.get(name, np.nan)
            self.values[name][idx] = value

    def term_series(self, name: str) -> tuple[list[int], list[float]]:
        """Return one term's data in ``matplotlib`` line format."""
        return self.steps, self.values.get(name, [])

    def save(self, path: Path) -> None:
        """Write this multi-term history as CSV."""
        if self.steps:
            columns = [np.asarray(self.steps, dtype=int)]
            columns.extend(np.asarray(self.values[name], dtype=float) for name in self.term_names)
            arr = np.column_stack(columns)
        else:
            arr = np.empty((0, 1 + len(self.term_names)))

        fmt = ["%d"] + ["%.6e"] * len(self.term_names)
        np.savetxt(
            path,
            arr,
            fmt=fmt,
            delimiter=",",
            header="Iteration," + ",".join(self.term_names),
            comments="",
        )


class Train(Routine):
    """
    ROM training routine.

    :param loss: loss function, callable as `loss(params, Any) -> float`
    :param init_params: PyTree of initial optimization parameters. Can specify as a callable to generate params.
    :param optimizer: Optax optimizer specification

    :param test: function to compute a validation test score, callable as `test(params) -> float`
    :param dataloader: Optionally load extra data for the loss function
    :param termination: stopping criteria
    :param diagnostics: configs for plotting and logging
    :param host_interval: global host synchronization interval; ``None`` synchronizes only at the final iteration
    :param checkpoint_interval: outer checkpoint check interval; ``None`` checks only at the final iteration

    :param root: run directory for checkpoints, logs, and history (optional)
    :param write_policy: ``reuse`` restores checkpoints, ``overwrite`` replaces artifacts, ``error`` fails
    :param checkpointer: Orbax policy checkpoint options
    :param load_orbax: optional params-only Orbax warm start used when no ``root`` checkpoint is resumed

    :param init_seed: random seed for initializing parameters (if init_params is callable)
    :param graph: graph object or YAML path (optional, for graph-related losses and dataloaders)
    """

    # Required
    loss: Callable[[PyTree, Any], float]
    init_params: Annotated[PyTree, BeforeValidator(from_yaml)]
    optimizer: GradientTransformation

    # Optional
    test: Callable[[PyTree], float] | None = None
    dataloader: Iterator[Any] = Field(default_factory=BatchLoader)  # empty loading by default
    termination: TerminationConfig = Field(default_factory=TerminationConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    host_interval: PositiveInt | None = 1
    checkpoint_interval: PositiveInt | None = 1

    # Persistence
    root: Path | None = None
    write_policy: Literal["reuse", "overwrite", "error"] = "reuse"
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)
    load_orbax: OrbaxParams | None = None

    # Other
    init_seed: int = 0
    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None

    @model_validator(mode="after")
    def _infer_train_intervals(self):
        """Infer cheap outer checkpoint cadence from a fixed-interval Orbax policy when possible."""
        if "checkpoint_interval" not in self.model_fields_set:
            interval = getattr(self.checkpointer.save_decision_policy, "interval", None)
            if isinstance(interval, int) and interval > 0:
                self.checkpoint_interval = interval
        return self

    @model_validator(mode="after")
    def _setup_extra_train_configs(self):
        """Do some assorted things after model validation."""
        if self.root is not None:
            self.root = self.root.resolve()

        # If init params implements a 'sample' function, then initialize the parameter pytree.
        # Pass graph object to loss, test, and dataloader if requested
        if self.graph is not None:
            
            self.graph.resolve_norms()

            for attr in ["loss", "test", "dataloader"]:
                if hasattr(ele := getattr(self, attr), "graph"):
                    if ele.graph is None:
                        ele.graph = self.graph
                
                        if isinstance(ele, GraphLoss):
                            ele._set_default_datasets()

            self.init_params = resolve_graph_refs(self.init_params, self.graph)

        sample_fn = getattr(self.init_params, "sample", None)
        if callable(sample_fn):
            self.init_params = sample_fn(jax.random.key(self.init_seed))
        
        # Start dataloader from current training step if applicable
        if self.root is not None and hasattr(self.dataloader, "set_iterator"):
            with ocp.training.Checkpointer(self.root, **dict(self.checkpointer)) as ckptr:
                if ckptr.latest is not None:
                    self.dataloader.set_iterator(ckptr.latest.step)
        
        return self

    @field_validator("write_policy", mode="after")
    @classmethod
    def _check_write_policy(cls, policy: bool, info):
        root = info.data["root"]
        if root is not None:
            if root.exists() and any(root.iterdir()):
                if policy == "error":
                    raise RoutineError(f"Training root already contains artifacts: {root} and policy is '{policy}'")
                if policy == "overwrite":
                    # Just rm previous train artifacts
                    for path in root.iterdir():
                        if path.is_dir() and path.joinpath("_CHECKPOINT_METADATA").is_file():
                            shutil.rmtree(path)
                        elif path.is_file() and path.suffix in {".csv", ".pdf"}:
                            path.unlink()
            root.mkdir(parents=True, exist_ok=True)
        
        return policy
    
    def run(self) -> int:
        """For compatibility with Routine."""
        self.__call__()
        return 0

    def __call__(self) -> PyTree:

        test_fn = self.test if self.test is not None else None
        _, static_params = eqx.partition(self.init_params, eqx.is_array)
        graph_loss = self.loss if isinstance(self.loss, GraphLoss) else None
        ema_enabled = graph_loss is not None and graph_loss.balancing.kind in {"ema", "ema_log"}
        term_names = graph_loss.term_names if graph_loss is not None else ()
        term_diagnostics_enabled = graph_loss is not None and (
            ema_enabled
            or self.diagnostics.raw_terms_plot.enabled
            or self.diagnostics.scaled_terms_plot.enabled
        )
        loss_state = _GraphLossEmaState.initialize(term_names) if ema_enabled else None
        ema_bootstrapped = False

        def _checkpoint_params(params: PyTree) -> PyTree:
            """Persist only array-valued leaves so Orbax never sees static callables/modules."""
            return eqx.filter(params, eqx.is_array)

        def _restore_params(dynamic_params: PyTree) -> PyTree:
            """Rebuild the full parameter pytree from checkpointed arrays and the init template."""
            return eqx.combine(dynamic_params, static_params)

        def _apply_optimizer_update(params: PyTree, opt_state: optax.OptState, grads: PyTree):
            updates, opt_state = self.optimizer.update(grads, opt_state, eqx.filter(params, eqx.is_array))
            params = eqx.apply_updates(params, updates)
            grad_norm = pytree_norm(grads) if self.termination.grad_tol else None
            return params, opt_state, grad_norm

        @eqx.filter_jit
        def _train_step(params: PyTree, opt_state: optax.OptState, batch: PyTree):
            loss, grads = eqx.filter_value_and_grad(self.loss)(params, batch)
            params, opt_state, grad_norm = _apply_optimizer_update(params, opt_state, grads)
            return params, opt_state, loss, grad_norm

        @eqx.filter_jit
        def _graph_train_step(
            params: PyTree,
            opt_state: optax.OptState,
            batch: PyTree,
            scales: Mapping[str, ArrayLike],
        ):
            def _loss(params: PyTree, batch: PyTree):
                return graph_loss(params, batch, scales=scales, return_aux=True)

            (loss, (raw_terms, scaled_terms)), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(params, batch)
            params, opt_state, grad_norm = _apply_optimizer_update(params, opt_state, grads)
            return params, opt_state, loss, raw_terms, scaled_terms, grad_norm

        def _checkpointables(params: PyTree, opt_state: optax.OptState) -> dict[str, PyTree]:
            """Build the checkpointable training state."""
            checkpointables = {"params": _checkpoint_params(params), "opt_state": opt_state}
            if loss_state is not None:
                checkpointables["loss_state"] = loss_state.checkpoint_tree()
            return checkpointables

        def _current_scales() -> dict[str, float]:
            """Return the current adaptive scales as host floats for diagnostics and checkpointing."""
            if loss_state is None:
                return {name: 1.0 for name in term_names}
            return dict(loss_state.scales)

        def _current_scale_arrays() -> dict[str, jax.Array]:
            """Return current adaptive scales as dynamic JAX leaves for jitted loss evaluation."""
            return {name: jnp.asarray(value, dtype=jnp.float32) for name, value in _current_scales().items()}

        def _host_float_terms(terms: Mapping[str, ArrayLike] | None) -> dict[str, float] | None:
            """Convert a ready term-value mapping to host floats."""
            if terms is None:
                return None
            return {name: float(np.asarray(value)) for name, value in terms.items()}

        def _compute_ema_scales() -> dict[str, float]:
            """Compute clipped and normalized adaptive scales from current EMA values."""
            if loss_state is None:
                return {}

            scales = {}
            for name in term_names:
                if loss_state.initialized[name]:
                    if graph_loss.balancing.kind == "ema_log":
                        scale = (
                            graph_loss.balancing.target / (
                                np.exp(np.clip(loss_state.ema[name], np.log(graph_loss.balancing.min_scale), 
                                               np.log(graph_loss.balancing.max_scale)))
                                + graph_loss.balancing.eps
                            )
                        )
                    else:
                        scale = graph_loss.balancing.target / (loss_state.ema[name] + graph_loss.balancing.eps)
                    scales[name] = float(np.clip(scale, graph_loss.balancing.min_scale, graph_loss.balancing.max_scale))
                else:
                    scales[name] = loss_state.scales[name]

            if not graph_loss.balancing.normalize or len(scales) == 0:
                return scales

            values = np.asarray(list(scales.values()), dtype=float)
            if graph_loss.balancing.kind == "ema":
                # Arithmetic mean
                denom = float(np.mean(values))
            elif graph_loss.balancing.kind == "ema_log":
                # Geometric mean
                if np.any(values <= 0.0):
                    logger.warning("Skipping GraphLoss log EMA scale normalization due to non-positive scale values.")
                    return scales
                denom = float(np.exp(np.mean(np.log(values))))
            else:
                raise ValueError(f"EMA kind '{graph_loss.balancing.kind}' not recognized.")
                                 
            if not np.isfinite(denom) or denom <= 0.0:
                logger.warning("Skipping GraphLoss EMA scale normalization due to non-positive scale denominator.")
                return scales

            factor = graph_loss.balancing.target / denom
            return {name: scale * factor for name, scale in scales.items()}

        def _ema_space_value(raw_value: float) -> float:
            """Map a raw loss magnitude into the space tracked by the EMA state."""
            if graph_loss.balancing.kind == "ema_log":
                return float(np.log(raw_value))
            return raw_value

        def _bootstrap_ema_state(raw_terms: Mapping[str, float]) -> None:
            """Initialize EMA state from the first batch before optimizer gradients are computed."""
            if loss_state is None:
                return

            for name in term_names:
                raw_value = raw_terms[name]
                if not np.isfinite(raw_value):
                    logger.warning(f"Skipping EMA bootstrap for non-finite GraphLoss term '{name}': {raw_value}")
                    continue
                if raw_value < 0.0:
                    logger.warning(f"EMA GraphLoss balancing assumes non-negative terms; got {name}={raw_value:.3e}")
                loss_state.ema[name] = _ema_space_value(raw_value)
                loss_state.initialized[name] = True
            loss_state.scales = _compute_ema_scales()

        def _update_ema_state(raw_terms: Mapping[str, float]) -> None:
            """Update EMA state from raw term values outside the differentiable path."""
            if loss_state is None:
                return
            if loss_state.step % graph_loss.balancing.update_interval != 0:
                loss_state.step += 1
                return

            new_step = loss_state.step + 1
            for name in term_names:
                raw_value = raw_terms[name]
                if not np.isfinite(raw_value):
                    logger.warning(f"Skipping EMA update for non-finite GraphLoss term '{name}': {raw_value}")
                    continue
                if raw_value < 0.0:
                    logger.warning(f"EMA GraphLoss balancing assumes non-negative terms; got {name}={raw_value:.3e}")
                
                if loss_state.initialized[name]:
                    ema = graph_loss.balancing.decay * loss_state.ema[name]
                    ema += (1.0 - graph_loss.balancing.decay) * _ema_space_value(raw_value)
                else:
                    ema = _ema_space_value(raw_value)
                    loss_state.initialized[name] = True
                
                loss_state.ema[name] = ema
            
            loss_state.scales = _compute_ema_scales()
            loss_state.step = new_step

        def _save_plot(fig):
            if fig is not None and self.root is not None:
                save_opts = dict(self.diagnostics.save_plot)
                fname = save_opts.pop("fname", "loss.pdf")
                fig.savefig(self.root / fname, **save_opts)
        
        def _load_scalar_history(fname: str) -> _ScalarHistory:
            if self.root is None:
                return _ScalarHistory()
            return _ScalarHistory.load(self.root / fname)

        def _load_term_history(fname: str) -> _TermHistory:
            if self.root is None:
                return _TermHistory(term_names)
            return _TermHistory.load(self.root / fname, term_names)

        def _save_histories() -> None:
            """Persist CSV histories independently of plot refresh cadence."""
            if self.root is None:
                return

            loss_hist.save(self.root / "loss.csv")
            if test_fn is not None:
                test_hist.save(self.root / "test.csv")
            if raw_terms_hist is not None:
                raw_terms_hist.save(self.root / "loss_terms_raw.csv")
            if scaled_terms_hist is not None:
                scaled_terms_hist.save(self.root / "loss_terms_scaled.csv")
            if term_scales_hist is not None:
                term_scales_hist.save(self.root / "loss_term_scales.csv")
        
        if self.root is not None:
            checkpointer_context = ocp.training.Checkpointer(self.root, **dict(self.checkpointer))
        else:
            checkpointer_context = _NullCheckpointer()

        with profile_trace("train", self.root):
            with checkpointer_context as ckptr:
                ## INITIALIZE/LOAD
                params, opt_state, curr_step, total_steps = None, None, None, None

                if ckptr.latest is None:
                    params = self.init_params
                    if self.load_orbax is not None:
                        params = self.load_orbax.resolve_params(self.init_params)
                    opt_state = self.optimizer.init(eqx.filter(params, eqx.is_array))
                    curr_step = 0
                    total_steps = self.termination.max_steps
                    logger.debug("Initialized train")
                else:
                    abstract_checkpointables = {
                        "params": _checkpoint_params(self.init_params),
                        "opt_state": self.optimizer.init(eqx.filter(self.init_params, eqx.is_array)),
                    }
                    if loss_state is not None:
                        abstract_checkpointables["loss_state"] = loss_state.checkpoint_tree()
                    
                    try:
                        _loaded = ckptr.load_checkpointables(abstract_checkpointables=abstract_checkpointables)
                    except Exception:
                        if loss_state is None:
                            raise
                        logger.warning(
                            "Could not restore GraphLoss EMA state from checkpoint; initializing fresh state."
                        )
                        _loaded = ckptr.load_checkpointables(
                            abstract_checkpointables={
                                "params": abstract_checkpointables["params"],
                                "opt_state": abstract_checkpointables["opt_state"],
                            }
                        )
                    else:
                        if loss_state is not None and "loss_state" in _loaded:
                            loss_state = _GraphLossEmaState.from_checkpoint(_loaded["loss_state"])
                            ema_bootstrapped = True
                    
                    params = _restore_params(_loaded["params"])
                    opt_state = _loaded["opt_state"]
                    curr_step = ckptr.latest.step  # checkpoint step is the number of completed optimizer updates
                    total_steps = self.termination.max_steps - curr_step

                    if total_steps <= 0:
                        logger.debug(
                            f"Training already reached max_steps={self.termination.max_steps} from checkpoint."
                        )
                        return 0
                    
                    logger.debug(f"Restarting train from step {curr_step}")
                
                log_interval = self.diagnostics.log_interval or float('inf')
                test_interval = self.diagnostics.test_interval or float('inf')
                plot_interval = self.diagnostics.plot_interval or float('inf')
                callback_interval = self.diagnostics.callback_interval or float('inf')
                
                loss_hist = _load_scalar_history("loss.csv")
                test_hist = _load_scalar_history("test.csv")
                raw_terms_hist = _load_term_history("loss_terms_raw.csv") if term_diagnostics_enabled else None
                scaled_terms_hist = _load_term_history("loss_terms_scaled.csv") if term_diagnostics_enabled else None
                term_scales_hist = _load_term_history("loss_term_scales.csv") if ema_enabled else None
                existing_checkpoint_steps = {checkpoint.step for checkpoint in ckptr.checkpoints}
                fig, axs, lines = None, None, None
                raw_term_lines, scaled_term_lines = {}, {}

                if 0 < plot_interval < float('inf'):
                    if self.diagnostics.live_plot:
                        plt.ion()

                    plot_specs = [self.diagnostics.loss_plot]

                    if test_fn is not None:
                        plot_specs.append(self.diagnostics.test_plot)

                    raw_plot_terms = ()
                    scaled_plot_terms = ()
                    if graph_loss is not None:
                        raw_plot_terms = self.diagnostics.raw_terms_plot.selected_terms(term_names)
                        scaled_plot_terms = self.diagnostics.scaled_terms_plot.selected_terms(term_names)
                    
                    if self.diagnostics.raw_terms_plot.enabled and raw_plot_terms:
                        plot_specs.append(tuple(
                            _merge_plot_spec(
                                self.diagnostics.raw_terms_plot.spec,
                                {
                                    "data": ([], []),
                                    "name": f"raw_terms_{name}",
                                    "opts": {"leg_label": name},
                                },
                            )
                            for name in raw_plot_terms
                        ))
                    
                    if self.diagnostics.scaled_terms_plot.enabled and scaled_plot_terms:
                        plot_specs.append(tuple(
                            _merge_plot_spec(
                                self.diagnostics.scaled_terms_plot.spec,
                                {
                                    "data": ([], []),
                                    "name": f"scaled_terms_{name}",
                                    "opts": {"leg_label": name},
                                },
                            )
                            for name in scaled_plot_terms
                        ))
                    
                    fig, axs = gridplot(plot_specs)
                    active_axes = axs.ravel()[:len(plot_specs)]
                    lines = [ax.lines[0] for ax in active_axes]
                    lines[0].set_data(*loss_hist.series())

                    if test_fn is not None:
                        lines[1].set_data(*test_hist.series())
                    
                    plot_axis_idx = 1 + int(test_fn is not None)
                    if self.diagnostics.raw_terms_plot.enabled and raw_plot_terms:
                        ax = active_axes[plot_axis_idx]
                        raw_term_lines = dict(zip(raw_plot_terms, ax.lines))
                        if raw_terms_hist is not None:
                            for name, line in raw_term_lines.items():
                                line.set_data(*raw_terms_hist.term_series(name))
                        plot_axis_idx += 1
                    
                    if self.diagnostics.scaled_terms_plot.enabled and scaled_plot_terms:
                        ax = active_axes[plot_axis_idx]
                        scaled_term_lines = dict(zip(scaled_plot_terms, ax.lines))
                        if scaled_terms_hist is not None:
                            for name, line in scaled_term_lines.items():
                                line.set_data(*scaled_terms_hist.term_series(name))
                
                def _save_final(metrics=None):
                    if 0 < plot_interval < float('inf') and self.diagnostics.live_plot:
                        plt.ioff()
                        plt.show()
                    with profile_annotation("checkpoint_save", env=os.environ):
                        ckptr.save_checkpointables(
                            step=curr_step,
                            checkpointables=_checkpointables(params, opt_state),
                            metrics=metrics,
                            force=True,
                            overwrite=True,
                        )
                    with profile_annotation("plot_save", env=os.environ):
                        _save_plot(fig)
                    _save_histories()

                t_start = time.time()
                metrics = None

                def _interval_active(interval: int | None, step: int) -> bool:
                    """Return whether host work for an interval should run at this optimizer step."""
                    final_step = step >= self.termination.max_steps
                    if interval is None:
                        return final_step
                    return step % interval == 0 or final_step

                ctxt = alive_bar(total_steps) if self.diagnostics.show_progress else _NullProgress()
                    
                with ctxt as bar:
                    try:
                        batch = next(self.dataloader)
                    except StopIteration:
                        logger.info(f"Train dataloader has stopped at step {curr_step}. Terminating...")
                        batch = None
                    last_batch = batch

                    while True:
                        if batch is None:
                            break

                        ## LOSS/VALIDATION EVALUATION
                        host_active = _interval_active(self.host_interval, curr_step)
                        next_batch = None
                        dataloader_stopped = False

                        try:
                            if (
                                host_active
                                and
                                ema_enabled
                                and graph_loss.balancing.bootstrap
                                and not ema_bootstrapped
                                and loss_state is not None
                            ):
                                _, _, _, bootstrap_terms, _, _ = jax.block_until_ready(
                                    _graph_train_step(params, opt_state, batch, _current_scale_arrays())
                                )
                                bootstrap_terms = _host_float_terms(bootstrap_terms)
                                _bootstrap_ema_state(bootstrap_terms)
                                ema_bootstrapped = True

                            with profile_step("train", step_num=curr_step, env=os.environ):
                                with profile_annotation("train_step", env=os.environ):
                                    if term_diagnostics_enabled:
                                        train_result = _graph_train_step(
                                            params,
                                            opt_state,
                                            batch,
                                            _current_scale_arrays(),
                                        )
                                    else:
                                        train_result = _train_step(params, opt_state, batch)

                            if curr_step < self.termination.max_steps:
                                try:
                                    next_batch = next(self.dataloader)
                                except StopIteration:
                                    dataloader_stopped = True
                                else:
                                    last_batch = next_batch

                            train_result = jax.block_until_ready(train_result)
                            if term_diagnostics_enabled:
                                next_params, next_opt_state, loss, raw_terms, scaled_terms, grad_norm_device = (
                                    train_result
                                )
                            else:
                                next_params, next_opt_state, loss, grad_norm_device = train_result
                                raw_terms, scaled_terms = None, None
                        except Exception as exc:
                            logger.exception(
                                f"Exception encountered during train step at step {curr_step}. "
                                "Saving checkpoint..."
                            )
                            _save_final()
                            raise RoutineError("Train step failure") from exc
                        
                        ## METRICS AND CHECKPOINT
                        t_diff = time.time() - t_start
                        runtime_reached = t_diff >= self.termination.max_runtime.total_seconds()
                        host_active = host_active or runtime_reached
                        test_score, grad_norm = None, None
                        loss_value = None
                        stats_str = f"step={curr_step}"

                        if host_active:
                            loss_value = float(loss)
                            raw_terms = _host_float_terms(raw_terms)
                            scaled_terms = _host_float_terms(scaled_terms)
                            loss_hist.record(curr_step, loss_value)
                            if raw_terms_hist is not None and raw_terms is not None:
                                raw_terms_hist.record(curr_step, raw_terms)
                            if scaled_terms_hist is not None and scaled_terms is not None:
                                scaled_terms_hist.record(curr_step, scaled_terms)
                            if term_scales_hist is not None:
                                term_scales_hist.record(curr_step, _current_scales())
                            if raw_terms is not None:
                                _update_ema_state(raw_terms)

                            metrics = {"loss": loss_value}

                            if (
                                test_fn is not None
                                and 0 < test_interval < float('inf')
                                and (
                                    curr_step % test_interval == 0
                                    or curr_step >= self.termination.max_steps
                                )
                            ):
                                with profile_annotation("validation", env=os.environ):
                                    test_score = float(test_fn(params))
                                test_hist.record(curr_step, test_score)
                                metrics["test_score"] = test_score

                            if self.termination.grad_tol and grad_norm_device is not None:
                                grad_norm = float(grad_norm_device)
                                metrics["grad_norm"] = grad_norm

                            with profile_annotation("checkpoint_save", env=os.environ):
                                if (
                                    _interval_active(self.checkpoint_interval, curr_step)
                                    and self.checkpointer.save_decision_policy is not None
                                    and curr_step not in existing_checkpoint_steps
                                ):
                                    saved_checkpoint = ckptr.save_checkpointables(
                                        step=curr_step,
                                        checkpointables=_checkpointables(params, opt_state),
                                        metrics=metrics,
                                    )
                                    if saved_checkpoint:
                                        _save_histories()
                                        existing_checkpoint_steps.add(curr_step)

                            ## DIAGNOSTICS
                            stats_str = f"loss={loss_value:.2e}"
                            if test_score is not None:
                                stats_str += f" test={test_score:.2e}"
                            if grad_norm is not None:
                                stats_str += f" grad={grad_norm:.2e}"
                        
                        bar.text = stats_str

                        if host_active and curr_step % log_interval == 0:
                            logger.debug(f"Elapsed: {_prettify_timedelta(time.time() - t_start)} "
                                        f"| step={curr_step} {stats_str}")
                        
                        if host_active and (
                            0 < plot_interval < float('inf')
                            and (curr_step % plot_interval == 0 or curr_step >= self.termination.max_steps)
                        ):
                            with profile_annotation("plot_update", env=os.environ):
                                lines[0].set_data(*loss_hist.series())

                                if test_fn is not None:
                                    lines[1].set_data(*test_hist.series())
                                if raw_terms_hist is not None:
                                    for name, line in raw_term_lines.items():
                                        line.set_data(*raw_terms_hist.term_series(name))
                                if scaled_terms_hist is not None:
                                    for name, line in scaled_term_lines.items():
                                        line.set_data(*scaled_terms_hist.term_series(name))
                                
                                for ax in axs.ravel():
                                    ax.relim()
                                    ax.autoscale_view()
                                fig.canvas.draw_idle()
                                fig.canvas.flush_events()
                                
                                _save_plot(fig)
                        
                        if (host_active
                            and self.diagnostics.progress_callback is not None
                            and 0 < callback_interval < float('inf')
                            and curr_step % callback_interval == 0):
                            with profile_annotation("progress_callback", env=os.environ):
                                self.diagnostics.progress_callback(params, self.graph, self.root)
                        
                        ## END CONDITIONS
                        if (
                            host_active
                            and
                            self.termination.test_tol
                            and test_score is not None
                            and test_score < self.termination.test_tol
                        ):
                            logger.info(f"Termination criteria reached: test score "
                                        f"{test_score:.2e} < {self.termination.test_tol:.2e}")
                            break
                            
                        if host_active and self.termination.grad_tol and grad_norm is not None:
                            if not jnp.isfinite(grad_norm):
                                logger.warning("Grad norm is not finite. Terminating...")
                                break

                            if grad_norm < self.termination.grad_tol:
                                logger.info(f"Termination criteria reached: gradient norm "
                                            f"{grad_norm:.2e} < {self.termination.grad_tol:.2e}")
                                break
                        
                        if host_active and self.termination.loss_tol and loss_value < self.termination.loss_tol:
                            logger.info(f"Termination criteria reached: loss "
                                        f"{loss_value:.2e} < {self.termination.loss_tol:.2e}")
                            break

                        if curr_step >= self.termination.max_steps:
                            logger.info(f"Termination criteria reached: "
                                        f"{curr_step} / {self.termination.max_steps} optimizer steps")
                            break

                        if runtime_reached:
                            logger.info(f"Termination criteria reached: max runtime "
                                        f"{_prettify_timedelta(t_diff)} / "
                                        f"{_prettify_timedelta(self.termination.max_runtime.total_seconds())}")
                            break

                        params, opt_state = next_params, next_opt_state
                        curr_step += 1
                        bar()

                        if dataloader_stopped:
                            if loss_hist.has_step(curr_step):
                                logger.info(f"Train dataloader has stopped at step {curr_step}. Terminating...")
                                break
                            logger.info(
                                f"Train dataloader has stopped at step {curr_step}. "
                                "Reusing the last batch for final metric evaluation..."
                            )
                            batch = last_batch
                        else:
                            batch = next_batch
                
                logger.debug(f"Train finished. Elapsed: {_prettify_timedelta(time.time()-t_start)}")
                _save_final(metrics)
        
        return params
