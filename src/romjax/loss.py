"""Loss functions for graphs."""
import functools
from inspect import Parameter, signature
from typing import Annotated, Any, Callable, Literal, Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, PyTree
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)

from romjax.data_gen import DataLoader, LoadDataConfig
from romjax.graph import FunctionGraph
from romjax.model import ImplicitSampleable, SourceSampleable
from romjax.operators import BinaryOp, UnaryOp
from romjax.tree import (
    TreePath,
    coerce_tree_paths,
    get_subtree,
    pytree_path_iter,
    pytree_resolve_refs,
    pytree_square_norm,
    set_subtree,
    shape_dtype_template_like,
)
from romjax.typing import CallableModel, from_registry, from_yaml

__all__ = ["GraphLoss", "GraphLossTerm", "GraphTest"]


def _aux_from_template(
    data: PyTree,
    template_paths: Sequence[TreePath] | None = None,
    aux_paths: Sequence[TreePath] | None = None
) -> PyTree | None:
    """Form graph aux data by templating from the input data.
    
    :param data: the input pytree containing array-like data
    :param template_paths: Target paths in data to form a template from
    :param aux_paths: Destination paths in aux graph input to place the formed template (e.g. for reconstruction)
    """
    template_paths = coerce_tree_paths(template_paths)
    aux_paths = coerce_tree_paths(aux_paths)
    aux = None

    if aux_paths:
        if len(template_paths) == 0:
            selected = data
        else:
            selected = None
            for p in template_paths:
                selected = set_subtree(selected, p, get_subtree(data, p))
        
        template = shape_dtype_template_like(selected)
        
        for p in aux_paths:
            aux = set_subtree(aux, p, template)
    
    return aux


def reconstruction_loss(
    params: PyTree, 
    single_data: PyTree, 
    graph: FunctionGraph, 
    path: list[str] | None = None,
    start: str | None = None,
    initial_path: list[str] | None = None,
    template_paths: Sequence[TreePath] | None = None,
    aux_paths: Sequence[TreePath] | None = None,
    error_op: BinaryOp | None = None,
    ignore: set | None = None,
):
    """State reconstruction objective. Minimize reconstruction error along a given path.

    Note that templates are generally handled internally by the graph, but sometimes you may need to pass it in
    a priori depending on which paths are being traversed, e.g. trying to reconstruct before compressing.
    
    :param params: optimization params
    :param single_data: single pytree sample payload
    :param graph: graph containing function/model definitions along edges
    :param path: the reconstruction path
    :param start: the starting node for the reconstruction path (defaults initial in the path)
    :param initial_path: optional transform for the input data before reconstruction (default uses raw data)
    :param template_paths: gather template from the input data
    :param aux_paths: insert template in the aux tree 
    :param error_op: optional override for node error
    :param ignore: optional override for node ignore paths
    """
    aux = _aux_from_template(single_data, template_paths, aux_paths)

    if initial_path is None:
        initial_data = single_data
    else:
        initial_data, aux = graph.push_path(
            single_data, initial_path, aux=aux, edge_payload_patches=params, return_aux=True
        )

    return graph.reconstruction_error(
        initial_data, path, edge_payload_patches=params, error_op=error_op, ignore=ignore, aux=aux, start=start
    )


def residual_loss(
    params: PyTree,
    single_data: PyTree,
    graph: FunctionGraph,
    path: list[str] = None,
    template_paths: Sequence[TreePath] | None = None,
    aux_paths: Sequence[TreePath] | None = None,
    error_op: BinaryOp | None = None,
    ignore: set | None = None,
):
    """Residual minimization objective. Minimize the result of a single forward path."""
    aux = _aux_from_template(single_data, template_paths, aux_paths)
    return graph.path_error(
        single_data,
        path_a=path,
        path_b=None,
        edge_payload_patches=params,
        error_op=error_op,
        ignore=ignore,
        aux_a=aux,
        aux_b=aux
    )


def similarity_loss(
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
    Similarity error objective. Minimize difference between a single similarity path and the data. Assumes data contains
    all inputs, outputs, and residuals (e.g. data from both sides of an implicit model edge).
    """
    aux = _aux_from_template(single_data, template_paths, aux_paths)
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
    "similarity": similarity_loss,
    "tikhonov": tikhonov_regularization,
    "orthogonal": orthogonal_regularization,
}


class GraphLossCallable(CallableModel):
    """Loss function for a single data sample."""

    callable: Callable[..., ArrayLike | tuple[ArrayLike, PyTree]]


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

    @property
    def accepts_aux(self) -> bool:
        """Whether the wrapped callable accepts the optional GraphLoss aux payload."""
        try:
            params = signature(self.term.callable).parameters.values()
        except (TypeError, ValueError):
            return False

        return any(param.name == "aux" or param.kind == Parameter.VAR_KEYWORD for param in params)

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
        graph: FunctionGraph,
        aux: PyTree | None = None,
        return_aux: bool = False,
    ) -> jax.Array | tuple[jax.Array, PyTree | None]:
        """Evaluate the unweighted, batch-reduced term value."""
        if self.batch_reduce is not None:
            if self.dataset is not None and self.dataset not in batch_data:
                value = jnp.asarray(0.0)  # if a dataset runs out during iteration
                return (value, aux) if return_aux else value

            term_batch = batch_data[self.dataset] if self.dataset is not None else batch_data
            if isinstance(term_batch, (list, tuple)):
                if len(term_batch) == 0:
                    value = jnp.asarray(0.0)
                    return (value, aux) if return_aux else value
                term_batch = self._stack_sequence_batch(term_batch)

            term_result = jax.lax.map(
                lambda single_data: self._call_term(params, single_data, graph, aux), term_batch,
                batch_size=self.batch_size
            )
            losses, aux = self._split_term_result(term_result, aux)
            value = self.batch_reduce(losses)
            return (value, aux) if return_aux else value

        value, aux = self._split_term_result(self._call_term(params, batch_data, graph, aux), aux)
        return (value, aux) if return_aux else value

    def _call_term(
        self,
        params: Mapping[str, PyTree],
        single_data: Mapping[str, PyTree],
        graph: FunctionGraph,
        aux: PyTree | None,
    ) -> jax.Array | tuple[jax.Array, PyTree]:
        """Call a scalar loss term with aux only when the callable declares support for it."""
        if self.accepts_aux:
            return self.term(params, single_data, graph, aux=aux)
        return self.term(params, single_data, graph)

    @staticmethod
    def _split_term_result(
        result: jax.Array | tuple[jax.Array, PyTree],
        aux: PyTree | None,
    ) -> tuple[jax.Array, PyTree | None]:
        """Split a term result into a scalar value and next aux carry."""
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, aux

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
    _term_weights: list[float] = PrivateAttr(default_factory=lambda: [])

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
        """Cache term metadata used during loss evaluation."""
        self._term_names = tuple(term.name or f"term_{idx}" for idx, term in enumerate(self.terms))
        self._term_weights = [term.weight for term in self.terms]

    @property
    def term_names(self) -> tuple[str, ...]:
        """Stable names for all terms in order."""
        return self._term_names

    def _raw_term_array(self, params: Mapping[str, PyTree], batch: Mapping[str, PyTree]) -> jax.Array:
        """Evaluate raw term values as an ordered JAX array."""
        if len(self.terms) == 0:
            return jnp.asarray([])

        values = []
        aux = None
        for term in self.terms:
            value, aux = term.raw_value(params, batch, self.graph, aux=aux, return_aux=True)
            values.append(value)

        return jnp.asarray(values)

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
        scaled_values = jnp.asarray(self._term_weights) * self._scale_array(scales) * raw_values
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
