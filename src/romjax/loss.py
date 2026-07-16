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
from romjax.graph import Edge, FunctionGraph, Node
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
    shape_dtype_like,
)
from romjax.typing import CallableModel, from_registry, from_yaml

__all__ = [
    "CyclicPathError",
    "GraphLoss",
    "GraphLossTerm",
    "GraphLossTermGenerator",
    "GraphTest",
    "cyclic_path_error_terms",
]


_GRAPH_PATH_PAYLOAD_CACHE = "graph_path_payloads"
_LOSS_TERM_GENERATOR_REGISTRY: dict[str, Callable[..., Sequence["GraphLossTerm"]]] = {}
_LOSS_REGISTRY: dict[str, Callable[[PyTree, PyTree, FunctionGraph], ArrayLike]] = {}


type _PathPayloadCacheKey = tuple[str, str, tuple[str, ...]]


def _coerce_optional_tree_paths(value: Any) -> Sequence[TreePath] | None:
    """Coerce configured tree paths while preserving ``None`` as an unset override."""
    return None if value is None else coerce_tree_paths(value)


def _default_dataset_edge(graph: FunctionGraph | None) -> str | None:
    """Return the first sampleable edge key in graph order."""
    if graph is None:
        return None
    for edge_name, edge in graph.edges.items():
        if isinstance(edge, ImplicitSampleable | SourceSampleable):
            return edge_name
    return None


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
        
        template = shape_dtype_like(selected)
        
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


_LOSS_REGISTRY.update({
    "reconstruction": reconstruction_loss,
    "residual": residual_loss,
    "similarity": similarity_loss,
    "tikhonov": tikhonov_regularization,
    "orthogonal": orthogonal_regularization,
})


class GraphLossCallable(CallableModel):
    """Loss function for a single data sample."""

    callable: Callable[..., ArrayLike | tuple[ArrayLike, PyTree]]


class GraphLossTermGenerator(CallableModel):
    """
    Configurable factory for one or more :class:`GraphLossTerm` objects.

    The wrapped callable is invoked with the active :class:`FunctionGraph` as its first argument. This allows
    YAML-configured loss terms to be generated after :class:`Train` binds a graph to a :class:`GraphLoss`.

    :param callable: callable returning one or more graph loss terms
    """

    callable: Callable[..., Sequence["GraphLossTerm"]]

    @model_validator(mode="before")
    @classmethod
    def _from_generator_config(cls, value):
        if callable(value) or isinstance(value, str):
            return {"callable": value}
        if isinstance(value, Mapping) and "generator" in value:
            opts = {key: item for key, item in value.items() if key != "generator"}
            return {"callable": value["generator"], **opts}
        return value

    @field_validator("callable", mode="before")
    @classmethod
    def _resolve_generator_callable(cls, value):
        if isinstance(value, str) and value in _LOSS_TERM_GENERATOR_REGISTRY:
            return _LOSS_TERM_GENERATOR_REGISTRY[value]
        return value


class _CyclicPathSpec(BaseModel):
    """Static path metadata for one generated cyclic path-error term.
    
    :ivar name: the generated name of the graph loss term
    :ivar start: the nominal start node of the path
    :ivar dest: the final node of the path
    :ivar logical_start: the dataset node to actually start from (seeded from data node->start node initially)
    :ivar path_a: the clockwise path from start->dest (including initial seeding)
    :ivar path_b: the counter-clockwise path from start->dest (including initial seeding)
    :ivar cache_keys: all the partial paths this graph loss term will encounter
    :ivar index: the global index in all generated cyclic path error terms
    :ivar last_use: the index of the last term that uses each partial cached path, use for cleaning the cache 
    """

    name: str
    start: str
    dest: str
    logical_start: str
    path_a: tuple[str, ...]
    path_b: tuple[str, ...]
    cache_keys: tuple[_PathPayloadCacheKey, ...] = ()
    index: int = 0
    last_use: dict[_PathPayloadCacheKey, int] = Field(default_factory=dict)


class _CyclicPathErrorCallable(BaseModel):
    """Single-sample callable used by generated cyclic path-error terms."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: _CyclicPathSpec
    dataset: str
    dataset_edge: str
    template_paths: Annotated[Sequence[TreePath] | None, BeforeValidator(coerce_tree_paths)] = None
    aux_paths: Annotated[Sequence[TreePath] | None, BeforeValidator(coerce_tree_paths)] = None
    cache_payloads: bool = False
    error_op: BinaryOp | None = None
    ignore: Annotated[Sequence[TreePath] | None, BeforeValidator(_coerce_optional_tree_paths)] = None

    @field_validator("error_op", mode="before")
    @classmethod
    def _coerce_error_op(cls, value: Any) -> BinaryOp | None:
        if value is None or isinstance(value, BinaryOp):
            return value
        return BinaryOp(value)

    def __call__(
        self,
        params: PyTree,
        single_data: PyTree,
        graph: FunctionGraph,
        aux: PyTree | None = None,
    ) -> jax.Array | tuple[jax.Array, PyTree]:
        """Evaluate one generated cyclic path-error term for a single sample."""
        graph_aux = _aux_from_template(single_data, self.template_paths, self.aux_paths)

        aux_out = {} if aux is None else dict(aux)
        payload_cache = dict(aux_out.get(_GRAPH_PATH_PAYLOAD_CACHE, {}))

        out_a, graph_aux, payload_cache = self._push_cached_path(
            single_data, self.spec.path_a, graph, params, graph_aux, payload_cache
        )
        out_b, graph_aux, payload_cache = self._push_cached_path(
            single_data, self.spec.path_b, graph, params, graph_aux, payload_cache
        )

        dest_node = graph._resolve_node(self.spec.dest)
        error_op = dest_node.error_op if self.error_op is None else self.error_op
        ignore = dest_node.ignore if self.ignore is None else self.ignore
        if not self.cache_payloads:
            return error_op(out_a, out_b, ignore=ignore)

        for key, last_index in self.spec.last_use.items():
            if last_index <= self.spec.index:
                payload_cache.pop(key, None)

        aux_out[_GRAPH_PATH_PAYLOAD_CACHE] = payload_cache
        return error_op(out_a, out_b, ignore=ignore), aux_out

    def _push_cached_path(
        self,
        single_data: PyTree,
        logical_path: tuple[str, ...],
        graph: FunctionGraph,
        params: PyTree,
        graph_aux: PyTree | None,
        payload_cache: dict[_PathPayloadCacheKey, PyTree],
    ) -> tuple[PyTree, PyTree | None, dict[_PathPayloadCacheKey, PyTree]]:
        """Push one logical path, reusing and appending path-prefix payloads."""
        prefix_len = self._longest_cached_prefix(logical_path, payload_cache)
        if prefix_len == 0:
            payload = single_data
            curr_node = graph._resolve_node(self.spec.logical_start)
        else:
            prefix = logical_path[:prefix_len]
            payload = payload_cache[self._cache_key(prefix)]
            curr_node = graph._path_end_node(list(prefix), start=self.spec.logical_start)

        for step_idx in range(prefix_len, len(logical_path)):
            edge_name = logical_path[step_idx]
            step_prefix = logical_path[: step_idx + 1]
            edge = graph._resolve_edge(edge_name)
            full_cycle_final_dataset_step = (
                len(logical_path) > 1
                and step_idx == len(logical_path) - 1
                and edge.name == self.dataset_edge
                and graph._path_end_node(list(logical_path), start=self.spec.logical_start) == self.spec.logical_start
            )

            if (
                step_idx == 0
                and edge.name == self.dataset_edge
                and curr_node in {edge.source, edge.target}
                and not full_cycle_final_dataset_step
            ):
                # The dataset already contains source/target payloads, so skip evaluation here.
                curr_node = graph._step_path_node(curr_node, edge)
            else:
                payload, graph_aux = graph.push_path(
                    payload,
                    [edge_name],
                    start=curr_node,
                    aux=graph_aux,
                    edge_payload_patches=params,
                    return_aux=True,
                )
                curr_node = graph._step_path_node(curr_node, edge)

            payload_cache[self._cache_key(step_prefix)] = payload

        return payload, graph_aux, payload_cache

    def _longest_cached_prefix(
        self,
        logical_path: tuple[str, ...],
        payload_cache: Mapping[_PathPayloadCacheKey, PyTree],
    ) -> int:
        """Return the length of the longest cached prefix for ``logical_path``."""
        for prefix_len in range(len(logical_path), 0, -1):
            if self._cache_key(logical_path[:prefix_len]) in payload_cache:
                return prefix_len
        return 0

    def _cache_key(self, path: tuple[str, ...]) -> _PathPayloadCacheKey:
        """Return the cross-term payload-cache key for one logical path prefix."""
        return (self.dataset, self.spec.logical_start, tuple(path))


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

            batch_size = self._batch_axis_size(term_batch)
            if self._aux_matches_batch(aux, batch_size):
                term_result = jax.lax.map(
                    lambda item: self._call_term(params, item[0], graph, item[1]),
                    (term_batch, aux),
                    batch_size=self.batch_size,  # batch over aux data too
                )
            else:
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
    def _batch_axis_size(term_batch: PyTree) -> int | None:
        """Return the leading batch-axis size for a mapped term batch when available."""
        leaves = jax.tree_util.tree_leaves(term_batch)
        for leaf in leaves:
            if hasattr(leaf, "shape") and len(leaf.shape) > 0:
                return int(leaf.shape[0])
        return None

    @staticmethod
    def _aux_matches_batch(aux: PyTree | None, batch_size: int | None) -> bool:
        """Return whether an aux tree appears to carry one entry per batch sample."""
        if aux is None or batch_size is None:
            return False
        if not isinstance(aux, Mapping) or _GRAPH_PATH_PAYLOAD_CACHE not in aux:
            return False
        leaves = jax.tree_util.tree_leaves(aux)
        sized_leaves = [leaf for leaf in leaves if hasattr(leaf, "shape") and len(leaf.shape) > 0]
        return len(sized_leaves) > 0 and all(int(leaf.shape[0]) == batch_size for leaf in sized_leaves)

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


def _connected_cycle_edge(graph: FunctionGraph, source: Node, target: Node) -> Edge:
    """Return the unique edge connecting two adjacent cycle nodes."""
    matches = [
        edge
        for edge in graph.edges.values()
        if (edge.source == source and edge.target == target) or (edge.source == target and edge.target == source)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one cycle edge between {source.name!r} and {target.name!r}, found {len(matches)}."
        )
    return matches[0]


def _cycle_edges(graph: FunctionGraph, node_names: Sequence[str] | None) -> tuple[list[Node], list[Edge]]:
    """Resolve and validate clockwise cycle nodes and their adjacent edges."""
    if node_names is None:
        node_names = list(graph.nodes.keys())
    if len(node_names) < 2:
        raise ValueError("cyclic_path_errors requires at least two cycle nodes.")

    nodes = [graph._resolve_node(name) for name in node_names]
    if len({node.name for node in nodes}) != len(nodes):
        raise ValueError("cyclic_path_errors nodes must be unique.")

    edges = [
        _connected_cycle_edge(graph, nodes[idx], nodes[(idx + 1) % len(nodes)])
        for idx in range(len(nodes))
    ]
    return nodes, edges


def _clockwise_path(cycle_edges: Sequence[Edge], start_idx: int, dest_idx: int) -> tuple[str, ...]:
    """Return clockwise edge names from one cycle index to another."""
    n_nodes = len(cycle_edges)
    step_count = (dest_idx - start_idx) % n_nodes
    if step_count == 0:
        step_count = n_nodes
    return tuple(cycle_edges[(start_idx + offset) % n_nodes].name for offset in range(step_count))


def _counter_clockwise_path(cycle_edges: Sequence[Edge], start_idx: int, dest_idx: int) -> tuple[str, ...]:
    """Return counter-clockwise edge names from one cycle index to another."""
    n_nodes = len(cycle_edges)
    step_count = (start_idx - dest_idx) % n_nodes
    if step_count == 0:
        step_count = n_nodes
    return tuple(cycle_edges[(start_idx - 1 - offset) % n_nodes].name for offset in range(step_count))


def _shortest_seed_path(
    cycle_edges: Sequence[Edge],
    dataset_indices: tuple[int, int],
    target_idx: int,
) -> tuple[int, tuple[str, ...]]:
    """Return the nearest dataset endpoint and shortest cycle path used to seed an outside node."""
    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    for endpoint_idx in dataset_indices:
        cw = _clockwise_path(cycle_edges, endpoint_idx, target_idx)
        ccw = _counter_clockwise_path(cycle_edges, endpoint_idx, target_idx)
        if endpoint_idx == target_idx:
            candidates.append((0, endpoint_idx, ()))
        else:
            candidates.append((len(cw), endpoint_idx, cw))
            candidates.append((len(ccw), endpoint_idx, ccw))

    best_distance = min(distance for distance, _, _ in candidates)
    best = [(endpoint_idx, path) for distance, endpoint_idx, path in candidates if distance == best_distance]
    if len(best) != 1:
        raise ValueError("Ambiguous shortest seeding path for cyclic_path_errors; provide a less ambiguous cycle.")
    return best[0]


def _cache_key(dataset: str, logical_start: str, path: tuple[str, ...]) -> _PathPayloadCacheKey:
    """Return a path payload cache key."""
    return dataset, logical_start, path


def _path_prefixes(dataset: str, logical_start: str, path: tuple[str, ...]) -> tuple[_PathPayloadCacheKey, ...]:
    """Return all cacheable prefixes for one logical path."""
    return tuple(_cache_key(dataset, logical_start, path[:idx]) for idx in range(1, len(path) + 1))


def _order_cyclic_specs(specs: Sequence[_CyclicPathSpec]) -> list[_CyclicPathSpec]:
    """Greedily order cyclic terms to improve path-prefix cache reuse."""
    pending = list(enumerate(specs))
    ordered: list[_CyclicPathSpec] = []
    live: set[_PathPayloadCacheKey] = set()

    while pending:
        best_position = 0
        best_score: tuple[int, int, int] | None = None
        for position, (original_idx, spec) in enumerate(pending):
            keys = set(spec.cache_keys)

            # Prefer large overlap with upstream cached paths, default to original order to initialize
            score = (len(keys & live), -len(keys - live), -original_idx)

            if best_score is None or score > best_score:
                best_position = position
                best_score = score

        _, spec = pending.pop(best_position)
        ordered.append(spec)
        live.update(spec.cache_keys)

    return ordered


def _finalize_cache_plan(specs: Sequence[_CyclicPathSpec]) -> list[_CyclicPathSpec]:
    """Assign term indices and last-use metadata for generated cache keys."""
    last_use: dict[_PathPayloadCacheKey, int] = {}
    for idx, spec in enumerate(specs):
        for key in spec.cache_keys:
            last_use[key] = idx

    finalized = []
    for idx, spec in enumerate(specs):
        spec.index = idx
        spec.last_use = {key: last_use[key] for key in spec.cache_keys}
        finalized.append(spec)
    return finalized


def _cyclic_path_specs(
    graph: FunctionGraph,
    *,
    nodes: Sequence[str] | None,
    dataset: str | None,
    cache_policy: Literal["none", "last_use"] = "last_use",
) -> tuple[list[_CyclicPathSpec], str]:
    """Build static term path specs for cyclic graph loss generation."""
    cycle_nodes, cycle_edges = _cycle_edges(graph, nodes)
    dataset = dataset or _default_dataset_edge(graph)
    if dataset is None:
        raise ValueError("cyclic_path_errors requires a dataset when the graph has no sampleable edges.")
    dataset_edge = graph._resolve_edge(dataset)
    endpoint_indices = tuple(
        idx
        for idx, node in enumerate(cycle_nodes)
        if node in {dataset_edge.source, dataset_edge.target}
    )
    if len(endpoint_indices) != 2:
        raise ValueError("cyclic_path_errors dataset edge must connect two nodes in the configured cycle.")

    specs: list[_CyclicPathSpec] = []
    for start_idx, start_node in enumerate(cycle_nodes):
        if start_idx in endpoint_indices:
            # Start the path from a dataset-producing node
            logical_start_idx = start_idx
            seed_path: tuple[str, ...] = ()
        else:
            # Otherwise, start from the nearest dataset-producing node, and go along a seed path to the start node
            logical_start_idx, seed_path = _shortest_seed_path(cycle_edges, endpoint_indices, start_idx)

        logical_start = cycle_nodes[logical_start_idx].name
        for dest_idx, dest_node in enumerate(cycle_nodes):
            path_a = seed_path + _clockwise_path(cycle_edges, start_idx, dest_idx)
            path_b = seed_path + _counter_clockwise_path(cycle_edges, start_idx, dest_idx)
            cache_keys = (
                *_path_prefixes(dataset, logical_start, path_a),
                *_path_prefixes(dataset, logical_start, path_b),
            )
            specs.append(
                _CyclicPathSpec(
                    name=f"{start_node.name}->{dest_node.name}",
                    start=start_node.name,
                    dest=dest_node.name,
                    logical_start=logical_start,
                    path_a=path_a,
                    path_b=path_b,
                    cache_keys=tuple(dict.fromkeys(cache_keys)),
                )
            )

    if cache_policy == "last_use":
        specs = _order_cyclic_specs(specs)
    elif cache_policy != "none":
        raise ValueError(f"Unknown cyclic_path_errors cache_policy: {cache_policy!r}")

    return _finalize_cache_plan(specs), dataset_edge.name


def cyclic_path_error_terms(
    graph: FunctionGraph,
    *,
    nodes: Sequence[str] | None = None,
    dataset: str | None = None,
    weight: float = 1.0,
    batch_reduce: UnaryOp | None = "mean",
    batch_size: int | None = None,
    template_paths: Sequence[TreePath] | None = None,
    aux_paths: Sequence[TreePath] | None = None,
    cache_payloads: bool = False,
    cache_policy: Literal["none", "last_use"] = "last_use",
    error_op: BinaryOp | None = None,
    ignore: Sequence[TreePath] | None = None,
) -> list[GraphLossTerm]:
    """
    Generate path-error terms for every ordered pair of nodes in a cyclic graph.

    :param graph: function graph containing a simple cycle
    :param nodes: optional clockwise node ordering; defaults to the graph node order
    :param dataset: edge name/key for the dataset-producing edge; defaults to the first sampleable graph edge
    :param weight: shared scalar weight for all generated terms
    :param batch_reduce: shared batch reduction for all generated terms
    :param batch_size: optional ``jax.lax.map`` batch size for all generated terms
    :param template_paths: optional source paths used to form graph auxiliary templates
    :param aux_paths: optional destination paths for graph auxiliary templates
    :param cache_payloads: whether generated terms should share intermediate path payloads
    :param cache_policy: deterministic term ordering/eviction strategy for shared payload cache
    :param error_op: optional shared override for every generated term's destination-node error operator
    :param ignore: optional shared override for every generated term's destination-node ignored paths
    :return: concrete graph loss terms
    """
    dataset = dataset or _default_dataset_edge(graph)
    if dataset is None:
        raise ValueError("cyclic_path_errors requires a dataset when the graph has no sampleable edges.")
    specs, dataset_edge = _cyclic_path_specs(graph, nodes=nodes, dataset=dataset, cache_policy=cache_policy)
    return [
        GraphLossTerm(
            name=spec.name,
            term=_CyclicPathErrorCallable(
                spec=spec,
                dataset=dataset,
                dataset_edge=dataset_edge,
                template_paths=template_paths,
                aux_paths=aux_paths,
                cache_payloads=cache_payloads,
                error_op=error_op,
                ignore=ignore,
            ),
            dataset=dataset,
            weight=weight,
            batch_reduce=batch_reduce,
            batch_size=batch_size,
        )
        for spec in specs
    ]


_LOSS_TERM_GENERATOR_REGISTRY.update({"commutativity": cyclic_path_error_terms})


class CyclicPathError(BaseModel):
    """
    Evaluate all cyclic path errors as a matrix, or reduce the matrix with a norm.

    Rows correspond to configured cycle start nodes and columns to destination nodes. Path errors use the same
    generation and optional payload-cache behavior as :func:`cyclic_path_error_terms`, but are neither weighted nor
    adaptively balanced.

    :param graph: function graph containing a simple cycle
    :param nodes: optional clockwise node ordering; defaults to graph node order
    :param dataset: edge name/key for the dataset-producing edge; defaults to the first sampleable graph edge
    :param batch_reduce: reduction applied to each path error over its dataset batch
    :param batch_size: optional ``jax.lax.map`` batch size for each path error
    :param template_paths: optional source paths used to form graph auxiliary templates
    :param aux_paths: optional destination paths for graph auxiliary templates
    :param cache_payloads: whether path errors should share intermediate path payloads
    :param cache_policy: deterministic term ordering/eviction strategy for shared payload cache
    :param error_op: optional shared replacement for destination-node error operators
    :param ignore: optional shared replacement for destination-node ignored paths
    :param norm: optional unary operator applied to the complete error matrix
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    graph: Annotated[FunctionGraph, BeforeValidator(from_yaml)] | None = None
    nodes: Sequence[str] | None = None
    dataset: str | None = None
    batch_reduce: UnaryOp | None = "mean"
    batch_size: int | None = None
    template_paths: Annotated[Sequence[TreePath] | None, BeforeValidator(coerce_tree_paths)] = None
    aux_paths: Annotated[Sequence[TreePath] | None, BeforeValidator(coerce_tree_paths)] = None
    cache_payloads: bool = False
    cache_policy: Literal["none", "last_use"] = "last_use"
    error_op: BinaryOp | None = None
    ignore: Annotated[Sequence[TreePath] | None, BeforeValidator(_coerce_optional_tree_paths)] = None
    norm: UnaryOp | None = None
    _terms: tuple[GraphLossTerm, ...] = PrivateAttr(default=())
    _matrix_indices: tuple[tuple[int, int], ...] = PrivateAttr(default=())
    _matrix_shape: tuple[int, int] = PrivateAttr(default=(0, 0))

    @field_validator("batch_reduce", "norm", mode="before")
    @classmethod
    def _coerce_unary_op(cls, value: Any) -> UnaryOp | None:
        if value is None or isinstance(value, UnaryOp):
            return value
        return UnaryOp(value)

    @field_validator("error_op", mode="before")
    @classmethod
    def _coerce_error_op(cls, value: Any) -> BinaryOp | None:
        if value is None or isinstance(value, BinaryOp):
            return value
        return BinaryOp(value)

    @model_validator(mode="after")
    def _check_graph(self) -> "CyclicPathError":
        if self.graph is not None:
            self._build_terms()
        return self

    def _build_terms(self):
        """Generate path-error terms and static matrix coordinates."""
        terms = cyclic_path_error_terms(
            self.graph,
            nodes=self.nodes,
            dataset=self.dataset,
            batch_reduce=self.batch_reduce,
            batch_size=self.batch_size,
            template_paths=self.template_paths,
            aux_paths=self.aux_paths,
            cache_payloads=self.cache_payloads,
            cache_policy=self.cache_policy,
            error_op=self.error_op,
            ignore=self.ignore,
        )
        cycle_nodes, _ = _cycle_edges(self.graph, self.nodes)
        node_indices = {node.name: idx for idx, node in enumerate(cycle_nodes)}

        self._terms = tuple(terms)
        self._matrix_indices = tuple(
            (node_indices[term.term.callable.spec.start], node_indices[term.term.callable.spec.dest])
            for term in terms
        )
        self._matrix_shape = (len(cycle_nodes), len(cycle_nodes))
    
    def bind_graph(self, graph: FunctionGraph):
        self.graph = graph
        self._build_terms()

    def __call__(self, params: Mapping[str, PyTree], batch: Mapping[str, PyTree]) -> jax.Array:
        """
        Evaluate cyclic path errors for a batch.

        :param params: edge-payload parameter patches
        :param batch: batch data passed to generated path-error terms
        :return: cyclic path-error matrix, or its configured norm
        """
        if self.graph is None:
            raise ValueError("Must specify a graph to evaluate cyclic path error")
        params = pytree_resolve_refs(params)
        aux = None
        values = []
        for term in self._terms:
            value, aux = term.raw_value(params, batch, self.graph, aux=aux, return_aux=True)
            values.append(value)

        flat_values = jnp.asarray(values)
        rows, columns = zip(*self._matrix_indices, strict=True)
        matrix = jnp.zeros(self._matrix_shape, dtype=flat_values.dtype).at[jnp.asarray(rows), jnp.asarray(columns)].set(
            flat_values
        )
        return matrix if self.norm is None else self.norm(matrix)


class GraphLoss(BaseModel):
    """
    Loss function for a `FunctionGraph`.

    :param terms: loss terms combined by weighted summation
    :param balancing: optional adaptive term scaling policy
    :param graph: the FunctionGraph, leave as None to defer to `Train.graph`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    terms: Sequence[GraphLossTerm | GraphLossTermGenerator]
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
        self._expand_generators()
        if self.graph is None and any(isinstance(term, GraphLossTermGenerator) for term in self.terms):
            self._refresh_term_cache()
            return self
        self._set_default_term_names()
        if self.graph is not None:
            self._set_default_datasets()
        self._refresh_term_cache()
        return self

    def bind_graph(self, graph: FunctionGraph) -> None:
        """
        Bind a graph and finalize any graph-dependent generated terms.

        :param graph: graph used by generated terms and default dataset inference
        """
        self.graph = graph
        self._expand_generators()
        self._set_default_term_names()
        self._set_default_datasets()
        self._refresh_term_cache()

    def _expand_generators(self) -> None:
        """Expand configured term generators when a graph is available."""
        expanded: list[GraphLossTerm | GraphLossTermGenerator] = []
        changed = False
        for item in self.terms:
            if not isinstance(item, GraphLossTermGenerator):
                expanded.append(item)
                continue
            if self.graph is None:
                expanded.append(item)
                continue
            generated = item(self.graph)
            expanded.extend(GraphLossTerm.model_validate(term) for term in generated)
            changed = True

        if changed:
            self.terms = expanded

    def _set_default_term_names(self) -> None:
        """Assign deterministic names to unnamed loss terms and require uniqueness."""
        names = []
        for idx, term in enumerate(self.terms):
            if isinstance(term, GraphLossTermGenerator):
                continue
            if term.name is None:
                term.name = f"term_{idx}"
            if term.name in names:
                raise ValueError(f"Duplicate GraphLoss term name: {term.name}")
            names.append(term.name)
    
    def _set_default_datasets(self):
        """Grab the first sampleable edge as the default dataset, i.e. typically there is only one."""
        if self.graph is not None:
            _default_edge = _default_dataset_edge(self.graph)
            for term in self.terms:
                if isinstance(term, GraphLossTermGenerator):
                    continue
                if term.dataset is None:
                    term.dataset = _default_edge
            self._refresh_term_cache()

    def _refresh_term_cache(self) -> None:
        """Cache term metadata used during loss evaluation."""
        concrete_terms = [term for term in self.terms if isinstance(term, GraphLossTerm)]
        self._term_names = tuple(term.name or f"term_{idx}" for idx, term in enumerate(concrete_terms))
        self._term_weights = [term.weight for term in concrete_terms]

    @property
    def term_names(self) -> tuple[str, ...]:
        """Stable names for all terms in order."""
        return self._term_names

    def _raw_term_array(self, params: Mapping[str, PyTree], batch: Mapping[str, PyTree]) -> jax.Array:
        """Evaluate raw term values as an ordered JAX array."""
        if len(self.terms) == 0:
            return jnp.asarray([])
        if any(isinstance(term, GraphLossTermGenerator) for term in self.terms):
            raise ValueError("GraphLoss contains unexpanded term generators; bind a FunctionGraph before evaluation.")

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
