from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextvars import ContextVar
from inspect import Parameter, signature
from typing import Annotated, Any, Hashable, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import networkx as nx
from jaxtyping import PyTree
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from romjax.norm import EdgeNormConfig, NormTree
from romjax.operators import BinaryOp
from romjax.tree import TreePath, coerce_tree_paths, pytree_merge
from romjax.typing import ListModel

type EdgePatch = Mapping[Edge, Mapping[str, PyTree]]  # Maps edge names to extra payload dict data
type EdgePyTree = Mapping[Edge, PyTree]               # Maps edge names to more general pytree data


__all__ = ['FunctionGraph', 'Node', 'Edge', 'CompositeEdge']


_EDGE_NORM_CONTEXT: ContextVar[tuple[tuple[int, str], ...]] = ContextVar(
    "romjax_edge_norm_context",
    default=(),
)
    

class Node(BaseModel, Hashable):
    """
    A Node in a FunctionGraph represents a vector space. This is essentially just a string identifier and
    a way to compute the size of errors in the space.

    Error defaults to `sum-square`, which is the squared L2 norm.

    Must be hashable to be usable with networkx. Just hashes using the string identifier.
    """
    name: str
    error_op: BinaryOp = Field(default_factory=lambda: BinaryOp("sum-square"))
    ignore: Annotated[Sequence[TreePath], BeforeValidator(coerce_tree_paths)] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value

    @field_validator("error_op", mode="before")
    @classmethod
    def _coerce_error_op(cls, value):
        if isinstance(value, BinaryOp):
            return value
        return BinaryOp(value)

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) ->str:
        return self.__str__()

    def error(self, value: PyTree, value_hat: PyTree) -> jax.Array:
        """
        Compute the pytree error at this node. Only consider shared paths, and optionally ignore paths via `self.ignore`
        """
        return self.error_op(value, value_hat, ignore=self.ignore)


class Edge(BaseModel, Hashable, ABC):
    """
    An Edge is the abstract class for function mappings between nodes (vector spaces) in a FunctionGraph.

    Must implement forward/backward calls to map vectors (PyTrees) between the source/target nodes.
    Hashable for easy access and consistency with Node via a string identifier.
    Can specify simply as "a->b" for convenience.
    """
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    source: Node
    target: Node
    name: str = ""
    norm: EdgeNormConfig = Field(default_factory=EdgeNormConfig)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for direction in ("forward", "backward"):
            cls._wrap_norm_map(direction)
            cls._wrap_norm_aux_map(direction)

    @classmethod
    def _wrap_norm_map(cls, direction: Literal["forward", "backward"]) -> None:
        method = getattr(cls, direction, None)
        if method is None:
            return
        raw_method = getattr(method, "_romjax_norm_raw", method)

        def wrapped(self: "Edge", x: PyTree) -> PyTree:
            if self._norm_context_active(direction):
                return raw_method(self, x)
            x_norm = self._apply_norm_stage(direction, "pre", x, aux=None)
            token = self._enter_norm_context(direction)
            try:
                y = raw_method(self, x_norm)
            finally:
                _EDGE_NORM_CONTEXT.reset(token)
            return self._apply_norm_stage(direction, "post", y, aux=None)

        wrapped.__name__ = getattr(method, "__name__", direction)
        wrapped.__qualname__ = getattr(method, "__qualname__", wrapped.__qualname__)
        wrapped.__doc__ = getattr(method, "__doc__", None)
        wrapped.__isabstractmethod__ = getattr(method, "__isabstractmethod__", False)
        wrapped._romjax_norm_wrapped = True
        wrapped._romjax_norm_raw = raw_method
        setattr(cls, direction, wrapped)

    @classmethod
    def _wrap_norm_aux_map(cls, direction: Literal["forward", "backward"]) -> None:
        method_name = f"{direction}_aux"
        method = getattr(cls, method_name, None)
        if method is None:
            return
        raw_method = getattr(method, "_romjax_norm_raw", method)

        def wrapped(self: "Edge", x: PyTree, aux: PyTree | None = None, *args: Any, **kwargs: Any):
            if self._norm_context_active(direction):
                return raw_method(self, x, aux, *args, **kwargs)
            norm_aux, edge_aux = self._split_norm_aux(aux)
            x_norm = self._apply_norm_stage(direction, "pre", x, aux=self._stage_norm_aux(norm_aux, direction, "pre"))
            token = self._enter_norm_context(direction)
            try:
                y, aux_out = raw_method(self, x_norm, edge_aux, *args, **kwargs)
            finally:
                _EDGE_NORM_CONTEXT.reset(token)
            y_norm = self._apply_norm_stage(direction, "post", y, aux=self._stage_norm_aux(norm_aux, direction, "post"))
            return y_norm, aux_out

        wrapped.__name__ = getattr(method, "__name__", method_name)
        wrapped.__qualname__ = getattr(method, "__qualname__", wrapped.__qualname__)
        wrapped.__doc__ = getattr(method, "__doc__", None)
        wrapped.__isabstractmethod__ = getattr(method, "__isabstractmethod__", False)
        wrapped._romjax_norm_wrapped = True
        wrapped._romjax_norm_raw = raw_method
        setattr(cls, method_name, wrapped)

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, value):
        if isinstance(value, str):
            if "->" not in value:
                raise ValueError("Can't create an edge from provided string. Must be of the form 'source->target'")
            _split = value.split("->")
            return {"source": _split[0].strip(), "target": _split[1].strip()}

        return value

    @model_validator(mode="after")
    def _set_default_name(self):
        if self.name == "" or self.name is None:
            self.name = f"{self.source}->{self.target}"
        return self

    def _norm_context_active(self, direction: Literal["forward", "backward"]) -> bool:
        """Return whether this edge direction is already inside a normalization wrapper."""
        return (id(self), direction) in _EDGE_NORM_CONTEXT.get()

    def _enter_norm_context(self, direction: Literal["forward", "backward"]):
        """Mark an edge direction as active while the raw implementation runs."""
        current = _EDGE_NORM_CONTEXT.get()
        return _EDGE_NORM_CONTEXT.set(current + ((id(self), direction),))

    @staticmethod
    def _split_norm_aux(aux: PyTree | None) -> tuple[PyTree | None, PyTree | None]:
        """Split reserved normalization aux overrides from edge-specific auxiliary state."""
        if not isinstance(aux, Mapping) or "norm" not in aux:
            return None, aux
        edge_aux = dict(aux)
        norm_aux = edge_aux.pop("norm")
        if len(edge_aux) == 0:
            return norm_aux, None
        return norm_aux, edge_aux

    @staticmethod
    def _stage_norm_aux(
        norm_aux: PyTree | None,
        direction: Literal["forward", "backward"],
        stage: Literal["pre", "post"],
    ) -> PyTree | None:
        """Return runtime normalization overrides for one direction/stage."""
        if not isinstance(norm_aux, Mapping):
            return None
        direction_aux = norm_aux.get(direction)
        if isinstance(direction_aux, Mapping):
            return direction_aux.get(stage)
        return None

    def _apply_norm_stage(
        self,
        direction: Literal["forward", "backward"],
        stage: Literal["pre", "post"],
        x: PyTree,
        aux: PyTree | None = None,
    ) -> PyTree:
        """Apply one configured normalization stage."""
        hook = getattr(self, f"_{direction}_{stage}_norm")
        try:
            hook_signature = signature(hook)
        except (TypeError, ValueError):
            return hook(x, aux=aux)

        aux_param = hook_signature.parameters.get("aux")
        accepts_aux = aux_param is not None and aux_param.kind in {
            Parameter.POSITIONAL_OR_KEYWORD,
            Parameter.KEYWORD_ONLY,
        }
        if accepts_aux:
            return hook(x, aux=aux)
        return hook(x)

    def resolve_norms(self) -> None:
        """
        Resolve and cache any artifact-backed normalization trees configured on this edge.

        Call this before entering ``jax.jit``/``jax.grad`` regions when a norm stage references an HDF5 artifact.
        Runtime edge evaluation also resolves lazily if this method was not called explicitly.
        """
        for direction in (self.norm.forward, self.norm.backward):
            for norm in (direction.pre, direction.post):
                if isinstance(norm, NormTree):
                    norm.resolve_root()

    def _forward_pre_norm(self, x: PyTree, aux: PyTree | None = None) -> PyTree:
        """
        Apply configured forward pre-normalization.

        :param x: edge input payload
        :param aux: optional runtime norm-constant overrides
        :return: normalized payload
        """
        norm = self.norm.forward.pre
        return x if norm is None else norm(x, aux=aux)

    def _forward_post_norm(self, x: PyTree, aux: PyTree | None = None) -> PyTree:
        """
        Apply configured forward post-normalization.

        :param x: edge output payload
        :param aux: optional runtime norm-constant overrides
        :return: normalized payload
        """
        norm = self.norm.forward.post
        return x if norm is None else norm(x, aux=aux)

    def _backward_pre_norm(self, x: PyTree, aux: PyTree | None = None) -> PyTree:
        """
        Apply configured backward pre-normalization.

        :param x: edge input payload
        :param aux: optional runtime norm-constant overrides
        :return: normalized payload
        """
        norm = self.norm.backward.pre
        return x if norm is None else norm(x, aux=aux)

    def _backward_post_norm(self, x: PyTree, aux: PyTree | None = None) -> PyTree:
        """
        Apply configured backward post-normalization.

        :param x: edge output payload
        :param aux: optional runtime norm-constant overrides
        :return: normalized payload
        """
        norm = self.norm.backward.post
        return x if norm is None else norm(x, aux=aux)

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False
    
    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.source}->{self.target}"

    def __call__(
        self,
        x: PyTree,
        direction: Literal["forward", "backward"] = "forward",
        *,
        aux: PyTree | None = None,
        return_aux: bool = False,
    ) -> PyTree | tuple[PyTree, PyTree | None]:
        if direction == "forward":
            if return_aux:
                return self.forward_aux(x, aux)
            return self.forward(x)
        if direction == "backward":
            if return_aux:
                return self.backward_aux(x, aux)
            return self.backward(x)
        raise ValueError(f"Unknown direction {direction}")
    
    @abstractmethod
    def forward(self, x: PyTree) -> PyTree:
        """Maps a vector `x` from source to target."""
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, x: PyTree) -> PyTree:
        """Maps a vector `x` from target to source."""
        raise NotImplementedError

    def forward_aux(self, x: PyTree, aux: PyTree | None = None) -> tuple[PyTree, PyTree | None]:
        """
        Auxiliary-data aware forward map.

        Edges that do not need auxiliary data can keep this default implementation.
        """
        del aux
        return self.forward(x), None

    def backward_aux(self, x: PyTree, aux: PyTree | None = None) -> tuple[PyTree, PyTree | None]:
        """
        Auxiliary-data aware backward map.

        Edges that do not need auxiliary data can keep this default implementation.
        """
        del aux
        return self.backward(x), None
    

class IdentityEdge(Edge):
    """Default edge mapping that acts as the identity in both directions."""

    def forward(self, x: PyTree) -> PyTree:
        return x

    def backward(self, x: PyTree) -> PyTree:
        return x


class CompositeEdge(Edge):
    """
    Reusable graph-native edge defined as a path through existing graph edges.

    The configured ``path`` is interpreted with the same mixed forward/backward semantics as
    :meth:`FunctionGraph.push_path`.

    :param source: composite path start node
    :param target: composite path end node
    :param name: edge identifier
    :param path: ordered list of existing graph edge names
    """

    path: list[str]

    _graph: "FunctionGraph | None" = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_path_config(self):
        if len(self.path) == 0:
            raise ValueError("CompositeEdge path must contain at least one edge name.")
        return self

    def _bind_graph(self, graph: "FunctionGraph") -> None:
        """Bind the parent graph after construction."""
        self._graph = graph

    def _require_graph(self) -> "FunctionGraph":
        if self._graph is None:
            raise ValueError(f"Composite edge {self.name!r} is not bound to a FunctionGraph.")
        return self._graph

    def _run_path(
        self,
        x: PyTree,
        *,
        path: list[str] | None = None,
        start: Node,
        aux: EdgePatch | None = None,
        edge_payload_patches: EdgePatch | None = None,
        return_aux: bool = False,
        composite_stack: tuple[str, ...] = (),
    ) -> PyTree | tuple[PyTree, EdgePatch]:
        graph = self._require_graph()
        return graph._push_path_internal(
            x,
            self.path if path is None else path,
            start=start,
            aux=aux,
            edge_payload_patches=edge_payload_patches,
            return_aux=return_aux,
            composite_stack=composite_stack + (self.name,),
        )

    def forward(self, x: PyTree) -> PyTree:
        """Evaluate the configured path from ``source`` to ``target``."""
        ret, aux = self.forward_aux(x, aux=None)
        return ret

    def backward(self, x: PyTree) -> PyTree:
        """Evaluate the configured path from ``target`` back to ``source``."""
        ret, aux = self.backward_aux(x, aux=None)
        return ret

    def forward_aux(
        self, 
        x: PyTree, 
        aux: PyTree | None = None, 
        edge_payload_patches: EdgePatch | None = None, 
        composite_stack: tuple[str, ...] = ()
    ) -> tuple[PyTree, PyTree | None]:
        """
        Evaluate the forward composite path and return the graph-managed auxiliary cache.

        :param x: payload at ``source``
        :param aux: optional precomputed graph aux cache
        :return: ``(payload, aux_cache)``
        """
        return self._run_path(x, start=self.source, aux=aux, return_aux=True, 
                              edge_payload_patches=edge_payload_patches, composite_stack=composite_stack)

    def backward_aux(
        self, 
        x: PyTree, 
        aux: PyTree | None = None,
        edge_payload_patches: EdgePatch | None = None, 
        composite_stack: tuple[str, ...] = ()
        ) -> tuple[PyTree, PyTree | None]:
        """
        Evaluate the backward composite path and return the graph-managed auxiliary cache.

        :param x: payload at ``target``
        :param aux: optional precomputed graph aux cache
        :return: ``(payload, aux_cache)``
        """
        return self._run_path(x, path=list(reversed(self.path)), start=self.target, aux=aux, return_aux=True,
                              edge_payload_patches=edge_payload_patches, composite_stack=composite_stack)


# Equivalent to the alias NodeList = ListModel[Node], but now others can use this by importing it
class NodeList(ListModel[Node]):
    """A list of graph nodes."""
    
    def __repr__(self):
        return f"[{', '.join([node.name for node in self.values()])}]"


class EdgeList(ListModel[Edge]):
    """A list of graph edges.

    Untyped edge inputs (e.g., dicts or strings) are parsed as :class:`IdentityEdge` by default.
    """

    def __repr__(self):
        return f"[{', '.join([repr(edge) for edge in self.values()])}]"

    def __setitem__(self, key: str | int, value: Any) -> None:
        if not isinstance(value, Edge):
            value = IdentityEdge.model_validate(value)
        super().__setitem__(key, value)


class FunctionGraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    nodes: NodeList = Field(default_factory=NodeList)
    edges: EdgeList = Field(default_factory=EdgeList)

    def __repr__(self):
        return f"FunctionGraph{repr(self.edges)}"

    @model_validator(mode='after')
    def _add_extra_nodes_from_edges(self):
        """If edges have source/targets not in the provided node list, add them"""
        for edge in self.edges.values():
            if edge.source not in self.nodes:
                self.nodes.append(edge.source)
            if edge.target not in self.nodes:
                self.nodes.append(edge.target)
        
        return self

    @model_validator(mode="after")
    def _bind_edge_nodes_to_graph_nodes(self):
        """Point edge endpoints at the canonical graph nodes when they are configured."""
        for edge in self.edges.values():
            edge.source = self._resolve_node(edge.source)
            edge.target = self._resolve_node(edge.target)
        return self

    @model_validator(mode="after")
    def _bind_and_validate_composite_edges(self):
        for edge in self.edges.values():
            if isinstance(edge, CompositeEdge):
                edge._bind_graph(self)

        for edge in self.edges.values():
            if isinstance(edge, CompositeEdge):
                self._validate_composite_edge(edge)

        return self

    def graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes.values())
        graph.add_edges_from(
            [(edge.source, edge.target, {"object": edge}) for edge in self.edges.values()]
        )
        return graph

    def _resolve_edge(self, edge_like: str | Edge) -> Edge:
        """Resolve edge handle from an edge object or edge name."""
        if isinstance(edge_like, Edge):
            return edge_like
        if edge_like in self.edges:
            return self.edges[edge_like]
        for edge in self.edges.values():
            if edge.name == edge_like:
                return edge
        raise KeyError(f"Edge {edge_like!r} not found in graph.")

    def _resolve_node(self, node_like: Node | str) -> Node:
        """Resolve a node handle to the canonical graph node when available."""
        if isinstance(node_like, Node):
            node_name = node_like.name
        else:
            node_name = node_like
        if node_name in self.nodes:
            return self.nodes[node_name]
        return node_like if isinstance(node_like, Node) else Node(name=node_name)

    @staticmethod
    def _copy_aux_cache(aux: EdgePatch | None) -> EdgePatch:
        if aux is None:
            return {}
        out: dict[str, dict[str, PyTree]] = {}
        for edge_name, edge_aux in aux.items():
            out[edge_name] = dict(edge_aux)
        return out

    @staticmethod
    def _normalize_path(path: list[str | Edge] | tuple[str | Edge, ...] | None) -> list[str | Edge]:
        """Normalize a path-like input to a concrete edge list."""
        if path is None:
            raise ValueError("Path may not be None.")
        if isinstance(path, str):
            return [path]
        return list(path)

    def _resolve_start_node(self, path: list[str | Edge], start: Node | str | None) -> Node:
        """Resolve the start node for a path, including empty loopback paths."""
        if start is not None:
            return self._resolve_node(start)
        if len(path) == 0:
            raise ValueError("Loopback paths require an explicit start node.")
        return self._resolve_node(self._resolve_edge(path[0]).source)

    def _path_end_node(self, path: list[str | Edge], start: Node | str | None = None) -> Node:
        """Resolve the terminal node reached by following a path from ``start``."""
        normalized = self._normalize_path(path)
        curr_node = self._resolve_start_node(normalized, start)
        for edge_ref in normalized:
            curr_node = self._step_path_node(curr_node, self._resolve_edge(edge_ref))
        return curr_node

    def _step_path_node(self, curr_node: Node, edge: Edge) -> Node:
        """Advance one graph step using ``push_path`` connectivity semantics."""
        if edge.source == curr_node:
            return self._resolve_node(edge.target)
        if edge.target == curr_node:
            return self._resolve_node(edge.source)
        raise ValueError(
            f"Path discontinuity at edge {edge.name!r}: edge does not connect to current node {curr_node!r}."
        )

    def _validate_composite_edge(self, edge: CompositeEdge, stack: tuple[str, ...] = ()) -> None:
        """
        Validate one composite edge against graph connectivity and recursive composition rules.

        :param edge: composite edge to validate
        :param stack: active composite recursion stack for cycle detection
        """
        if edge.name in stack:
            cycle = " -> ".join((*stack, edge.name))
            raise ValueError(f"Composite edge recursion cycle detected: {cycle}")

        curr_node = edge.source
        for edge_name in edge.path:
            if edge_name == edge.name:
                raise ValueError(f"Composite edge {edge.name!r} cannot include itself in its own path.")

            try:
                step_edge = self._resolve_edge(edge_name)
            except KeyError as exc:
                raise ValueError(
                    f"Composite edge {edge.name!r} references unknown edge {edge_name!r}."
                ) from exc

            curr_node = self._step_path_node(curr_node, step_edge)

            if isinstance(step_edge, CompositeEdge):
                self._validate_composite_edge(step_edge, stack=stack + (edge.name,))

        if curr_node != edge.target:
            raise ValueError(
                f"Composite edge {edge.name!r} ends at node {curr_node!r}, expected target {edge.target!r}."
            )

    def _push_path_internal(
        self,
        payload: PyTree,
        path: list[str | Edge],
        *,
        start: Node | str | None = None,
        aux: EdgePatch | None = None,
        edge_payload_patches: EdgePatch | None = None,
        return_aux: bool = False,
        composite_stack: tuple[str, ...] = (),
    ) -> PyTree | tuple[PyTree, EdgePatch]:
        """Internal path executor with recursive composite-edge cycle tracking."""
        curr_node = self._resolve_start_node(path, start)
        aux_cache = self._copy_aux_cache(aux)

        if len(path) == 0:
            if return_aux:
                return payload, aux_cache
            return payload

        for edge_ref in path:
            edge = self._resolve_edge(edge_ref)

            if edge.source == curr_node:
                direction: Literal["forward", "backward"] = "forward"
                next_node = edge.target
                produced_key = "backward"
            elif edge.target == curr_node:
                direction = "backward"
                next_node = edge.source
                produced_key = "forward"
            else:
                raise ValueError(
                    f"Path discontinuity at edge {edge.name!r}: edge does not connect to current node {curr_node!r}."
                )

            if isinstance(edge, CompositeEdge) and edge.name in composite_stack:
                cycle = " -> ".join((*composite_stack, edge.name))
                raise ValueError(f"Composite edge recursion cycle detected: {cycle}")

            payload_in = payload
            edge_payload_patch = None if edge_payload_patches is None else edge_payload_patches.get(edge.name)
            if edge_payload_patch is not None:
                if not isinstance(payload_in, Mapping):
                    raise TypeError(
                        f"Edge payload patches require mapping payloads, but edge {edge.name!r} received "
                        f"{type(payload_in).__name__}."
                    )
                payload_in = pytree_merge(payload_in, edge_payload_patch)

            composite = isinstance(edge, CompositeEdge)
            if composite:
                edge_aux_in = aux_cache
                extra_kwargs = {"edge_payload_patches": edge_payload_patches, "composite_stack": composite_stack}
            else:
                edge_aux_in = aux_cache.get(edge.name, {}).get(direction)
                extra_kwargs = {}

            if direction == "forward":
                payload, edge_aux_out = edge.forward_aux(payload_in, edge_aux_in, **extra_kwargs)
            else:
                payload, edge_aux_out = edge.backward_aux(payload_in, edge_aux_in, **extra_kwargs)

            if composite:
                aux_cache = edge_aux_out
            else:
                if edge_aux_out is not None:
                    edge_cache = aux_cache.setdefault(edge.name, {})
                    edge_cache[produced_key] = edge_aux_out

            curr_node = next_node

        if return_aux:
            return payload, aux_cache
        return payload

    def push_path(
        self,
        payload: PyTree,
        path: list[str | Edge] | tuple[str | Edge, ...],
        *,
        start: Node | str | None = None,
        aux: EdgePatch | None = None,
        edge_payload_patches: EdgePatch | None = None,
        return_aux: bool = False,
    ) -> PyTree | tuple[PyTree, EdgePatch]:
        """
        Push one payload along a path of graph edges with transparent auxiliary-data handling.

        Auxiliary cache format:
        ``{edge_name: {"forward": aux_needed_for_forward, "backward": aux_needed_for_backward}}``

        The cache is maintained by the graph:
        - when traversing an edge forward, any produced auxiliary data is stored for that edge's backward direction
        - when traversing an edge backward, any produced auxiliary data is stored for that edge's forward direction
        - when ``edge_payload_patches`` provides a patch for an edge name, that patch is merged into the current
          payload only for that edge evaluation, including recursive traversals inside :class:`CompositeEdge`

        :param payload: payload to propagate along the path
        :param path: ordered edges to traverse (edge names or edge objects)
        :param start: starting node for the path (defaults to first node of first edge in path)
        :param aux: optional precomputed auxiliary cache
        :param edge_payload_patches: optional payload patches keyed by edge name
        :param return_aux: if True, return both payload and updated auxiliary cache
        :return: payload at path end, or ``(payload, aux_cache)`` when ``return_aux=True``
        """
        normalized_path = self._normalize_path(path)
        return self._push_path_internal(
            payload,
            normalized_path,
            start=start,
            aux=aux,
            edge_payload_patches=edge_payload_patches,
            return_aux=return_aux,
        )

    def resolve_norms(self) -> None:
        """
        Resolve and cache artifact-backed normalization trees for all graph edges.

        This is useful before tracing graph computations with ``jax.jit`` or ``jax.grad`` so file IO does not happen
        during tracing.
        """
        for edge in self.edges.values():
            edge.resolve_norms()

    def path_error(
        self,
        payload: PyTree,
        path_a: list[str | Edge] | tuple[str | Edge, ...] | None,
        path_b: list[str | Edge] | tuple[str | Edge, ...] | None,
        *,
        start: Node | str | None = None,
        aux_a: EdgePatch | None = None,
        aux_b: EdgePatch | None = None,
        edge_payload_patches: EdgePatch | None = None,
        error_op: BinaryOp | None = None,
        ignore: set | None = None,
    ) -> jax.Array:
        """
        Propagate one payload along two paths and compare the results at the common destination node.

        ``None`` denotes a zero path and an empty path denotes an identity path.  A path used as an identity
        loopback inherits its start node from the other path, so loopback paths do not require an explicit ``start``.
        If both paths are absent or empty, the error is zero.
        """
        if path_a is None and path_b is None:
            return jnp.asarray(0.0)

        normalized_a = self._normalize_path(path_a) if path_a is not None else None
        normalized_b = self._normalize_path(path_b) if path_b is not None else None
        nonempty_path = next(
            (path for path in (normalized_a, normalized_b) if path),
            None,
        )
        if nonempty_path is None:
            return jnp.asarray(0.0)

        if start is None:
            start_a = self._resolve_start_node(normalized_a or nonempty_path, None)
            start_b = self._resolve_start_node(normalized_b or nonempty_path, None)
        else:
            start_a = start_b = self._resolve_node(start)

        end_a = self._path_end_node(normalized_a, start=start_a) if normalized_a else None
        end_b = self._path_end_node(normalized_b, start=start_b) if normalized_b else None
        if end_a is not None and end_b is not None and end_a != end_b:
            raise ValueError(
                f"Path error requires matching destinations, but got {end_a!r} and {end_b!r}."
            )

        out_a = self.push_path(
            payload,
            [] if normalized_a is None else normalized_a,
            start=start_a,
            aux=aux_a,
            edge_payload_patches=edge_payload_patches,
        )
        if normalized_b is not None:
            out_b = self.push_path(
                payload,
                normalized_b,
                start=start_b,
                aux=aux_b,
                edge_payload_patches=edge_payload_patches,
            )
        else:
            out_b = jax.tree_util.tree_map(
                lambda leaf: jnp.zeros_like(jnp.asarray(leaf)) if eqx.is_array(leaf) else leaf,
                out_a,
            )

        if normalized_a is None:
            out_a = jax.tree_util.tree_map(
                lambda leaf: jnp.zeros_like(jnp.asarray(leaf)) if eqx.is_array(leaf) else leaf,
                out_b,
            )

        end_node = end_a if end_a is not None else end_b
        assert end_node is not None
        op = end_node.error_op if error_op is None else BinaryOp(error_op)
        ignore = ignore or end_node.ignore
        
        return op(out_a, out_b, ignore=ignore)

    def reconstruction_error(
        self,
        payload: PyTree,
        path: list[str | Edge] | tuple[str | Edge, ...],
        *,
        start: Node | str | None = None,
        aux: EdgePatch | None = None,
        edge_payload_patches: EdgePatch | None = None,
        error_op: BinaryOp | None = None,
        ignore: set | None = None,
    ) -> jax.Array:
        """
        Compute reconstruction error by comparing a loopback path against forward-then-backward traversal.

        The payload is pushed forward along ``path`` and then backward along the reversed path back to its start.
        """
        normalized_path = self._normalize_path(path)
        start_node = self._resolve_start_node(normalized_path, start)
        forward_out, aux_cache = self.push_path(
            payload,
            normalized_path,
            start=start_node,
            aux=aux,
            edge_payload_patches=edge_payload_patches,
            return_aux=True,
        )
        destination = self._path_end_node(normalized_path, start=start_node)
        reconstructed = self.push_path(
            forward_out,
            list(reversed(normalized_path)),
            start=destination,
            aux=aux_cache,
            edge_payload_patches=edge_payload_patches,
        )

        op = start_node.error_op if error_op is None else BinaryOp(error_op)
        ignore = ignore or start_node.ignore

        return op(payload, reconstructed, ignore=ignore)
