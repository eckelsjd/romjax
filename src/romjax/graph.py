from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Hashable, Literal

import jax
import networkx as nx
from jaxtyping import PyTree
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from romjax.tree import TreeErrorOperator, get_tree_operator, pytree_merge
from romjax.typing import ListModel

type EdgePatch = Mapping[Edge, Mapping[str, PyTree]]  # Maps edge names to extra payload dict data
type EdgePyTree = Mapping[Edge, PyTree]               # Maps edge names to more general pytree data


__all__ = ['FunctionGraph', 'Node', 'Edge', 'CompositeEdge']
    

class Node(BaseModel, Hashable):
    """
    A Node in a FunctionGraph represents a vector space. This is essentially just a string identifier and
    a way to compute the size of errors in the space.

    Error defaults to `mean-relative`, which averages per-leaf relative error.

    Must be hashable to be usable with networkx. Just hashes using the string identifier.
    """
    name: str
    error_op: TreeErrorOperator = Field(default_factory=lambda: get_tree_operator("mean-relative"))

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value

    @field_validator("error_op", mode="before")
    @classmethod
    def _coerce_error_op(cls, value):
        if isinstance(value, TreeErrorOperator):
            return value
        return get_tree_operator(value)

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
        Compute the pytree error at this node. Only consider entries present in `value_hat`
        (i.e. ignore extra entries in `value`).
        """
        def _select_matching(tree: PyTree, template: PyTree) -> PyTree:
            if isinstance(template, Mapping):
                return {key: _select_matching(tree[key], template[key]) for key in template}
            if isinstance(template, tuple):
                return tuple(_select_matching(tree[idx], template[idx]) for idx in range(len(template)))
            if isinstance(template, list):
                return [_select_matching(tree[idx], template[idx]) for idx in range(len(template))]
            return tree

        value_filter = _select_matching(value, value_hat)
        return self.error_op(value_filter, value_hat)


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
        return f"{self.name}: {self.source}->{self.target}"

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

    def forward_aux(self, x: PyTree, aux: PyTree | None = None) -> tuple[PyTree, PyTree | None]:
        """
        Evaluate the forward composite path and return the graph-managed auxiliary cache.

        :param x: payload at ``source``
        :param aux: optional precomputed graph aux cache
        :return: ``(payload, aux_cache)``
        """
        return self._run_path(x, start=self.source, aux=aux, return_aux=True)

    def backward_aux(self, x: PyTree, aux: PyTree | None = None) -> tuple[PyTree, PyTree | None]:
        """
        Evaluate the backward composite path and return the graph-managed auxiliary cache.

        :param x: payload at ``target``
        :param aux: optional precomputed graph aux cache
        :return: ``(payload, aux_cache)``
        """
        return self._run_path(x, path=list(reversed(self.path)), start=self.target, aux=aux, return_aux=True)


# Equivalent to the alias NodeList = ListModel[Node], but now others can use this by importing it
class NodeList(ListModel[Node]):
    """A list of graph nodes."""
    pass


class EdgeList(ListModel[Edge]):
    """A list of graph edges.

    Untyped edge inputs (e.g., dicts or strings) are parsed as :class:`IdentityEdge` by default.
    """

    def __setitem__(self, key: str | int, value: Any) -> None:
        if not isinstance(value, Edge):
            value = IdentityEdge.model_validate(value)
        super().__setitem__(key, value)


class FunctionGraph(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    nodes: NodeList = Field(default_factory=NodeList)
    edges: EdgeList = Field(default_factory=EdgeList)

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

            edge_aux_in = aux_cache.get(edge.name, {}).get(direction)
            if direction == "forward":
                if isinstance(edge, CompositeEdge):
                    payload, aux_cache = edge._run_path(
                        payload_in,
                        start=edge.source,
                        aux=aux_cache,
                        edge_payload_patches=edge_payload_patches,
                        return_aux=True,
                        composite_stack=composite_stack,
                    )
                else:
                    payload, edge_aux_out = edge.forward_aux(payload_in, edge_aux_in)
            else:
                if isinstance(edge, CompositeEdge):
                    payload, aux_cache = edge._run_path(
                        payload_in,
                        path=list(reversed(edge.path)),
                        start=edge.target,
                        aux=aux_cache,
                        edge_payload_patches=edge_payload_patches,
                        return_aux=True,
                        composite_stack=composite_stack,
                    )
                else:
                    payload, edge_aux_out = edge.backward_aux(payload_in, edge_aux_in)

            if not isinstance(edge, CompositeEdge) and edge_aux_out is not None:
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

    def path_error(
        self,
        payload: PyTree,
        path_a: list[str | Edge] | tuple[str | Edge, ...],
        path_b: list[str | Edge] | tuple[str | Edge, ...],
        *,
        start: Node | str | None = None,
        aux_a: EdgePatch | None = None,
        aux_b: EdgePatch | None = None,
        edge_payload_patches: EdgePatch | None = None,
    ) -> jax.Array:
        """
        Propagate one payload along two paths and compare the results at the common destination node.

        Empty paths are treated as loopbacks and require an explicit ``start`` node.
        """
        normalized_a = self._normalize_path(path_a)
        normalized_b = self._normalize_path(path_b)
        start_node = self._resolve_start_node(normalized_a if normalized_a else normalized_b, start)
        end_a = self._path_end_node(normalized_a, start=start_node)
        end_b = self._path_end_node(normalized_b, start=start_node)
        if end_a != end_b:
            raise ValueError(
                f"Path error requires matching destinations, but got {end_a!r} and {end_b!r}."
            )

        out_a = self.push_path(
            payload,
            normalized_a,
            start=start_node,
            aux=aux_a,
            edge_payload_patches=edge_payload_patches,
        )
        out_b = self.push_path(
            payload,
            normalized_b,
            start=start_node,
            aux=aux_b,
            edge_payload_patches=edge_payload_patches,
        )
        return end_a.error(out_a, out_b)

    def reconstruction_error(
        self,
        payload: PyTree,
        path: list[str | Edge] | tuple[str | Edge, ...],
        *,
        start: Node | str | None = None,
        aux: EdgePatch | None = None,
        edge_payload_patches: EdgePatch | None = None,
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
        return start_node.error(payload, reconstructed)
