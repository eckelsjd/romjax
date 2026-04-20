from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Hashable, Literal

import networkx as nx
from jaxtyping import PyTree
from pydantic import BaseModel, ConfigDict, Field, model_validator

from romjax.typing import ListModel, RoxObject


class Node(BaseModel, Hashable, RoxObject):
    """
    A Node in a FunctionGraph represents a vector space. At the moment, this is essentially just a string identifier.

    Must be hashable to be usable with networkx. Just hashes using the string identifier currently.
    """
    name: str 

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value

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


class Edge(BaseModel, Hashable, RoxObject, ABC):
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


class FunctionGraph(BaseModel, RoxObject):
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

    @staticmethod
    def _copy_aux_cache(
        aux: Mapping[str, Mapping[str, PyTree]] | None,
    ) -> dict[str, dict[str, PyTree]]:
        if aux is None:
            return {}
        out: dict[str, dict[str, PyTree]] = {}
        for edge_name, edge_aux in aux.items():
            out[edge_name] = dict(edge_aux)
        return out

    def push_path(
        self,
        x: PyTree,
        *,
        start: Node | str,
        path: list[str | Edge],
        aux: Mapping[str, Mapping[str, PyTree]] | None = None,
        return_aux: bool = False,
    ) -> PyTree | tuple[PyTree, dict[str, dict[str, PyTree]]]:
        """
        Push one payload along a path of graph edges with transparent auxiliary-data handling.

        Auxiliary cache format:
        ``{edge_name: {"forward": aux_needed_for_forward, "backward": aux_needed_for_backward}}``

        The cache is maintained by the graph:
        - when traversing an edge forward, any produced auxiliary data is stored for that edge's backward direction
        - when traversing an edge backward, any produced auxiliary data is stored for that edge's forward direction

        :param x: payload to propagate along the path
        :param start: starting node for the path
        :param path: ordered edges to traverse (edge names or edge objects)
        :param target: optional expected final node
        :param aux: optional precomputed auxiliary cache
        :param return_aux: if True, return both payload and updated auxiliary cache
        :return: payload at path end, or ``(payload, aux_cache)`` when ``return_aux=True``
        """
        curr_node = start if isinstance(start, Node) else Node(name=start)
        payload = x
        aux_cache = self._copy_aux_cache(aux)

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

            edge_aux_in = aux_cache.get(edge.name, {}).get(direction)
            if direction == "forward":
                payload, edge_aux_out = edge.forward_aux(payload, edge_aux_in)
            else:
                payload, edge_aux_out = edge.backward_aux(payload, edge_aux_in)

            if edge_aux_out is not None:
                edge_cache = aux_cache.setdefault(edge.name, {})
                edge_cache[produced_key] = edge_aux_out

            curr_node = next_node

        if return_aux:
            return payload, aux_cache
        return payload
