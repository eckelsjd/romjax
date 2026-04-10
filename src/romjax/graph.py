from typing import Hashable, Literal, Any
from abc import abstractmethod, ABC

import networkx as nx
from pydantic import BaseModel, model_validator, Field, ConfigDict
from jaxtyping import PyTree

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
                raise ValueError(f"Can't create an edge from provided string. Must be of the form 'source->target'")
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
        if isinstance(other, Node):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            return False
    
    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.name}: {self.source}->{self.target}"

    def __call__(self, x: PyTree, direction: Literal["forward", "backward"] = "forward") -> PyTree:
        if direction == "forward":
            return self.forward(x)
        elif direction == "backward":
            return self.backward(x)
        else:
            raise ValueError(f"Unknown direction {direction}")
    
    @abstractmethod
    def forward(self, x: PyTree) -> PyTree:
        """Maps a vector `x` from source to target."""
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, x: PyTree) -> PyTree:
        """Maps a vector `x` from target to source."""
        raise NotImplementedError
    

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
