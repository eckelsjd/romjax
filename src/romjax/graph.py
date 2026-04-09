from typing import Hashable, Callable, Literal

import networkx as nx
from pydantic import BaseModel, model_validator, Field, ConfigDict
from jaxtyping import PyTree

from romjax.typing import ListModel, RoxObject


class Node(BaseModel, Hashable, RoxObject):
    """Hashable node to be used with networkx. Essentially just a string with some extra validated fields."""
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


class Edge(BaseModel, Hashable, RoxObject):
    """Hashable edge to be used with networkx. Uses pydantic validation to support convenient a->b specification."""
    source: Node
    target: Node
    name: str = ""
    forward: Callable[[PyTree], PyTree] | None = None
    backward: Callable[[PyTree], PyTree] | None = None

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
            if self.forward is None:
                raise RuntimeError(f"Must define a forward function for edge {self.name}")
            return self.forward(x)
        elif direction == "backward":
            if self.backward is None:
                raise RuntimeError(f"Must define a backward function for edge {self.name}")
            return self.backward(x)
        else:
            raise ValueError(f"Unknown direction {direction}")


# Equivalent to the alias NodeList = ListModel[Node], but now others can use this by importing it
class NodeList(ListModel[Node]):
    """A list of graph nodes."""
    pass


class EdgeList(ListModel[Edge]):
    """A list of graph edges."""
    pass


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
