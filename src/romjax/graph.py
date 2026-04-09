from typing import Hashable, Mapping, Any, Callable, Literal

from pydantic import BaseModel, model_validator, field_validator, Field, ConfigDict, ValidationInfo
from jaxtyping import PyTree

from romjax.typing import DictModel, RoxObject


class Node(Hashable, BaseModel, RoxObject):
    """Hashable node to be used with networkx. Allows pydantic validation of other node configs."""
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


class Edge(Hashable, BaseModel, RoxObject):
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

    @field_validator("name", mode="after")
    @classmethod
    def _non_empty_name(cls, value, info: ValidationInfo) -> str:
        if value == "" or value is None:
            return f"{str(info.data['source'])}->{str(info.data['target'])}"
        
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
        

class NodeList(DictModel):
    """
    Allow list- or dict-like access of graph nodes with validation.
    
    Primarily acts like a MutableMapping and a Pydantic model, whose elements are validated Nodes.
    Also for convenience acts like an ordered list with integer/slice indexing.
    """

    def __init__(self, data: Mapping | list | tuple | None = None, **kwargs):
        super().__init__()
        self.update(data, **kwargs)  # Triggers validation of all elements
    
    def __setitem__(self, key: str | int, value: Any) -> None:
        if isinstance(key, int):
            key = list(self.keys())[key]
        super().__setitem__(str(key), Node.model_validate(value))
    
    def __getitem__(self, key) -> Node | list[Node]:
        if isinstance(key, list | tuple):
            return [self.__getitem__(ele) for ele in key]
        if isinstance(key, int | slice):
            return list(self.values())[key]
        return super().__getitem__(str(key))
    
    def __delitem__(self, key) -> None:
        if isinstance(key, list | tuple):
            _keys = list(self.keys())
            _del_keys = [_keys[ele] for ele in key]
            for ele in _del_keys:
                self.__delitem__(ele)
        elif isinstance(key, int | slice):
            ele = list(self.keys())[key]
            if isinstance(ele, list):
                for item in ele:
                    super().__delitem__(item)
            else:
                super().__delitem__(ele)
        else:
            super().__delitem__(str(key))

    # Override dict to work with lists or dicts
    def update(self, data: Mapping | list | tuple | None = None, **kwargs):
        if data is not None:
            if isinstance(data, Mapping):
                super().update(data)
            else:
                data = [data] if not isinstance(data, list | tuple) else data
                for ele in data:
                    self.__setitem__(str(ele), ele)
        if kwargs:
            super().update(kwargs)
    
    # Some extra list-like methods for convenience (since we can)
    def append(self, data):
        self.update(data)

    def extend(self, data):
        self.update(data)
    
    def index(self, key):
        for i, k in enumerate(self.keys()):
            if k == key:
                return i
        raise ValueError(f"'{key}' is not in list")


class FunctionGraph(BaseModel, RoxObject):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    nodes: NodeList = Field(default_factory=NodeList)

    def graph(self):
        # Build nx graph on the fly from nodes/edges
        pass