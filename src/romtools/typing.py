from typing import Any, Iterator, MutableMapping, Callable

from pydantic import BaseModel, ConfigDict
from jax.typing import ArrayLike


type PyTree = Any  # Python containers for use with jax

# For PDEs
type Coordinates = tuple[ArrayLike, ...]
type ForcingCallable = Callable[[PyTree, PyTree], ArrayLike]
type BoundaryCallable = Callable[[PyTree], PyTree]


class DictModel(BaseModel, MutableMapping):
    """Allow dict-like access of pydantic models."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow", validate_assignment=True)

    @classmethod
    def yaml_tag(cls) -> str:
        """YAML tag used by YamlLoader for this model class."""
        return f"!model:{cls.__module__}.{cls.__name__}"

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __delitem__(self, key: str) -> None:
        if not hasattr(self, key):
            raise KeyError(key)
        delattr(self, key)

    def __iter__(self) -> Iterator[str]:
        for k, _ in super().__iter__():
            yield k

    def __len__(self) -> int:
        return len(dict(self))
