from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, Callable, Iterator, MutableMapping, Protocol, Annotated, TypeAlias
from weakref import WeakKeyDictionary

import lineax as lx
import optimistix as optx
from jax.typing import ArrayLike
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, AfterValidator
from pydantic_core import core_schema

__all__ = ['PyTree', 'Coordinates', 'ForcingCallable', 'BoundaryCallable', 'DictModel', 
           'LxObject', 'OptxObject', 'IterativeSolver', 'AdjointMethod']

type PyTree = Any  # Python containers for use with jax

# For PDEs
type Coordinates = tuple[ArrayLike, ...]
type ForcingCallable = Callable[[PyTree, PyTree], ArrayLike]
type BoundaryCallable = Callable[[PyTree], PyTree]
type InitialCallable = Callable[[Coordinates], ArrayLike]


class DictModel(BaseModel, MutableMapping):
    """Allow dict-like access of pydantic models."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        extra="allow", 
        validate_assignment=True,
        use_enum_values=True
    )

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


class ModuleObjectSpec(BaseModel):
    """Specification for constructing module objects from configs."""

    name: str
    opts: dict[str, Any] = Field(default_factory=dict)


class ModuleObjectBuilder(Protocol):
    """Callable protocol for constructing an object from name and opts."""

    def __call__(self, name: str, opts: dict[str, Any]) -> Any: ...


_SPEC_REGISTRY: WeakKeyDictionary[object, dict[str, Any]] = WeakKeyDictionary()


def build_from_module(module) -> ModuleObjectBuilder:
    """
    Create a builder that constructs objects from a Python module by name.

    :param module: Python module containing the target classes.
    :return: Builder callable.
    """
    def _build(name: str, opts: dict[str, Any]) -> Any:
        if not hasattr(module, name):
            raise ValueError(f"Module '{module.__name__}' does not contain attribute '{name}'")
        return getattr(module, name)(**opts)
    return _build


def _store_spec(obj: object, name: str, opts: dict[str, Any]) -> None:
    spec = {"name": name, "opts": opts}
    try:
        _SPEC_REGISTRY[obj] = spec
    except TypeError:
        pass
    try:
        setattr(obj, "__romtools_spec__", spec)
    except Exception:
        pass


def _get_spec(obj: object) -> dict[str, Any] | None:
    if hasattr(obj, "__romtools_spec__"):
        return getattr(obj, "__romtools_spec__")
    try:
        return _SPEC_REGISTRY.get(obj)
    except TypeError:
        return None


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_serialize_value(v) for v in value)

    spec = _get_spec(value)
    if spec is None:
        try:
            return _serialize_value(_infer_spec(value))
        except Exception:
            return value
    return {
        "name": spec["name"],
        "opts": {k: _serialize_value(v) for k, v in spec["opts"].items()},
    }


def _infer_spec(value: Any) -> dict[str, Any]:
    name = value.__class__.__name__
    opts: dict[str, Any] = {}
    try:
        sig = signature(value.__class__.__init__)
    except (TypeError, ValueError):
        return {"name": name, "opts": opts}

    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        try:
            if hasattr(value, param.name):
                opts[param.name] = _serialize_value(getattr(value, param.name))
        except Exception:
            continue

    return {"name": name, "opts": opts}


def _module_object_schema(builder: ModuleObjectBuilder, opts_adapter: TypeAdapter | None) -> core_schema.CoreSchema:
    def validate(value: Any) -> Any:
        if isinstance(value, dict):
            spec = ModuleObjectSpec.model_validate(value)
            opts = spec.opts
            if opts_adapter is not None:
                opts = opts_adapter.validate_python(opts)
            obj = builder(spec.name, opts)
            _store_spec(obj, spec.name, opts)
            return obj
        return value

    def serialize(value: Any) -> Any:
        spec = _get_spec(value)
        if spec is None:
            spec = _infer_spec(value)
        return _serialize_value(spec)

    return core_schema.no_info_plain_validator_function(
        validate,
        serialization=core_schema.plain_serializer_function_ser_schema(serialize),
    )


def module_object_type(builder: ModuleObjectBuilder, *, opts_adapter: TypeAdapter | None = None) -> type:
    """
    Create a Pydantic-compatible type that constructs module objects from dict specs.

    :param builder: Callable that constructs an object from name and opts.
    :param opts_adapter: Optional adapter to validate nested opts.
    :return: A custom type usable in Pydantic models.
    """
    class ModuleObject:
        @classmethod
        def __get_pydantic_core_schema__(cls, _source, _handler):
            return _module_object_schema(builder, opts_adapter)

    return ModuleObject


def _require_type(value: Any, required_type: type):
    if not isinstance(value, required_type):
        raise TypeError(f"Expected {required_type}, got {type(value).__name__}")
    return value


# Probably a thousand better ways to do this, but here we are
# Essentially I just wanted custom pydantic validation/serialization for third-party objects
type LxObject = module_object_type(build_from_module(lx))
type OptxObject = module_object_type(build_from_module(optx), opts_adapter=TypeAdapter(dict[str, LxObject | Any]))    
type IterativeSolver = Annotated[OptxObject, AfterValidator(lambda v: _require_type(v, optx.AbstractIterativeSolver))]
type AdjointMethod = Annotated[OptxObject, AfterValidator(lambda v: _require_type(v, optx.AbstractAdjoint))]
