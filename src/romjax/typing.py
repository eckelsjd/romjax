from __future__ import annotations

from abc import ABC, abstractmethod
from inspect import Parameter, signature
from pathlib import Path
from typing import Annotated, Any, Iterator, Mapping, MutableMapping, Protocol, get_args
from weakref import WeakKeyDictionary

import lineax as lx
import optimistix as optx
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SerializationInfo,
    TypeAdapter,
    model_serializer,
    model_validator,
)
from pydantic_core import core_schema

__all__ = ['DictModel', 'ListModel', 'LxObject', 'OptxObject', 'IterativeSolver', 'AdjointMethod',
           'romjax_from_file', 'Routine', 'RoutineError']


def romjax_from_file(value: str | Path | bytes | Any) -> Any:
    """Try to load a romjax object from config file. Useful as a pydantic validator."""
    if isinstance(value, str | Path | bytes):
        import romjax
        return romjax.load(value)
    return value


class Routine(BaseModel, ABC):
    """Base class mixin that provides the `run()` method."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError


class RoutineError(RuntimeError):
    """Raised when routines encounter invalid local state."""
    

class DictModel(BaseModel, MutableMapping):
    """Allow dict-like access of pydantic models."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        extra="allow", 
        validate_assignment=True,
        use_enum_values=True
    )

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
        return len(type(self).model_fields) + len(self.model_extra or ())


class ListModel[T: BaseModel](DictModel):
    """
    Allow list- or dict-like access/storage of generic pydantic models (T), and validates them when setting.
    
    Primarily acts like a MutableMapping and a Pydantic model, whose elements are validated pydantic models of type T.
    Also for convenience acts like an ordered list with integer/slice indexing.
    """

    _adapter: TypeAdapter | None = PrivateAttr(default=None)

    def _item_type(self) -> type[BaseModel]:
        args = self.__pydantic_generic_metadata__.get('args', ())
        if args:
            return args[0]

        for cls in type(self).mro():
            meta = getattr(cls, "__pydantic_generic_metadata__", None)
            if not meta:
                continue
            base_args = meta.get("args", ())
            if base_args:
                return base_args[0]

        for base in getattr(type(self), "__orig_bases__", ()):
            base_args = get_args(base)
            if base_args:
                return base_args[0]

        raise TypeError(f"Could not infer item type for {type(self).__name__}")

    def _get_adapter(self) -> TypeAdapter:
        """Helper for using type T as a way to validate internal data when setting."""
        if self._adapter is None:
            self._adapter = TypeAdapter(self._item_type())
        return self._adapter
    
    @model_validator(mode='before')
    @classmethod
    def _from_list(cls, value):
        if isinstance(value, list):
            return cls(value)  # avoids dict unpacking
        return value
    
    @model_serializer(mode="plain")
    def _serialize_as_list(self, info: SerializationInfo) -> list[Any]:
        """Serialize as a list of T-serialized items in key iteration order."""
        adapter = self._get_adapter()
        mode = "json" if info.mode == "json" else "python"
        return [adapter.dump_python(value, mode=mode) for value in self.values()]

    def __init__(self, data: Mapping | list | tuple | None = None, **kwargs):
        super().__init__()
        self.update(data, **kwargs)  # Triggers validation of all elements
    
    def __setitem__(self, key: str | int, value: Any) -> None:
        if isinstance(key, int):
            key = list(self.keys())[key]
        super().__setitem__(str(key), self._get_adapter().validate_python(value))
    
    def __getitem__(self, key) -> T | list[T]:
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
        setattr(obj, "__romjax_spec__", spec)
    except Exception:
        pass


def _get_spec(obj: object) -> dict[str, Any] | None:
    if hasattr(obj, "__romjax_spec__"):
        return getattr(obj, "__romjax_spec__")
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
