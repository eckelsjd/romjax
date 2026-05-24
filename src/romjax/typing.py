from __future__ import annotations

import sys
from copy import deepcopy
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Annotated, Any, Callable, Iterator, Mapping, MutableMapping, Sequence, TypeVar, get_args
from weakref import WeakKeyDictionary

import lineax as lx
import optax
import optimistix as optx
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    PrivateAttr,
    SerializationInfo,
    TypeAdapter,
    model_serializer,
    model_validator,
)

__all__ = ['DictModel', 'ListModel', 'CallableModel', 'ThirdPartyType', 'WriteStream',
           'from_yaml', 'from_registry', 'require_type', 'from_module_spec', 'to_module_spec']


_SPEC_REGISTRY: WeakKeyDictionary[object, dict[str, Any]] = WeakKeyDictionary()
_SPEC_ID_REGISTRY: dict[int, tuple[type, dict[str, Any]]] = {}
_THIRD_PARTY_MODULES = (lx, optx, optax)
type _DefaultModules = str | ModuleType | Sequence[str | ModuleType] | None

T = TypeVar("T")


def require_type(required_type: type, value: Any):
    if not isinstance(value, required_type):
        raise ValueError(f"Expected {required_type}, got {type(value).__name__}")
    return value


def require_attr(required_attr: str, value: Any):
    if not hasattr(value, required_attr):
        raise ValueError(f"Expected attribute '{required_attr}'")
    return value


def from_yaml(value: str | Path | bytes | Any) -> Any:
    """Try to load from yaml. Useful as a pydantic validator."""
    if isinstance(value, str | Path | bytes):
        import romjax
        return romjax.load(value)
    return value


def from_registry(registry: dict[str, T], key: str | Any) -> T:
    """Try to load an object by key from a registry. Useful as a pydantic before validator."""
    if isinstance(key, Mapping):
        for selector in ("callable", "name"):
            value = key.get(selector)
            if isinstance(value, str) and value in registry:
                target = registry[value]
                opts = {k: v for k, v in key.items() if k != selector}
                if isinstance(target, type):
                    return target(**opts)
                if opts:
                    return {"callable": target, **opts}
                return target
        return key

    if not isinstance(key, str):
        return key
    
    if key not in registry:
        raise KeyError(f"Option '{key}' not found in registry: {list(registry.keys())}")

    if isinstance(registry[key], type):
        return registry[key]()
    
    return registry[key]
    

class CallableModel(BaseModel):
    """
    Parent class for a pydantic callable. Allows configuration of a callable and kwargs via extra pydantic fields.
    
    When calling a child class, the flow will go `child(*args)->super(**extra)->child.callable(*args, **extra)`,
    effectively storing kwargs via pydantic extra fields at creation, then using them at runtime.

    The intended use case is to allow configuring a whole function f(**kwargs) in a single object, and then still have
    the usual call syntax f() at runtime. Can think of this as a validated callable pydantic model.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        validate_assignment=True, 
        validate_default=True,
        extra="allow"
    )

    callable: Callable

    @model_validator(mode="before")
    @classmethod
    def _from_callable(cls, value):
        if callable(value):
            return {"callable": value}
        return value
    
    @property
    def opts(self):
        return self.model_extra
    
    def __call__(self, *args, **kwargs):
        opts = dict(self.opts)
        opts.update(kwargs)
        callable_obj = self.callable
        try:
            first_param = next(iter(signature(callable_obj).parameters.values()))
        except (TypeError, ValueError, StopIteration):
            first_param = None

        if isinstance(first_param, Parameter) and first_param.name == "self":
            return callable_obj(self, *args, **opts)

        return callable_obj(*args, **opts)


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
    

# Three ways to try and store/retrieve a spec: 1) by ID, 2) by weak ref, 3) by monkeypatch special __romjax attr
def _store_spec(value: object, spec: dict[str, Any]) -> None:
    """Attach a normalized construction spec to a validated object when possible."""
    _SPEC_ID_REGISTRY[id(value)] = (type(value), deepcopy(spec))

    try:
        _SPEC_REGISTRY[value] = spec
    except TypeError:
        pass

    try:
        setattr(value, "__romjax_third_party_spec__", spec)
    except Exception:
        pass


def _get_spec(value: object) -> dict[str, Any] | None:
    """Retrieve a previously cached construction spec."""
    if hasattr(value, "__romjax_third_party_spec__"):
        return deepcopy(getattr(value, "__romjax_third_party_spec__"))

    spec_entry = _SPEC_ID_REGISTRY.get(id(value))
    if spec_entry is not None:
        spec_type, spec = spec_entry
        if type(value) is spec_type:
            return deepcopy(spec)

    try:
        spec = _SPEC_REGISTRY.get(value)
    except TypeError:
        spec = None

    return deepcopy(spec) if spec is not None else None


def _is_primitive(value: Any) -> bool:
    return isinstance(value, str | int | float | bool | type(None))


def _qualname(value: Any) -> str:
    return f"{value.__module__}.{value.__qualname__}"


def _public_qualname(value: Any) -> str:
    module_name = value.__module__.split(".", maxsplit=1)[0]
    return f"{module_name}.{value.__qualname__}"


def _is_importable_object(value: Any) -> bool:
    if not isinstance(value, FunctionType | BuiltinFunctionType | type):
        return False

    try:
        module = import_module(value.__module__)
    except ModuleNotFoundError:
        return False

    resolved: Any = module
    for attr in value.__qualname__.split("."):
        if attr == "<locals>" or not hasattr(resolved, attr):
            return False
        resolved = getattr(resolved, attr)

    return resolved is value


def _normalize_default_modules(default_modules: _DefaultModules = None) -> tuple[ModuleType, ...]:
    """Normalize default module hints used when resolving short third-party names."""
    if default_modules is None:
        return ()
    if isinstance(default_modules, str | ModuleType):
        default_modules = (default_modules,)

    modules: list[ModuleType] = []
    for module in default_modules:
        modules.append(import_module(module) if isinstance(module, str) else module)
    return tuple(modules)


def _resolve_name(name: str, *, default_modules: _DefaultModules = None) -> Any:
    """Resolve an import path, optionally using a parent module as context."""
    if "." in name:
        module_name, _, attr_path = name.rpartition(".")
        module = import_module(module_name)
        resolved: Any = module
        for attr in attr_path.split("."):
            resolved = getattr(resolved, attr)
        return resolved

    for module in _normalize_default_modules(default_modules):
        if hasattr(module, name):
            return getattr(module, name)

    matches = [
        getattr(module, name)
        for module in _THIRD_PARTY_MODULES
        if isinstance(module, ModuleType) and hasattr(module, name)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous third-party name {name!r}; use a fully qualified import path.")

    raise ValueError(f"Could not resolve third-party name {name!r}.")


def _try_normalize_spec_string(value: str, *, default_modules: _DefaultModules = None) -> dict[str, Any] | None:
    try:
        _resolve_name(value, default_modules=default_modules)
    except Exception:
        return None

    return {"name": value}


def _normalize_value(value: Any, *, default_modules: _DefaultModules = None) -> Any:
    """Normalize nested values, converting nested specs and cached objects recursively."""
    if _is_primitive(value):
        if isinstance(value, str):
            spec = _try_normalize_spec_string(value, default_modules=default_modules)
            if spec is not None:
                return spec
        return value

    spec = _get_spec(value)
    if spec is not None:
        return spec

    if isinstance(value, Mapping):
        if "name" in value:
            return _normalize_spec_data(value, default_modules=default_modules)
        return {str(key): _normalize_value(val, default_modules=default_modules) for key, val in value.items()}

    if isinstance(value, list):
        return [_normalize_value(item, default_modules=default_modules) for item in value]

    if isinstance(value, tuple):
        return tuple(_normalize_value(item, default_modules=default_modules) for item in value)

    if _is_importable_object(value):
        return {"name": _qualname(value), "args": [], "kwargs": {}}

    return value


def _normalize_spec_data(data: str | Mapping[str, Any], *, default_modules: _DefaultModules = None) -> dict[str, Any]:
    """Normalize shorthand or mapping specs into a canonical recursive form."""
    if isinstance(data, str):
        spec = _try_normalize_spec_string(data, default_modules=default_modules)
        if spec is None:
            raise ValueError(f"Could not resolve third-party shorthand {data!r}.")
        return spec

    if not isinstance(data, Mapping):
        raise TypeError(f"Expected third-party spec mapping or string, got {type(data).__name__}")

    if "name" not in data:
        raise ValueError("Third-party spec mappings must include a 'name' field.")

    target = _resolve_name(str(data["name"]), default_modules=default_modules)
    nested_default_module = getattr(target, "__module__", None)

    spec: dict[str, Any] = {"name": str(data["name"])}
    if "args" in data:
        spec["args"] = [_normalize_value(arg, default_modules=nested_default_module) for arg in data.get("args", ())]
    if "kwargs" in data:
        spec["kwargs"] = {
            str(key): _normalize_value(val, default_modules=nested_default_module)
            for key, val in data.get("kwargs", {}).items()
        }
    return spec


def _construct_value(value: Any, *, default_modules: _DefaultModules = None) -> Any:
    """Recursively construct nested spec values."""
    if _is_primitive(value):
        if isinstance(value, str):
            spec = _try_normalize_spec_string(value, default_modules=default_modules)
            if spec is None:
                return value
            return _construct_spec(spec, default_modules=default_modules)
        return value

    spec = _get_spec(value)
    if spec is not None:
        return value

    if isinstance(value, Mapping):
        if "name" in value:
            return _construct_spec(
                _normalize_spec_data(value, default_modules=default_modules),
                default_modules=default_modules,
            )
        return {str(key): _construct_value(val, default_modules=default_modules) for key, val in value.items()}

    if isinstance(value, list):
        return [_construct_value(item, default_modules=default_modules) for item in value]

    if isinstance(value, tuple):
        return tuple(_construct_value(item, default_modules=default_modules) for item in value)

    return value


def _construct_spec(spec: dict[str, Any], *, default_modules: _DefaultModules = None) -> Any:
    """Construct an object from a normalized spec and cache its serialized form."""
    target = _resolve_name(spec["name"], default_modules=default_modules)
    target_module = getattr(target, "__module__", None)
    args = [_construct_value(arg, default_modules=target_module) for arg in spec.get("args", ())]
    kwargs = {
        key: _construct_value(val, default_modules=target_module)
        for key, val in spec.get("kwargs", {}).items()
    }
    value = target(*args, **kwargs)
    _store_spec(value, spec)
    return value


def from_module_spec(value: Any, *, default_modules: _DefaultModules = None):
    """Recursively convert a module spec into a third-party object."""
    if isinstance(value, str | Mapping):
        try:
            spec = _normalize_spec_data(value, default_modules=default_modules)
        except Exception:
            return value
        return _construct_spec(spec, default_modules=default_modules)
    return value


def to_module_spec(value: Any):
    """Recursively serialize a third-party object into a module spec."""
    if _is_primitive(value):
        return value

    spec = _get_spec(value)
    if spec is not None:
        return spec

    if isinstance(value, list):
        return [to_module_spec(item) for item in value]

    if isinstance(value, tuple):
        return tuple(to_module_spec(item) for item in value)

    if isinstance(value, Mapping):
        return {str(key): to_module_spec(val) for key, val in value.items()}

    if _is_importable_object(value):
        return {"name": _qualname(value), "args": [], "kwargs": {}}

    try:
        return _infer_spec(value)
    except Exception:
        return value


def _infer_spec(value: Any) -> dict[str, Any]:
    """Best-effort serialization for third-party objects without cached specs."""
    kwargs: dict[str, Any] = {}
    try:
        init_params = signature(type(value).__init__).parameters.values()
    except (TypeError, ValueError):
        init_params = ()

    for param in init_params:
        if param.name == "self":
            continue
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue
        if not hasattr(value, param.name):
            continue

        try:
            kwargs[param.name] = to_module_spec(getattr(value, param.name))
        except Exception:
            continue

    return {"name": _public_qualname(type(value)), "kwargs": kwargs}


class _ThirdPartyType:
    """Pydantic-compatible third-party object spec annotation factory."""

    def __call__(self, *, default_modules: _DefaultModules = None):
        return Annotated[
            Any,
            BeforeValidator(lambda value: from_module_spec(value, default_modules=default_modules)),
            PlainSerializer(to_module_spec),
        ]

    def __or__(self, other: Any) -> Any:
        return self() | other

    def __ror__(self, other: Any) -> Any:
        return other | self()

    def __get_pydantic_core_schema__(self, source: Any, handler: Any) -> Any:
        del source
        return handler.generate_schema(self())


ThirdPartyType = _ThirdPartyType()


def _resolve_stream_or_path(value):
    if value in ("stdout", "sys.stdout", sys.stdout):
        return sys.stdout
    if value in ("stderr", "sys.stderr", sys.stderr):
        return sys.stderr
    if isinstance(value, str):
        return Path(value)  # treat all plain strings as files
    return value


type WriteStream = Annotated[
    ThirdPartyType(),
    BeforeValidator(_resolve_stream_or_path),
    # AfterValidator(partial(require_attr, "write"))
]
