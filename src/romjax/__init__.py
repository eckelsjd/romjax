"""Reduced-order modeling in jax.

- Author - Joshua Eckels (eckelsjd@umich.edu)
- License - MIT
"""
__version__ = "0.0.1"


from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from functools import partial as _partial
from importlib import import_module as _import_module
from inspect import getattr_static as _getattr_static
from os import PathLike as _PathLike
from pathlib import Path as _Path
from types import BuiltinFunctionType as _BuiltinFunctionType
from types import FunctionType as _FunctionType
from typing import IO as _IO
from typing import TYPE_CHECKING
from typing import Any as _Any
from typing import Literal as _Literal
from typing import Optional as _Optional
from typing import Type as _Type
from typing import Union as _Union

import yaml as _yaml
from pydantic import BaseModel as _BaseModel
from pydantic.fields import PydanticUndefined as _PydanticUndefined

type _Stream = _Union[str, bytes, bytearray, _PathLike[str], _IO[_Any]]

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "random": ("romjax.rng", None),
    "tree": ("romjax.tree", None),
    "CompositeEdge": ("romjax.graph", "CompositeEdge"),
    "FunctionGraph": ("romjax.graph", "FunctionGraph"),
    "ExplicitModel": ("romjax.model", "ExplicitModel"),
    "FilterModel": ("romjax.model", "FilterModel"),
    "ImplicitModel": ("romjax.model", "ImplicitModel"),
    "eqx_evaluate": ("romjax.model", "eqx_evaluate"),
    "LinearProjection": ("romjax.nn", "LinearProjection"),
    "ImplicitIterativeGalerkin": ("romjax.pde", "ImplicitIterativeGalerkin"),
    "gridplot": ("romjax.plotting", "gridplot"),
    "Poisson2D": ("romjax.poisson", "Poisson2D"),
    "gen_keys": ("romjax.rng", "gen_keys"),
    "PyTreeSampler": ("romjax.rng", "PyTreeSampler"),
    "NearSolutionSampler": ("romjax.rng", "NearSolutionSampler"),
    "get_unary_operator": ("romjax.tree", "get_unary_operator"),
    "get_error_operator": ("romjax.tree", "get_error_operator"),
    "get_tree_operator": ("romjax.tree", "get_tree_operator"),
    "DictModel": ("romjax.typing", "DictModel"),
    "ListModel": ("romjax.typing", "ListModel"),
    "CallableModel": ("romjax.typing", "CallableModel"),
    "ThirdPartyType": ("romjax.typing", "ThirdPartyType"),
    "Routine": ("romjax.routine", "Routine"),
    "RoutineConfig": ("romjax.routine", "RoutineConfig"),
    "RoutineError": ("romjax.routine", "RoutineError"),
    "load_h5": ("romjax.utils", "load_h5"),
    "save_h5": ("romjax.utils", "save_h5"),
    "DataGeneration": ("romjax.data_gen", "DataGeneration"),
    "Train": ("romjax.train", "Train"),
    "GraphLoss": ("romjax.train", "GraphLoss"),
    "GraphTest": ("romjax.train", "GraphTest"),
    "GraphDataLoader": ("romjax.train", "GraphDataLoader"),
    "BatchDataLoader": ("romjax.train", "BatchDataLoader")
}

__all__ = list(_LAZY_EXPORTS.keys())

#ruff: noqa F401
if TYPE_CHECKING:
    from . import rng as random
    from . import tree as tree
    from .data_gen import DataGeneration
    from .graph import CompositeEdge, FunctionGraph
    from .model import ExplicitModel, FilterModel, ImplicitModel, eqx_evaluate
    from .nn import LinearProjection
    from .pde import ImplicitIterativeGalerkin
    from .plotting import gridplot
    from .poisson import Poisson2D
    from .rng import NearSolutionSampler, PyTreeSampler, gen_keys
    from .routine import Routine, RoutineConfig, RoutineError
    from .train import GraphDataLoader, GraphLoss, GraphTest, Train, BatchDataLoader
    from .tree import get_error_operator, get_tree_operator, get_unary_operator
    from .typing import CallableModel, DictModel, ListModel, ThirdPartyType
    from .utils import load_h5, save_h5

def __getattr__(name: str) -> _Any:
    """Lazily load internal package symbols to avoid circular dependencies."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = _import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


class ConfigLoader(_ABC):
    """Common interface for loading and dumping configs to/from files, streams, etc."""

    @classmethod
    @_abstractmethod
    def load(cls, stream: _Stream, **kwargs: _Any) -> _Any:
        """Load an object from a stream. If a file path is given, will attempt to open the file."""
        raise NotImplementedError

    @classmethod
    @_abstractmethod
    def dump(cls, obj: _Any, stream: _Stream, **kwargs: _Any) -> _Any:
        """Save an object to a stream. If a file path is given, will attempt to write to the file."""
        raise NotImplementedError


class YamlLoader(ConfigLoader):
    """YAML configs. 

    - Represent pydantic classes with `!pd:Path.To.SubClass`. Must be subclass of BaseModel.
    - Represent any root-level builtin `romjax` object with `!romx:SomeObject`
    - Supports basic Pydantic model_dump() to dictionary.
    - Supports !!python/name tag for functions and other importable names
    """
    PYDANTIC_TAG = "!pd:"
    ROMX_TAG = "!romx:"

    @classmethod
    def get_tag(cls, data: _Any) -> str:
        t = type(data)
        import romjax
        tag = cls.ROMX_TAG if hasattr(romjax, t.__name__) else cls.PYDANTIC_TAG
        return f"{tag}{t.__module__}.{t.__name__}"

    @classmethod
    def get_loader(cls) -> _Type[_yaml.SafeLoader]:
        """Return a custom yaml loader that handles callables and pydantic models."""
        class _Loader(_yaml.SafeLoader):
            pass

        def _import_python_name(value: str) -> _Any:
            parts = value.split(".")
            for i in range(len(parts) - 1, 0, -1):
                module_name = ".".join(parts[:i])
                attr_path = parts[i:]
                try:
                    module = _import_module(module_name)
                except ModuleNotFoundError:
                    continue

                obj: _Any = module
                for attr in attr_path:
                    try:
                        obj = getattr(obj, attr)
                    except AttributeError:
                        try:
                            obj = _getattr_static(obj, attr)
                        except AttributeError:
                            fields = getattr(obj, "model_fields", None)
                            if fields is None or attr not in fields:
                                raise
                            obj = fields[attr].default
                            if obj is _PydanticUndefined:
                                raise
                        if isinstance(obj, staticmethod | classmethod):
                            obj = obj.__func__
                return obj

            raise ValueError(f"Invalid python/name tag: {value!r}")

        def _construct_python_name_multi(loader: _yaml.SafeLoader, tag_suffix: str, node: _yaml.Node) -> _Any:
            value = tag_suffix or loader.construct_scalar(node)
            return _import_python_name(value)

        def _construct_python_name(loader: _yaml.SafeLoader, node: _yaml.Node) -> _Any:
            """Any importable name (e.g. functions)."""
            value = loader.construct_scalar(node)
            return _construct_python_name_multi(loader, value, node)

        def _construct_base_model(loader: _yaml.SafeLoader, tag_suffix: str, node: _yaml.Node,
                                  default_module: str = "__main__") -> _BaseModel:
            """Essentially just a convenience to automatically construct Pydantic models when loading."""
            if not tag_suffix:
                raise ValueError("Missing class path in YAML tag.")
            module_name, _, class_name = tag_suffix.rpartition(".")
            
            if module_name == '':
                module_name = default_module  # Try to load from a default module
            if not module_name or not class_name:
                raise ValueError(f"Invalid pydantic tag suffix: {tag_suffix!r}")
            
            module = _import_module(module_name)
            cls_obj = getattr(module, class_name, None)

            if cls_obj is None:
                raise ValueError(f"Class not found: {tag_suffix!r}")
            if not isinstance(cls_obj, type) or not issubclass(cls_obj, _BaseModel):
                raise TypeError(f"Tagged class is not an instance of required pydantic BaseModel: {tag_suffix!r}")
            
            if isinstance(node, _yaml.MappingNode):
                data = loader.construct_mapping(node, deep=True)
                return cls_obj(**data)
            if isinstance(node, _yaml.SequenceNode):
                data = loader.construct_sequence(node, deep=True)
                return cls_obj(data)
            data = loader.construct_scalar(node)
            if isinstance(data, str):
                try:
                    data = _import_python_name(data)
                except Exception:
                    pass

            return cls_obj(data)

        _Loader.add_constructor("tag:yaml.org,2002:python/name", _construct_python_name)
        _Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", _construct_python_name_multi)
        _Loader.add_multi_constructor(cls.PYDANTIC_TAG, _construct_base_model)
        _Loader.add_multi_constructor(cls.ROMX_TAG, _partial(_construct_base_model, default_module="romjax"))
        return _Loader

    @classmethod
    def get_dumper(cls) -> _Type[_yaml.SafeDumper]:
        """Return a custom yaml dumper that handles callables and pydantic models."""
        class _Dumper(_yaml.SafeDumper):
            pass

        def _represent_python_name(dumper: _yaml.SafeDumper, data: _Any) -> _yaml.Node:
            name = f"{data.__module__}.{data.__qualname__}"
            return dumper.represent_scalar("tag:yaml.org,2002:python/name", name)

        def _represent_base_model(dumper: _yaml.SafeDumper, data: _BaseModel) -> _yaml.Node:
            """Essentially a convenience that defers serialization to pydantic model_dump"""
            tag = cls.get_tag(data)
            payload = data.model_dump()
            if isinstance(payload, list | tuple):
                return dumper.represent_sequence(tag, payload)
            if isinstance(payload, _FunctionType | _BuiltinFunctionType):
                name = f"{payload.__module__}.{payload.__qualname__}"
                return dumper.represent_scalar(tag, name)
            if isinstance(payload, (str, int, float, bool)) or payload is None:
                return dumper.represent_scalar(tag, str(payload) if payload is not None else "null")
            return dumper.represent_mapping(tag, payload)

        _Dumper.add_representer(_FunctionType, _represent_python_name)
        _Dumper.add_representer(_BuiltinFunctionType, _represent_python_name)
        _Dumper.add_multi_representer(_BaseModel, _represent_base_model)
        return _Dumper

    @classmethod
    def load(cls, stream: _Stream, **kwargs: _Any) -> _Any:
        """Load a configuration from a yaml-like stream. Small wrapper around yaml.load
        
        :param stream: A string, path, file-stream, byte-stream, or similar.
        :return: the configuration
        """
        if "Loader" not in kwargs:
            kwargs["Loader"] = cls.get_loader()

        if isinstance(stream, (_PathLike, _Path)):
            with _Path(stream).open("r", encoding="utf-8") as fh:
                return _yaml.load(fh, **kwargs)
        if isinstance(stream, str):
            try:
                exists = _Path(stream).exists()
                looks_like_path = (
                    stream.endswith((".yml", ".yaml"))
                    or _Path(stream).is_absolute()
                    or stream.startswith(".")
                )
            except OSError:
                return _yaml.load(stream, **kwargs)
            else: 
                if exists or looks_like_path:
                    with _Path(stream).open("r", encoding="utf-8") as fh:
                        return _yaml.load(fh, **kwargs)
            return _yaml.load(stream, **kwargs)
        if isinstance(stream, (bytes, bytearray)):
            return _yaml.load(stream.decode("utf-8"), **kwargs)
        if hasattr(stream, "read"):
            return _yaml.load(stream, **kwargs)
        raise TypeError("Unsupported stream type for YAML load.")

    @classmethod
    def dump(cls, obj: _Any, stream: _Stream | None = None, **kwargs: _Any) -> _Optional[str]:
        """Dump a configuration to a yaml-like stream. Small wrapper around yaml.dump
        
        :param obj: the yaml configuration
        :param stream: A string, path, file-stream, byte-stream, or similar.
        """
        if "Dumper" not in kwargs:
            kwargs["Dumper"] = cls.get_dumper()

        if stream is None:
            return _yaml.dump(obj, **kwargs)
        if isinstance(stream, (_PathLike, _Path)) or isinstance(stream, str):
            with _Path(stream).open("w", encoding="utf-8") as fh:
                _yaml.dump(obj, fh, **kwargs)
            return None
        if hasattr(stream, "write"):
            _yaml.dump(obj, stream, **kwargs)
            return None
        raise TypeError("Unsupported stream type for YAML dump.")


def load(stream: _Stream, method: _Literal["yaml"] = "yaml", **kwargs):
    """Load an object from a stream. If a file path is given, will attempt to open the file. Only yaml supported."""
    if method == "yaml":
        return YamlLoader.load(stream, **kwargs)
    else:
        raise ValueError(f"Load method '{method}' unknown")


def dump(obj: _Any, stream: _Stream, method: _Literal["yaml"] = "yaml", **kwargs):
    """Save an object to a stream. If a file path is given, will attempt to write to the file. Only yaml supported."""
    if method == "yaml":
        return YamlLoader.dump(obj, stream, **kwargs)
    else:
        raise ValueError(f"Dump method '{method}' unknown")
    
