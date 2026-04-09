"""Reduced-order modeling in jax.

- Author - Joshua Eckels (eckelsjd@umich.edu)
- License - MIT
"""
__version__ = "0.0.1"
__all__ = ['ConfigLoader', 'YamlLoader', 'DictModel', 'ImplicitModel', 'gridplot', 'load_h5', 'save_h5',
           'gen_keys', 'train', 'random']

from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from importlib import import_module as _import_module
from os import PathLike as _PathLike
from pathlib import Path as _Path
from types import BuiltinFunctionType as _BuiltinFunctionType, FunctionType as _FunctionType
from types import FunctionType as _FunctionType
from typing import IO as _IO, Any as _Any, Optional as _Optional, Type as _Type, Union as _Union

import yaml as _yaml

from .plotting import gridplot
from .utils import load_h5, save_h5
from . import rng as random
from .rng import gen_keys
from .optim import train
from .typing import DictModel, ImplicitModel, RoxObject


class ConfigLoader(_ABC):
    """Common interface for loading and dumping configs to/from files, streams, etc."""

    @classmethod
    @_abstractmethod
    def load(cls, stream: _Any, **kwargs: _Any) -> _Any:
        """Load an object from a stream. If a file path is given, will attempt to open the file."""
        raise NotImplementedError

    @classmethod
    @_abstractmethod
    def dump(cls, obj: _Any, stream: _Any, **kwargs: _Any) -> _Any:
        """Save an object to a stream. If a file path is given, will attempt to write to the file."""
        raise NotImplementedError


class YamlLoader(ConfigLoader):
    """YAML configs. 
    
    **romjax objects**
    - Represent `romjax` objects with `!rox:path.to.Subclass` tag. Must be subclass of `RoxObject`
    - Supports basic Pydantic model_dump() to dictionary.
    - Supports !!python/name tag for functions and other importable names
    """

    @staticmethod
    def _yaml_loader() -> _Type[_yaml.SafeLoader]:
        """Return a custom yaml loader that handles romtool objects and callables."""
        class _Loader(_yaml.SafeLoader):
            pass

        def _construct_python_name_multi(loader: _yaml.SafeLoader, tag_suffix: str, node: _yaml.Node) -> _Any:
            value = tag_suffix or loader.construct_scalar(node)
            module_name, _, attr_path = value.rpartition(".")
            if not module_name or not attr_path:
                raise ValueError(f"Invalid python/name tag: {value!r}")
            module = _import_module(module_name)
            obj: _Any = module
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            return obj

        def _construct_python_name(loader: _yaml.SafeLoader, node: _yaml.Node) -> _Any:
            """Any importable name (e.g. functions)."""
            value = loader.construct_scalar(node)
            return _construct_python_name_multi(loader, value, node)

        def _construct_romjax_object(loader: _yaml.SafeLoader, tag_suffix: str, node: _yaml.Node) -> RoxObject:
            """Essentially just a convenience to automatically construct dataclasses when loading."""
            if not tag_suffix:
                raise ValueError("Missing class path in YAML tag.")
            module_name, _, class_name = tag_suffix.rpartition(".")
            if not module_name or not class_name:
                raise ValueError(f"Invalid romjax tag: {tag_suffix!r}")
            module = _import_module(module_name)
            cls_obj = getattr(module, class_name, None)
            if cls_obj is None:
                raise ValueError(f"Class not found: {tag_suffix!r}")
            if not isinstance(cls_obj, type) or not issubclass(cls_obj, RoxObject):
                raise TypeError(f"Tagged class is not an instance of required RoxObject: {tag_suffix!r}")
            if isinstance(node, _yaml.MappingNode):
                data = loader.construct_mapping(node, deep=True)
                return cls_obj(**data)
            if isinstance(node, _yaml.SequenceNode):
                data = loader.construct_sequence(node, deep=True)
                return cls_obj(data)
            data = loader.construct_scalar(node)
            return cls_obj(data)

        _Loader.add_constructor("tag:yaml.org,2002:python/name", _construct_python_name)
        _Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", _construct_python_name_multi)
        _Loader.add_multi_constructor(RoxObject.YAML_TAG, _construct_romjax_object)
        return _Loader

    @staticmethod
    def _yaml_dumper() -> _Type[_yaml.SafeDumper]:
        """Return a custom yaml dumper that handles romtool objects and callables."""
        class _Dumper(_yaml.SafeDumper):
            pass

        def _represent_python_name(dumper: _yaml.SafeDumper, data: _Any) -> _yaml.Node:
            name = f"{data.__module__}.{data.__qualname__}"
            return dumper.represent_scalar("tag:yaml.org,2002:python/name", name)

        def _represent_romjax_object(dumper: _yaml.SafeDumper, data: RoxObject) -> _yaml.Node:
            tag = data.yaml_tag()
            payload = data.model_dump()
            if isinstance(payload, list | tuple):
                return dumper.represent_sequence(tag, payload)
            return dumper.represent_mapping(tag, payload)

        _Dumper.add_representer(_FunctionType, _represent_python_name)
        _Dumper.add_representer(_BuiltinFunctionType, _represent_python_name)
        _Dumper.add_multi_representer(RoxObject, _represent_romjax_object)
        return _Dumper

    @classmethod
    def load(cls, 
             stream: _Union[str, bytes, bytearray, _PathLike[str], _IO[_Any]], 
             **kwargs: _Any
             ) -> _Any:
        """Load a configuration from a yaml-like stream.
        
        :param stream: A string, path, file-stream, byte-stream, or similar.
        :return: the configuration
        """
        loader = cls._yaml_loader()
        if isinstance(stream, (_PathLike, _Path)):
            with _Path(stream).open("r", encoding="utf-8") as fh:
                return _yaml.load(fh, Loader=loader, **kwargs)
        if isinstance(stream, str):
            looks_like_path = (
                stream.endswith((".yml", ".yaml"))
                or _Path(stream).is_absolute()
                or stream.startswith(".")
            )
            if _Path(stream).exists() or looks_like_path:
                with _Path(stream).open("r", encoding="utf-8") as fh:
                    return _yaml.load(fh, Loader=loader, **kwargs)
            return _yaml.load(stream, Loader=loader, **kwargs)
        if isinstance(stream, (bytes, bytearray)):
            return _yaml.load(stream.decode("utf-8"), Loader=loader, **kwargs)
        if hasattr(stream, "read"):
            return _yaml.load(stream, Loader=loader, **kwargs)
        raise TypeError("Unsupported stream type for YAML load.")

    @classmethod
    def dump(cls,
             obj: _Any,
             stream: _Optional[_Union[str, _PathLike[str], _IO[_Any]]] = None,
             **kwargs: _Any,
             ) -> _Optional[str]:
        """Dump a configuration to a yaml-like stream.
        
        :param obj: the yaml configuration
        :param stream: A string, path, file-stream, byte-stream, or similar.
        """
        dumper = cls._yaml_dumper()
        if stream is None:
            return _yaml.dump(obj, Dumper=dumper, **kwargs)
        if isinstance(stream, (_PathLike, _Path)) or isinstance(stream, str):
            with _Path(stream).open("w", encoding="utf-8") as fh:
                _yaml.dump(obj, fh, Dumper=dumper, **kwargs)
            return None
        if hasattr(stream, "write"):
            _yaml.dump(obj, stream, Dumper=dumper, **kwargs)
            return None
        raise TypeError("Unsupported stream type for YAML dump.")
