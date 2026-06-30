"""Reduced-order modeling in jax.

- Author - Joshua Eckels (eckelsjd@umich.edu)
- License - MIT
"""
__version__ = "0.0.1"


from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from contextvars import ContextVar as _ContextVar
from copy import deepcopy as _deepcopy
from functools import partial as _partial
from importlib import import_module as _import_module
from inspect import getattr_static as _getattr_static
from io import StringIO as _StringIO
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


class YamlSource:
    """In-memory YAML source produced by nested ``!overrides:`` tags.

    :param node: raw YAML node to construct when the source is loaded
    :param source_path: path of the YAML file that declared this source, when known
    :param label: optional diagnostic label for the source
    """

    def __init__(self, node: _yaml.Node | None, source_path: _Path | None, label: str | None = None) -> None:
        self.node = node
        self.source_path = source_path
        self.label = label


type _Stream = _Union[str, bytes, bytearray, _PathLike[str], _IO[_Any], YamlSource]

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "random": ("romjax.rng", None),
    "tree": ("romjax.tree", None),
    "Compression": ("romjax.compression", "Compression"),
    "CompositeEdge": ("romjax.graph", "CompositeEdge"),
    "FunctionGraph": ("romjax.graph", "FunctionGraph"),
    "GridSearch": ("romjax.grid_search", "GridSearch"),
    "ExplicitModel": ("romjax.model", "ExplicitModel"),
    "FilterModel": ("romjax.model", "FilterModel"),
    "ImplicitModel": ("romjax.model", "ImplicitModel"),
    "eqx_evaluate": ("romjax.model", "eqx_evaluate"),
    "LinearProjection": ("romjax.nn", "LinearProjection"),
    "ImplicitIterativeGalerkin": ("romjax.pde", "ImplicitIterativeGalerkin"),
    "AliveProgressMeter": ("romjax.pde", "AliveProgressMeter"),
    "gridplot": ("romjax.plotting", "gridplot"),
    "Poisson2D": ("romjax.poisson", "Poisson2D"),
    "Vlasov1D1V": ("romjax.vlasov", "Vlasov1D1V"),
    "gen_keys": ("romjax.rng", "gen_keys"),
    "PyTreeSampler": ("romjax.rng", "PyTreeSampler"),
    "NearSolutionSampler": ("romjax.rng", "NearSolutionSampler"),
    "GraphRef": ("romjax.typing", "GraphRef"),
    "UnaryOp": ("romjax.operators", "UnaryOp"),
    "BinaryOp": ("romjax.operators", "BinaryOp"),
    "DictModel": ("romjax.typing", "DictModel"),
    "ListModel": ("romjax.typing", "ListModel"),
    "CallableModel": ("romjax.typing", "CallableModel"),
    "ThirdPartyType": ("romjax.typing", "ThirdPartyType"),
    "CompositeRoutine": ("romjax.routine", "CompositeRoutine"),
    "Routine": ("romjax.routine", "Routine"),
    "RoutineConfig": ("romjax.routine", "RoutineConfig"),
    "RoutineError": ("romjax.routine", "RoutineError"),
    "load_h5": ("romjax.utils", "load_h5"),
    "save_h5": ("romjax.utils", "save_h5"),
    "DataGeneration": ("romjax.data_gen", "DataGeneration"),
    "DataLoader": ("romjax.data_gen", "DataLoader"),
    "CompareOrbax": ("romjax.compare", "CompareOrbax"),
    "CompareTable": ("romjax.compare", "CompareTable"),
    "OrbaxParams": ("romjax.train", "OrbaxParams"),
    "Train": ("romjax.train", "Train"),
    "GraphLoss": ("romjax.train", "GraphLoss"),
    "GraphTest": ("romjax.train", "GraphTest"),
    "BatchLoader": ("romjax.train", "BatchLoader"),
}

__all__ = list(_LAZY_EXPORTS.keys())

#ruff: noqa F401
if TYPE_CHECKING:
    from . import rng as random
    from . import tree as tree
    from .compression import Compression
    from .compare import CompareOrbax, CompareTable
    from .data_gen import DataGeneration, DataLoader
    from .graph import CompositeEdge, FunctionGraph
    from .grid_search import GridSearch
    from .model import ExplicitModel, FilterModel, ImplicitModel, eqx_evaluate
    from .nn import LinearProjection
    from .pde import ImplicitIterativeGalerkin, AliveProgressMeter
    from .plotting import gridplot
    from .poisson import Poisson2D
    from .vlasov import Vlasov1D1V
    from .rng import NearSolutionSampler, PyTreeSampler, gen_keys
    from .routine import CompositeRoutine, Routine, RoutineConfig, RoutineError
    from .operators import BinaryOp, UnaryOp
    from .train import (
        BatchLoader,
        GraphLoss,
        GraphTest,
        OrbaxParams,
        Train,
    )
    from .typing import CallableModel, DictModel, GraphRef, ListModel, ThirdPartyType
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
    OVERRIDES_TAG = "!overrides:"
    _SOURCE_PATH: _ContextVar[_Path | None] = _ContextVar("romjax_yaml_source_path", default=None)

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
                return cls_obj.model_validate(data)
            if isinstance(node, _yaml.SequenceNode):
                data = loader.construct_sequence(node, deep=True)
                return cls_obj.model_validate(data)
            data = loader.construct_scalar(node)
            if isinstance(data, str):
                try:
                    data = _import_python_name(data)
                except Exception:
                    pass

            return cls_obj.model_validate(data)

        def _construct_overrides(loader: _yaml.SafeLoader, tag_suffix: str, node: _yaml.Node) -> YamlSource:
            """Construct a nested override as a loadable in-memory YAML source."""
            source_path = cls.current_source_path()
            override_path = cls._resolve_override_path(tag_suffix, source_path)
            base_node, _ = cls._compose_resolved_node(override_path)
            override_node = cls._copy_with_tag(node, cls._default_node_tag(node))
            merged_node = cls._merge_nodes(base_node, override_node)
            return YamlSource(merged_node, source_path, tag_suffix)

        _Loader.add_constructor("tag:yaml.org,2002:python/name", _construct_python_name)
        _Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", _construct_python_name_multi)
        _Loader.add_multi_constructor(cls.PYDANTIC_TAG, _construct_base_model)
        _Loader.add_multi_constructor(cls.ROMX_TAG, _partial(_construct_base_model, default_module="romjax"))
        _Loader.add_multi_constructor(cls.OVERRIDES_TAG, _construct_overrides)
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

        def _represent_yaml_source(dumper: _yaml.SafeDumper, data: YamlSource) -> _yaml.Node:
            return _deepcopy(data.node) if data.node is not None else dumper.represent_none(None)

        _Dumper.add_representer(_FunctionType, _represent_python_name)
        _Dumper.add_representer(_BuiltinFunctionType, _represent_python_name)
        _Dumper.add_representer(YamlSource, _represent_yaml_source)
        _Dumper.add_multi_representer(_BaseModel, _represent_base_model)
        return _Dumper

    @classmethod
    def _read_stream(cls, stream: _Stream) -> tuple[str, _Path | None]:
        """Read YAML input while preserving path context for override resolution.

        :param stream: A YAML string, path, byte buffer, or file-like object.
        :return: the YAML text and the source path when one is known.
        """
        if isinstance(stream, (_PathLike, _Path)):
            path = _Path(stream)
            return path.read_text(encoding="utf-8"), path
        if isinstance(stream, str):
            try:
                path = _Path(stream)
                exists = path.exists()
                looks_like_path = stream.endswith((".yml", ".yaml")) or path.is_absolute() or stream.startswith(".")
            except OSError:
                return stream, None
            if exists or looks_like_path:
                return path.read_text(encoding="utf-8"), path
            return stream, None
        if isinstance(stream, (bytes, bytearray)):
            return stream.decode("utf-8"), None
        if hasattr(stream, "read"):
            data = stream.read()
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return data, None
        raise TypeError("Unsupported stream type for YAML load.")

    @classmethod
    def _resolve_override_path(cls, path: str, source_path: _Path | None) -> _Path:
        """Resolve an override target path.

        :param path: The path suffix from a root ``!overrides:`` tag.
        :param source_path: Path of the YAML file declaring the override, when known.
        :return: the resolved path to load.
        """
        normalized = path.replace("\\", "/")
        prefix = "__parent__/"
        if normalized.startswith(prefix):
            if source_path is None:
                raise ValueError("The __parent__ override path requires loading from a file path.")
            return source_path.parent / normalized[len(prefix):]
        return _Path(normalized)

    @classmethod
    def _resolve_parent_scalar(cls, value: str, source_path: _Path | None) -> str:
        """Resolve a scalar that may use the ``__parent__/`` prefix.

        :param value: scalar text
        :param source_path: path of the YAML file that declared the scalar
        :return: resolved scalar text
        """
        normalized = value.replace("\\", "/")
        if not normalized.startswith("__parent__/"):
            return value
        return cls._resolve_override_path(value, source_path).resolve().as_posix()

    @classmethod
    def current_source_path(cls) -> _Path | None:
        """Return the YAML file path currently being constructed, if available.

        :return: the current YAML source path
        """
        return cls._SOURCE_PATH.get()

    @classmethod
    def resolve_parent_path(cls, path: str | _Path) -> _Path:
        """Resolve a path that may use the ``__parent__/`` prefix.

        :param path: path string to resolve
        :return: path relative to the active YAML source when prefixed, otherwise unchanged
        """
        return cls._resolve_override_path(str(path), cls.current_source_path())

    @classmethod
    def _is_override_node(cls, node: _yaml.Node | None) -> bool:
        """Return whether a raw YAML node declares a root-level override.

        :param node: A raw YAML node.
        :return: whether the node tag is an override tag.
        """
        return node is not None and node.tag.startswith(cls.OVERRIDES_TAG)

    @classmethod
    def _is_constructed_tag_node(cls, node: _yaml.Node) -> bool:
        """Return whether a node should override as a complete tagged object.

        :param node: A raw YAML node.
        :return: whether the node is tagged for immediate romjax or pydantic construction.
        """
        return node.tag.startswith((cls.ROMX_TAG, cls.PYDANTIC_TAG))

    @classmethod
    def _is_null_node(cls, node: _yaml.Node) -> bool:
        """Return whether a raw scalar node is YAML null.

        :param node: A raw YAML node.
        :return: whether the node is tagged as null.
        """
        return isinstance(node, _yaml.ScalarNode) and node.tag == "tag:yaml.org,2002:null"

    @classmethod
    def _copy_with_tag(cls, node: _yaml.Node, tag: str) -> _yaml.Node:
        """Return a deep copy of a YAML node with a replacement tag.

        :param node: The node to copy.
        :param tag: The tag to apply to the copied node.
        :return: the copied node.
        """
        copied = _deepcopy(node)
        copied.tag = tag
        return copied

    @classmethod
    def _default_node_tag(cls, node: _yaml.Node) -> str:
        """Return the default YAML tag for a node's structural type.

        :param node: The node whose type determines the tag.
        :return: the default scalar, sequence, or mapping YAML tag.
        """
        if isinstance(node, _yaml.MappingNode):
            return _yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
        if isinstance(node, _yaml.SequenceNode):
            return _yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG
        return _yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG

    @classmethod
    def _mapping_pairs_by_key(cls, node: _yaml.MappingNode) -> dict[tuple[str, str], tuple[_yaml.Node, _yaml.Node]]:
        """Index mapping node pairs by scalar key tag and value.

        :param node: A raw mapping node.
        :return: mapping from comparable scalar keys to original key/value node pairs.
        """
        pairs: dict[tuple[str, str], tuple[_yaml.Node, _yaml.Node]] = {}
        for key_node, value_node in node.value:
            if not isinstance(key_node, _yaml.ScalarNode):
                raise TypeError("YAML override merging only supports scalar mapping keys.")
            pairs[(key_node.tag, key_node.value)] = (key_node, value_node)
        return pairs

    @classmethod
    def _merge_nodes(cls, base: _yaml.Node | None, override: _yaml.Node) -> _yaml.Node:
        """Merge raw YAML override nodes into raw base nodes.

        :param base: The base YAML node, if one exists.
        :param override: The override YAML node.
        :return: a merged YAML node.
        """
        if base is None or cls._is_constructed_tag_node(override):
            return _deepcopy(override)
        if isinstance(base, _yaml.MappingNode) and isinstance(override, _yaml.MappingNode):
            base_pairs = cls._mapping_pairs_by_key(base)
            override_pairs = cls._mapping_pairs_by_key(override)
            merged_pairs: list[tuple[_yaml.Node, _yaml.Node]] = []

            for base_key, base_value in base.value:
                key = (base_key.tag, base_key.value)
                if key in override_pairs:
                    _, override_value = override_pairs.pop(key)
                    merged_pairs.append((_deepcopy(base_key), cls._merge_nodes(base_value, override_value)))
                else:
                    merged_pairs.append((_deepcopy(base_key), _deepcopy(base_value)))

            for override_key, override_value in override_pairs.values():
                merged_pairs.append((_deepcopy(override_key), _deepcopy(override_value)))

            return _yaml.MappingNode(base.tag, merged_pairs, base.start_mark, base.end_mark, base.flow_style)
        if isinstance(base, _yaml.SequenceNode) and isinstance(override, _yaml.SequenceNode):
            merged_items = [_deepcopy(item) for item in base.value]
            for index, override_item in enumerate(override.value):
                if cls._is_null_node(override_item):
                    continue
                if index < len(merged_items):
                    merged_items[index] = cls._merge_nodes(merged_items[index], override_item)
                else:
                    merged_items.append(_deepcopy(override_item))
            return _yaml.SequenceNode(base.tag, merged_items, base.start_mark, base.end_mark, base.flow_style)
        return _deepcopy(override)

    @classmethod
    def _resolve_parent_refs(cls, node: _yaml.Node | None, source_path: _Path | None) -> _yaml.Node | None:
        """Return a node copy with ``__parent__/`` scalar and override-tag references resolved.

        :param node: raw YAML node to rewrite
        :param source_path: path used to resolve parent-relative references
        :return: copied node with parent-relative references rewritten
        """
        if node is None:
            return None

        copied = _deepcopy(node)
        if copied.tag.startswith(cls.OVERRIDES_TAG):
            suffix = copied.tag[len(cls.OVERRIDES_TAG):]
            copied.tag = f"{cls.OVERRIDES_TAG}{cls._resolve_parent_scalar(suffix, source_path)}"

        if isinstance(copied, _yaml.ScalarNode):
            copied.value = cls._resolve_parent_scalar(copied.value, source_path)
            return copied
        if isinstance(copied, _yaml.SequenceNode):
            copied.value = [cls._resolve_parent_refs(item, source_path) for item in copied.value]
            return copied
        if isinstance(copied, _yaml.MappingNode):
            copied.value = [
                (
                    cls._resolve_parent_refs(key_node, source_path),
                    cls._resolve_parent_refs(value_node, source_path),
                )
                for key_node, value_node in copied.value
            ]
            return copied
        return copied

    @classmethod
    def _compose_resolved_node(cls, stream: _Stream) -> tuple[_yaml.Node | None, _Path | None]:
        """Compose YAML into a raw node after resolving any root-level overrides.

        :param stream: A YAML string, path, byte buffer, or file-like object.
        :return: the composed YAML node with overrides already merged and its source path.
        """
        text, source_path = cls._read_stream(stream)
        node = _yaml.compose(text, Loader=_yaml.SafeLoader)
        if not cls._is_override_node(node):
            return node, source_path

        if node is None:
            return None, source_path
        override_path = cls._resolve_override_path(node.tag[len(cls.OVERRIDES_TAG):], source_path)
        base_node, _ = cls._compose_resolved_node(override_path)
        override_node = cls._copy_with_tag(node, cls._default_node_tag(node))
        return cls._merge_nodes(base_node, override_node), source_path

    @classmethod
    def _node_to_yaml(cls, node: _yaml.Node | None) -> str:
        """Serialize a raw YAML node to text without constructing custom objects.

        :param node: The YAML node to serialize.
        :return: YAML text.
        """
        if node is None:
            return ""
        stream = _StringIO()
        _yaml.serialize(node, stream=stream)
        return stream.getvalue()

    @classmethod
    def load(cls, stream: _Stream, **kwargs: _Any) -> _Any:
        """Load a configuration from a yaml-like stream. Small wrapper around yaml.load
        
        :param stream: A string, path, file-stream, byte-stream, or similar.
        :return: the configuration
        """
        if "Loader" not in kwargs:
            kwargs["Loader"] = cls.get_loader()

        if isinstance(stream, YamlSource):
            node, source_path = stream.node, stream.source_path
        else:
            node, source_path = cls._compose_resolved_node(stream)
        token = cls._SOURCE_PATH.set(source_path)
        try:
            return _yaml.load(cls._node_to_yaml(node), **kwargs)
        finally:
            cls._SOURCE_PATH.reset(token)

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
    
