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
from re import compile as _re_compile
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


_TEMPLATE_PATTERN = _re_compile(r"{{\s*(.*?)\s*}}")
_INTEGER_PATTERN = _re_compile(r"[+-]?\d+")
_FLOAT_PATTERN = _re_compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)")


class _TemplatePathError(ValueError):
    """Raised internally when a template path cannot be resolved."""


class _DeleteMarker:
    """Internal sentinel for a ``!delete`` configuration override."""


_DELETE = _DeleteMarker()


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
    "SVD": ("romjax.compression", "SVD"),
    "BlockLinearCompression": ("romjax.compression", "BlockLinearCompression"),
    "IdentityEdge": ("romjax.graph", "IdentityEdge"),
    "CompositeEdge": ("romjax.graph", "CompositeEdge"),
    "FunctionGraph": ("romjax.graph", "FunctionGraph"),
    "GridSearch": ("romjax.grid_search", "GridSearch"),
    "ExecutorConfig": ("romjax.execution", "ExecutorConfig"),
    "mpi_context": ("romjax.mpi_runtime", "mpi_context"),
    "ExplicitModel": ("romjax.model", "ExplicitModel"),
    "FilterModel": ("romjax.model", "FilterModel"),
    "ImplicitModel": ("romjax.model", "ImplicitModel"),
    "eqx_evaluate": ("romjax.model", "eqx_evaluate"),
    "Affine": ("romjax.nn", "Affine"),
    "LinearProjection": ("romjax.nn", "LinearProjection"),
    "BlockLinearAutoencoder": ("romjax.nn", "BlockLinearAutoencoder"),
    "ImplicitAffine": ("romjax.pde", "ImplicitAffine"),
    "ImplicitIterativeGalerkin": ("romjax.pde", "ImplicitIterativeGalerkin"),
    "AliveProgressMeter": ("romjax.pde", "AliveProgressMeter"),
    "IterativeSolver": ("romjax.pde", "IterativeSolver"),
    "LinearSolver": ("romjax.pde", "LinearSolver"),
    "RandomNewton": ("romjax.pde", "RandomNewton"),
    "gridplot": ("romjax.plotting", "gridplot"),
    "AdvectionDiffusion2D": ("romjax.transport", "AdvectionDiffusion2D"),
    "Vlasov1D1V": ("romjax.vlasov", "Vlasov1D1V"),
    "gen_keys": ("romjax.rng", "gen_keys"),
    "PyTreeSampler": ("romjax.rng", "PyTreeSampler"),
    "NearSolutionSampler": ("romjax.rng", "NearSolutionSampler"),
    "SolverSampler": ("romjax.rng", "SolverSampler"),
    "CompressionSampler": ("romjax.rng", "CompressionSampler"),
    "TreeRef": ("romjax.tree", "TreeRef"),
    "UnaryOp": ("romjax.operators", "UnaryOp"),
    "AffineUnaryOp": ("romjax.operators", "AffineUnaryOp"),
    "BinaryOp": ("romjax.operators", "BinaryOp"),
    "DictModel": ("romjax.typing", "DictModel"),
    "ListModel": ("romjax.typing", "ListModel"),
    "CallableModel": ("romjax.typing", "CallableModel"),
    "ThirdPartyType": ("romjax.typing", "ThirdPartyType"),
    "from_yaml": ("romjax.typing", "from_yaml"),
    "CompositeRoutine": ("romjax.routine", "CompositeRoutine"),
    "Routine": ("romjax.routine", "Routine"),
    "RoutineConfig": ("romjax.routine", "RoutineConfig"),
    "RoutineError": ("romjax.routine", "RoutineError"),
    "load_h5": ("romjax.utils", "load_h5"),
    "save_h5": ("romjax.utils", "save_h5"),
    "DataGeneration": ("romjax.data_gen", "DataGeneration"),
    "DataLoader": ("romjax.data_gen", "DataLoader"),
    "CompareOrbax": ("romjax.compare", "CompareOrbax"),
    "CompareMetric": ("romjax.compare", "CompareMetric"),
    "OrbaxRef": ("romjax.train", "OrbaxRef"),
    "resolve_orbax_params": ("romjax.train", "resolve_orbax_params"),
    "Train": ("romjax.train", "Train"),
    "BatchLoader": ("romjax.train", "BatchLoader"),
    "GraphLoss": ("romjax.loss", "GraphLoss"),
    "CyclicPathError": ("romjax.loss", "CyclicPathError"),
    "GraphLossTerm": ("romjax.loss", "GraphLossTerm"),
    "GraphLossTermGenerator": ("romjax.loss", "GraphLossTermGenerator"),
    "GraphTest": ("romjax.loss", "GraphTest"),
    "ProjectionAutoencoder": ("romjax.loss", "ProjectionAutoencoder"),
    "ProjectionRegularization": ("romjax.loss", "ProjectionRegularization"),
    "GridBoundaryInputs": ("romjax.pde", "GridBoundaryInputs")
}

__all__ = list(_LAZY_EXPORTS.keys())

#ruff: noqa F401
if TYPE_CHECKING:
    from . import rng as random
    from . import tree as tree
    from .compression import BlockLinearCompression, Compression, SVD
    from .compare import CompareMetric, CompareOrbax
    from .data_gen import DataGeneration, DataLoader
    from .execution import ExecutorConfig
    from .graph import CompositeEdge, FunctionGraph, IdentityEdge
    from .grid_search import GridSearch
    from .model import ExplicitModel, FilterModel, ImplicitModel, eqx_evaluate
    from .nn import Affine, BlockLinearAutoencoder, LinearProjection
    from .loss import (
        CyclicPathError,
        GraphLoss,
        GraphLossTerm,
        GraphLossTermGenerator,
        GraphTest,
        ProjectionAutoencoder,
        ProjectionRegularization,
    )
    from .pde import (
        AliveProgressMeter,
        GridBoundaryInputs,
        ImplicitAffine,
        ImplicitIterativeGalerkin,
        IterativeSolver,
        LinearSolver,
        RandomNewton,
    )
    from .plotting import gridplot
    from .transport import AdvectionDiffusion2D
    from .vlasov import Vlasov1D1V
    from .rng import CompressionSampler, NearSolutionSampler, PyTreeSampler, SolverSampler, gen_keys
    from .routine import CompositeRoutine, Routine, RoutineConfig, RoutineError
    from .operators import AffineUnaryOp, BinaryOp, UnaryOp
    from .train import BatchLoader, OrbaxRef, Train, resolve_orbax_params
    from .tree import TreeRef
    from .typing import CallableModel, DictModel, ListModel, ThirdPartyType, from_yaml
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
    - Supports ``!delete`` within overrides to remove a mapping key or sequence item. In YAML flow collections,
      write the empty tagged scalar as ``!delete ''`` so the tag is delimited.
    - Supports ``{{ dotted.path }}`` references and ``{{ resolver.path: arg1, arg2 }}`` calls.
      A complete template preserves the referenced or returned value's type; embedded templates interpolate text.
    """
    PYDANTIC_TAG = "!pd:"
    ROMX_TAG = "!romx:"
    OVERRIDES_TAG = "!overrides:"
    DELETE_TAG = "!delete"
    YAML_SOURCE_TAG = "!romx:yaml-source"
    _SOURCE_PATH: _ContextVar[_Path | None] = _ContextVar("romjax_yaml_source_path", default=None)

    @classmethod
    def get_tag(cls, data: _Any) -> str:
        t = type(data)
        import romjax
        tag = cls.ROMX_TAG if hasattr(romjax, t.__name__) else cls.PYDANTIC_TAG
        return f"{tag}{t.__module__}.{t.__qualname__}"

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
            name = tag_suffix if '.' in tag_suffix else f'{default_module}.{tag_suffix}'
            try:
                cls_obj = _import_python_name(name)
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"Class not found: {tag_suffix!r}") from exc
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

        def _construct_yaml_source(loader: _yaml.SafeLoader, node: _yaml.Node) -> YamlSource:
            """Preserve an internally serialized source for deferred loading."""
            if not isinstance(node, _yaml.MappingNode):
                raise ValueError("Deferred YAML sources must contain a mapping payload.")
            fields = {
                key_node.value: value_node
                for key_node, value_node in node.value
                if isinstance(key_node, _yaml.ScalarNode)
            }
            source_node = fields.get("node")
            if source_node is None:
                raise ValueError("Deferred YAML sources must contain a 'node' payload.")
            label_node = fields.get("label")
            label = label_node.value if isinstance(label_node, _yaml.ScalarNode) else None
            source_path_node = fields.get("source_path")
            source_path = (
                _Path(source_path_node.value)
                if isinstance(source_path_node, _yaml.ScalarNode) and source_path_node.value
                else cls.current_source_path()
            )
            return YamlSource(source_node, source_path, label)

        def _construct_delete(loader: _yaml.SafeLoader, node: _yaml.Node) -> _DeleteMarker:
            """Construct the deferred deletion marker used by composite configurations."""
            cls._validate_delete_node(node)
            return _DELETE

        _Loader.add_constructor("tag:yaml.org,2002:python/name", _construct_python_name)
        _Loader.add_multi_constructor("tag:yaml.org,2002:python/name:", _construct_python_name_multi)
        _Loader.add_multi_constructor(cls.PYDANTIC_TAG, _construct_base_model)
        _Loader.add_multi_constructor(cls.ROMX_TAG, _partial(_construct_base_model, default_module="romjax"))
        _Loader.add_multi_constructor(cls.OVERRIDES_TAG, _construct_overrides)
        _Loader.add_constructor(cls.DELETE_TAG, _construct_delete)
        _Loader.add_constructor(cls.YAML_SOURCE_TAG, _construct_yaml_source)
        return _Loader

    @classmethod
    def _restore_model_values(cls, original: _Any, payload: _Any) -> _Any:
        """Restore nested model types without undoing explicit Pydantic serializers."""
        if type(original).__pydantic_decorators__.model_serializers:
            return payload
        schema = type(original).__pydantic_core_schema__
        definitions = {item['ref']: item for item in schema.get('definitions', ())}
        while schema['type'] != 'model':
            schema = definitions[schema['schema_ref']] if schema['type'] == 'definition-ref' else schema['schema']
        if schema.get('root_model'):
            return cls._restore_model_value(original.root, payload, schema['schema'], definitions)
        if not isinstance(payload, dict):
            return payload
        while schema['type'] != 'model-fields':
            schema = definitions[schema['schema_ref']] if schema['type'] == 'definition-ref' else schema['schema']
        fields = schema['fields']
        aliases = {field.get('serialization_alias', name): name for name, field in fields.items()}
        for key in payload:
            name = aliases.get(key, key) if type(original).model_config.get('serialize_by_alias') else key
            if hasattr(original, name):
                payload[key] = cls._restore_model_value(
                    getattr(original, name), payload[key], fields.get(name, {}).get('schema', {}), definitions
                )
        return payload

    @classmethod
    def _schema_value_type(cls, schema: dict, definitions: dict) -> type | None:
        """Identify a union branch's Python type without running user validators."""
        while schema:
            if schema['type'] == 'model':
                return schema['cls']
            if schema['type'] == 'definition-ref':
                schema = definitions.get(schema['schema_ref'], {})
            elif 'schema' in schema:
                schema = schema['schema']
            else:
                return {'list': list, 'tuple': tuple, 'dict': dict, 'str': str, 'int': int,
                        'float': float, 'bool': bool, 'bytes': bytes, 'none': type(None)}.get(schema['type'])
        return None

    @classmethod
    def _restore_model_value(cls, original: _Any, payload: _Any, schema: dict, definitions: dict) -> _Any:
        """Follow field schemas so annotated serializers also control nested containers."""
        while schema:
            # Stop before the model itself: its serializer owns its tagged YAML body,
            # whereas serializers on fields/annotations own the entire field value.
            if schema['type'] == 'model':
                break
            if 'serialization' in schema:
                return payload
            if schema['type'] == 'definition-ref':
                schema = definitions.get(schema['schema_ref'], {})
            elif 'schema' in schema:
                schema = schema['schema']
            elif schema['type'] in ('union', 'tagged-union'):
                choices = schema['choices']
                choices = choices.values() if isinstance(choices, dict) else choices
                matches = []
                for choice in choices:
                    choice = choice[0] if isinstance(choice, tuple) else choice
                    value_type = cls._schema_value_type(choice, definitions)
                    if value_type is not None and isinstance(original, value_type):
                        matches.append((value_type is type(original), choice))
                if not matches:
                    # An unknown/custom branch may have serialized the value. Keep
                    # its output rather than guess and undo that transformation.
                    return payload
                schema = max(matches, key=lambda item: item[0])[1]
            else:
                break
        if isinstance(original, _BaseModel):
            return original
        if isinstance(original, dict) and isinstance(payload, dict):
            for key in payload.keys() & original.keys():
                payload[key] = cls._restore_model_value(
                    original[key], payload[key], schema.get('values_schema', {}), definitions
                )
        elif isinstance(original, (list, tuple)) and isinstance(payload, (list, tuple)):
            items = schema.get('items_schema', {})
            restored = []
            for index, (value, item) in enumerate(zip(original, payload)):
                item_schema = items
                if isinstance(items, list):
                    item_index = index
                    variadic = schema.get('variadic_item_index')
                    if variadic is not None and index >= variadic:
                        tail_start = len(original) - (len(items) - variadic - 1)
                        item_index = variadic if index < tail_start else len(items) - len(original) + index
                    item_schema = items[item_index]
                restored.append(cls._restore_model_value(value, item, item_schema, definitions))
            payload = restored
        return payload

    @classmethod
    def get_dumper(cls, *, preserve_yaml_sources: bool = False, preserve_model_types: bool = False) -> _Type[_yaml.SafeDumper]:
        """Return a YAML dumper preserving model types and explicit serializers.

        ``preserve_model_types`` remains accepted for compatibility; model types
        are now preserved in ordinary dumps as well.
        """
        class _Dumper(_yaml.SafeDumper):
            pass

        def _represent_python_name(dumper: _yaml.SafeDumper, data: _Any) -> _yaml.Node:
            name = f"{data.__module__}.{data.__qualname__}"
            return dumper.represent_scalar("tag:yaml.org,2002:python/name", name)

        def _represent_base_model(dumper: _yaml.SafeDumper, data: _BaseModel) -> _yaml.Node:
            """Essentially a convenience that defers serialization to pydantic model_dump"""
            from romjax.typing import ListModel

            tag = cls.get_tag(data)
            serializers = type(data).__pydantic_decorators__.model_serializers.values()
            if isinstance(data, ListModel) and all(item.func is ListModel._serialize_as_list for item in serializers):
                # A sequence loses explicit storage keys; sorting a mapping changes
                # integer indexing. A list of pairs preserves both independently of
                # the dumper's sort_keys option for ordinary dictionaries.
                return dumper.represent_mapping(tag, list(data.items()))
            payload = cls._restore_model_values(data, data.model_dump())
            if isinstance(payload, list | tuple):
                return dumper.represent_sequence(tag, payload)
            if isinstance(payload, _FunctionType | _BuiltinFunctionType):
                name = f"{payload.__module__}.{payload.__qualname__}"
                return dumper.represent_scalar(tag, name)
            if isinstance(payload, (str, int, float, bool)) or payload is None:
                return dumper.represent_scalar(tag, str(payload) if payload is not None else "null")
            return dumper.represent_mapping(tag, payload)

        def _represent_yaml_source(dumper: _yaml.SafeDumper, data: YamlSource) -> _yaml.Node:
            if data.node is None:
                return dumper.represent_none(None)
            node = _deepcopy(data.node)
            if preserve_yaml_sources:
                key_node = _yaml.ScalarNode(_yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG, "node")
                pairs: list[tuple[_yaml.Node, _yaml.Node]] = [(key_node, node)]
                if data.label is not None:
                    pairs.append(
                        (
                            _yaml.ScalarNode(_yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG, "label"),
                            _yaml.ScalarNode(_yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG, data.label),
                        )
                    )
                if data.source_path is not None:
                    pairs.append(
                        (
                            _yaml.ScalarNode(_yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG, "source_path"),
                            _yaml.ScalarNode(
                                _yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG,
                                data.source_path.as_posix(),
                            ),
                        )
                    )
                return _yaml.MappingNode(cls.YAML_SOURCE_TAG, pairs)
            return node

        def _represent_delete(dumper: _yaml.SafeDumper, data: _DeleteMarker) -> _yaml.Node:
            """Serialize the deferred deletion marker without converting it to text."""
            return dumper.represent_scalar(cls.DELETE_TAG, "")

        _Dumper.add_representer(_FunctionType, _represent_python_name)
        _Dumper.add_representer(_BuiltinFunctionType, _represent_python_name)
        _Dumper.add_multi_representer(_Path, lambda dumper, data: dumper.represent_str(str(data)))
        _Dumper.add_representer(YamlSource, _represent_yaml_source)
        _Dumper.add_representer(_DeleteMarker, _represent_delete)
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
    def _is_delete_node(cls, node: _yaml.Node) -> bool:
        """Return whether a raw YAML node is the deletion marker.

        :param node: A raw YAML node.
        :return: whether ``node`` is a valid bare ``!delete`` scalar.
        """
        if node.tag != cls.DELETE_TAG:
            return False
        cls._validate_delete_node(node)
        return True

    @classmethod
    def _validate_delete_node(cls, node: _yaml.Node) -> None:
        """Validate the syntax of a raw ``!delete`` YAML node.

        :param node: A YAML node tagged as a deletion marker.
        :raises ValueError: if the marker is not a bare scalar.
        """
        if not isinstance(node, _yaml.ScalarNode) or node.value:
            raise ValueError("!delete must be a bare scalar marker.")

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
                    if cls._is_delete_node(override_value):
                        continue
                    merged_pairs.append((_deepcopy(base_key), cls._merge_nodes(base_value, override_value)))
                else:
                    merged_pairs.append((_deepcopy(base_key), _deepcopy(base_value)))

            for override_key, override_value in override_pairs.values():
                if cls._is_delete_node(override_value):
                    raise ValueError(f"Cannot delete missing mapping key {override_key.value!r}.")
                merged_pairs.append((_deepcopy(override_key), _deepcopy(override_value)))

            return _yaml.MappingNode(base.tag, merged_pairs, base.start_mark, base.end_mark, base.flow_style)
        if isinstance(base, _yaml.SequenceNode) and isinstance(override, _yaml.SequenceNode):
            merged_items: list[_yaml.Node] = []
            for index, override_item in enumerate(override.value):
                if cls._is_delete_node(override_item):
                    if index >= len(base.value):
                        raise ValueError(f"Cannot delete sequence item at index {index}; base sequence is too short.")
                    continue
                if cls._is_null_node(override_item):
                    if index < len(base.value):
                        merged_items.append(_deepcopy(base.value[index]))
                    continue
                if index < len(base.value):
                    merged_items.append(cls._merge_nodes(base.value[index], override_item))
                else:
                    merged_items.append(_deepcopy(override_item))
            if len(base.value) > len(override.value):
                merged_items.extend(_deepcopy(item) for item in base.value[len(override.value):])
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
        seen: set[int] = set()

        def resolve(current: _yaml.Node) -> None:
            if id(current) in seen:
                return
            seen.add(id(current))
            # Deferred sources have their own source_path. Resolve their contents
            # only when that source is loaded, not relative to this wrapper.
            if current.tag == cls.YAML_SOURCE_TAG:
                return
            if current.tag.startswith(cls.OVERRIDES_TAG):
                suffix = current.tag[len(cls.OVERRIDES_TAG):]
                current.tag = f"{cls.OVERRIDES_TAG}{cls._resolve_parent_scalar(suffix, source_path)}"
            if isinstance(current, _yaml.ScalarNode):
                current.value = cls._resolve_parent_scalar(current.value, source_path)
            elif isinstance(current, _yaml.SequenceNode):
                for item in current.value:
                    resolve(item)
            elif isinstance(current, _yaml.MappingNode):
                for key, value in current.value:
                    resolve(key)
                    resolve(value)

        resolve(copied)
        return copied

    @classmethod
    def _template_location(cls, node: _yaml.Node, path: tuple[str, ...]) -> str:
        """Return a human-readable location for a template node.

        :param node: raw YAML node containing a template
        :param path: dotted path to the node within its document
        :return: diagnostic location text
        """
        location = ".".join(path) or "<root>"
        if node.start_mark is None:
            return location
        return f"{location} (line {node.start_mark.line + 1}, column {node.start_mark.column + 1})"

    @classmethod
    def _template_path_node(cls, root: _yaml.Node, expression: str) -> tuple[_yaml.Node, tuple[str, ...]]:
        """Find one dotted template path in a raw YAML document.

        :param root: root YAML node used as template context
        :param expression: dotted mapping path, with numeric sequence indexes
        :return: referenced node and its path
        :raises _TemplatePathError: if the expression does not identify a node
        """
        parts = expression.split(".")
        if not expression or any(not part for part in parts):
            raise _TemplatePathError(f"Invalid template path {expression!r}.")

        current = root
        resolved: list[str] = []
        for part in parts:
            if isinstance(current, _yaml.MappingNode):
                value_node = next(
                    (
                        candidate
                        for key_node, candidate in current.value
                        if isinstance(key_node, _yaml.ScalarNode) and key_node.value == part
                    ),
                    None,
                )
                if value_node is None:
                    raise _TemplatePathError(f"Template path {expression!r} does not exist.")
                current = value_node
            elif isinstance(current, _yaml.SequenceNode):
                try:
                    index = int(part)
                except ValueError as error:
                    raise _TemplatePathError(
                        f"Template path {expression!r} uses non-numeric index {part!r}."
                    ) from error
                if index < 0 or index >= len(current.value):
                    raise _TemplatePathError(f"Template path {expression!r} has an out-of-range index.")
                current = current.value[index]
            else:
                raise _TemplatePathError(f"Template path {expression!r} cannot descend through a scalar value.")
            resolved.append(part)
        return current, tuple(resolved)

    @classmethod
    def _template_node_value(cls, node: _yaml.Node) -> _Any:
        """Construct one fully rendered raw YAML node for template evaluation.

        :param node: rendered node to construct
        :return: Python value represented by ``node``
        """
        return _yaml.load(cls._node_to_yaml(node), Loader=cls.get_loader())

    @classmethod
    def _template_value_node(cls, value: _Any) -> _yaml.Node:
        """Represent one resolver result as a raw YAML node.

        :param value: resolver return value
        :return: YAML node that reconstructs to ``value``
        :raises ValueError: if the return value cannot be represented as YAML
        """
        node = _yaml.compose(cls.dump(value, sort_keys=False), Loader=_yaml.SafeLoader)
        if node is None:
            raise ValueError("A template resolver returned an empty YAML document.")
        return node

    @classmethod
    def _coerce_template_argument(cls, value: str) -> _Any:
        """Coerce the limited numeric literals allowed in resolver arguments.

        :param value: unquoted, non-reference resolver argument
        :return: an integer, float, or unchanged string
        """
        if _INTEGER_PATTERN.fullmatch(value):
            return int(value)
        if _FLOAT_PATTERN.fullmatch(value):
            return float(value)
        return value

    @classmethod
    def _render_templates(cls, node: _yaml.Node | None) -> _yaml.Node | None:
        """Render basic references and resolver calls in one composed YAML document.

        Rendering occurs on YAML nodes, after override composition and before Pydantic construction. This preserves
        standalone reference types and lets inline nested overrides receive values from their declaring document.

        :param node: composed YAML document to render
        :return: rendered YAML document
        :raises ValueError: if a template is malformed, circular, or cannot be resolved
        """
        if node is None:
            return None

        root = _deepcopy(node)
        rendered: dict[int, _yaml.Node] = {}
        rendering: set[int] = set()

        def fail(message: str, current: _yaml.Node, path: tuple[str, ...]) -> ValueError:
            return ValueError(f"{message} at {cls._template_location(current, path)}.")

        def resolve_expression(expression: str, current: _yaml.Node, path: tuple[str, ...]) -> _yaml.Node:
            expression = expression.strip()
            resolver_path, separator, raw_arguments = expression.partition(":")
            resolver_path = resolver_path.strip()
            if not resolver_path:
                raise fail(f"Invalid template expression {expression!r}", current, path)

            if not separator:
                try:
                    referenced, referenced_path = cls._template_path_node(root, resolver_path)
                except _TemplatePathError as error:
                    raise fail(str(error), current, path) from error
                return render_node(referenced, referenced_path)

            try:
                resolver_node, resolver_node_path = cls._template_path_node(root, resolver_path)
            except _TemplatePathError as error:
                raise fail(str(error), current, path) from error
            resolver = cls._template_node_value(render_node(resolver_node, resolver_node_path))
            if not callable(resolver):
                raise fail(f"Template resolver {resolver_path!r} is not callable", current, path)

            arguments: list[_Any] = []
            if raw_arguments.strip():
                for argument in raw_arguments.split(","):
                    argument = argument.strip()
                    if not argument:
                        raise fail(f"Invalid empty argument in template expression {expression!r}", current, path)
                    try:
                        argument_node, argument_path = cls._template_path_node(root, argument)
                    except _TemplatePathError:
                        arguments.append(cls._coerce_template_argument(argument))
                    else:
                        arguments.append(cls._template_node_value(render_node(argument_node, argument_path)))
            try:
                return cls._template_value_node(resolver(*arguments))
            except Exception as error:
                raise fail(f"Template resolver {resolver_path!r} failed: {error}", current, path) from error

        def render_scalar(current: _yaml.ScalarNode, path: tuple[str, ...]) -> _yaml.Node:
            matches = list(_TEMPLATE_PATTERN.finditer(current.value))
            if not matches:
                return current
            if len(matches) == 1 and matches[0].span() == (0, len(current.value)):
                return resolve_expression(matches[0].group(1), current, path)

            pieces: list[str] = []
            position = 0
            for match in matches:
                pieces.append(current.value[position:match.start()])
                value_node = resolve_expression(match.group(1), current, path)
                pieces.append(str(cls._template_node_value(value_node)))
                position = match.end()
            pieces.append(current.value[position:])
            return _yaml.ScalarNode(_yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG, "".join(pieces), style=current.style)

        def render_node(current: _yaml.Node, path: tuple[str, ...]) -> _yaml.Node:
            identifier = id(current)
            if identifier in rendered:
                return rendered[identifier]
            if identifier in rendering:
                raise fail("Circular template reference", current, path)
            rendering.add(identifier)
            try:
                if current.tag == cls.YAML_SOURCE_TAG:
                    result = current
                elif isinstance(current, _yaml.ScalarNode):
                    result = render_scalar(current, path)
                elif isinstance(current, _yaml.SequenceNode):
                    current.value = [render_node(item, path + (str(index),)) for index, item in enumerate(current.value)]
                    result = current
                elif isinstance(current, _yaml.MappingNode):
                    pairs: list[tuple[_yaml.Node, _yaml.Node]] = []
                    for key_node, value_node in current.value:
                        value_path = path + (key_node.value,) if isinstance(key_node, _yaml.ScalarNode) else path
                        pairs.append((key_node, render_node(value_node, value_path)))
                    current.value = pairs
                    result = current
                else:
                    result = current
            finally:
                rendering.remove(identifier)
            rendered[identifier] = result
            return result

        return render_node(root, ())

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
            node = cls._render_templates(node)
            node = cls._resolve_parent_refs(node, source_path)
            return _yaml.load(cls._node_to_yaml(node), **kwargs)
        finally:
            cls._SOURCE_PATH.reset(token)

    @classmethod
    def dump(cls, obj: _Any, stream: _Stream | None = None, **kwargs: _Any) -> _Optional[str]:
        """Dump a configuration to a yaml-like stream. Small wrapper around yaml.dump

        Model types and explicit Pydantic serializers are preserved. Custom graph
        edges must be importable and reconstructible from their serialized
        configuration; exclude runtime caches and provide serializers for state
        that cannot be represented directly in YAML.
        
        :param obj: the yaml configuration
        :param stream: A string, path, file-stream, byte-stream, or similar.
        """
        preserve_yaml_sources = kwargs.pop("_preserve_yaml_sources", False)
        preserve_model_types = kwargs.pop("_preserve_model_types", False)
        if "Dumper" not in kwargs:
            kwargs["Dumper"] = cls.get_dumper(preserve_yaml_sources=preserve_yaml_sources,
                                              preserve_model_types=preserve_model_types)

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
    
