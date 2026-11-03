"""Base routine classes."""

import ast
import re
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from importlib import import_module as _import_module
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Callable, Iterator, Literal, Mapping

import matplotlib
import matplotlib.pyplot as plt
import yaml
from alive_progress import config_handler
from loguru import logger
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    field_validator,
    model_validator,
)

from romjax.execution import ExecutionAction, ExecutorConfig, ProgressBarConfig, TaskContext, TaskSpec, TaskWorker
from romjax.plotting import GridplotConfig
from romjax.runtime import RuntimeContext, runtime_context
from romjax.typing import DictModel, WriteStream, from_yaml

if TYPE_CHECKING:
    from romjax import YamlSource

__all__ = [
    "CompositeRoutine",
    "Routine",
    "RoutineConfig",
    "RoutineError",
    "RoutineSpec",
    "LoggerConfig",
    "ProgressBarConfig",
]

_LAZY_EXPORTS = {
    "DataGeneration": ("romjax.data_gen", "DataGeneration"),
    "GridSearch": ("romjax.grid_search", "GridSearch"),
    "Train": ("romjax.train", "Train"),
    "CompareMetric": ("romjax.compare", "CompareMetric")
}

available = list(_LAZY_EXPORTS.keys())

_RUNTIME_ROOT: ContextVar[Path | None] = ContextVar("romjax_runtime_root", default=None)
_TIMESTAMP_PATTERN = re.compile(r"\{\{\s*timestamp\s*\}\}")

def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = _import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


class LoggerConfig(BaseModel):
    """
    Loguru diagnostics configuration. See options of :meth:`loguru.Logger.configure`.

    :param handlers: loguru handler config dicts
    :param levels: configs for logging levels
    :param extra: extra parameters for loguru logger
    :param patcher: will be applied to record dicts of each logged message
    :param activation: list of tuples denoting which loggers should be enabled
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True, extra="forbid")

    handlers: list[dict[str, Any]] | None = None
    levels: list[dict[str, Any]] | None = None
    extra: dict[str, Any] | None = None
    patcher: Callable[[dict], None] | None = None
    activation: list[tuple[str, bool]] | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_handlers(cls, value):
        if (isinstance(value, Mapping) and "sink" in value) or not isinstance(value, Mapping):
            return {"handlers": value}
        return value

    @field_validator("handlers", mode="before")
    @classmethod
    def _normalize_sink(cls, value):
        """Normalize handler sinks. Will make sure each handler is either a Path or a write-able stream,
        including the special sys.stdout and sys.stderr streams."""
        if value is None:
            return value
        if not isinstance(value, list):
            value = [value]
        
        write_adapter = TypeAdapter(WriteStream)

        normalized = []
        for handler in value:
            if isinstance(handler, Mapping):
                item = dict(handler)
                if sink := item.get("sink"):
                    item["sink"] = write_adapter.validate_python(sink)
                normalized.append(item)
            else:
                normalized.append({"sink": write_adapter.validate_python(handler)})

        return normalized


class RoutineConfig(BaseModel):
    """
    Global configurations for routines. Will toggle global settings during model validation.
    
    :ivar jax_platforms: JAX platform(s), as accepted by the ``jax_platforms`` config
    :ivar jax_enable_x64: whether JAX should enable 64-bit floating-point values
    :ivar mplstyle: matplotlib plot style
    :ivar gridplot: gridplot style
    :ivar logger: loguru configuration
    :ivar progress_bar: alive_bar configuration
    :ivar extra: anything you want to write/reuse in yaml but don't want to exist on its own
    :ivar runtime_root: execution directory inherited by descendants; supports ``{{timestamp}}``
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True, extra="forbid")

    jax_platforms: str | None = None
    jax_enable_x64: bool | None = None
    mplstyle: str | Path | Mapping | None = None
    gridplot: GridplotConfig | None = None
    logger: LoggerConfig | None = None
    progress_bar: ProgressBarConfig | None = None
    extra: Any | None = Field(exclude=True, default=None)
    runtime_root: Path | None = None

    @field_validator("runtime_root", mode="before")
    @classmethod
    def _validate_runtime_root(cls, value: Any) -> Any:
        if value is not None:
            if not str(value).strip():
                raise ValueError("runtime_root must be a nonempty path.")
            remaining = _TIMESTAMP_PATTERN.sub("timestamp", str(value))
            if "{{" in remaining or "}}" in remaining:
                raise ValueError("Unresolved runtime_root template; only {{timestamp}} is resolved at runtime.")
        return value

    @model_validator(mode="before")
    @classmethod
    def _configure_jax(cls, value):
        if isinstance(value, Mapping):
            jax_platforms = value.get("jax_platforms")
            enable_x64 = value.get("jax_enable_x64")
        else:
            jax_platforms = getattr(value, "jax_platforms", None)
            enable_x64 = getattr(value, "jax_enable_x64", None)

        import jax

        if enable_x64 is not None:
            jax.config.update("jax_enable_x64", enable_x64)
        if jax_platforms is not None:
            jax.config.update("jax_platforms", jax_platforms)
        return value

    @model_validator(mode="after")
    def configure(self):
        if (style := self.mplstyle) is not None:    # plt style
            if isinstance(style, str | Path):
                plt.style.use(style)
            else:
                if (file := style.pop("file", None)) is not None:
                    plt.style.use(file)
                matplotlib.rcParams.update(style)
        
        if (plot := self.gridplot) is not None:     # gridplot style
            import romjax.plotting
            names = plot.model_fields_set if plot.override_none else type(plot).model_fields
            romjax.plotting.set_global(**{name: getattr(plot, name) for name in names})
        
        if (log := self.logger) is not None:        # loguru
            logger.configure(**dict(log))

        if (bar := self.progress_bar) is not None:  # alive bar
            config_handler.set_global(**{key: getattr(bar, key) for key in (bar.model_extra or {})})
        
        return self

    
class Routine(BaseModel, ABC):
    """Base class providing shared configuration and runtime scopes for ``run()``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True, extra="allow")
    
    routine_config: Annotated[RoutineConfig | None, BeforeValidator(from_yaml)] = None
    _runtime_active: bool = PrivateAttr(default=False)
    _resolved_runtime_root: Path | None = PrivateAttr(default=None)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Apply the common execution scope to custom and built-in routines."""
        super().__init_subclass__(**kwargs)
        implementation = cls.__dict__.get("run")
        if implementation is not None:
            @wraps(implementation)
            def scoped_run(self: "Routine", *args: Any, **kwargs: Any) -> int:
                with self.run_context():
                    return implementation(self, *args, **kwargs)
            cls.run = scoped_run

    @property
    def resolved_runtime_root(self) -> Path | None:
        """Runtime directory selected for the current or most recent invocation."""
        return self._resolved_runtime_root

    @staticmethod
    def _create_runtime_root(template: Path) -> Path:
        """Resolve and reserve a runtime directory without reusing timestamped runs."""
        template = template.expanduser().absolute()
        text = str(template)
        if not _TIMESTAMP_PATTERN.search(text):
            template.mkdir(parents=True, exist_ok=True)
            return template
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        # Reserve the first timestamp-bearing ancestor, not just a trailing subdirectory.
        index = next(i for i, part in enumerate(template.parts) if _TIMESTAMP_PATTERN.search(part))
        counter = 0
        while True:
            stamp = timestamp if counter == 0 else f"{timestamp}-{counter}"
            root = Path(_TIMESTAMP_PATTERN.sub(stamp, text))
            reservation = Path(*root.parts[:index + 1])
            try:
                reservation.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                counter += 1
                continue
            root.mkdir(parents=True, exist_ok=True)
            return root

    @contextmanager
    def run_context(self, *, parent_root: Path | None = None) -> Iterator[None]:
        """Resolve roots for one invocation, restoring configured paths on exit.

        Nested calls on this instance reuse the active scope. Worker launchers may
        explicitly carry the parent's scope across thread or process boundaries.

        :param parent_root: resolved runtime directory inherited from a parent worker
        :return: context manager covering artifact setup and execution
        """
        if self._runtime_active:
            yield
            return
        runtime_root = parent_root if parent_root is not None else _RUNTIME_ROOT.get()
        if runtime_root is None and self.routine_config is not None:
            template = self.routine_config.runtime_root
            if template is not None:
                runtime_root = self._create_runtime_root(template)
        configured_root = getattr(self, "root", None)
        token = _RUNTIME_ROOT.set(runtime_root)
        self._runtime_active = True
        self._resolved_runtime_root = runtime_root
        if configured_root is None:
            artifact_root = None
        elif runtime_root is None:
            artifact_root = Path(configured_root).expanduser().absolute()
        else:
            artifact_root = (runtime_root / configured_root).absolute()
        try:
            with runtime_context(RuntimeContext(runtime_root, artifact_root)):
                if runtime_root is not None and configured_root is not None:
                    self.root = runtime_root / configured_root
                self._prepare_run()
                yield
        finally:
            if configured_root is not None:
                self.root = configured_root
            self._runtime_active = False
            _RUNTIME_ROOT.reset(token)

    def _prepare_run(self) -> None:
        """Prepare artifacts after resolving roots and before executing the routine."""

    @model_validator(mode="wrap")
    @classmethod
    def _get_routine_configs_from_extra(cls, value, handler):
        validated_extra = None

        if isinstance(value, Mapping):
            value = dict(value)
            config_data = {
                key: value[key]
                for key in RoutineConfig.model_fields
                if key != "routine_config" and key in value and key not in cls.model_fields
            }
            if config_data:
                validated_extra = RoutineConfig.model_validate(config_data)

        self = handler(value)

        if self.model_extra is None or len(self.model_extra) == 0:
            return self

        for key in self.model_extra:
            if key not in RoutineConfig.model_fields:
                raise ValueError(
                    f"Extra Routine argument '{key}' not recognized. Only global configs: "
                    f"{RoutineConfig.model_fields} allowed."
                )

        if validated_extra is None:
            return self

        if self.routine_config is None:
            object.__setattr__(self, "routine_config", validated_extra)
        else:
            for attr in RoutineConfig.model_fields:
                if (ele := getattr(validated_extra, attr, None)) is not None:
                    # Merge gridplot configs, all others will override completely.
                    if attr == "gridplot" and (curr_ele := self.routine_config.gridplot) is not None:
                        GridplotConfig.merge(curr_ele, ele, in_place=True)
                    else:
                        object.__setattr__(self.routine_config, attr, ele)

        object.__setattr__(self, "__pydantic_extra__", {})
        return self

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError

    def _jax_child_env(self) -> dict[str, str]:
        """Return explicit routine JAX options as child-process environment variables.

        :return: environment variables required to reproduce configured JAX options
        """
        if self.routine_config is None:
            return {}

        env: dict[str, str] = {}
        if self.routine_config.jax_platforms is not None:
            env["JAX_PLATFORMS"] = self.routine_config.jax_platforms
        if self.routine_config.jax_enable_x64 is not None:
            env["JAX_ENABLE_X64"] = str(self.routine_config.jax_enable_x64).lower()
        return env


class RoutineSpec(DictModel):
    """Immutable YAML recipe that materializes a fresh routine per invocation.

    :param recipe: copied :class:`romjax.YamlSource` retained for repeatable runs
    :param runtime_root: optional shared runtime-root override
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    recipe: Any
    runtime_root: str | Path | None = None
    _template: Any = PrivateAttr()
    _resolved_runtime_root: Path | None = PrivateAttr(default=None)

    @classmethod
    def from_yaml(
        cls,
        source: str | Path | bytes | "YamlSource",
        *,
        runtime_root: str | Path | None = None,
    ) -> "RoutineSpec":
        """Capture YAML without constructing the configured routine.

        :param source: YAML text, path, bytes, or a preserved YAML source
        :param runtime_root: optional CLI-style shared-root override
        :return: repeatable routine recipe
        """
        from romjax.configuration import YamlTemplate

        template = YamlTemplate.from_source(source, preserve_sources=True)
        return cls(recipe=deepcopy(template.source), runtime_root=runtime_root)

    @model_validator(mode="after")
    def _capture_template(self) -> "RoutineSpec":
        from romjax.configuration import YamlTemplate

        self._template = YamlTemplate.from_source(deepcopy(self.recipe), preserve_sources=True)
        return self

    @property
    def resolved_runtime_root(self) -> Path | None:
        """Return the most recently selected shared runtime directory."""
        return self._resolved_runtime_root

    @staticmethod
    def _mapping_value(node: Any, key: str) -> Any | None:
        import yaml

        if not isinstance(node, yaml.MappingNode):
            return None
        return next((value for item, value in node.value
                     if isinstance(item, yaml.ScalarNode) and item.value == key), None)

    @staticmethod
    def _node_value(node: Any) -> Any:
        import romjax

        return romjax.YamlLoader._template_node_value(node) if node is not None else None

    def _static_template(self):
        import romjax
        from romjax.configuration import YamlTemplate

        node = romjax.YamlLoader._render_templates(self._template.source.node, defer_runtime=True)
        return YamlTemplate.from_source(
            romjax.YamlSource(node, self._template.source.source_path, self._template.source.label),
            preserve_sources=True,
        )

    def _selected_root(self, parent_root: Path | None) -> tuple[Path | None, bool]:
        static = self._static_template()
        node = static.source.node
        configured = self.runtime_root
        if configured is None:
            configured = self._node_value(self._mapping_value(node, "runtime_root"))
        if configured is None:
            return (parent_root.absolute() if parent_root is not None else None), False
        text = str(configured)
        if "{{runtime." in text:
            raise ValueError("runtime_root cannot depend on runtime context.")
        template = Path(text)
        if parent_root is not None:
            return parent_root.absolute(), False
        root = Routine._create_runtime_root(template)
        return root, "{{timestamp" in text

    def _materialize(self, shared_root: Path | None) -> Routine:
        import yaml

        import romjax
        from romjax.configuration import ConfigAssignment, YamlTemplate

        static = self._static_template()
        raw_root = self._node_value(self._mapping_value(static.source.node, "root"))
        artifact_root: Path | None = None
        if raw_root is not None:
            root_source = static.render_runtime(RuntimeContext(shared_root, None))
            resolved_root = self._node_value(self._mapping_value(root_source.source.node, "root"))
            artifact_root = Path(resolved_root).expanduser()
            if not artifact_root.is_absolute():
                artifact_root = (shared_root / artifact_root if shared_root is not None else artifact_root.absolute())
            artifact_root = artifact_root.absolute()

        context = RuntimeContext(shared_root, artifact_root)
        assignments: list[ConfigAssignment] = []
        if artifact_root is not None:
            assignments.append(ConfigAssignment(path=("root",), value=str(artifact_root)))
        if shared_root is not None:
            assignments.append(ConfigAssignment(path=("runtime_root",), value=str(shared_root)))
        source = static.render(assignments) if assignments else static.source
        source_node = deepcopy(source.node)

        def defer_children(node: yaml.Node) -> None:
            if not isinstance(node, yaml.MappingNode):
                return
            for key, value in node.value:
                if (
                    isinstance(key, yaml.ScalarNode)
                    and key.value == "routines"
                    and isinstance(value, yaml.SequenceNode)
                ):
                    for index, child in enumerate(value.value):
                        if child.tag != romjax.YamlLoader.YAML_SOURCE_TAG and child.tag.startswith(
                            (romjax.YamlLoader.ROMX_TAG, romjax.YamlLoader.PYDANTIC_TAG)
                        ):
                            payload = yaml.MappingNode(
                                "tag:yaml.org,2002:map",
                                [
                                    (yaml.ScalarNode("tag:yaml.org,2002:str", "node"), child),
                                    (yaml.ScalarNode("tag:yaml.org,2002:str", "source_path"),
                                     yaml.ScalarNode("tag:yaml.org,2002:str", str(source.source_path or ""))),
                                ],
                            )
                            payload.tag = romjax.YamlLoader.YAML_SOURCE_TAG
                            value.value[index] = payload
                elif isinstance(value, yaml.MappingNode):
                    defer_children(value)

        defer_children(source_node)
        deferred = romjax.YamlSource(source_node, source.source_path, source.label)
        rendered = YamlTemplate.from_source(deferred, preserve_sources=True).render_runtime(context)
        with runtime_context(context):
            routine = romjax.load(rendered.source)
        if not isinstance(routine, Routine):
            raise ValueError(f"CompositeRoutine child must validate to Routine, got {type(routine).__name__}.")
        return routine

    @contextmanager
    def execution(self, *, parent_root: Path | None = None) -> Iterator[Routine]:
        """Materialize and execute one fresh routine in its runtime scope.

        :param parent_root: inherited shared runtime directory
        :return: the concrete routine for this invocation
        """
        shared_root, owns_timestamp_root = self._selected_root(parent_root)
        self._resolved_runtime_root = shared_root
        started = False
        try:
            routine = self._materialize(shared_root)
            with routine.run_context(parent_root=shared_root):
                started = True
                yield routine
        except BaseException:
            if not started and owns_timestamp_root and shared_root is not None:
                shutil.rmtree(shared_root, ignore_errors=True)
            raise

    def run(self) -> int:
        """Materialize one routine and call its ``run`` method once."""
        with self.execution() as routine:
            return routine.run()


class CompositeOverrideCase(BaseModel):
    """One named configuration value in a composite override.

    :param name: case name used in ``case_root``
    :param value: arbitrary YAML value merged into the base configuration
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)

    name: str
    value: Any = None


class CompositeOverride(BaseModel):
    """One dimension of a composite routine Cartesian product.

    :param name: override dimension name
    :param cases: named candidate values for this dimension
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)

    name: str
    cases: tuple[CompositeOverrideCase, ...]

    @field_validator("cases", mode="before")
    @classmethod
    def _coerce_cases(cls, value: Any) -> Any:
        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise ValueError("Composite override cases must be a non-empty sequence.")
        if not value:
            raise ValueError("Composite override cases must be a non-empty sequence.")
        return value

    @model_validator(mode="after")
    def _validate_case_names(self) -> "CompositeOverride":
        if len({case.name for case in self.cases}) != len(self.cases):
            raise ValueError(f"Composite override {self.name!r} has duplicate case names.")
        return self


@dataclass(frozen=True)
class _CompositeCase:
    """Prepared child routine input and its display label."""

    label: str
    value: Any


@dataclass(frozen=True)
class _CompositeCaseResult:
    """Result of executing one prepared child routine."""

    case: _CompositeCase
    exit_code: int | None = None
    routine_name: str | None = None


@contextmanager
def _composite_execution(value: Any, parent_root: Path | None) -> Iterator[Routine]:
    """Materialize one YAML child at the boundary immediately before execution."""
    import romjax

    if isinstance(value, romjax.YamlSource | str | Path | bytes):
        with RoutineSpec.from_yaml(value).execution(parent_root=parent_root) as routine:
            yield routine
        return
    if not isinstance(value, Routine):
        raise ValueError(f"CompositeRoutine child must validate to Routine, got {type(value).__name__}.")
    with value.run_context(parent_root=parent_root):
        yield value


def _load_composite_routine(value: Any) -> Routine:
    """Load a legacy child immediately for direct validation callers."""
    import romjax

    routine = romjax.load(value) if isinstance(value, romjax.YamlSource | str | Path | bytes) else value
    if not isinstance(routine, Routine):
        raise ValueError(f"CompositeRoutine child must validate to Routine, got {type(routine).__name__}.")
    return routine


class _CompositeWorker(TaskWorker):
    def __init__(self, cases: Sequence[_CompositeCase] = (), *, write_resolved: bool = False,
                 runtime_root: Path | None = None) -> None:
        self.cases = cases
        self.write_resolved = write_resolved
        self.runtime_root = runtime_root

    def prepare(self, *, remote: bool) -> "_CompositeWorker":
        if remote and any(isinstance(case.value, Routine) for case in self.cases):
            raise ValueError("Remote CompositeRoutine execution requires YAML-backed routine specifications.")
        return _CompositeWorker(write_resolved=self.write_resolved, runtime_root=self.runtime_root)

    def run(self, case: _CompositeCase, context: TaskContext) -> _CompositeCaseResult:
        with _composite_execution(case.value, self.runtime_root) as routine:
            root = getattr(routine, "root", None)
            if self.write_resolved and root is not None:
                import romjax

                root_path = Path(root)
                root_path.mkdir(parents=True, exist_ok=True)
                romjax.dump(routine, root_path / "resolved.yml", sort_keys=False,
                            _preserve_yaml_sources=True, _preserve_model_types=True)
            return _CompositeCaseResult(case, exit_code=int(routine.run()), routine_name=type(routine).__name__)


_CASE_ROOT_PATTERN = re.compile(r"\{\{\s*case_root(?P<index>\s*\[[^\]]+\])?\s*\}\}")


class CompositeRoutine(Routine):
    """Run child routines directly or from a Cartesian product of YAML overrides.

    :param routines: legacy child routine specifications
    :param base: base YAML source to merge in base/overrides mode
    :param overrides: named override dimensions in base/overrides mode
    :param executor: case scheduling backend
    :param failure_policy: handling for non-zero child exits or raised exceptions
    :param write_resolved: write each child to ``root/resolved.yml`` before running;
        defaults to false for ``routines`` and true for ``base``/``overrides``.
        Enabled snapshots require YAML-serializable child state.
    """

    routines: list[Any] | None = None
    base: Any | None = None
    overrides: tuple[CompositeOverride, ...] | None = None
    executor: Annotated[ExecutorConfig, BeforeValidator(ExecutorConfig.from_dict)] = Field(
        default_factory=ExecutorConfig
    )
    failure_policy: Literal["stop", "continue", "force"] = "stop"
    write_resolved: bool | None = None
    _source_path: Path | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _from_plain_list(cls, value: Any) -> Any:
        """Allow direct validation from a list of routines."""
        return {"routines": value} if isinstance(value, list | tuple) else value

    @field_validator("routines", mode="before")
    @classmethod
    def _validate_routines(cls, value: Any) -> Any:
        if value is None:
            return value
        return list(value) if isinstance(value, list | tuple) else [value]

    @model_validator(mode="after")
    def _validate_mode(self) -> "CompositeRoutine":
        import romjax

        object.__setattr__(self, "_source_path", romjax.YamlLoader.current_source_path())
        has_legacy = self.routines is not None
        has_expansion = self.base is not None or self.overrides is not None
        if has_legacy == has_expansion:
            raise ValueError("CompositeRoutine requires either routines or both base and overrides.")
        if has_expansion and (self.base is None or not self.overrides):
            raise ValueError("CompositeRoutine base/overrides mode requires both base and a non-empty overrides list.")
        if self.overrides is not None and len({item.name for item in self.overrides}) != len(self.overrides):
            raise ValueError("CompositeRoutine override names must be unique.")
        return self

    def _resolve_child_spec(self, value: Any) -> Any:
        """Resolve a legacy YAML path relative to this composite's source file."""
        import romjax

        if isinstance(value, str | Path):
            if self._source_path is not None:
                return romjax.YamlLoader._resolve_override_path(str(value), self._source_path)
            return romjax.YamlLoader.resolve_parent_path(value)
        return value

    def _validate_routine(self, value: Any) -> Routine:
        """Resolve and validate one legacy child routine specification.

        :param value: routine instance or YAML-backed child specification
        :return: validated child routine
        """
        return _load_composite_routine(self._resolve_child_spec(value))

    @staticmethod
    def _render_case_root(text: str, parts: tuple[str, ...]) -> str:
        """Replace supported case-root templates in one YAML scalar."""
        def integer(node: ast.expr | None) -> int | None:
            if node is None:
                return None
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                return node.value
            if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, int)
            ):
                return -node.operand.value
            raise ValueError("case_root indices and slices only support integers.")

        def replace(match: re.Match[str]) -> str:
            index = match.group("index")
            selected: str | tuple[str, ...] = parts
            if index:
                expression = ast.parse(f"value{index}", mode="eval").body
                if not isinstance(expression, ast.Subscript) or not isinstance(expression.value, ast.Name):
                    raise ValueError("Invalid case_root template expression.")
                index_node = expression.slice
                if not isinstance(index_node, ast.Slice):
                    selected = parts[integer(index_node)]
                elif isinstance(index_node, ast.Slice):
                    bounds = (index_node.lower, index_node.upper, index_node.step)
                    selected = parts[slice(*(integer(node) for node in bounds))]
            return selected if isinstance(selected, str) else "/".join(selected)

        return _CASE_ROOT_PATTERN.sub(replace, text)

    def _render_node(self, node: yaml.Node, parts: tuple[str, ...]) -> yaml.Node:
        """Return a copied YAML node with case-root templates rendered."""
        import copy

        result = copy.deepcopy(node)
        def visit(current: yaml.Node) -> None:
            if isinstance(current, yaml.ScalarNode) and isinstance(current.value, str):
                current.value = self._render_case_root(current.value, parts)
            elif isinstance(current, yaml.SequenceNode):
                for item in current.value:
                    visit(item)
            elif isinstance(current, yaml.MappingNode):
                for key, value in current.value:
                    visit(key)
                    visit(value)
        visit(result)
        return result

    def _base_source(self) -> tuple[yaml.Node, Path | None]:
        """Load the raw base node without constructing its routine."""
        import romjax

        if isinstance(self.base, romjax.YamlSource):
            if self.base.node is None:
                raise RoutineError("CompositeRoutine base source is empty.")
            return self.base.node, self.base.source_path
        if isinstance(self.base, str | Path):
            path = romjax.YamlLoader._resolve_override_path(str(self.base), self._source_path)
            node, source_path = romjax.YamlLoader._compose_resolved_node(path)
            if node is None:
                raise RoutineError("CompositeRoutine base source is empty.")
            return node, source_path
        raise ValueError("CompositeRoutine base must be a YAML path or YamlSource.")

    def _expanded_cases(self) -> list[_CompositeCase]:
        """Build merged YAML sources for every selected override combination."""
        import romjax

        base_node, base_path = self._base_source()
        cases: list[_CompositeCase] = []
        assert self.overrides is not None
        for selected in product(*(item.cases for item in self.overrides)):
            parts = tuple(f"{override.name}={case.name}" for override, case in zip(self.overrides, selected))
            merged = self._render_node(base_node, parts)
            merged = romjax.YamlLoader._resolve_parent_refs(merged, base_path)
            for case in selected:
                if case.value is None:
                    continue
                if isinstance(case.value, romjax._DeleteMarker):
                    raise ValueError("CompositeRoutine override cases cannot delete the root configuration.")
                override_text = romjax.YamlLoader.dump(
                    case.value,
                    sort_keys=False,
                    _preserve_yaml_sources=True,
                )
                override_node = yaml.compose(override_text, Loader=yaml.SafeLoader)
                if override_node is not None:
                    override_node = self._render_node(override_node, parts)
                    override_node = romjax.YamlLoader._resolve_parent_refs(override_node, self._source_path)
                    merged = romjax.YamlLoader._merge_nodes(merged, override_node)
            cases.append(_CompositeCase("/".join(parts), romjax.YamlSource(merged, base_path)))
        return cases

    def _cases(self) -> list[_CompositeCase]:
        """Return all child inputs in the selected composite mode."""
        if self.routines is not None:
            return [
                _CompositeCase(f"child {index}", self._resolve_child_spec(value))
                for index, value in enumerate(self.routines)
            ]
        return self._expanded_cases()

    @staticmethod
    def _failure_summary(failures: list[str]) -> str:
        lines = ["CompositeRoutine failures:"]
        lines.extend(f"- {failure}" for failure in failures)
        return "\n".join(lines)

    def run(self) -> int:
        """Run all generated or explicitly supplied child routines."""
        failures: list[str] = []
        exit_code = 0
        cases = self._cases()

        def handle(outcome):
            nonlocal exit_code
            label = outcome.task.label
            result = outcome.value
            if outcome.error is not None:
                value = outcome.task.payload.value
                name = f" ({type(value).__name__})" if isinstance(value, Routine) else ""
                summary = f"{label}{name} raised {type(outcome.error).__name__}: {outcome.error}"
                logger.exception("CompositeRoutine {}", summary)
                code = 1
            elif result.exit_code == 0:
                return ExecutionAction.CONTINUE
            else:
                summary = f"{label} ({result.routine_name}) exited with code {result.exit_code}"
                code = result.exit_code or 1
            failures.append(summary)
            if exit_code == 0:
                exit_code = code
            return ExecutionAction.STOP if self.failure_policy == "stop" else ExecutionAction.CONTINUE

        tasks = [TaskSpec(str(index), case.label, case) for index, case in enumerate(cases)]
        write_resolved = self.write_resolved if self.write_resolved is not None else self.routines is None
        worker = _CompositeWorker(cases, write_resolved=write_resolved, runtime_root=self.resolved_runtime_root)
        self.executor.execute(tasks, worker, on_result=handle,
                              error_policy="collect" if self.failure_policy == "force" else "raise",
                              jax_env=self._jax_child_env(), collect_results=False)
        if failures:
            logger.error(self._failure_summary(failures))
        return exit_code


class RoutineError(RuntimeError):
    """Raised when routines encounter invalid local state."""
    pass
