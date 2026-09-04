"""Base routine classes."""

import ast
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import import_module as _import_module
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Mapping

import matplotlib
import matplotlib.pyplot as plt
import yaml
from alive_progress import alive_bar, config_handler
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

from romjax.plotting import GridplotConfig
from romjax.typing import DictModel, WriteStream, from_yaml
from romjax.utils import _NullProgress

__all__ = [
    "CompositeRoutine",
    "Routine",
    "RoutineConfig",
    "RoutineError",
    "LoggerConfig",
    "ProgressBarConfig",
]

_LAZY_EXPORTS = {
    "DataGeneration": ("romjax.data_gen", "DataGeneration"),
    "GridSearch": ("romjax.grid_search", "GridSearch"),
    "Train": ("romjax.train", "Train"),
    "CompareMetric": ("romjax.compare", "CompareMetric")
}

_COMPOSITE_PROGRESS_ACTIVE: ContextVar[bool] = ContextVar("romjax_composite_progress_active", default=False)

available = list(_LAZY_EXPORTS.keys())

def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = _import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


class ProgressBarConfig(DictModel):
    """Store configuration options for alive_bar."""

    @model_validator(mode="after")
    def _normalize_file(self):
        """Allow any write-able file stream."""
        ta = TypeAdapter(WriteStream)
        if hasattr(self, "file"):
            object.__setattr__(self, "file", ta.validate_python(self.file))
        return self


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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True, extra="forbid")

    jax_platforms: str | None = None
    jax_enable_x64: bool | None = None
    mplstyle: str | Path | Mapping | None = None
    gridplot: GridplotConfig | None = None
    logger: LoggerConfig | None = None
    progress_bar: ProgressBarConfig | None = None
    extra: Any | None = Field(exclude=True, default=None)

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
            romjax.plotting.set_global(**{name: getattr(plot, name) for name in type(plot).model_fields})
        
        if (log := self.logger) is not None:        # loguru
            logger.configure(**dict(log))

        if (bar := self.progress_bar) is not None:  # alive bar
            config_handler.set_global(**{key: getattr(bar, key) for key in (bar.model_extra or {})})
        
        return self

    
class Routine(BaseModel, ABC):
    """Base class mixin that provides the `run()` method. Can only pass extra options for global configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True, extra="allow")
    
    routine_config: Annotated[RoutineConfig | None, BeforeValidator(from_yaml)] = None

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
    error: Exception | None = None
    routine_name: str | None = None
    stage: Literal["validation", "run"] = "validation"


def _load_composite_routine(value: Any) -> Routine:
    """Load a routine from an already-resolved composite child value."""
    import romjax

    routine = romjax.load(value) if isinstance(value, romjax.YamlSource | str | Path | bytes) else value
    if not isinstance(routine, Routine):
        raise ValueError(f"CompositeRoutine child must validate to Routine, got {type(routine).__name__}.")
    return routine


def _run_composite_case(case: _CompositeCase) -> _CompositeCaseResult:
    """Run one child and write its resolved config when it owns a root directory."""
    try:
        routine = _load_composite_routine(case.value)
    except Exception as exc:
        return _CompositeCaseResult(case, error=exc)
    try:
        root = getattr(routine, "root", None)
        if root is not None:
            import romjax

            root_path = Path(root)
            root_path.mkdir(parents=True, exist_ok=True)
            romjax.dump(
                routine,
                root_path / "resolved.yml",
                sort_keys=False,
                _preserve_yaml_sources=True,
            )
        return _CompositeCaseResult(
            case,
            exit_code=int(routine.run()),
            routine_name=type(routine).__name__,
            stage="run",
        )
    except Exception as exc:
        return _CompositeCaseResult(case, error=exc, routine_name=type(routine).__name__, stage="run")


def _configure_composite_process(jax_env: Mapping[str, str]) -> None:
    """Apply parent routine JAX settings before a composite worker loads a child."""
    os.environ.update(jax_env)


class CompositeExecutorConfig(BaseModel, ABC):
    """Abstract scheduler configuration for composite routine cases."""

    show_progress: bool = True

    @classmethod
    def from_dict(cls, value: Any) -> "CompositeExecutorConfig":
        """Construct a concrete executor from YAML-friendly input.

        :param value: executor name, mapping, or existing executor config
        :return: normalized executor configuration
        """
        if isinstance(value, CompositeExecutorConfig):
            return value
        aliases = {
            "serial": "serial",
            "thread": "thread",
            "threads": "thread",
            "process": "process",
            "processes": "process",
        }
        if isinstance(value, str):
            value = {"kind": aliases.get(value, value)}
        if not isinstance(value, Mapping):
            raise ValueError("CompositeRoutine executor must be a string, mapping, or CompositeExecutorConfig.")
        kind = aliases.get(str(value.get("kind", "serial")))
        if kind == "serial":
            return CompositeSerialExecutorConfig.model_validate(value)
        if kind in {"thread", "process"}:
            return CompositeConcurrentExecutorConfig.model_validate({**value, "kind": kind})
        raise ValueError(f"Unknown CompositeRoutine executor: {value.get('kind')!r}.")

    @contextmanager
    def progress_context(self, total: int):
        """Return the master progress-bar context."""
        if not self.show_progress or _COMPOSITE_PROGRESS_ACTIVE.get():
            yield _NullProgress()
            return
        token = _COMPOSITE_PROGRESS_ACTIVE.set(True)
        try:
            with alive_bar(total) as bar:
                yield bar
        finally:
            _COMPOSITE_PROGRESS_ACTIVE.reset(token)

    @abstractmethod
    def run_cases(self, cases: Sequence[_CompositeCase], stop_on_failure: bool) -> list[_CompositeCaseResult]:
        """Run prepared cases and return completed results."""
        raise NotImplementedError


class CompositeSerialExecutorConfig(CompositeExecutorConfig):
    """Serial composite routine executor."""

    kind: Literal["serial"] = "serial"

    def run_cases(self, cases: Sequence[_CompositeCase], stop_on_failure: bool) -> list[_CompositeCaseResult]:
        results: list[_CompositeCaseResult] = []
        with self.progress_context(len(cases)) as bar:
            for case in cases:
                bar.text = case.label
                result = _run_composite_case(case)
                results.append(result)
                bar()
                if stop_on_failure and (result.error is not None or result.exit_code != 0):
                    break
        return results


class CompositeConcurrentExecutorConfig(CompositeExecutorConfig):
    """Thread- or process-based composite routine executor.

    :param kind: concurrent backend kind
    :param max_workers: worker count for the executor
    """

    kind: Literal["thread", "process"] = "process"
    max_workers: int | None = Field(default=None, gt=0)

    def _context(self, jax_env: Mapping[str, str]) -> ThreadPoolExecutor | ProcessPoolExecutor:
        if self.kind == "thread":
            return ThreadPoolExecutor(max_workers=self.max_workers)
        return ProcessPoolExecutor(
            self.max_workers,
            initializer=_configure_composite_process,
            initargs=(dict(jax_env),),
        )

    def run_cases(
        self,
        cases: Sequence[_CompositeCase],
        stop_on_failure: bool,
        jax_env: Mapping[str, str] | None = None,
    ) -> list[_CompositeCaseResult]:
        results: list[_CompositeCaseResult] = []
        pending = iter(cases)
        workers = self.max_workers or min(32, max(1, len(cases)))
        with self.progress_context(len(cases)) as bar, self._context(jax_env or {}) as executor:
            active: dict[Future[_CompositeCaseResult], _CompositeCase] = {}
            for _ in range(min(workers, len(cases))):
                case = next(pending, None)
                if case is not None:
                    active[executor.submit(_run_composite_case, case)] = case
            stopped = False
            while active:
                future = next(as_completed(active))
                case = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = _CompositeCaseResult(case, error=exc)
                results.append(result)
                bar.text = case.label
                bar()
                failed = result.error is not None or result.exit_code != 0
                if stop_on_failure and failed:
                    stopped = True
                if not stopped:
                    next_case = next(pending, None)
                    if next_case is not None:
                        active[executor.submit(_run_composite_case, next_case)] = next_case
        return results


_CASE_ROOT_PATTERN = re.compile(r"\{\{\s*case_root(?P<index>\s*\[[^\]]+\])?\s*\}\}")


class CompositeRoutine(Routine):
    """Run child routines directly or from a Cartesian product of YAML overrides.

    :param routines: legacy child routine specifications
    :param base: base YAML source to merge in base/overrides mode
    :param overrides: named override dimensions in base/overrides mode
    :param executor: case scheduling backend
    :param failure_policy: handling for non-zero child exits or raised exceptions
    """

    routines: list[Any] | None = None
    base: Any | None = None
    overrides: tuple[CompositeOverride, ...] | None = None
    executor: CompositeExecutorConfig = Field(default_factory=CompositeSerialExecutorConfig)
    failure_policy: Literal["stop", "continue", "force"] = "stop"
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

    @field_validator("executor", mode="before")
    @classmethod
    def _coerce_executor(cls, value: Any) -> CompositeExecutorConfig:
        return CompositeExecutorConfig.from_dict(value)

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
        if isinstance(self.executor, CompositeConcurrentExecutorConfig) and self.executor.kind == "process":
            if self.routines is not None and any(isinstance(item, Routine) for item in self.routines):
                raise ValueError("CompositeRoutine process executor requires YAML-backed routine specifications.")
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
        if isinstance(self.executor, CompositeConcurrentExecutorConfig):
            results = self.executor.run_cases(
                cases,
                stop_on_failure=self.failure_policy == "stop",
                jax_env=self._jax_child_env(),
            )
        else:
            results = self.executor.run_cases(cases, stop_on_failure=self.failure_policy == "stop")
        for result in results:
            label = result.case.label
            if result.error is not None:
                if self.failure_policy != "force":
                    raise result.error
                if result.stage == "validation":
                    summary = f"{label} failed validation with {type(result.error).__name__}: {result.error}"
                else:
                    summary = f"{label} ({result.routine_name}) raised {type(result.error).__name__}: {result.error}"
                logger.exception("CompositeRoutine {}", summary)
                code = 1
            elif result.exit_code == 0:
                continue
            else:
                summary = f"{label} ({result.routine_name}) exited with code {result.exit_code}"
                code = result.exit_code or 1
            failures.append(summary)
            if exit_code == 0:
                exit_code = code
            if self.failure_policy == "stop":
                break
        if failures:
            logger.error(self._failure_summary(failures))
        return exit_code


class RoutineError(RuntimeError):
    """Raised when routines encounter invalid local state."""
    pass
