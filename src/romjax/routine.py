"""Base routine classes."""

from abc import ABC, abstractmethod
from importlib import import_module as _import_module
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Mapping

import matplotlib
import matplotlib.pyplot as plt
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

from romjax.plotting import GridplotConfig
from romjax.typing import DictModel, WriteStream, from_yaml

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
    "CompareTable": ("romjax.compare", "CompareTable")
}

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


class CompositeRoutine(Routine):
    """
    Run a sequence of child routines.

    :param routines: routines, inline routine objects, or YAML files resolving to routines
    :param failure_policy: handling for non-zero child exits or raised exceptions
    """

    routines: list[Any]
    failure_policy: Literal["stop", "continue", "force"] = "stop"
    _source_path: Path | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _from_plain_list(cls, value):
        """Allow direct validation from a list of routines."""
        if isinstance(value, list | tuple):
            return {"routines": value}
        return value

    @field_validator("routines", mode="before")
    @classmethod
    def _validate_routines(cls, value):
        """Normalize child routine inputs without validating them yet."""
        if not isinstance(value, list | tuple):
            value = [value]

        return list(value)

    @model_validator(mode="after")
    def _capture_source_path(self):
        """Remember the YAML source path for runtime child resolution."""
        import romjax

        object.__setattr__(self, "_source_path", romjax.YamlLoader.current_source_path())
        return self

    def _validate_routine(self, value: Any) -> Routine:
        """Validate one child routine specification.

        :param value: routine instance or YAML path
        :return: validated routine
        """
        if isinstance(value, Routine):
            return value

        import romjax

        if isinstance(value, romjax.YamlSource):
            value = romjax.load(value)
        elif isinstance(value, str | Path | bytes):
            if isinstance(value, str | Path):
                source_path = self._source_path
                if source_path is not None:
                    stream = romjax.YamlLoader._resolve_override_path(str(value), source_path)
                else:
                    stream = romjax.YamlLoader.resolve_parent_path(value)
            else:
                stream = value
            value = romjax.load(stream)

        if not isinstance(value, Routine):
            raise ValueError(f"CompositeRoutine child must validate to Routine, got {type(value).__name__}.")

        return value

    @staticmethod
    def _failure_summary(failures: list[str]) -> str:
        """Return a compact multi-line failure summary.

        :param failures: collected failure descriptions
        :return: formatted summary
        """
        lines = ["CompositeRoutine failures:"]
        lines.extend(f"- {failure}" for failure in failures)
        return "\n".join(lines)

    def run(self) -> int:
        """Run child routines sequentially."""
        failures: list[str] = []
        exit_code = 0

        for index, routine_spec in enumerate(self.routines):
            routine_label = f"child {index}"
            try:
                routine = self._validate_routine(routine_spec)
            except Exception as exc:
                if self.failure_policy != "force":
                    raise

                summary = f"{routine_label} failed validation with {type(exc).__name__}: {exc}"
                failures.append(summary)
                if exit_code == 0:
                    exit_code = 1
                logger.exception("CompositeRoutine {}", summary)
                continue

            routine_label = f"{routine_label} ({type(routine).__name__})"
            try:
                child_code = int(routine.run())
            except Exception as exc:
                if self.failure_policy != "force":
                    raise

                summary = f"{routine_label} raised {type(exc).__name__}: {exc}"
                failures.append(summary)
                if exit_code == 0:
                    exit_code = 1
                logger.exception("CompositeRoutine {}", summary)
                continue

            if child_code == 0:
                continue

            summary = f"{routine_label} exited with code {child_code}"
            failures.append(summary)
            if exit_code == 0:
                exit_code = child_code

            if self.failure_policy == "stop":
                logger.error(self._failure_summary(failures))
                return child_code

        if failures:
            logger.error(self._failure_summary(failures))

        return exit_code


class RoutineError(RuntimeError):
    """Raised when routines encounter invalid local state."""
    pass
