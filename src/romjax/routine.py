"""Base routine classes."""

from abc import ABC, abstractmethod
from importlib import import_module as _import_module
from pathlib import Path
from typing import Annotated, Any, Callable, Mapping

import matplotlib
import matplotlib.pyplot as plt
from alive_progress import config_handler
from loguru import logger
from pydantic import BaseModel, BeforeValidator, ConfigDict, TypeAdapter, field_validator, model_validator

from romjax.plotting import GridplotConfig
from romjax.typing import DictModel, WriteStream, from_yaml

__all__ = ["Routine", "RoutineConfig", "RoutineError", "LoggerConfig", "ProgressBarConfig"]

_LAZY_EXPORTS = {
    "DataGeneration": ("romjax.data_gen", "DataGeneration"),
    "GraphTrain": ("romjax.train", "GraphTrain"),
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
    
    :ivar mplstyle: matplotlib plot style
    :ivar gridplot: gridplot style
    :ivar logger: loguru configuration
    :ivar progress_bar: alive_bar configuration
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True, extra="forbid")

    mplstyle: str | Path | Mapping | None = None
    gridplot: GridplotConfig | None = None
    logger: LoggerConfig | None = None
    progress_bar: ProgressBarConfig | None = None

    @model_validator(mode="after")
    def configure(self):
        if (style := self.mplstyle) is not None:    # plt style
            if isinstance(style, str | Path):
                plt.style.use(style)
            else:
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

    @model_validator(mode="after")
    def _get_routine_configs_from_extra(self):
        if self.model_extra is None or len(self.model_extra) == 0:
            return self
        
        for k in self.model_extra:
            if k not in RoutineConfig.model_fields:
                raise ValueError(f"Extra Routine argument '{k}' not recognized. Only global configs: "
                                 f"{RoutineConfig.model_fields} allowed.")
        
        validated_extra = RoutineConfig.model_validate(self.model_extra)  # updates globals

        # Update the instance config just for consistency
        if self.routine_config is None:
            object.__setattr__(self, "routine_config", validated_extra)
        else:
            for attr in RoutineConfig.model_fields:
                if (ele := getattr(validated_extra, attr, None)) is not None:
                    # Merge gridplot configs, all others will override completely
                    if attr == "gridplot" and (curr_ele := self.routine_config.gridplot) is not None:
                        GridplotConfig.merge(curr_ele, ele, in_place=True)
                    else:
                        object.__setattr__(self.routine_config, attr, ele)

        object.__setattr__(self, "__pydantic_extra__", {})
        return self

    @abstractmethod
    def run(self) -> int:
        raise NotImplementedError


class RoutineError(RuntimeError):
    """Raised when routines encounter invalid local state."""
    pass
