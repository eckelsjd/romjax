"""Explicit runtime context shared by recipe materialization and workers."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_ACTIVE_RUNTIME_CONTEXT: ContextVar["RuntimeContext | None"] = ContextVar(
    "romjax_active_runtime_context", default=None
)


@dataclass(frozen=True)
class RuntimeContext:
    """Concrete paths available while one routine is materialized.

    :param root: absolute shared workflow directory, if one was selected
    :param artifact_root: absolute artifact directory owned by the current routine
    """

    root: Path | None = None
    artifact_root: Path | None = None

    def __post_init__(self) -> None:
        for name in ("root", "artifact_root"):
            value = getattr(self, name)
            if value is None:
                continue
            path = Path(value).expanduser().absolute()
            object.__setattr__(self, name, path)


def current_runtime_context() -> RuntimeContext | None:
    """Return the context inherited by nested YAML construction, if any."""
    return _ACTIVE_RUNTIME_CONTEXT.get()


@contextmanager
def runtime_context(context: RuntimeContext) -> Iterator[None]:
    """Make one concrete runtime context available to nested YAML validators."""
    token = _ACTIVE_RUNTIME_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_RUNTIME_CONTEXT.reset(token)
