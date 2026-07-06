"""JAX profiler helpers for ROM training and grid-search runs.

The helpers in this module are opt-in and controlled by environment variables so
they can be used without changing existing YAML configurations.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import jax

_TRUTHY_VALUES = {"1", "true", "yes", "on", "trace"}
_FALSEY_VALUES = {"0", "false", "no", "off", "none", ""}

__all__ = [
    "build_profile_env",
    "profile_enabled",
    "profile_label",
    "profile_options",
    "profile_step",
    "profile_trace",
    "profile_trace_dir",
    "profile_trace_root",
]


def _env_bool(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().lower()
    if normalized in _FALSEY_VALUES:
        return False
    if normalized in _TRUTHY_VALUES:
        return True
    return True


def _sanitize_label(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "profile"


def _trace_run_id() -> str:
    """Return a timestamped run identifier suitable for a trace directory name."""
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S") + f"-pid{os.getpid()}"


def _env_int(value: str | None, name: str) -> int | None:
    """Parse an optional integer environment value."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def profile_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether ROMJAX profiling is enabled in ``env`` or the current process."""
    source = os.environ if env is None else env
    return _env_bool(source.get("ROMJAX_PROFILE"))


def profile_label(default: str, env: Mapping[str, str] | None = None) -> str:
    """Return the current profiler label, falling back to ``default``."""
    source = os.environ if env is None else env
    return _sanitize_label(source.get("ROMJAX_PROFILE_LABEL", default))


def profile_trace_root(default_root: Path | None, env: Mapping[str, str] | None = None) -> Path:
    """Return the directory under which traces should be written."""
    source = os.environ if env is None else env
    if (explicit := source.get("ROMJAX_PROFILE_DIR")):
        return Path(explicit).expanduser().resolve()
    if default_root is not None:
        return Path(default_root).expanduser().resolve() / "profiles"
    return Path.cwd().resolve() / "romjax-profiles"


def profile_trace_dir(
    default_label: str,
    default_root: Path | None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Return the concrete trace directory for the current process, if enabled."""
    if not profile_enabled(env):
        return None
    source = os.environ if env is None else env
    if (explicit := source.get("ROMJAX_PROFILE_DIR")):
        return Path(explicit).expanduser().resolve()
    root = profile_trace_root(default_root, source)
    return root / f"{profile_label(default_label, source)}-{_trace_run_id()}"


def build_profile_env(
    default_label: str,
    default_root: Path | None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build child-process profiling overrides for a GridSearch case."""
    if not profile_enabled(env):
        return {}
    source = os.environ if env is None else env
    if (explicit := source.get("ROMJAX_PROFILE_DIR")):
        root = Path(explicit).expanduser().resolve()
        if default_root is not None:
            root = root / Path(default_root).name
    elif default_root is not None:
        root = Path(default_root).expanduser().resolve() / "profiles"
    else:
        root = Path.cwd().resolve() / "romjax-profiles"
    trace_dir = root / f"{profile_label(default_label, env)}-{_trace_run_id()}"
    return {
        "ROMJAX_PROFILE": "1",
        "ROMJAX_PROFILE_LABEL": profile_label(default_label, env),
        "ROMJAX_PROFILE_DIR": str(trace_dir),
    }


def profile_options(env: Mapping[str, str] | None = None) -> jax.profiler.ProfileOptions | None:
    """Return JAX profiler options configured by ROMJAX profile environment variables.

    :param env: environment mapping to read; defaults to :data:`os.environ`
    :return: configured JAX profiler options, or ``None`` when no option override is set
    """
    source = os.environ if env is None else env
    host_level = _env_int(source.get("ROMJAX_PROFILE_HOST_TRACER_LEVEL"), "ROMJAX_PROFILE_HOST_TRACER_LEVEL")
    device_level = _env_int(source.get("ROMJAX_PROFILE_DEVICE_TRACER_LEVEL"), "ROMJAX_PROFILE_DEVICE_TRACER_LEVEL")
    python_level = _env_int(source.get("ROMJAX_PROFILE_PYTHON_TRACER_LEVEL"), "ROMJAX_PROFILE_PYTHON_TRACER_LEVEL")

    if host_level is None and device_level is None and python_level is None:
        return None

    options = jax.profiler.ProfileOptions()
    if host_level is not None:
        options.host_tracer_level = host_level
    if device_level is not None:
        options.device_tracer_level = device_level
    if python_level is not None:
        options.python_tracer_level = python_level
    return options


@contextmanager
def profile_trace(
    default_label: str,
    default_root: Path | None,
    env: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Trace the enclosed block with ``jax.profiler`` when profiling is enabled."""
    trace_dir = profile_trace_dir(default_label, default_root, env)
    if trace_dir is None:
        with nullcontext():
            yield
        return

    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(trace_dir, profiler_options=profile_options(env)):
        yield


@contextmanager
def profile_step(
    default_label: str,
    step_num: int | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Annotate a single profiler step when tracing is enabled."""
    if not profile_enabled(env):
        with nullcontext():
            yield
        return

    if step_num is None:
        with jax.profiler.StepTraceAnnotation(default_label):
            yield
        return

    with jax.profiler.StepTraceAnnotation(default_label, step_num=step_num):
        yield


@contextmanager
def profile_annotation(label: str, env: Mapping[str, str] | None = None, **kwargs: Any) -> Iterator[None]:
    """Emit a lightweight nested profiler annotation when tracing is enabled."""
    if not profile_enabled(env):
        with nullcontext():
            yield
        return

    with jax.profiler.TraceAnnotation(label, **kwargs):
        yield
