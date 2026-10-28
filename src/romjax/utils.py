"""Module for assorted processing utilities."""
import asyncio
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from functools import wraps
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
from orbax.checkpoint import v1 as ocp
from pydantic import BaseModel

__all__ = ['get_gpu_memory', 'print_gpu_memory', 'monitor_gpu_memory', 'save_h5', 'load_h5', 'required_fields']

_ASYNCIO_WAKEUP_CHECKED = False
_ASYNCIO_WAKEUP_BLOCKED = False
_ASYNCIO_LOOP_FACTORY_LOCK = threading.RLock()


def _get_uvloop() -> Any | None:
    """Import uvloop when available."""
    try:
        import uvloop
    except ImportError:
        return None
    return uvloop


def _check_asyncio_thread_wakeup() -> Any | None:
    """Return uvloop when local socket wakeups are denied by the runtime.

    Orbax's background runner schedules work onto a selector event loop from
    another thread. In restricted Linux runtimes, socketpair sends can fail,
    leaving checkpoint operations waiting for callbacks that never wake the
    loop. uvloop uses a different wakeup mechanism. Its loop factories are
    applied only around synchronous Checkpointer methods.
    """
    global _ASYNCIO_WAKEUP_CHECKED, _ASYNCIO_WAKEUP_BLOCKED
    if _ASYNCIO_WAKEUP_CHECKED or sys.platform == "win32":
        return _get_uvloop() if _ASYNCIO_WAKEUP_BLOCKED else None
    _ASYNCIO_WAKEUP_CHECKED = True

    try:
        reader, writer = socket.socketpair()
        try:
            writer.send(b"x")
            reader.recv(1)
        finally:
            reader.close()
            writer.close()
    except OSError:
        _ASYNCIO_WAKEUP_BLOCKED = True
        return _get_uvloop()
    return None


@contextmanager
def _uvloop_checkpoint_call(uvloop: Any):
    """Run one Orbax call with uvloop factories, restoring asyncio afterward."""
    with _ASYNCIO_LOOP_FACTORY_LOCK:
        original_run = asyncio.run
        original_new_loop = asyncio.new_event_loop
        original_events_new_loop = asyncio.events.new_event_loop
        asyncio.run = uvloop.run
        asyncio.new_event_loop = uvloop.new_event_loop
        asyncio.events.new_event_loop = uvloop.new_event_loop
        try:
            yield
        finally:
            asyncio.run = original_run
            asyncio.new_event_loop = original_new_loop
            asyncio.events.new_event_loop = original_events_new_loop


def _wrap_checkpointer_method(checkpointer: Any, name: str, uvloop: Any) -> None:
    """Use a wakeup-compatible event loop for one synchronous Orbax method."""
    method = getattr(checkpointer, name, None)
    if method is None:
        return

    @wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with _uvloop_checkpoint_call(uvloop):
            return method(*args, **kwargs)

    setattr(checkpointer, name, wrapped)


class _NullProgress:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        return False
    def __call__(self):
        pass
    def text(self, msg):
        pass


def _create_checkpointer(directory: Path, **kwargs: Any) -> Any:
    """Create a checkpointer with synchronous directory setup.

    Orbax's background directory creation relies on an event-loop wakeup
    between threads. Synchronous setup avoids stalls on platforms where that
    wakeup mechanism is unavailable while retaining asynchronous checkpoint
    writes.
    """
    uvloop = _check_asyncio_thread_wakeup()

    context = ocp.Context()
    if hasattr(context, "asynchronous"):
        context.asynchronous.create_directories_asynchronously = False
    else:
        # Orbax versions before the mutable Context configuration API require
        # options to be passed at construction.
        context = ocp.Context(
            async_options=ocp.options.AsyncOptions(create_directories_asynchronously=False)
        )

    with context:
        if uvloop is None:
            checkpointer = ocp.training.Checkpointer(directory, **kwargs)
        else:
            with _uvloop_checkpoint_call(uvloop):
                checkpointer = ocp.training.Checkpointer(directory, **kwargs)
    if uvloop is not None:
        for method_name in ("save_checkpointables", "load_checkpointables", "close"):
            _wrap_checkpointer_method(checkpointer, method_name, uvloop)
    return checkpointer
    

def required_fields(model_cls: type[BaseModel], inherited: bool = True) -> set[str]:
    """Get the required fields of a pydantic model."""
    ignore = set()

    # Ignore inherited fields
    if not inherited:
        for base in model_cls.__mro__[1:]:
            if issubclass(base, BaseModel):
                ignore.update(getattr(base, "model_fields", {}))

    return {
        name
        for name, field in model_cls.model_fields.items()
        if name not in ignore and field.is_required()
    }


def format_time_engineering(seconds: float):
    """Helper to format times in common engineering magnitudes."""
    prefixes = [
        (1e-12, "ps"),
        (1e-9, "ns"),
        (1e-6, "μs"),
        (1e-3, "ms"),
        (1.0,  "s"),
        (1e3, "ks")
    ]

    # Find the appropriate prefix and scale
    for factor, suffix in prefixes:
        scaled = seconds / factor
        if 1 <= scaled < 1000:
            return f"{scaled:.1f} {suffix}"
    
    # Fall back to scientific notation if out of normal range
    return f"{seconds:.1e} s"


def _parse_nvidia_smi_output(output: str) -> list[tuple[int, int]]:
    """
    Parse the output of an nvidia-smi memory query.

    :param output: Raw stdout from nvidia-smi.
    :return: List of (used_mib, total_mib) tuples for each visible GPU.
    """
    entries: list[tuple[int, int]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        used_mib = int(parts[0])
        total_mib = int(parts[1])
        entries.append((used_mib, total_mib))
    return entries


def get_gpu_memory() -> list[tuple[int, int]]:
    """
    Query GPU memory usage via nvidia-smi.

    :return: List of (used_mib, total_mib) tuples for each visible GPU.
    :raises FileNotFoundError: If nvidia-smi is not available.
    :raises subprocess.CalledProcessError: If nvidia-smi fails.
    """
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        usage = _parse_nvidia_smi_output(result.stdout)
    except FileNotFoundError:
        print("GPU memory: nvidia-smi not found in PATH.")
        return
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown error"
        print(f"GPU memory: nvidia-smi failed ({stderr}).")
        return
    
    return usage


def print_gpu_memory(logger: Any | None = None) -> None:
    """
    Print the current GPU memory usage and total memory in MiB.

    :param logger: logging object to use (if None, just prints to console)
    """
    print_fn = print if logger is None else logger.info
    usage = get_gpu_memory()
    
    if not usage:
        print_fn("GPU memory: no GPUs detected.")
        return

    if len(usage) == 1:
        used_mib, total_mib = usage[0]
        print_fn(f"GPU memory: {used_mib} MiB / {total_mib} MiB")
        return

    for index, (used_mib, total_mib) in enumerate(usage):
        print_fn(f"GPU {index} memory: {used_mib} MiB / {total_mib} MiB")


def _monitor_loop(stop_event: threading.Event, interval_seconds: float) -> None:
    """
    Continuously print GPU memory usage until stop_event is set.

    :param stop_event: Event used to stop the loop.
    :param interval_seconds: Sleep interval between samples.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")

    while not stop_event.is_set():
        print_gpu_memory()
        stop_event.wait(interval_seconds)


def monitor_gpu_memory(interval_seconds: float = 5.0) -> tuple[threading.Thread, threading.Event]:
    """
    Start a background thread that periodically prints GPU memory usage.

    :param interval_seconds: Time between samples in seconds.
    :return: Tuple of (thread, stop_event).
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_monitor_loop,
        args=(stop_event, interval_seconds),
        daemon=True,
    )
    thread.start()
    return thread, stop_event


def save_h5(data: dict[str, Any], filename: str | PathLike, mode: str = 'a'):
    """Save data to h5 file."""

    def _recursively_save(h5group, obj):
        """Helper to recursively save dictionary to h5 file."""
        for key, val in obj.items():
            if isinstance(val, dict):
                subgroup = h5group[key] if key in h5group else h5group.create_group(key, track_order=True)
                _recursively_save(subgroup, val)
            else:
                if key in h5group:
                    del h5group[key]
                h5group.create_dataset(key, data=np.asarray(val))

    with h5py.File(Path(filename), mode, track_order=True) as f:
        for key in data:
            if isinstance(data[key], dict):  # recurse
                group = f[key] if key in f else f.create_group(key, track_order=True)
                _recursively_save(group, data[key])
            else:
                if key in f:
                    del f[key]
                f.create_dataset(key, data=np.asarray(data[key]))


def load_h5(data: dict[str, Any], filename: str | PathLike, mode: str = 'r', jax: bool = False):
    """Load data from h5 file into a dictionary. An empty dictionary will load everything.
    Selectively mark data to load in the dictionary with None.
    """

    def _recursively_load(node):
        """Return Python objects for an h5 node (Group -> dict, Dataset -> np.ndarray)."""
        if isinstance(node, h5py.Group):
            out = {}
            for k in node:
                out[k] = _recursively_load(node[k])
            return out
        else:
            return jnp.asarray(node[()]) if jax else node[()]

    def _recursively_fill(pattern: dict[str, Any], node):
        """Fill a pattern dict in-place from the corresponding h5 group/node."""
        for k in list(pattern.keys()):
            if k not in node:
                continue
            if pattern[k] is None:
                pattern[k] = _recursively_load(node[k])
            elif isinstance(pattern[k], dict) and isinstance(node[k], h5py.Group):
                _recursively_fill(pattern[k], node[k])
            # if pattern[k] is not None and not a dict, leave as-is

    with h5py.File(Path(filename), mode, track_order=True) as f:
        # Load everything
        if len(data) == 0:
            for key in f:
                data[key] = _recursively_load(f[key])
        # Selectively load requested data (supports nested dict patterns)
        else:
            _recursively_fill(data, f)

    return data
