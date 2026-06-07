"""Module for assorted processing utilities."""
import subprocess
import threading
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel

__all__ = ['get_gpu_memory', 'print_gpu_memory', 'monitor_gpu_memory', 'save_h5', 'load_h5', 'required_fields']


class _NullProgress:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        return False
    def __call__(self):
        pass
    def text(self, msg):
        pass
    

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
