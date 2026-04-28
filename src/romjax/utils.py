"""Module for assorted processing utilities."""
import logging
import subprocess
import sys
import threading
from os import PathLike
from pathlib import Path
from typing import Any, Generator, Mapping

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

__all__ = ['to_pytree', 'merge_pytrees', 'iter_pytree', 'pytree_at', 'get_logger', 'tree_l2_norm', 
           'get_gpu_memory', 'print_gpu_memory', 'monitor_gpu_memory', 'save_h5', 'load_h5', 'Logger']


LOG_FORMATTER = logging.Formatter(u"%(asctime)s — [%(levelname)s] — %(name)-10s — %(message)s")

# TODO: maybe some interesting ideas with a custom PyTree object that implements magic methods
# could support iter, index, len as well as add, sub, mult, broadcast and other array operations

@jax.jit
def tree_l2_norm(tree: PyTree):
    return jnp.sqrt(jax.tree.reduce(lambda acc, x: acc + jnp.sum(x**2), tree, 0.0))


def to_pytree(value: PyTree) -> PyTree:
    """Convert nested pydantic models and dicts to a PyTree of just 
    dicts,tuples,lists -- anything else is left as a leaf node (i.e. jax arrays)."""
    if isinstance(value, BaseModel):
        data = value.model_dump()
        return {k: to_pytree(v) for k, v in data.items()}
    if isinstance(value, Mapping):
        return {k: to_pytree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_pytree(v) for v in value)
    if isinstance(value, list):
        return [to_pytree(v) for v in value]
    
    return value


def merge_pytrees(defaults: PyTree, overrides: PyTree) -> PyTree:
    """Merge pytrees, overwriting existing paths and adding any new ones.
    
    :param defaults: the existing pytree
    :param overrides: the pytree to merge
    :return: a new merged pytree
    """
    if overrides is None:
        return defaults
    if defaults is None:
        return overrides
    
    if isinstance(defaults, Mapping) and isinstance(overrides, Mapping):
        merged: dict = dict(defaults)
        for key, value in overrides.items():
            if key in merged:
                merged[key] = merge_pytrees(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    if isinstance(defaults, tuple) and isinstance(overrides, tuple):
        merged = []
        common = min(len(defaults), len(overrides))
        for idx in range(common):
            merged.append(merge_pytrees(defaults[idx], overrides[idx]))
        if len(defaults) > common:
            merged.extend(defaults[common:])
        if len(overrides) > common:
            merged.extend(overrides[common:])
        return tuple(merged)
    
    if isinstance(defaults, list) and isinstance(overrides, list):
        merged_list: list = []
        common = min(len(defaults), len(overrides))
        for idx in range(common):
            merged_list.append(merge_pytrees(defaults[idx], overrides[idx]))
        if len(defaults) > common:
            merged_list.extend(defaults[common:])
        if len(overrides) > common:
            merged_list.extend(overrides[common:])
        return merged_list
    
    return overrides


def iter_pytree(tree: PyTree) -> Generator[PyTree, None, None]:
    """Yield per-sample pytrees from a batched pytree with a leading batch axis."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    if not leaves:
        return
    batch_size = leaves[0].shape[0]
    for i in range(batch_size):
        yield jax.tree_util.tree_unflatten(treedef, [leaf[i] for leaf in leaves])


def pytree_at(tree: PyTree, index: int) -> PyTree:
    """Return a pytree with each leaf at the provided index."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return jax.tree_util.tree_unflatten(treedef, [leaf[index] for leaf in leaves])


def get_logger(name: str, stdout: bool = True, log_file: str | Path = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a file/stdout logger with the given name.

    :param name: the name of the logger to return
    :param stdout: whether to add a stdout stream handler to the logger
    :param log_file: add file logging to this file (optional)
    :param level: the logging level to set
    :returns: the logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    if stdout:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(std_handler)
    if log_file is not None:
        f_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        f_handler.setLevel(level)
        f_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(f_handler)

    return logger


class Logger(BaseModel):
    """Simple logging with pydantic validation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    name: str
    stdout: bool = True
    file: Path | None = None
    _logger: logging.Logger | None = PrivateAttr(default=None)

    @model_validator(mode='after')
    def _set_logger(self):
        self._logger = get_logger(self.name, self.stdout, self.file)
        return self

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying ``logging.Logger``."""
        logger = self.__pydantic_private__.get("_logger")
        if logger is not None:
            return getattr(logger, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
    

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


def print_gpu_memory(logger: logging.Logger | None = None) -> None:
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
            
