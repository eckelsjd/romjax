"""Optional MPI context shared by the CLI and routine executors."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from concurrent.futures import Executor
from contextlib import contextmanager
from typing import Any

from romjax.execution import ExecutionError

_MPI_LAUNCH_KEYS = (
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_RANK",
    "PMI_SIZE",
    "PMI_RANK",
    "MV2_COMM_WORLD_SIZE",
    "MV2_COMM_WORLD_RANK",
    "PMIX_NAMESPACE",
    "PMIX_RANK",
)
_MPI_MODES = frozenset({"auto", "off", "on"})
_active_executor: Executor | None = None
_active_size = 0


def active_executor() -> Executor | None:
    """Return the MPI pool owned by the current CLI invocation, if any."""
    return _active_executor


def active_size() -> int:
    """Return the active MPI world size."""
    return _active_size


def _mpi_mode() -> str:
    """Return the user-selected MPI launch policy."""
    mode = os.environ.get("ROMJAX_MPI", "auto").strip().lower()
    if mode not in _MPI_MODES:
        choices = ", ".join(sorted(_MPI_MODES))
        raise ExecutionError(f"ROMJAX_MPI must be one of: {choices}.")
    return mode


def _has_parallel_world(size_key: str, rank_key: str) -> bool:
    """Return whether a complete launcher signature describes more than one rank."""
    size = os.environ.get(size_key)
    rank = os.environ.get(rank_key)
    if size is None or rank is None:
        return False
    try:
        return int(size) > 1 and int(rank) >= 0
    except ValueError:
        return False


def _has_mpi_launcher_environment() -> bool:
    """Return whether the environment contains a complete MPI launcher signature.

    Slurm allocation and task variables are intentionally excluded: they identify
    scheduled processes, not whether an application will use MPI.
    """
    return (
        _has_parallel_world("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK")
        or _has_parallel_world("PMI_SIZE", "PMI_RANK")
        or _has_parallel_world("MV2_COMM_WORLD_SIZE", "MV2_COMM_WORLD_RANK")
        or all(os.environ.get(key) for key in ("PMIX_NAMESPACE", "PMIX_RANK"))
    )


def _require_mpi() -> tuple[Any, Any]:
    """Import the optional MPI dependency with an actionable error."""
    try:
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
    except (ImportError, OSError, RuntimeError) as exc:
        message = "MPI execution requires the mpi extra and a working MPI runtime (install with uv sync --extra mpi)."
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise ExecutionError(message) from exc
    return MPI, MPICommExecutor


@contextmanager
def mpi_context() -> Iterator[bool]:
    """Own one MPI pool on rank zero and reserve other launched ranks for task work.

    Enter this context on every launched rank, before loading routine configuration.
    Only the controller should load configurations or call ``run()``. Outside an
    MPI launch it yields ``True`` without creating a pool. Control detection with
    ``ROMJAX_MPI``: ``"auto"`` (the default) recognizes complete implementation-
    specific signatures, ``"on"`` always initializes MPI, and ``"off"`` never
    initializes MPI. Slurm variables alone never trigger MPI initialization.

    :return: ``True`` on the controlling rank, ``False`` on workers
    """
    global _active_executor, _active_size

    if _active_executor is not None:
        yield True
        return

    mode = _mpi_mode()
    if (
        os.environ.get("ROMJAX_MPI_CASE_CHILD") == "1"
        or mode == "off"
        or (mode == "auto" and not _has_mpi_launcher_environment())
    ):
        yield True
        return

    MPI, MPICommExecutor = _require_mpi()
    with MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        if executor is None:
            yield False
            return
        _active_executor = executor
        _active_size = MPI.COMM_WORLD.Get_size()
        try:
            yield True
        finally:
            _active_executor = None
            _active_size = 0


# Backward-compatible CLI entry point.
cli_context = mpi_context
