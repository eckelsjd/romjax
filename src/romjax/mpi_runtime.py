"""Optional MPI context shared by the CLI and GridSearch executor."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from romjax.routine import RoutineError

_MPI_LAUNCH_KEYS = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMI_RANK",
    "PMIX_RANK",
    "MV2_COMM_WORLD_SIZE",
    "SLURM_PROCID",
)
_active_executor: Any | None = None
_active_size = 0


def active_executor() -> Any | None:
    """Return the MPI pool owned by the current CLI invocation, if any."""
    return _active_executor


def active_size() -> int:
    """Return the active MPI world size."""
    return _active_size


def _require_mpi() -> tuple[Any, Any]:
    """Import the optional MPI dependency with an actionable error."""
    try:
        from mpi4py import MPI
        from mpi4py.futures import MPICommExecutor
    except (ImportError, OSError, RuntimeError) as exc:
        message = "MPI GridSearch requires the mpi extra and a working MPI runtime (install with uv sync --extra mpi)."
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        raise RoutineError(message) from exc
    return MPI, MPICommExecutor


@contextmanager
def cli_context() -> Iterator[bool]:
    """Give rank zero the CLI and reserve other launched ranks for case work.

    :return: ``True`` on the controlling rank, ``False`` on workers
    """
    global _active_executor, _active_size

    if os.environ.get("ROMJAX_MPI_CASE_CHILD") == "1" or not any(key in os.environ for key in _MPI_LAUNCH_KEYS):
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
