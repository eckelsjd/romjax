"""A tiny deterministic case routine for the MPI GridSearch demo."""

import os
import socket
from pathlib import Path

import jax.numpy as jnp

from romjax.routine import Routine


class DemoCase(Routine):
    """Write one JAX-computed scalar loss in GridSearch's standard CSV format.

    :param root: case output directory
    :param value: grid candidate to evaluate
    """

    root: Path
    value: float

    def run(self) -> int:
        """Evaluate the case and save its loss.

        :return: zero on success
        """
        self.root.mkdir(parents=True, exist_ok=True)
        loss = float(jnp.square(jnp.asarray(self.value) - 2.0))
        (self.root / "loss.csv").write_text(f"step,loss\n0,{loss}\n", encoding="utf-8")

        mpi_rank = next(
            (
                os.environ[key]
                for key in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID")
                if key in os.environ
            ),
            "unknown",
        )
        mpi_size = next(
            (
                os.environ[key]
                for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE", "SLURM_NTASKS")
                if key in os.environ
            ),
            "unknown",
        )
        local_rank = next(
            (
                os.environ[key]
                for key in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "SLURM_LOCALID")
                if key in os.environ
            ),
            "unknown",
        )
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        print(
            f"MPI launcher: rank={mpi_rank}, size={mpi_size}, local_rank={local_rank}; "
            f"host={socket.gethostname()}, pid={os.getpid()}; "
            f"CPUs: system={os.cpu_count()}, accessible={len(affinity) if affinity is not None else 'unknown'}, "
            f"affinity={affinity if affinity is not None else 'unavailable'}",
            flush=True,
        )
        return 0
