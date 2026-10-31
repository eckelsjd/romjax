"""Tests for MPI launch detection at the CLI boundary."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from romjax import mpi_runtime
from romjax.execution import ExecutionError


@pytest.fixture
def clean_mpi_environment(monkeypatch):
    """Clear environment values consulted by the MPI context."""
    for key in (*mpi_runtime._MPI_LAUNCH_KEYS, "SLURM_PROCID", "ROMJAX_MPI"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_mpi_runtime(monkeypatch):
    """Replace MPI bootstrap with a two-rank in-memory context."""
    calls: list[None] = []

    class FakeWorld:
        @staticmethod
        def Get_size() -> int:
            return 2

    @contextmanager
    def fake_executor(_world, *, root: int):
        assert root == 0
        yield object()

    def fake_require_mpi():
        calls.append(None)
        return SimpleNamespace(COMM_WORLD=FakeWorld()), fake_executor

    monkeypatch.setattr(mpi_runtime, "_require_mpi", fake_require_mpi)
    return calls


def test_mpi_context_ignores_slurm_task_rank(clean_mpi_environment, monkeypatch):
    """A normal Slurm task must not bootstrap MPI merely because it has a rank."""
    monkeypatch.setenv("SLURM_PROCID", "0")

    with mpi_runtime.mpi_context() as is_master:
        assert is_master is True


@pytest.mark.parametrize(
    ("environment", "should_bootstrap"),
    [
        ({"OMPI_COMM_WORLD_SIZE": "4"}, False),
        ({"OMPI_COMM_WORLD_RANK": "0"}, False),
        ({"OMPI_COMM_WORLD_SIZE": "4", "OMPI_COMM_WORLD_RANK": "0"}, True),
        ({"PMI_SIZE": "4"}, False),
        ({"PMI_RANK": "0"}, False),
        ({"PMI_SIZE": "4", "PMI_RANK": "0"}, True),
        ({"PMIX_NAMESPACE": "job.0", "PMIX_RANK": "0"}, True),
    ],
)
def test_mpi_context_auto_requires_complete_launcher_signature(
    clean_mpi_environment, fake_mpi_runtime, monkeypatch, environment, should_bootstrap
):
    """Auto mode must use complete MPI launcher signatures, never a lone rank field."""
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    with mpi_runtime.mpi_context() as is_master:
        assert is_master is True

    assert bool(fake_mpi_runtime) is should_bootstrap


def test_mpi_context_off_skips_detected_mpi_launch(clean_mpi_environment, fake_mpi_runtime, monkeypatch):
    """The explicit off mode must suppress otherwise valid launcher detection."""
    monkeypatch.setenv("ROMJAX_MPI", "off")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")

    with mpi_runtime.mpi_context() as is_master:
        assert is_master is True

    assert fake_mpi_runtime == []


def test_mpi_context_on_bootstraps_without_launcher_environment(clean_mpi_environment, fake_mpi_runtime, monkeypatch):
    """The explicit on mode must bootstrap MPI without an auto-detected launcher."""
    monkeypatch.setenv("ROMJAX_MPI", "on")

    with mpi_runtime.mpi_context() as is_master:
        assert is_master is True

    assert fake_mpi_runtime == [None]


def test_mpi_context_rejects_unknown_mode(clean_mpi_environment, monkeypatch):
    """Unknown MPI modes must fail rather than silently selecting a launch policy."""
    monkeypatch.setenv("ROMJAX_MPI", "sometimes")

    with pytest.raises(ExecutionError, match="ROMJAX_MPI must be one of"):
        with mpi_runtime.mpi_context():
            pass
