"""MPI GridSearch behavior, including a real local MPI launch when available."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

import romjax
from romjax.execution import ExecutionError, ExecutorConfig
from romjax.grid_search import GridSearch
from romjax.routine import Routine


class MpiEnvRoutine(Routine):
    """Record per-case values and launcher-provided process environment."""

    root: Path
    resolved: int

    def run(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "resolved_value.txt").write_text(str(self.resolved), encoding="utf-8")
        (self.root / "rank.txt").write_text(
            os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "")), encoding="utf-8"
        )
        (self.root / "jax_env.txt").write_text(os.environ.get("JAX_ENABLE_X64", ""), encoding="utf-8")
        (self.root / "gpu_env.txt").write_text(os.environ.get("CUDA_VISIBLE_DEVICES", ""), encoding="utf-8")
        time.sleep(0.1)
        return 0


def test_grid_search_accepts_mpi_executor_from_yaml(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    base.write_text("root: original\nresolved: 0\n", encoding="utf-8")
    config = tmp_path / "grid.yml"
    config.write_text(
        f"!romx:GridSearch\nroot: {tmp_path / 'grid'}\nbase: __parent__/base.yml\n"
        "override:\n  - path: [resolved]\n    cases: [1, 2]\nexecutor: mpi\n",
        encoding="utf-8",
    )

    search = romjax.load(config)

    assert isinstance(search, GridSearch)
    assert isinstance(search.executor, ExecutorConfig)
    assert search.executor.kind == "mpi" and search.executor.show_progress


def test_mpi_grid_search_requires_mpi_launch(tmp_path: Path) -> None:
    pytest.importorskip("mpi4py")
    base = tmp_path / "base.yml"
    base.write_text("root: original\nresolved: 0\n", encoding="utf-8")
    search = GridSearch(
        root=tmp_path / "grid",
        base=base,
        override=[{"path": ["resolved"], "cases": [1]}],
        executor="mpi",
    )

    with pytest.raises(Exception, match="MPI launch|mpiexec|srun"):
        search.run()


def test_mpi_grid_search_warns_if_mpi_extra_is_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import builtins

    original_import = builtins.__import__

    def missing_mpi(name: str, *args, **kwargs):
        if name == "mpi4py" or name.startswith("mpi4py."):
            raise ImportError("mpi4py missing")
        return original_import(name, *args, **kwargs)

    base = tmp_path / "base.yml"
    base.write_text("root: original\nresolved: 0\n", encoding="utf-8")
    search = GridSearch(
        root=tmp_path / "grid",
        base=base,
        override=[{"path": ["resolved"], "cases": [1]}],
        executor="mpi",
    )
    monkeypatch.setattr(builtins, "__import__", missing_mpi)

    with pytest.warns(RuntimeWarning, match="mpi extra"), pytest.raises(ExecutionError, match="mpi extra"):
        search.run()


@pytest.mark.skipif(
    importlib.util.find_spec("mpi4py") is None or shutil.which("mpiexec") is None,
    reason="requires the mpi extra and an MPI launcher",
)
@pytest.mark.parametrize(
    "composite_executor",
    [None, "serial", "thread", "process"],
    ids=["direct", "nested_serial", "nested_thread", "nested_process_rejected"],
)
def test_mpi_grid_search_runs_nested_base_overrides(tmp_path: Path, composite_executor: str | None) -> None:
    base = tmp_path / "base.yml"
    base.write_text(
        "!pd:tests.test_grid_search_mpi.MpiEnvRoutine\nroot: original\nresolved: 0\n",
        encoding="utf-8",
    )
    grid = tmp_path / "grid.yml"
    grid.write_text(
        "!romx:GridSearch\n"
        f"root: {tmp_path / 'grid'}\nbase: __parent__/base.yml\n"
        "override:\n  - path: [resolved]\n    cases: [1, 2, 3, 4]\n"
        "executor: {kind: mpi, show_progress: false}\n"
        "jax_enable_x64: true\n",
        encoding="utf-8",
    )
    composite = tmp_path / "composite.yml"
    composite.write_text(
        "!romx:CompositeRoutine\n"
        "base: __parent__/grid.yml\n"
        "overrides:\n  - name: trial\n    cases:\n"
        f"      - name: one\n        value: {{root: {tmp_path / 'grid_one'}}}\n"
        f"      - name: two\n        value: {{root: {tmp_path / 'grid_two'}}}\n"
        f"executor: {{kind: {composite_executor or 'serial'}, show_progress: false}}\n",
        encoding="utf-8",
    )

    command = [
        shutil.which("mpiexec"),
        "-n",
        "4",
        sys.executable,
        "-c",
        "from romjax.romx_cli import main; main()",
        "run",
        str(composite if composite_executor else grid),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "7"
    result = subprocess.run(command, capture_output=True, text=True, timeout=90, env=env, check=False)

    if composite_executor == "process":
        assert result.returncode != 0
        assert "Nested MPI execution" in result.stderr
        return

    assert result.returncode == 0, result.stdout + result.stderr
    roots = (tmp_path / "grid_one", tmp_path / "grid_two") if composite_executor else (tmp_path / "grid",)
    for search_root in roots:
        manifest = yaml.safe_load((search_root / "grid_search_manifest.yml").read_text(encoding="utf-8"))
        assert len(manifest["cases"]) == 4
        assert all(item["status"] == "succeeded" for item in manifest["cases"].values())
        assert [
            (search_root / "cases" / f"case_{index:04d}" / "resolved_value.txt").read_text(encoding="utf-8")
            for index in range(4)
        ] == ["1", "2", "3", "4"]
        assert {
            (search_root / "cases" / f"case_{index:04d}" / "jax_env.txt").read_text(encoding="utf-8")
            for index in range(4)
        } == {"true"}
        assert {
            (search_root / "cases" / f"case_{index:04d}" / "gpu_env.txt").read_text(encoding="utf-8")
            for index in range(4)
        } == {"7"}
        assert len({
            (search_root / "cases" / f"case_{index:04d}" / "rank.txt").read_text(encoding="utf-8")
            for index in range(4)
        }) > 1
