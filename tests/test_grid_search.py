from __future__ import annotations

import threading
from pathlib import Path, PureWindowsPath

import pytest
import yaml

import romjax
from romjax.grid_search import (
    GridSearch,
    GridSearchCaseResult,
    HybridExecutorConfig,
    SerialExecutorConfig,
    _set_override_path,
    _yaml_path_text,
    orbax_metric,
)
from romjax.routine import RoutineError


def test_sparse_override_sets_mapping_and_sequence_paths() -> None:
    tree = {}
    tree = _set_override_path(tree, ("loss", "terms", 1, "weight"), 0.3)
    tree = _set_override_path(tree, ("loss", "terms", 0, "alpha"), 0.01)

    assert tree == {
        "loss": {
            "terms": [
                {"alpha": 0.01},
                {"weight": 0.3},
            ]
        }
    }


def test_grid_search_writes_loadable_override_yaml(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text(
        """
root: original
loss:
  terms:
    - {alpha: 1.0}
    - {weight: 1.0}
""",
        encoding="utf-8",
    )
    search = GridSearch(
        root=tmp_path / "grid",
        base=base_path,
        override=[
            {"path": ["loss", "terms", 1, "weight"], "cases": [0.3]},
            {"path": ["loss", "terms", 0, "alpha"], "cases": [0.01]},
        ],
    )

    case_root = search.root / "cases" / "case_0000"
    config_path = search._write_case_config(case_root, (0.3, 0.01))
    loaded = romjax.YamlLoader.load(config_path)

    assert Path(loaded["root"]) == case_root
    assert loaded["loss"]["terms"][0]["alpha"] == 0.01
    assert loaded["loss"]["terms"][1]["weight"] == 0.3


def test_yaml_path_text_normalizes_windows_paths() -> None:
    path = PureWindowsPath(r"C:\Users\alice\grid\cases\case_0000")

    assert _yaml_path_text(path) == "C:/Users/alice/grid/cases/case_0000"


def test_grid_search_loads_from_yaml(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")
    config_path = tmp_path / "grid.yml"
    config_path.write_text(
        """
!romx:GridSearch
root: grid
base: __parent__/base.yml
override:
  - path: [value]
    cases: [1, 2]
executor: serial
""",
        encoding="utf-8",
    )

    search = romjax.load(config_path)

    assert isinstance(search, GridSearch)
    assert search.base == base_path.resolve()
    assert search.executor == SerialExecutorConfig()
    assert search.case_root_path == (("root",),)


def test_grid_search_writes_case_from_inline_base_override(tmp_path: Path) -> None:
    train_path = tmp_path / "train.yml"
    config_path = tmp_path / "grid.yml"
    train_path.write_text(
        """
root: original
value: train
artifact: __parent__/artifact.txt
""",
        encoding="utf-8",
    )
    config_path.write_text(
        f"""
!romx:GridSearch
root: {tmp_path / "grid"}
base: !overrides:__parent__/train.yml
  value: inline
override:
  - path: [value]
    cases: [case]
executor: serial
""",
        encoding="utf-8",
    )

    search = romjax.load(config_path)
    case_root = search.root / "cases" / "case_0000"
    case_config = search._write_case_config(case_root, ("case",))
    loaded = romjax.YamlLoader.load(case_config)

    assert isinstance(search.base, romjax.YamlSource)
    assert Path(loaded["root"]) == case_root
    assert loaded["value"] == "case"
    assert loaded["artifact"] == (tmp_path / "artifact.txt").resolve().as_posix()


def test_grid_search_loads_inline_override(tmp_path: Path) -> None:
    grid_path = tmp_path / "grid.yml"
    train_path = tmp_path / "train.yml"
    parent_path = tmp_path / "all-feat.yml"
    grid_path.write_text(
        """
!romx:GridSearch
root: grid
base: __parent__/train.yml
override:
  - path: [loss, 0, callable]
    cases: [default]
executor: serial
""",
        encoding="utf-8",
    )
    train_path.write_text(
        """
root: train
loss:
  - {callable: default, path: [original]}
""",
        encoding="utf-8",
    )
    parent_path.write_text(
        f"""
!romx:CompositeRoutine
- !overrides:__parent__/grid.yml
  root: {tmp_path / "cases" / "grid-sr"}
  base: !overrides:__parent__/train.yml
    loss:
      - {{callable: reconstruction, path: [coordinate transform]}}
""",
        encoding="utf-8",
    )

    composite = romjax.load(parent_path)
    search = composite._validate_routine(composite.routines[0])
    case_root = search.root / "cases" / "case_0000"
    case_config = search._write_case_config(case_root, ("residual",))
    loaded = romjax.YamlLoader.load(case_config)

    assert isinstance(search, GridSearch)
    assert isinstance(search.base, romjax.YamlSource)
    assert loaded["loss"] == [{"callable": "residual", "path": ["coordinate transform"]}]


def test_grid_search_loads_hybrid_executor_from_yaml(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")
    config_path = tmp_path / "grid.yml"
    config_path.write_text(
        """
!romx:GridSearch
root: grid
base: __parent__/base.yml
override:
  - path: [value]
    cases: [1, 2]
executor:
  show_progress: false
  kind: hybrid
  gpu:
    devices: [0, 1]
    workers_per_device: 2
    memory_fraction: 0.5
    preallocate: false
  cpu:
    max_workers: 3
""",
        encoding="utf-8",
    )

    search = romjax.load(config_path)

    assert isinstance(search, GridSearch)
    assert isinstance(search.executor, HybridExecutorConfig)
    assert search.executor.kind == "hybrid"
    assert search.executor.gpu.devices == (0, 1)
    assert search.executor.gpu.workers_per_device == 2
    assert search.executor.gpu.memory_fraction == 0.5
    assert search.executor.gpu.preallocate is False
    assert search.executor.cpu.max_workers == 3


def test_grid_search_loads_rolling_save_policy_from_yaml(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")
    config_path = tmp_path / "grid.yml"
    config_path.write_text(
        """
!romx:GridSearch
root: grid
base: __parent__/base.yml
override:
  - path: [value]
    cases: [1, 2]
save_policy:
  rolling: 2
""",
        encoding="utf-8",
    )

    search = romjax.load(config_path)

    assert isinstance(search, GridSearch)
    assert search.save_policy.mode == "rolling"
    assert search.save_policy.count == 2


def test_grid_search_writes_windows_style_parent_override_as_posix_yaml(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "grid" / "base.yml"
    base_path.parent.mkdir(parents=True)
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")
    search = GridSearch(
        root=tmp_path / "grid",
        base=base_path,
        override=[{"path": ["value"], "cases": [10]}],
    )

    monkeypatch.setattr("romjax.grid_search.os.path.relpath", lambda *_args, **_kwargs: "..\\..\\base.yml")
    case_root = search.root / "cases" / "case_0000"
    config_path = search._write_case_config(case_root, (10,))

    assert config_path.read_text(encoding="utf-8").startswith("!overrides:__parent__/../../base.yml\n")
    assert romjax.YamlLoader.load(config_path)["root"] == str(case_root).replace("\\", "/")


def test_hybrid_grid_search_rejects_scheduler_owned_child_env(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")

    with pytest.raises(RoutineError, match="Hybrid GridSearch controls"):
        GridSearch(
            root=tmp_path / "grid",
            base=base_path,
            override=[{"path": ["value"], "cases": [10]}],
            executor={"kind": "hybrid", "show_progress": False},
            child_env={"JAX_PLATFORMS": "cpu"},
        )


def test_build_hybrid_slots_for_explicit_gpus_and_cpu() -> None:
    config = HybridExecutorConfig.model_validate(
        {
            "gpu": {
                "devices": [2, 3],
                "workers_per_device": 2,
                "memory_fraction": 0.75,
                "preallocate": False,
            },
            "cpu": {"max_workers": 1},
        }
    )
    slots = config.build_slots()

    assert [slot.manifest() for slot in slots] == [
        {"kind": "gpu", "index": 2},
        {"kind": "gpu", "index": 2},
        {"kind": "gpu", "index": 3},
        {"kind": "gpu", "index": 3},
        {"kind": "cpu", "index": None},
    ]
    assert slots[0].env == {
        "CUDA_VISIBLE_DEVICES": "2",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    assert slots[-1].env == {"JAX_PLATFORMS": "cpu"}


def test_build_hybrid_slots_allows_cpu_only() -> None:
    config = HybridExecutorConfig.model_validate({"gpu": {"devices": []}, "cpu": {"max_workers": 2}})
    slots = config.build_slots()

    assert [slot.manifest() for slot in slots] == [
        {"kind": "cpu", "index": None},
        {"kind": "cpu", "index": None},
    ]


def test_default_orbax_metric_reads_loss_csv(tmp_path: Path) -> None:
    (tmp_path / "loss.csv").write_text("Iteration,Value\n0,3.0\n1,1.5\n2,2.0\n", encoding="utf-8")

    assert orbax_metric(tmp_path) == (3.0 + 1.5 + 2.0) / 3


def test_grid_search_run_writes_manifest_and_copies_best(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")

    def fake_run_case(spec):
        loss = {"case_0000": 3.0, "case_0001": 1.0}[spec.name]
        (spec.root / "loss.csv").write_text(f"Iteration,Value\n0,{loss}\n", encoding="utf-8")
        return GridSearchCaseResult(
            name=spec.name,
            root=spec.root,
            config_path=spec.config_path,
            exit_code=0,
            start_time="2026-06-09T10:00:00-04:00",
            end_time="2026-06-09T10:00:01-04:00",
            stdout_path=spec.root / "stdout.log",
            stderr_path=spec.root / "stderr.log",
        )

    monkeypatch.setattr("romjax.grid_search._run_case_subprocess", fake_run_case)
    search = GridSearch(
        root=tmp_path / "grid",
        base=base_path,
        override=[{"path": ["value"], "cases": [10, 20]}],
        save_policy={"best": 1},
    )

    search.executor.show_progress = False
    assert search.run() == 0

    manifest = yaml.safe_load((search.root / "grid_search_manifest.yml").read_text(encoding="utf-8"))
    assert manifest["best"] == "case_0001"
    assert manifest["ranking"] == [{"case": "case_0001", "metric": 1.0}, {"case": "case_0000", "metric": 3.0}]
    assert manifest["cases"]["case_0000"]["retained"] is False
    assert manifest["cases"]["case_0001"]["retained"] is True
    assert manifest["cases"]["case_0001"]["start_time"] == "2026-06-09T10:00:00-04:00"
    assert manifest["cases"]["case_0001"]["end_time"] == "2026-06-09T10:00:01-04:00"
    assert (search.root / "best" / "case.yml").exists()
    assert not (search.root / "cases" / "case_0000").exists()


def test_grid_search_rolling_save_policy_deletes_losers_as_new_cases_start(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")

    def fake_run_case(spec):
        if spec.name == "case_0002":
            assert not (tmp_path / "grid" / "cases" / "case_0000").exists()
        loss = {"case_0000": 3.0, "case_0001": 2.0, "case_0002": 1.0}[spec.name]
        (spec.root / "loss.csv").write_text(f"Iteration,Value\n0,{loss}\n", encoding="utf-8")
        return GridSearchCaseResult(
            name=spec.name,
            root=spec.root,
            config_path=spec.config_path,
            exit_code=0,
            start_time="2026-06-09T10:00:00-04:00",
            end_time="2026-06-09T10:00:01-04:00",
            stdout_path=spec.root / "stdout.log",
            stderr_path=spec.root / "stderr.log",
        )

    monkeypatch.setattr("romjax.grid_search._run_case_subprocess", fake_run_case)
    search = GridSearch(
        root=tmp_path / "grid",
        base=base_path,
        override=[{"path": ["value"], "cases": [10, 20, 30]}],
        save_policy={"rolling": 1},
    )

    search.executor.show_progress = False
    assert search.run() == 0

    manifest = yaml.safe_load((search.root / "grid_search_manifest.yml").read_text(encoding="utf-8"))
    assert manifest["best"] == "case_0002"
    assert manifest["cases"]["case_0000"]["retained"] is False
    assert manifest["cases"]["case_0001"]["retained"] is False
    assert manifest["cases"]["case_0002"]["retained"] is True
    assert (search.root / "best" / "case.yml").exists()
    assert not (search.root / "cases" / "case_0000").exists()
    assert (search.root / "cases" / "case_0002").exists()


def test_grid_search_rolling_save_policy_works_with_out_of_order_completion(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")
    release_slow_case = threading.Event()

    def fake_run_case(spec):
        if spec.name == "case_0000":
            release_slow_case.wait(timeout=5.0)
        if spec.name == "case_0001":
            release_slow_case.set()
        loss = {"case_0000": 2.0, "case_0001": 1.0}[spec.name]
        (spec.root / "loss.csv").write_text(f"Iteration,Value\n0,{loss}\n", encoding="utf-8")
        return GridSearchCaseResult(
            name=spec.name,
            root=spec.root,
            config_path=spec.config_path,
            exit_code=0,
            start_time="2026-06-09T10:00:00-04:00",
            end_time="2026-06-09T10:00:01-04:00",
            stdout_path=spec.root / "stdout.log",
            stderr_path=spec.root / "stderr.log",
        )

    monkeypatch.setattr("romjax.grid_search._run_case_subprocess", fake_run_case)
    search = GridSearch(
        root=tmp_path / "grid",
        base=base_path,
        override=[{"path": ["value"], "cases": [10, 20]}],
        save_policy={"rolling": 1},
        executor={"kind": "thread", "max_workers": 2, "show_progress": False},
    )

    assert search.run() == 0

    manifest = yaml.safe_load((search.root / "grid_search_manifest.yml").read_text(encoding="utf-8"))
    assert manifest["best"] == "case_0001"
    assert manifest["cases"]["case_0000"]["retained"] is False
    assert manifest["cases"]["case_0001"]["retained"] is True
    assert not (search.root / "cases" / "case_0000").exists()
    assert (search.root / "cases" / "case_0001").exists()


def test_grid_search_reuse_policy_skips_existing_cases(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")
    case_root = tmp_path / "grid" / "cases" / "case_0000"
    case_root.mkdir(parents=True)
    (case_root / "case.yml").write_text("existing: true\n", encoding="utf-8")
    (case_root / "loss.csv").write_text("Iteration,Value\n0,0.5\n", encoding="utf-8")

    def fail_run_case(spec):
        raise AssertionError("existing cases should be reused, not relaunched")

    monkeypatch.setattr("romjax.grid_search._run_case_subprocess", fail_run_case)
    search = GridSearch(
        root=tmp_path / "grid",
        base=base_path,
        override=[{"path": ["value"], "cases": [10]}],
        write_policy="reuse",
    )

    search.executor.show_progress = False
    assert search.run() == 0

    manifest = yaml.safe_load((search.root / "grid_search_manifest.yml").read_text(encoding="utf-8"))
    assert manifest["cases"]["case_0000"]["metric"] == 0.5
    assert (case_root / "case.yml").read_text(encoding="utf-8") == "existing: true\n"


def test_hybrid_grid_search_dynamically_reuses_finished_slots(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")
    release_gpu = threading.Event()
    launched: dict[str, dict[str, str]] = {}
    devices: dict[str, dict[str, int | str | None] | None] = {}

    def fake_run_case(spec):
        launched[spec.name] = dict(spec.env)
        devices[spec.name] = spec.device
        if spec.name == "case_0000":
            release_gpu.wait(timeout=5.0)
        if spec.name == "case_0002":
            release_gpu.set()
        loss = {"case_0000": 3.0, "case_0001": 2.0, "case_0002": 1.0}[spec.name]
        (spec.root / "loss.csv").write_text(f"Iteration,Value\n0,{loss}\n", encoding="utf-8")
        return GridSearchCaseResult(
            name=spec.name,
            root=spec.root,
            config_path=spec.config_path,
            exit_code=0,
            start_time="2026-06-09T10:00:00-04:00",
            end_time="2026-06-09T10:00:01-04:00",
            stdout_path=spec.root / "stdout.log",
            stderr_path=spec.root / "stderr.log",
            device=spec.device,
        )

    monkeypatch.setattr("romjax.grid_search._run_case_subprocess", fake_run_case)
    search = GridSearch(
        root=tmp_path / "grid",
        base=base_path,
        override=[{"path": ["value"], "cases": [10, 20, 30]}],
        executor={
            "kind": "hybrid",
            "gpu": {"devices": [0], "workers_per_device": 1, "memory_fraction": 0.5},
            "cpu": {"max_workers": 1},
            "show_progress": False
        },
        child_env={"ROMJAX_TEST": "1"},
    )

    assert search.run() == 0

    assert launched["case_0000"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert launched["case_0000"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"
    assert "JAX_PLATFORMS" not in launched["case_0000"]
    assert launched["case_0001"]["JAX_PLATFORMS"] == "cpu"
    assert launched["case_0002"]["JAX_PLATFORMS"] == "cpu"
    assert all(env["ROMJAX_TEST"] == "1" for env in launched.values())
    assert devices["case_0000"] == {"kind": "gpu", "index": 0}
    assert devices["case_0001"] == {"kind": "cpu", "index": None}
    assert devices["case_0002"] == {"kind": "cpu", "index": None}

    manifest = yaml.safe_load((search.root / "grid_search_manifest.yml").read_text(encoding="utf-8"))
    assert manifest["cases"]["case_0000"]["device"] == {"kind": "gpu", "index": 0}
    assert manifest["cases"]["case_0001"]["device"] == {"kind": "cpu", "index": None}
    assert manifest["cases"]["case_0002"]["device"] == {"kind": "cpu", "index": None}
