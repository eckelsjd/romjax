from __future__ import annotations

from pathlib import Path

import yaml

import romjax
from romjax.grid_search import (
    ExecutorConfig,
    GridSearch,
    GridSearchCaseResult,
    _set_override_path,
    orbax_metric,
)


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
        show_progress=False,
    )

    case_root = search.root / "cases" / "case_0000"
    config_path = search._write_case_config(case_root, (0.3, 0.01))
    loaded = romjax.YamlLoader.load(config_path)

    assert loaded["root"] == str(case_root)
    assert loaded["loss"]["terms"][0]["alpha"] == 0.01
    assert loaded["loss"]["terms"][1]["weight"] == 0.3


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
show_progress: false
executor: serial
""",
        encoding="utf-8",
    )

    search = romjax.load(config_path)

    assert isinstance(search, GridSearch)
    assert search.base == base_path.resolve()
    assert search.executor == ExecutorConfig(kind="serial")
    assert search.case_root_path == (("root",),)


def test_default_orbax_metric_reads_loss_csv(tmp_path: Path) -> None:
    (tmp_path / "loss.csv").write_text("Iteration,Value\n0,3.0\n1,1.5\n2,2.0\n", encoding="utf-8")

    assert orbax_metric(tmp_path) == 1.5


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
        show_progress=False,
    )

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
        show_progress=False,
    )

    assert search.run() == 0

    manifest = yaml.safe_load((search.root / "grid_search_manifest.yml").read_text(encoding="utf-8"))
    assert manifest["cases"]["case_0000"]["metric"] == 0.5
    assert (case_root / "case.yml").read_text(encoding="utf-8") == "existing: true\n"
