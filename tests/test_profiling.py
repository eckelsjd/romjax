from __future__ import annotations

from pathlib import Path

import jax
import pytest

from romjax.grid_search import GridSearch
from romjax.profiling import build_profile_env, profile_enabled, profile_trace


def test_profile_enabled_respects_truthy_and_falsey_flags() -> None:
    assert profile_enabled({"ROMJAX_PROFILE": "1"})
    assert profile_enabled({"ROMJAX_PROFILE": "true"})
    assert not profile_enabled({"ROMJAX_PROFILE": "0"})
    assert not profile_enabled({"ROMJAX_PROFILE": "false"})


def test_profile_trace_uses_jax_profiler_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class DummyTrace:
        def __init__(self, log_dir, **kwargs):
            seen["log_dir"] = Path(log_dir)
            seen["kwargs"] = kwargs

        def __enter__(self):
            seen["entered"] = True
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            seen["exited"] = True
            return False

    monkeypatch.setenv("ROMJAX_PROFILE", "1")
    monkeypatch.setattr("romjax.profiling._trace_run_id", lambda: "20260616-120000-pid123")
    monkeypatch.setattr(jax.profiler, "trace", lambda log_dir, **kwargs: DummyTrace(log_dir, **kwargs))

    with profile_trace("train", tmp_path):
        pass

    assert seen["entered"] is True
    assert seen["exited"] is True
    assert seen["log_dir"] == (tmp_path.resolve() / "profiles" / "train-20260616-120000-pid123")


def test_build_profile_env_is_case_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROMJAX_PROFILE", "1")
    monkeypatch.setattr("romjax.profiling._trace_run_id", lambda: "fixed")

    env = build_profile_env("train", tmp_path / "grid" / "cases" / "case_0007")

    assert env["ROMJAX_PROFILE"] == "1"
    assert env["ROMJAX_PROFILE_LABEL"] == "train"
    assert Path(env["ROMJAX_PROFILE_DIR"]) == (tmp_path / "grid" / "cases" / "case_0007" / "profiles" / "train-fixed")


def test_build_profile_env_respects_explicit_profile_root_for_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ROMJAX_PROFILE", "1")
    monkeypatch.setenv("ROMJAX_PROFILE_DIR", str(tmp_path / "traces"))
    monkeypatch.setattr("romjax.profiling._trace_run_id", lambda: "fixed")

    env = build_profile_env("train", tmp_path / "grid" / "cases" / "case_0007")

    assert Path(env["ROMJAX_PROFILE_DIR"]) == (tmp_path / "traces" / "case_0007" / "train-fixed")


def test_grid_search_prepares_profiled_case_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_path = tmp_path / "base.yml"
    base_path.write_text("root: run\nvalue: 1\n", encoding="utf-8")
    monkeypatch.setenv("ROMJAX_PROFILE", "1")
    monkeypatch.setattr("romjax.profiling._trace_run_id", lambda: "fixed")

    search = GridSearch(
        root=tmp_path / "grid",
        base=base_path,
        override=[{"path": ["value"], "cases": [10]}],
    )

    specs, _ = search._prepare_cases()

    assert specs[0].env["ROMJAX_PROFILE"] == "1"
    assert Path(specs[0].env["ROMJAX_PROFILE_DIR"]) == (
        tmp_path / "grid" / "cases" / "case_0000" / "profiles" / "train-fixed"
    )
