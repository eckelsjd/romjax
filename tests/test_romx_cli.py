from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import ClassVar

import matplotlib
from alive_progress import config_handler
from loguru import logger

import romjax
from romjax import plotting
from romjax.romx_cli import cli
from romjax.routine import Routine

MODULE_NAME = __name__


class DemoRoutine(Routine):
    """Minimal routine used to exercise the CLI loader."""

    root: Path
    message: str = "demo"
    observed: ClassVar[dict[str, object] | None] = None
    observed_messages: ClassVar[list[str]] = []

    def run(self) -> int:
        gridplot = self.routine_config.gridplot if self.routine_config is not None else None
        DemoRoutine.observed_messages.append(self.message)
        DemoRoutine.observed = {
            "root": self.root,
            "message": self.message,
            "subplot_size_in": None if gridplot is None else gridplot.subplot_size_in,
            "xlabel": None if gridplot is None else gridplot.global_axis_opts.xlabel,
            "series_lw": None if gridplot is None else gridplot.local_plot_kwargs["series"]["lw"],
        }
        return 0


def test_run_custom_routine(tmp_path, capsys):
    root = tmp_path / "runs"
    root.mkdir()
    config_path = tmp_path / "routine.yaml"

    config_path.write_text(
        "\n".join(
            [
                f"!pd:{MODULE_NAME}.DemoRoutine",
                f"root: {root}",
                "message: loaded",
            ]
        ),
        encoding="utf-8",
    )

    assert cli(["run", str(config_path)]) == 0
    assert DemoRoutine.observed == {
        "root": root,
        "message": "loaded",
        "subplot_size_in": None,
        "xlabel": None,
        "series_lw": None,
    }
    assert (root / config_path.name).exists()

    missing = tmp_path / "missing.yaml"
    assert cli(["run", str(missing)]) == 2
    assert f"Config file '{missing}' not found" in capsys.readouterr().err


def test_run_with_globals(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    log_file = tmp_path / "run.log"
    config_path = tmp_path / "routine.yaml"

    calls: dict[str, object] = {}

    def capture(name: str):
        def _capture(**kwargs):
            calls[name] = kwargs

        return _capture

    monkeypatch.setattr(plotting, "set_global", capture("gridplot"))
    monkeypatch.setattr(logger, "configure", capture("logger"))
    monkeypatch.setattr(config_handler, "set_global", capture("bar"))
    monkeypatch.setattr(matplotlib.rcParams, "update", lambda settings: calls.__setitem__("rc", dict(settings)))

    config_path.write_text(
        "\n".join(
            [
                f"!pd:{MODULE_NAME}.DemoRoutine",
                f"root: {root}",
                "message: configured",
                "routine_config:",
                "  gridplot:",
                "    subplot_size_in: [2.0, 2.5]",
                "    global_axis_opts:",
                "      xlabel: time",
                "    local_plot_kwargs:",
                "      series:",
                "        lw: 3",
                "  logger:",
                "    handlers:",
                "      - sink: stdout",
                f"      - sink: {log_file}",
                "  progress_bar:",
                "    file: stderr",
                "  mplstyle:",
                "    axes.facecolor: black",
            ]
        ),
        encoding="utf-8",
    )

    assert cli(["run", str(config_path)]) == 0
    assert (root / config_path.name).exists()
    assert DemoRoutine.observed == {
        "root": root,
        "message": "configured",
        "subplot_size_in": (2.0, 2.5),
        "xlabel": "time",
        "series_lw": 3,
    }

    assert calls["gridplot"]["subplot_size_in"] == (2.0, 2.5)
    assert calls["gridplot"]["global_axis_opts"].xlabel == "time"
    assert calls["logger"]["handlers"][0]["sink"] is sys.stdout
    assert calls["logger"]["handlers"][1]["sink"] == log_file
    assert calls["bar"]["file"] is sys.stderr
    assert calls["rc"] == {"axes.facecolor": "black"}


def test_run_profile_flags_set_environment(tmp_path, monkeypatch):
    config_path = tmp_path / "routine.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"!pd:{MODULE_NAME}.DemoRoutine",
                f"root: {tmp_path / 'runs'}",
            ]
        ),
        encoding="utf-8",
    )

    observed: dict[str, str | None] = {}

    class DummyRoutine:
        def run(self) -> int:
            observed["run_profile"] = os.environ.get("ROMJAX_PROFILE")
            observed["run_dir"] = os.environ.get("ROMJAX_PROFILE_DIR")
            observed["run_label"] = os.environ.get("ROMJAX_PROFILE_LABEL")
            return 0

    def fake_load(path):
        observed["load_profile"] = os.environ.get("ROMJAX_PROFILE")
        observed["load_dir"] = os.environ.get("ROMJAX_PROFILE_DIR")
        observed["load_label"] = os.environ.get("ROMJAX_PROFILE_LABEL")
        return DummyRoutine()

    monkeypatch.setattr(romjax, "load", fake_load)
    try:
        assert (
            cli(
                [
                    "run",
                    "--profile",
                    "--profile-dir",
                    str(tmp_path / "traces"),
                    "--profile-label",
                    "cli-trace",
                    str(config_path),
                ]
            )
            == 0
        )
        assert observed["load_profile"] == "1"
        assert Path(observed["load_dir"]) == (tmp_path / "traces")
        assert observed["load_label"] == "cli-trace"
        assert observed["run_profile"] == "1"
        assert Path(observed["run_dir"]) == (tmp_path / "traces")
        assert observed["run_label"] == "cli-trace"
    finally:
        for key in ("ROMJAX_PROFILE", "ROMJAX_PROFILE_DIR", "ROMJAX_PROFILE_LABEL"):
            monkeypatch.delenv(key, raising=False)


def test_run_composite_routine(tmp_path):
    DemoRoutine.observed_messages = []
    root = tmp_path / "runs"
    root.mkdir()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    child_path = config_dir / "child.yaml"
    composite_path = config_dir / "composite.yaml"

    child_path.write_text(
        "\n".join(
            [
                f"!pd:{MODULE_NAME}.DemoRoutine",
                f"root: {root}",
                "message: child",
            ]
        ),
        encoding="utf-8",
    )
    composite_path.write_text(
        "\n".join(
            [
                "!romx:CompositeRoutine",
                "routines:",
                "  - __parent__/child.yaml",
                f"  - !pd:{MODULE_NAME}.DemoRoutine",
                f"    root: {root}",
                "    message: inline",
            ]
        ),
        encoding="utf-8",
    )

    assert cli(["run", str(composite_path)]) == 0
    assert DemoRoutine.observed_messages == ["child", "inline"]
