from __future__ import annotations

import sys
from typing import ClassVar

import matplotlib
import matplotlib.pyplot as plt
import pytest
from alive_progress import config_handler
from loguru import logger
from pydantic import ValidationError

import romjax
from romjax import plotting
from romjax.routine import CompositeRoutine, LoggerConfig, ProgressBarConfig, Routine, RoutineConfig

MODULE_NAME = __name__


class DemoRoutine(Routine):
    """Minimal routine used to exercise configuration validation."""

    name: str = "demo"
    exit_code: int = 0
    fail: bool = False
    observed: ClassVar[list[str]] = []

    def run(self) -> int:
        DemoRoutine.observed.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return self.exit_code


def test_logger_config(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(logger, "configure", lambda **kwargs: captured.update(kwargs))

    config = LoggerConfig.model_validate(
        {
            "handlers": [
                {"sink": "stdout", "format": "{message}"},
                {"sink": "stderr", "level": "INFO"},
            ],
            "levels": [{"name": "AUDIT", "no": 25, "color": "<cyan>"}],
            "extra": {"job": "demo"},
        }
    )

    assert config.handlers is not None
    assert len(config.handlers) == 2
    assert config.handlers[0]["sink"] is sys.stdout
    assert config.handlers[1]["sink"] is sys.stderr
    assert config.levels == [{"name": "AUDIT", "no": 25, "color": "<cyan>"}]

    RoutineConfig(logger=config)

    assert captured["handlers"] == config.handlers
    assert captured["levels"] == config.levels
    assert captured["extra"] == {"job": "demo"}


def test_progress_bar_config(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(config_handler, "set_global", lambda **kwargs: captured.update(kwargs))

    config = ProgressBarConfig.model_validate({"title": "Training", "file": "stdout"})

    assert config["file"] is sys.stdout

    RoutineConfig(progress_bar=config)

    assert captured["title"] == "Training"
    assert captured["file"] is sys.stdout


def test_routine_config(monkeypatch):
    calls: dict[str, list[object]] = {"gridplot": [], "logger": [], "bar": [], "style": [], "rc": []}

    monkeypatch.setattr(plotting, "set_global", lambda **kwargs: calls["gridplot"].append(kwargs))
    monkeypatch.setattr(logger, "configure", lambda **kwargs: calls["logger"].append(kwargs))
    monkeypatch.setattr(config_handler, "set_global", lambda **kwargs: calls["bar"].append(kwargs))
    monkeypatch.setattr(plt.style, "use", lambda style: calls["style"].append(style))
    monkeypatch.setattr(matplotlib.rcParams, "update", lambda settings: calls["rc"].append(dict(settings)))

    routine = DemoRoutine(
        routine_config={
            "gridplot": {
                "subplot_size_in": [1.0, 1.0],
                "global_axis_opts": {"xlabel": "base x"},
                "local_plot_kwargs": {"series": {"lw": 1}},
            },
            "logger": {"handlers": [{"sink": "stdout"}]},
            "progress_bar": {"file": "stdout"},
            "mplstyle": {"axes.facecolor": "white"},
        },
        gridplot={
            "subplot_size_in": [2.0, 3.0],
            "global_axis_opts": {"ylabel": "override y"},
            "local_plot_kwargs": {"series": {"color": "red"}},
        },
        logger={"handlers": [{"sink": "stderr", "format": "{message}"}]},
        progress_bar={"file": "stderr", "title": "override"},
        mplstyle="dark_background",
    )

    assert routine.model_extra == {}
    assert routine.routine_config is not None
    assert routine.routine_config.gridplot is not None
    assert routine.routine_config.gridplot.subplot_size_in == (2.0, 3.0)
    assert routine.routine_config.gridplot.global_axis_opts.xlabel == "base x"
    assert routine.routine_config.gridplot.global_axis_opts.ylabel == "override y"
    assert routine.routine_config.gridplot.local_plot_kwargs["series"]["lw"] == 1
    assert routine.routine_config.gridplot.local_plot_kwargs["series"]["color"] == "red"
    assert routine.routine_config.logger is not None
    assert routine.routine_config.logger.handlers[0]["sink"] is sys.stderr
    assert routine.routine_config.progress_bar is not None
    assert routine.routine_config.progress_bar["file"] is sys.stderr
    assert routine.routine_config.mplstyle == "dark_background"

    assert len(calls["gridplot"]) == 2
    assert len(calls["logger"]) == 2
    assert len(calls["bar"]) == 2
    assert calls["rc"][0] == {"axes.facecolor": "white"}
    assert calls["style"] == ["dark_background"]


def test_composite_routine_validates_from_plain_list() -> None:
    DemoRoutine.observed = []
    composite = CompositeRoutine.model_validate([DemoRoutine(name="one"), DemoRoutine(name="two")])

    assert composite.failure_policy == "stop"
    assert composite.run() == 0
    assert DemoRoutine.observed == ["one", "two"]


def test_composite_routine_loads_mixed_yaml_and_inline_children(tmp_path, monkeypatch) -> None:
    DemoRoutine.observed = []
    child_path = tmp_path / "child.yml"
    child_path.write_text(
        "\n".join(
            [
                f"!pd:{MODULE_NAME}.DemoRoutine",
                "name: file",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    composite = romjax.load(
        "\n".join(
            [
                "!romx:CompositeRoutine",
                "routines:",
                "  - child.yml",
                f"  - !pd:{MODULE_NAME}.DemoRoutine",
                "    name: inline",
            ]
        )
    )

    assert isinstance(composite, CompositeRoutine)
    assert composite.run() == 0
    assert DemoRoutine.observed == ["file", "inline"]


def test_composite_routine_parent_relative_child_path(tmp_path) -> None:
    DemoRoutine.observed = []
    config_dir = tmp_path / "configs"
    child_dir = config_dir / "children"
    child_dir.mkdir(parents=True)
    child_path = child_dir / "child.yml"
    composite_path = config_dir / "composite.yml"

    child_path.write_text(
        "\n".join(
            [
                f"!pd:{MODULE_NAME}.DemoRoutine",
                "name: parent-relative",
            ]
        ),
        encoding="utf-8",
    )
    composite_path.write_text(
        "\n".join(
            [
                "!romx:CompositeRoutine",
                "routines:",
                "  - __parent__/children/child.yml",
            ]
        ),
        encoding="utf-8",
    )

    composite = romjax.load(composite_path)

    assert composite.run() == 0
    assert DemoRoutine.observed == ["parent-relative"]


def test_composite_routine_rejects_non_routine_child(tmp_path) -> None:
    child_path = tmp_path / "child.yml"
    child_path.write_text("not: a routine\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="must validate to Routine"):
        CompositeRoutine.model_validate({"routines": [child_path]})


def test_composite_routine_failure_policy_stop() -> None:
    DemoRoutine.observed = []
    composite = CompositeRoutine(
        routines=[
            DemoRoutine(name="one", exit_code=7),
            DemoRoutine(name="two"),
        ],
        failure_policy="stop",
    )

    assert composite.run() == 7
    assert DemoRoutine.observed == ["one"]


def test_composite_routine_failure_policy_continue() -> None:
    DemoRoutine.observed = []
    composite = CompositeRoutine(
        routines=[
            DemoRoutine(name="one", exit_code=7),
            DemoRoutine(name="two"),
            DemoRoutine(name="three", exit_code=3),
        ],
        failure_policy="continue",
    )

    assert composite.run() == 7
    assert DemoRoutine.observed == ["one", "two", "three"]


def test_composite_routine_failure_policy_force_logs_exceptions(monkeypatch) -> None:
    DemoRoutine.observed = []
    logs: list[str] = []

    monkeypatch.setattr(logger, "exception", lambda message, *args: logs.append(message.format(*args)))
    monkeypatch.setattr(logger, "error", lambda message, *args: logs.append(message.format(*args)))

    composite = CompositeRoutine(
        routines=[
            DemoRoutine(name="one", fail=True),
            DemoRoutine(name="two"),
            DemoRoutine(name="three", exit_code=4),
        ],
        failure_policy="force",
    )

    assert composite.run() == 1
    assert DemoRoutine.observed == ["one", "two", "three"]
    assert any("child 0 (DemoRoutine) raised RuntimeError: one failed" in message for message in logs)
    assert any("child 2 (DemoRoutine) exited with code 4" in message for message in logs)
    assert any("CompositeRoutine failures:" in message for message in logs)


def test_nested_composite_routines_run_child_before_parent_continues(tmp_path) -> None:
    DemoRoutine.observed = []
    child_dir = tmp_path / "child"
    child_dir.mkdir()
    grandchild_path = child_dir / "grandchild.yml"
    child_path = child_dir / "composite_child.yml"
    parent_path = tmp_path / "composite_parent.yml"

    grandchild_path.write_text(
        "\n".join(
            [
                f"!pd:{MODULE_NAME}.DemoRoutine",
                "name: child-one",
            ]
        ),
        encoding="utf-8",
    )
    child_path.write_text(
        "\n".join(
            [
                "!romx:CompositeRoutine",
                "routines:",
                "  - __parent__/grandchild.yml",
                f"  - !pd:{MODULE_NAME}.DemoRoutine",
                "    name: child-two",
            ]
        ),
        encoding="utf-8",
    )
    parent_path.write_text(
        "\n".join(
            [
                "!romx:CompositeRoutine",
                "routines:",
                "  - __parent__/child/composite_child.yml",
                f"  - !pd:{MODULE_NAME}.DemoRoutine",
                "    name: parent-after",
            ]
        ),
        encoding="utf-8",
    )

    composite = romjax.load(parent_path)

    assert composite.run() == 0
    assert DemoRoutine.observed == ["child-one", "child-two", "parent-after"]
