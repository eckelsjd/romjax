from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
from alive_progress import config_handler
from loguru import logger

from romjax import plotting
from romjax.routine import LoggerConfig, ProgressBarConfig, Routine, RoutineConfig


class DemoRoutine(Routine):
    """Minimal routine used to exercise configuration validation."""

    name: str = "demo"

    def run(self) -> int:
        return 0


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
