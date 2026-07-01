from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest

from romjax import plotting
from romjax.plotting import AxisOptions, GridplotConfig, PlotSpec, gridplot

matplotlib.use("Agg")


def test_basic_gridplot(tmp_path):
    t = np.linspace(0, 2*np.pi, 30)
    def generate_sinusoid():
        for i in range(len(t)):
            yield t[:i+1], np.sin(t[:i+1])
    
    sin_spec = PlotSpec(
        kind='line',
        data=generate_sinusoid(),
        opts=AxisOptions(xlabel="t", ylabel="y(t)", animate=True, ylim=(-1, 1)),
        kwargs=dict(color="red", ls="--"),
        name="my_plot"
    )

    fig, _, _ = gridplot(
        sin_spec, 
        scheme='dark', 
        local_plot_kwargs={"my_plot": {"lw": 3}},
        animate_opts={"writer": "pillow", "dpi": 100, "fps": 15, "blit": False},
        save=Path(tmp_path) / "sine.gif",
    )

    assert (Path(tmp_path) / "sine.gif").exists()
    plt.close(fig)


def test_gridplot_cscale_sets_colorbar_normalization():
    x, y = np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    z = np.asarray([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]])

    spec = PlotSpec(
        kind="pcolor",
        data=(x, y, z),
        opts=AxisOptions(clim=(1.0, 300.0), cscale="log"),
    )

    fig, axs = gridplot(spec)
    mesh = axs[0, 0].collections[0]

    assert isinstance(mesh.norm, mcolors.LogNorm)
    assert mesh.norm.vmin == pytest.approx(1.0)
    assert mesh.norm.vmax == pytest.approx(300.0)
    assert fig.axes[1].get_yscale() == "log"

    plt.close(fig)


def test_global_override(monkeypatch):
    monkeypatch.setattr(
        plotting,
        "global_config",
        GridplotConfig(
            scheme="white",
            subplot_size_in=(2.0, 3.0),
            animate_opts={"blit": False, "fps": 10, "writer": "pillow"},
            subplots_kwargs={"squeeze": False},
        ),
    )

    plotting.set_global(
        subplot_size_in=(4.0, 5.0),
        global_axis_opts={"xlabel": "time"},
        global_plot_kwargs={"lw": 5},
    )

    spec = PlotSpec(
        kind="line",
        data=(np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0])),
        opts=AxisOptions(ylabel="value"),
        kwargs={"color": "red"},
        name="series",
    )

    fig, axs = gridplot(
        spec,
        global_axis_opts={"title": "global title", "xscale": "linear"},
        local_axis_opts={"series": {"ylabel": "local value", "title": "local title"}},
        global_plot_kwargs={"marker": "o"},
        local_plot_kwargs={"series": {"ls": "--"}},
    )

    ax = axs[0, 0]
    line = ax.lines[0]

    assert tuple(fig.get_size_inches()) == pytest.approx((4.0, 5.0))
    assert ax.get_xlabel() == "time"
    assert ax.get_ylabel() == "local value"
    assert ax.get_title() == "local title"
    assert line.get_color() == "red"
    assert line.get_linewidth() == 5
    assert line.get_marker() == "o"
    assert line.get_linestyle() == "--"

    plt.close(fig)
