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
        savefig={"fname": Path(tmp_path) / "sine.gif"},
    )

    assert (Path(tmp_path) / "sine.gif").exists()
    plt.close(fig)


def test_gridplot_savefig_forwards_static_figure_options(monkeypatch, tmp_path):
    calls = []

    def savefig(self, fname, **kwargs):
        calls.append((self, fname, kwargs))

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", savefig)
    fname = tmp_path / "plot.png"
    fig, _ = gridplot(
        PlotSpec(kind="line", data=(np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0]))),
        savefig={"fname": fname, "dpi": 300, "transparent": True},
    )

    assert calls == [(fig, fname, {"dpi": 300, "transparent": True})]
    plt.close(fig)


def test_gridplot_config_savefig_requires_path_or_string_fname():
    assert GridplotConfig(savefig={"fname": "plot.png"}).savefig == {"fname": "plot.png"}

    with pytest.raises(ValueError, match="fname"):
        GridplotConfig(savefig={"fname": 1})


def test_gridplot_savefig_merges_animation_options(monkeypatch, tmp_path):
    calls = []

    def save(self, fname, **kwargs):
        self._draw_was_started = True
        calls.append((fname, kwargs))

    monkeypatch.setattr(plotting.FuncAnimation, "save", save)
    fname = tmp_path / "plot.gif"
    fig, _, _ = gridplot(
        PlotSpec(
            kind="line",
            data=iter([(np.asarray([0.0]), np.asarray([1.0]))]),
            opts={"animate": True},
        ),
        animate_opts={"writer": "pillow", "dpi": 72, "fps": 24},
        savefig={"fname": fname, "dpi": 300, "transparent": True},
    )

    assert calls[0][0] == fname
    assert calls[0][1]["dpi"] == 72
    assert calls[0][1]["fps"] == 24
    assert calls[0][1]["writer"] == "pillow"
    assert calls[0][1]["transparent"] is True
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


def test_gridplot_hist_uses_log_bins_for_log_xscale():
    spec = PlotSpec(
        kind="hist",
        data=np.asarray([1.0, 10.0, 100.0, 1000.0]),
        opts=AxisOptions(xscale="log"),
        kwargs={"bins": 3},
    )

    fig, axs = gridplot(spec)
    patches = axs[0, 0].patches
    edges = np.asarray([patches[0].get_x(), *(patch.get_x() + patch.get_width() for patch in patches)])

    np.testing.assert_allclose(edges, np.geomspace(1.0, 1000.0, 4))
    assert axs[0, 0].get_xscale() == "log"

    plt.close(fig)


def test_gridplot_hist_auto_bins_are_inferred_in_log_space_for_log_xscale():
    spec = PlotSpec(
        kind="hist",
        data=np.geomspace(1.0, 1000.0, 16),
        opts=AxisOptions(xscale="log"),
        kwargs={"bins": "auto"},
    )

    fig, axs = gridplot(spec)
    patches = axs[0, 0].patches
    edges = np.asarray([patches[0].get_x(), *(patch.get_x() + patch.get_width() for patch in patches)])

    np.testing.assert_allclose(np.diff(np.log10(edges)), np.diff(np.log10(edges))[0])
    assert axs[0, 0].get_xscale() == "log"

    plt.close(fig)


def test_gridplot_hist_stepfilled_handles_list_artists():
    spec = PlotSpec(
        kind="hist",
        data=np.asarray([0.0, 1.0, 2.0, 3.0]),
        kwargs={"bins": 2, "histtype": "stepfilled"},
    )

    fig, axs = gridplot(spec)

    assert len(axs[0, 0].patches) == 1

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
