from pathlib import Path

import numpy as np

from romtools.plotting import gridplot, PlotSpec, PlotOpts


def test_basic_gridplot(tmp_path):
    t = np.linspace(0, 2*np.pi, 30)
    def generate_sinusoid():
        for i in range(len(t)):
            yield t[:i+1], np.sin(t[:i+1])
    
    sin_spec = PlotSpec(
        kind='line',
        data=generate_sinusoid(),
        opts=PlotOpts(xlabel="t", ylabel="y(t)", animate=True, ylim=(-1, 1)),
        kwargs=dict(color="red", ls="--"),
        name="my_plot"
    )

    fig, ax = gridplot(
        sin_spec, 
        scheme='dark', 
        plot_kwargs={"my_plot": {"lw": 3}},
        animate_opts={"writer": "pillow", "dpi": 100, "fps": 15, "blit": False},
        save=Path(tmp_path) / "sine.gif",
    )

    assert (Path(tmp_path) / "sine.gif").exists()
