import numpy as np
import matplotlib.pyplot as plt

from romtools.plotting import gridplot, PlotOpts, PlotSpec

type PcolorFrame = tuple[np.ndarray, np.ndarray, np.ndarray]
type LineFrame = tuple[np.ndarray, np.ndarray]
type DatasetFrames = tuple[
    list[PcolorFrame],
    list[LineFrame],
    list[np.ndarray],
    float,
    tuple[float, float],
    tuple[float, float],
]


def generate_dataset_frames(
    num_frames: int,
    x: np.ndarray,
    y: np.ndarray,
    phase_offset: float,
    seed: int,
) -> DatasetFrames:
    """Generate animated frames for a 2D field, a 1D slice, and its histogram.

    :param num_frames: number of animation frames
    :param x: 1D x-coordinates
    :param y: 1D y-coordinates
    :param phase_offset: phase shift for the field evolution
    :param seed: RNG seed for small stochastic texture
    :return: (pcolor_frames, line_frames, hist_frames, x_slice, (vmin, vmax), (ymin, ymax))
    """
    rng = np.random.default_rng(seed)
    xg, yg = np.meshgrid(x, y)
    phases = np.linspace(0.0, 2.0 * np.pi, num_frames, endpoint=False)
    slice_idx = int(0.35 * (x.size - 1))
    x_slice = float(x[slice_idx])

    pcolor_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    line_frames: list[tuple[np.ndarray, np.ndarray]] = []
    hist_frames: list[np.ndarray] = []
    vmin, vmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf

    for phase in phases:
        swirl = np.sin(2.0 * np.pi * (xg + 0.15 * np.sin(phase + phase_offset)))
        ripple = np.cos(2.0 * np.pi * (yg - 0.12 * np.cos(phase - phase_offset)))
        bump = np.exp(-8.0 * ((xg - 0.65 - 0.12 * np.cos(phase)) ** 2 + (yg - 0.45) ** 2))
        texture = 0.05 * rng.standard_normal(size=xg.shape)
        field = 1.2 * swirl * ripple + 0.6 * bump + texture

        pcolor_frames.append((xg, yg, field))

        slice_vals = field[:, slice_idx]
        line_frames.append((y, slice_vals))
        hist_frames.append(slice_vals)

        vmin = min(vmin, float(np.nanmin(field)))
        vmax = max(vmax, float(np.nanmax(field)))
        ymin = min(ymin, float(np.nanmin(slice_vals)))
        ymax = max(ymax, float(np.nanmax(slice_vals)))

    pad = 0.1 * (ymax - ymin) if ymax > ymin else 0.5
    return pcolor_frames, line_frames, hist_frames, x_slice, (vmin, vmax), (ymin - pad, ymax + pad)


def main() -> None:
    num_frames = 120
    x = np.linspace(0.0, 1.0, 80)
    y = np.linspace(0.0, 1.0, 70)

    (
        pcolor_a,
        line_a,
        hist_a,
        x_slice_a,
        clim_a,
        line_ylim_a,
    ) = generate_dataset_frames(num_frames, x, y, phase_offset=0.0, seed=3)

    (
        pcolor_b,
        line_b,
        hist_b,
        x_slice_b,
        clim_b,
        line_ylim_b,
    ) = generate_dataset_frames(num_frames, x, y, phase_offset=1.4, seed=9)

    vline_a = PlotSpec(
        kind="line",
        data=(np.full_like(y, x_slice_a), y),
        opts=PlotOpts(),
        kwargs={"color": "k", "lw": 2.0, "alpha": 0.85},
        name="vline_a",
    )
    vline_b = PlotSpec(
        kind="line",
        data=(np.full_like(y, x_slice_b), y),
        opts=PlotOpts(),
        kwargs={"color": "k", "lw": 2.0, "alpha": 0.85},
        name="vline_b",
    )

    pcolor_spec_a = PlotSpec(
        kind="pcolor",
        data=pcolor_a,
        opts=PlotOpts(
            animate=True,
            xlabel="x",
            ylabel="y",
            xlim=(float(x.min()), float(x.max())),
            ylim=(float(y.min()), float(y.max())),
            clim=clim_a,
            cbar_label="field",
        ),
        kwargs={"cmap": "viridis", "shading": "auto"},
        name="field_a",
    )
    pcolor_spec_b = PlotSpec(
        kind="pcolor",
        data=pcolor_b,
        opts=PlotOpts(
            animate=True,
            xlabel="x",
            ylabel="y",
            xlim=(float(x.min()), float(x.max())),
            ylim=(float(y.min()), float(y.max())),
            clim=clim_b,
            cbar_label="field",
        ),
        kwargs={"cmap": "magma", "shading": "auto"},
        name="field_b",
    )

    line_spec_a = PlotSpec(
        kind="line",
        data=line_a,
        opts=PlotOpts(
            animate=True,
            xlabel="y",
            ylabel="field(x_slice, y)",
            xlim=(float(y.min()), float(y.max())),
            ylim=line_ylim_a,
        ),
        kwargs={"color": "tab:blue", "lw": 2.0},
        name="slice_a",
    )
    line_spec_b = PlotSpec(
        kind="line",
        data=line_b,
        opts=PlotOpts(
            animate=True,
            xlabel="y",
            ylabel="field(x_slice, y)",
            xlim=(float(y.min()), float(y.max())),
            ylim=line_ylim_b,
        ),
        kwargs={"color": "tab:orange", "lw": 2.0},
        name="slice_b",
    )

    hist_spec_a = PlotSpec(
        kind="hist",
        data=hist_a,
        opts=PlotOpts(
            animate=True,
            xlabel="slice value",
            ylabel="density",
            xlim=line_ylim_a,
            ylim=(0,1),
        ),
        kwargs={
            "bins": 28,
            "range": line_ylim_a,
            "color": "tab:blue",
            "alpha": 0.85,
            "edgecolor": "white",
            "density": True,
        },
        name="hist_a",
    )
    hist_spec_b = PlotSpec(
        kind="hist",
        data=hist_b,
        opts=PlotOpts(
            animate=True,
            xlabel="slice value",
            ylabel="density",
            xlim=line_ylim_b,
            ylim=(0, 1),
        ),
        kwargs={
            "bins": 28,
            "range": line_ylim_b,
            "color": "tab:orange",
            "alpha": 0.85,
            "edgecolor": "white",
            "density": True,
        },
        name="hist_b",
    )

    # titles = [f"Frame {i + 1} / {num_frames}" for i in range(num_frames)]

    fig, axs, ani = gridplot(
        [
            [(pcolor_spec_a, vline_a), (pcolor_spec_b, vline_b)],
            [line_spec_a, line_spec_b],
            [hist_spec_a, hist_spec_b],
        ],
        scheme="dark",
        subplot_size_in=(4, 3),
        # title=titles,
        animate_opts={"writer": "ffmpeg", "dpi": 120, "fps": 15, "blit": False},
        save="gridplot.mp4",
    )

    plt.show()


if __name__ == "__main__":
    main()
