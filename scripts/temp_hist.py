import numpy as np

from romtools.plotting import gridplot, PlotSpec, PlotOpts


def generate_hist_frames(num_frames: int, samples_per_frame: int) -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    frames = []
    means = np.linspace(-2.0, 2.0, num_frames)
    scales = 0.6 + 0.2 * np.sin(np.linspace(0.0, 2.0 * np.pi, num_frames))
    for mean, scale in zip(means, scales):
        frames.append(rng.normal(loc=mean, scale=scale, size=samples_per_frame))
    return frames


def main() -> None:
    num_frames = 60
    samples_per_frame = 1500
    frames = generate_hist_frames(num_frames, samples_per_frame)

    hist_spec = PlotSpec(
        kind="hist",
        data=frames,
        opts=PlotOpts(
            animate=True,
            xlabel="x",
            ylabel="count",
            xlim=(-4.0, 4.0),
            ylim=(0, 1),
        ),
        kwargs={
            "bins": 30,
            "range": (-4.0, 4.0),
            "color": "tab:blue",
            "alpha": 0.8,
            "edgecolor": "white",
            "density": True
        },
        name="hist_demo",
    )

    gridplot(
        hist_spec,
        scheme="dark",
        subplot_size_in=(5, 3.5),
        animate_opts={"writer": "pillow", "dpi": 120, "fps": 15, "blit": False},
        save="hist.gif",
    )


if __name__ == "__main__":
    main()
