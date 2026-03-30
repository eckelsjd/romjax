import numpy as np

from romtools.plotting import gridplot, PlotSpec, PlotOpts


def generate_hist2d_frames(num_frames: int, samples_per_frame: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(11)
    frames = []
    angles = np.linspace(0.0, 2.0 * np.pi, num_frames)
    base_cov = np.array([[1.0, 0.6], [0.6, 1.2]])

    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        cov = rot @ base_cov @ rot.T
        mean = np.array([1.5 * np.cos(0.5 * angle), 1.5 * np.sin(0.5 * angle)])
        samples = rng.multivariate_normal(mean, cov, size=samples_per_frame)
        frames.append((samples[:, 0], samples[:, 1]))

    return frames


def main() -> None:
    num_frames = 60
    samples_per_frame = 2500
    frames = generate_hist2d_frames(num_frames, samples_per_frame)

    hist2d_spec = PlotSpec(
        kind="hist2d",
        data=frames,
        opts=PlotOpts(
            animate=True,
            xlabel="x",
            ylabel="y",
            clim=(0, 0.3),
            cbar_label="count",
            xlim=(-4.0, 4.0),
            ylim=(-4.0, 4.0),
        ),
        kwargs={
            "bins": (35, 35),
            "range": ((-4.0, 4.0), (-4.0, 4.0)),
            "cmap": "viridis",
            "density": True,
            "shading": "auto"
        },
        name="hist2d_demo",
    )

    gridplot(
        hist2d_spec,
        scheme="white",
        subplot_size_in=(5, 4),
        animate_opts={"writer": "pillow", "dpi": 120, "fps": 15, "blit": False},
        save="hist2d.gif",
    )


if __name__ == "__main__":
    main()
