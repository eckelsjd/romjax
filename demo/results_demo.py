from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import romjax as romx
from romjax.plotting import AxisOptions, PlotSpec
from romjax.rng import Distribution

num_samples = 4
grid_size = (32, 32)
xlin = jnp.linspace(0.0, 1.0, grid_size[0])
ylin = jnp.linspace(0.0, 1.0, grid_size[1])
xg, yg = jnp.meshgrid(xlin, ylin, indexing="xy")


def gaussian(x: jnp.ndarray, y: jnp.ndarray, cx: float, cy: float, width: float) -> jnp.ndarray:
    """Return a smooth Gaussian bump."""
    radius_sq = (x - cx) ** 2 + (y - cy) ** 2
    return jnp.exp(-radius_sq / (2.0 * width**2))


def input_fun(x: jnp.ndarray, y: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Sample a smooth forcing-like field with mild sample-to-sample variation."""
    key_amp, key_phase, key_center = jax.random.split(key, 3)
    amps = jax.random.uniform(key_amp, (3,), minval=0.8, maxval=1.2)
    phases = jax.random.uniform(key_phase, (2,), minval=-0.15, maxval=0.15)
    centers = jax.random.uniform(key_center, (2, 2), minval=0.2, maxval=0.8)

    field = (
        0.55 * amps[0] * jnp.sin(jnp.pi * (x + phases[0])) * jnp.sin(jnp.pi * (y + phases[1]))
        + 0.18 * amps[1] * jnp.cos(2.0 * jnp.pi * x) * jnp.sin(jnp.pi * y)
        + 0.16 * amps[2] * gaussian(x, y, centers[0, 0], centers[0, 1], 0.12)
        - 0.12 * gaussian(x, y, centers[1, 0], centers[1, 1], 0.16)
    )
    return field


def true_fun(input_grid: jnp.ndarray) -> jnp.ndarray:
    """Map the input field to a smooth reference solution."""
    base = 0.72 * input_grid
    low_mode = 0.22 * jnp.sin(jnp.pi * xg) * jnp.sin(2.0 * jnp.pi * yg)
    interaction = 0.08 * jnp.cos(jnp.pi * (xg - 0.1)) * input_grid
    return base + low_mode + interaction


def error_fun(approx: jnp.ndarray, true: jnp.ndarray) -> jnp.ndarray:
    """Return pointwise absolute error."""
    return jnp.abs(approx - true)


def sample_kle(
    key: jax.Array,
    *,
    variance: float,
    corr_x: float,
    corr_y: float,
    decay: float,
    mean: float = 0.0,
) -> jnp.ndarray:
    """Sample a smooth stochastic field using romjax's KLE distribution."""
    return Distribution(
        callable="kle",
        shape=grid_size,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        truncation=(18, 18),
        correlation_lengths=(corr_x, corr_y),
        variance=variance,
        spectral_decay=decay,
        mean=mean,
        weight="smooth",
        weight_opts={"low": 0.2, "high": 1.0, "length_rel": 0.18},
    ).sample(key)


def approx_from_noise(true: jnp.ndarray, method_idx: int, key: jax.Array) -> jnp.ndarray:
    """Construct a stand-in model prediction with smooth, stochastic failure modes."""
    key_a, key_b = jax.random.split(key)
    envelope = 0.65 + 0.35 * gaussian(xg, yg, 0.5, 0.5, 0.33)

    if method_idx == 0:
        coarse = sample_kle(key_a, variance=1.0, corr_x=0.28, corr_y=0.20, decay=2.8)
        fine = sample_kle(key_b, variance=1.0, corr_x=0.12, corr_y=0.14, decay=3.4)
        residual = 0.018 * coarse + 0.009 * fine
        return 0.972 * true + envelope * residual

    if method_idx == 1:
        anisotropic = sample_kle(key_a, variance=1.0, corr_x=0.36, corr_y=0.10, decay=2.5)
        local = sample_kle(key_b, variance=1.0, corr_x=0.09, corr_y=0.11, decay=3.6)
        residual = 0.017 * anisotropic - 0.008 * local * jnp.cos(2.0 * jnp.pi * xg)
        return true + envelope * residual

    if method_idx == 2:
        broad = sample_kle(key_a, variance=1.0, corr_x=0.24, corr_y=0.24, decay=2.6)
        patchy = sample_kle(key_b, variance=1.0, corr_x=0.10, corr_y=0.08, decay=4.0)
        mask = 0.45 + 0.55 * gaussian(xg, yg, 0.72, 0.34, 0.18)
        residual = 0.015 * broad + 0.011 * mask * patchy
        return 0.986 * true + residual

    if method_idx == 3:
        skewed = sample_kle(key_a, variance=1.0, corr_x=0.14, corr_y=0.30, decay=2.9)
        centered = sample_kle(key_b, variance=1.0, corr_x=0.11, corr_y=0.11, decay=3.8)
        residual = 0.016 * skewed - 0.010 * centered * gaussian(xg, yg, 0.35, 0.70, 0.20)
        return true + envelope * residual

    if method_idx == 4:
        smooth = sample_kle(key_a, variance=1.0, corr_x=0.30, corr_y=0.26, decay=3.6)
        residual = 0.0065 * smooth
        return 0.996 * true + residual

    if method_idx == 5:
        smooth = sample_kle(key_a, variance=1.0, corr_x=0.34, corr_y=0.32, decay=4.2)
        detail = sample_kle(key_b, variance=1.0, corr_x=0.14, corr_y=0.14, decay=4.8)
        residual = 0.0040 * smooth + 0.0015 * detail
        return true + residual

    raise ValueError(f"Unknown method index {method_idx}.")


def style_axes(fig, axs: Axes, artists, cbars) -> None:
    """Hide ticks and spines while keeping any top-row titles."""
    del fig, artists, cbars
    for ax in axs.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", length=0, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)


base_key = jax.random.key(0)
true_inputs = [input_fun(xg, yg, key) for key in jax.random.split(base_key, num_samples)]
true_outputs = [true_fun(true_input) for true_input in true_inputs]
# method_names = [
#     r"$J_{SR}$",
#     r"$J_{RM}$",
#     r"$J_{SE}$",
#     r"$\tilde{J}_{SE}$",
#     r"$J_{GC}$",
#     r"$\tilde{J}_{GC}$",
# ]
method_names = [
    "State reconstruction",
    "Residual minimization",
    "Solution error",
    "Solution error (approx)",
    "Graph commutativity",
    "Graph commutativity (approx)"
]
sample_keys = jax.random.split(jax.random.key(11), num_samples * len(method_names)).reshape(
    num_samples,
    len(method_names),
)

error_specs: list[list[PlotSpec]] = []

for i, true_input in enumerate(true_inputs):
    sample_errors = [
        error_fun(approx_from_noise(true_outputs[i], j, sample_keys[i, j]), true_outputs[i])
        for j in range(len(method_names))
    ]
    row_vmax = float(max(jnp.max(err) for err in sample_errors))

    sample_specs: list[PlotSpec] = []
    for j, error in enumerate(sample_errors):
        sample_specs.append(
            PlotSpec(
                kind="pcolor",
                data=(xg, yg, error),
                opts=AxisOptions(
                    title=method_names[j] if i == 0 else None,
                    grid=False
                ),
                kwargs={"cmap": "viridis", "vmin": 0.0, "vmax": row_vmax},
            )
        )

    error_specs.append(sample_specs)


output_path = Path(__file__).with_name("poisson-grid.png")
fig, ax = romx.gridplot(
    error_specs,
    scheme="white",
    subplot_size_in=(2.4, 2.2),
    save=output_path,
    sharex=True,
    sharey=True,
    adjust=style_axes,
)
plt.close(fig)

print(f"Saved demo plot to {output_path}")
