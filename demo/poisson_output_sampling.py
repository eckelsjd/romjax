"""Visualize near-solution output sampling for the Poisson2D example."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from romjax.plotting import gridplot
from romjax.poisson import Poisson2D
from romjax.random_field import kle, darcy


def main() -> None:
    """Solve one Poisson problem and draw several nearby output samples."""
    shape = (48, 48)
    bounds = ((0.0, 1.0), (0.0, 1.0))
    model = Poisson2D(
        config={
            "grid": {"shape": shape, "bounds": bounds},
            "solver": {"name": "Newton", "opts": {"rtol": 1.0, "atol": 1e-5}},
            "max_steps": 20,
            "throw": False,
        },
        # forcing="sinusoid",
        forcing_defaults={'const': -1.0},
        conductivity_sampler='parametric',
        conductivity_sampler_opts={'const': {'distribution': darcy, 'shape': shape, 'bounds': bounds}},
        outputs_sampler="near_solution",
        outputs_sampler_opts={
            "phi": {
                # "distribution": "normal",
                # "mean": 0.0,
                # "std": 1.0,
                # "shape": (48, 48),
                "distribution": kle,
                "shape": shape,
                "bounds": bounds,
                "truncation": (8, 8),
                "correlation_lengths": (0.15, 0.15),
                "variance": 1,
            },
            "scale": ("mean", 0.2)
        },
        # forcing_defaults={"A0": 0.8, "sigma": 0.02, "mu_x": 0.45, "mu_y": 0.55},
    )

    # inputs = {
    #     "forcing": {"A0": 0.9, "sigma": 0.015, "mu_x": 0.35, "mu_y": 0.65},
    # }
    inputs = model.sample_inputs(jax.random.key(0))
    solution = model.solve(inputs)
    # merged_inputs = model._merge_inputs(inputs)
    # forcing = jnp.asarray(model.forcing(merged_inputs["forcing"], solution))
    samples = [model.sample_outputs(jax.random.key(i), inputs=inputs, solution=solution)["phi"] for i in range(4)]

    x, y = model.config.grid.coords
    phi_stack = jnp.stack([solution["phi"], *samples])
    phi_clim = (float(jnp.min(phi_stack)), float(jnp.max(phi_stack)))
    plots = [
        {
            "kind": "pcolor",
            "data": (x, y, inputs['conductivity']['const']),
            "opts": {"title": "Input conductivity"},
            "kwargs": {"cmap": "viridis"},
        },
        {
            "kind": "pcolor",
            "data": (x, y, solution["phi"]),
            "opts": {"title": "Reference Solution", "clim": phi_clim},
            "kwargs": {"cmap": "coolwarm"},
        },
    ]
    plots.extend(
        {
            "kind": "pcolor",
            "data": (x, y, sample),
            "opts": {"title": f"Sample {idx + 1}", "clim": phi_clim},
            "kwargs": {"cmap": "coolwarm"},
        }
        for idx, sample in enumerate(samples)
    )

    gridplot(plots, shape=(2, 3), sharex="all", sharey="all", subplot_size_in=(4, 3))
    plt.show()


if __name__ == "__main__":
    main()
