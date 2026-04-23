"""Plot a small grid of KLE random field samples."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from romjax.pde import UniformGrid
from romjax.plotting import gridplot
from romjax.random_field import kle
from romjax.rng import parametric_sampler


def main() -> None:
    """Draw several KLE samples and plot them with gridplot."""
    key = jax.random.key(0)
    shape = (32, 32)
    trunc = (32, 32)
    nsamples = 6
    grid = UniformGrid(bounds=((-2.0, -1.0), (1.0, 2.0)), shape=shape)
    sample = parametric_sampler(
        key,
        conductivity={
            "distribution": kle,
            "shape": grid.shape,
            "bounds": grid.bounds,
            "truncation": trunc,
            "correlation_lengths": (0.2, 0.2),
            "variance": 0.5,
            "mean": 4.0,
            "nsamples": nsamples,
            "random_override": jax.random.normal(key, (nsamples, *shape))[:, :trunc[0], :trunc[1]]
        },
    )

    fields = jnp.asarray(sample["conductivity"])
    x, y = grid.coords
    clim = (float(jnp.min(fields)), float(jnp.max(fields)))
    plots = [
        {
            "kind": "pcolor",
            "data": (x, y, field),
            "opts": {"clim": clim},
            "kwargs": {"cmap": "jet"},
            # "kwargs": {"shading": "gouraud"},
        }
        for idx, field in enumerate(fields)
    ]
    gridplot(plots, subplot_size_in=(4, 3), scheme='dark', shape=(2, 3), sharex="all", sharey="all")
    plt.show()


if __name__ == "__main__":
    main()
