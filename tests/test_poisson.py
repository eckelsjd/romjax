import jax
import jax.numpy as jnp
import numpy as np

from romtools.typing import DictModel
from romtools.solvers.poisson import Poisson2D, gaussian_forcing, nonlinear_conductivity
from romtools.solvers.poisson import PoissonConfig, GaussianForcingInputs

# TODO: update these and test_loader and make sure full poisson config loads from file

def test_gaussian_forcing_jit_and_grad() -> None:
    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([0.0, 1.0, 2.0])

    def f(A0: jnp.ndarray) -> jnp.ndarray:
        inputs = {
            "A0": A0,
            "sigma": 1.5,
            "mu_x": 0.0,
            "mu_y": 0.0,
            "x": x,
            "y": y,
        }
        return jnp.sum(gaussian_forcing(inputs, {"phi": 0.0}))

    value = jax.jit(f)(1.0)
    grad = jax.grad(f)(1.0)

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_nonlinear_conductivity_jit_vmap_and_grad() -> None:
    phis = jnp.array([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])

    def k_fn(phi: jnp.ndarray) -> jnp.ndarray:
        inputs = {"k0": 2.0, "alpha": 0.5}
        return nonlinear_conductivity(inputs, {"phi": phi})

    vmap_out = jax.vmap(k_fn)(phis)
    jit_out = jax.jit(k_fn)(phis[0])

    def g(alpha: jnp.ndarray) -> jnp.ndarray:
        inputs = {"k0": 2.0, "alpha": alpha}
        return jnp.sum(nonlinear_conductivity(inputs, {"phi": phis[0]}))

    grad = jax.grad(g)(0.5)

    assert vmap_out.shape == phis.shape
    assert jnp.isfinite(vmap_out).all()
    assert jnp.isfinite(jit_out).all()
    assert jnp.isfinite(grad)


def test_poisson_coercion() -> None:
    model = Poisson2D(
        config={'grid': {'shape': (2, 4), 'bounds': [[0, 1], [1, 2]]}},
        forcing="gaussian",
        forcing_defaults=GaussianForcingInputs(A0=0.),
        conductivity="nonlinear",
        conductivity_defaults={"alpha": 0.25},
        boundary_defaults={'boundary': (({"type": "dirichlet", "value": 1.1}, 
                                         {"literally": "whatever"}),)}
    )

    assert isinstance(model.config, PoissonConfig)
    assert np.allclose(model.config.grid.spacing, (0.5, 0.25)) 
    assert model.forcing is gaussian_forcing
    assert model.conductivity is nonlinear_conductivity
    assert isinstance(model.forcing_defaults, DictModel)
    assert isinstance(model.conductivity_defaults, DictModel)
    assert model.boundary_defaults.boundary[0][1]['literally'] == 'whatever'
    assert model.forcing_defaults['sigma'] == 1.0
    assert model.conductivity_defaults['k0'] == 1.0
