from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from romjax import YamlLoader
from romjax.pde import UniformGrid, homogeneous_boundary
from romjax.plotting import gridplot
from romjax.poisson import (
    GaussianForcingInputs,
    Poisson2D,
    PoissonConfig,
    gaussian_forcing,
    nonlinear_conductivity,
)
from romjax.random_field import darcy
from romjax.rng import Distribution, gen_keys, near_solution_sampler
from romjax.typing import DictModel


def test_gaussian_forcing_jit_and_grad() -> None:
    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([0.0, 1.0, 2.0])

    def f(A0: jnp.ndarray) -> jnp.ndarray:
        inputs = {
            "A0": A0,
            "sigma": 1.5,
            "mu_x": 0.0,
            "mu_y": 0.0,
            "coords": (x, y),
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
    assert jnp.allclose(jnp.array([2, 3, 6]), jit_out)
    assert jnp.allclose(jnp.array([10]), grad)


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


def test_poisson_evaluate():
    # Homogeneous dirichlet BCs, no conductivity, sinusoidal manufactured solution
    fixture_path = Path("tests/fixtures_poisson.yml")
    model = YamlLoader.load(fixture_path)['solver']

    grid = model.config.grid
    coords = grid.coords
    x, y = coords

    # Manufactured solution verification
    phi_exact = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)

    manufactured = Poisson2D(
        config=model.config,
        forcing='sinusoid'
    )

    inputs_exact = {
        "conductivity": {"k0": 1.0, "alpha": 0.0},
        "boundary": homogeneous_boundary(ndim=2),
    }
    residual = manufactured.evaluate(inputs_exact, {"phi": phi_exact})['phi_residual']
    assert jnp.max(jnp.abs(residual)) < 1e-2

    # JIT, VMAP, and GRAD compatibility with Gaussian forcing
    inputs = {
        "forcing": {
            "A0": 0.5,
            "sigma": 0.1,
            "mu_x": 0.5,
            "mu_y": 0.5,
        },
        "conductivity": {"k0": 1.0, "alpha": 0.0},
        "boundary": homogeneous_boundary(ndim=2),
    }
    phi0 = jnp.ones_like(x)

    def eval_sum(forcing_amp: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
        local_inputs = {
            **inputs,
            "forcing": {**inputs["forcing"], "A0": forcing_amp},
        }
        return jnp.sum(model.evaluate(local_inputs, {"phi": phi})['phi_residual'])

    jit_out = jax.jit(eval_sum)(inputs["forcing"]["A0"], phi0)
    vmap_out = jax.vmap(lambda p: eval_sum(inputs["forcing"]["A0"], p))(
        jnp.stack([phi0, 0.5 * phi0])
    )
    grad_forcing = jax.grad(lambda a0: eval_sum(a0, phi0))(inputs["forcing"]["A0"])
    grad_phi = jax.grad(lambda p: eval_sum(inputs["forcing"]["A0"], p))(phi0)

    assert jnp.allclose(jit_out, vmap_out[0])
    assert jnp.all(jit_out <= vmap_out[1])  # negative forcing is halved
    assert jnp.isfinite(grad_forcing)
    assert jnp.isfinite(grad_phi).all()
    assert grad_phi.shape == phi0.shape


def get_laplace_solver():
    """Basic laplace solver on unit square."""
    laplace = Poisson2D(
        config={
            "grid": UniformGrid(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(50, 50)),
            "solver": {
                "name": "Newton", 
                "opts": {"rtol": 1, "atol": 1e-4} # only look at residual atol essentially
            },
            "max_steps": 20,
            "initial_guess": lambda coords: jnp.ones_like(coords[0])
        }
    )

    return laplace


def get_small_poisson(
    **kwargs,
) -> Poisson2D:
    """Construct a small Poisson model for sampling tests."""
    return Poisson2D(
        config={
            "grid": UniformGrid(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(8, 8)),
            "solver": {
                "name": "Newton",
                "opts": {"rtol": 1.0, "atol": 1e-4},
            },
            "max_steps": 5,
            "throw": False,
        },
        **kwargs,
    )


def test_laplace_solve():
    """Run Poisson with no forcing and unity conductivity. Initial Gaussian bump should smooth to 0."""
    laplace = get_laplace_solver()
    out = laplace.solve()
    assert jnp.max(jnp.abs(out["phi"])) < 1e-4


def test_laplace_solve_jit_and_grad():
    """Add constant forcing and take gradients of the sum over the grid."""
    laplace = get_laplace_solver()

    @jax.jit
    def solve_sum(A0: jnp.ndarray) -> jnp.ndarray:
        inputs = {
            "forcing": {"const": A0},
        }
        phi = laplace.solve(inputs)["phi"]
        return jnp.sum(phi)

    center = 0.5
    grad = jax.grad(solve_sum)(jnp.array(center))
    eps = 1e-2
    fd = (solve_sum(center + eps) - solve_sum(center - eps)) / (2 * eps)
    assert jnp.allclose(grad, fd, atol=1e-2, rtol=1e-2)


def test_poisson_manufactured_solve(show_plot=False):
    """Test analytical solution of manufactured sinusoid forcing."""
    shape = (50, 50)
    dx_error = (1/shape[0]) ** 2
    mms = Poisson2D(
        config={
            "grid": {"shape": shape, "bounds": ((0, 1), (0, 1))},
            "max_steps": 50,
            "solver": dict(name='Newton', opts={'rtol': 1e2, 'atol': dx_error*1.5}),
            "initial_guess": lambda coords: jnp.ones_like(coords[0]),
            "throw": False
        }, 
        forcing="sinusoid",
    )

    x, y = mms.config.grid.coords
    phi = mms.solve()["phi"]
    phi_exact = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    error = jnp.abs(phi - phi_exact)
    vmin = min(float(jnp.min(phi)), float(jnp.min(phi_exact)))
    vmax = max(float(jnp.max(phi)), float(jnp.max(phi_exact)))
    clim = (vmin, vmax)

    if show_plot:
        opts = {'xlabel': "$x$", 'ylabel': "$y$"}
        kwargs = {'shading': 'gouraud'}
        approx_spec = {'kind': 'pcolor', 'data': (x, y, phi), 'opts': {**opts, 'clim': clim}, 'kwargs': kwargs}
        true_spec = {'kind': 'pcolor', 'data': (x, y, phi_exact), 'opts': {**opts, 'clim': clim}, 'kwargs': kwargs}
        error_spec = {'kind': 'pcolor', 'data': (x, y, error), 'opts': {**opts, 'clim': 'auto'}, 
                      'kwargs': {**kwargs, 'cmap': 'bwr'}}
        gridplot([approx_spec, true_spec, error_spec], scheme='dark', shape=(1, 3), sharey='row')
        plt.show()

    assert jnp.max(error) < dx_error*2


def test_poisson_sample_inputs():
    fixture_path = Path("tests/fixtures_poisson.yml")
    model = YamlLoader.load(fixture_path)['solver']

    for key in gen_keys(3, seed=122):
        sample = model.sample_inputs(key)
        forcing_key, conductivity_key, _ = jax.random.split(key, 3)
        forcing_subkeys = jax.random.split(forcing_key, 2)
        expected_mu_x = jax.random.uniform(forcing_subkeys[0], minval=0.4, maxval=0.6, shape=())
        expected_a0 = jax.random.normal(forcing_subkeys[1], shape=()) * 0.1 + 0.5

        assert "forcing" in sample
        assert "mu_x" in sample["forcing"]
        assert "A0" in sample["forcing"]
        assert sample["forcing"]["mu_x"] >= 0.4
        assert sample["forcing"]["mu_x"] < 0.6
        assert np.isclose(sample["forcing"]["mu_x"], float(expected_mu_x), rtol=1e-5, atol=1e-6)
        assert np.isclose(sample["forcing"]["A0"], float(expected_a0), rtol=1e-5, atol=1e-6)

        expected_k0 = darcy(
            jax.random.split(conductivity_key, 1)[0],
            shape=(50, 50),
            bounds=((0, 1), (0, 1)),
        )

        assert "conductivity" in sample
        assert "k0" in sample["conductivity"]
        assert sample["conductivity"]["k0"].shape == (50, 50)
        assert np.allclose(np.asarray(sample["conductivity"]["k0"]), np.asarray(expected_k0))


def test_poisson_outputs_sampler_resolution_and_validation() -> None:
    model = get_small_poisson(
        outputs_sampler="near_solution",
        outputs_sampler_opts={
            "phi": {
                "distribution": "normal",
                "std": 0.1,
                "shape": (8, 8),
            },
        },
    )

    assert model.outputs_sampler is near_solution_sampler
    assert isinstance(model.outputs_sampler_opts["phi"], Distribution)

    with pytest.raises(TypeError):
        get_small_poisson(
            outputs_sampler="near_solution",
            outputs_sampler_opts={"phi": 0.1},
        )


def test_poisson_sample_outputs_near_solution_is_deterministic() -> None:
    model = get_small_poisson(
        outputs_sampler="near_solution",
        outputs_sampler_opts={"phi": {"distribution": "normal", "std": 0.1, "shape": (8, 8)}},
    )
    key = jax.random.key(7)
    solution = {"phi": jnp.ones(model.config.grid.shape)}

    sample_a = model.sample_outputs(key, solution=solution)
    sample_b = model.sample_outputs(key, solution=solution)

    assert sample_a["phi"].shape == model.config.grid.shape
    assert jnp.allclose(sample_a["phi"], sample_b["phi"])


def test_poisson_sample_outputs_smooth_mode_preserves_shape() -> None:
    model = get_small_poisson(
        outputs_sampler="near_solution",
        outputs_sampler_opts={
            "phi": {
                "distribution": "uniform",
                "minval": -0.2,
                "maxval": 0.2,
                "shape": (8, 8),
            },
        },
    )
    solution = {"phi": jnp.zeros(model.config.grid.shape)}
    sample = model.sample_outputs(jax.random.key(3), solution=solution)

    assert sample["phi"].shape == model.config.grid.shape
    assert jnp.isfinite(sample["phi"]).all()


def test_poisson_sample_outputs_reuses_solution_before_solving(monkeypatch: pytest.MonkeyPatch) -> None:
    model = get_small_poisson(
        outputs_sampler="near_solution",
        outputs_sampler_opts={"phi": {"distribution": "uniform", "minval": -0.1, "maxval": 0.1, "shape": (8, 8)}},
    )
    calls = {"count": 0}
    solved = {"phi": 2.0 * jnp.ones(model.config.grid.shape)}

    def solve_spy(self, inputs=None, residuals=None, return_sol=False):
        del inputs, residuals, return_sol
        calls["count"] += 1
        return solved

    monkeypatch.setattr(Poisson2D, "solve", solve_spy)

    provided = {"phi": jnp.ones(model.config.grid.shape)}
    sample_with_solution = model.sample_outputs(jax.random.key(0), solution=provided)
    assert calls["count"] == 0
    assert not jnp.allclose(sample_with_solution["phi"], solved["phi"])

    sample_without_solution = model.sample_outputs(jax.random.key(0))
    assert calls["count"] == 1
    assert sample_without_solution["phi"].shape == model.config.grid.shape


def test_poisson_sample_outputs_custom_callable_support() -> None:
    seen: dict[str, object] = {}

    def custom_sampler(key, *, inputs=None, solution=None, bias=0.0):
        del key
        seen["inputs"] = inputs
        seen["solution"] = solution
        return jnp.asarray(solution["phi"]) + bias

    model = get_small_poisson(
        outputs_sampler=custom_sampler,
        outputs_sampler_opts={"bias": 0.25},
    )
    inputs = {"forcing": {"const": 1.5}}
    solution = {"phi": jnp.zeros(model.config.grid.shape)}

    sample = model.sample_outputs(jax.random.key(9), inputs=inputs, solution=solution)

    assert seen["inputs"] == inputs
    assert seen["solution"] == solution
    assert jnp.allclose(sample["phi"], 0.25)
