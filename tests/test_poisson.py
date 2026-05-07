from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from romjax import YamlLoader
from romjax.pde import UniformGrid, homogeneous_boundary
from romjax.plotting import gridplot
from romjax.poisson import GaussianForcing, NonlinearConductivity, Poisson2D, PoissonConfig
from romjax.rng import Distribution, NearSolutionSampler, PyTreeSampler, gen_keys
from romjax.typing import DictModel


def test_gaussian_forcing_jit_and_grad() -> None:
    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([0.0, 1.0, 2.0])
    forcing = GaussianForcing()

    def f(a0: jnp.ndarray) -> jnp.ndarray:
        inputs = {
            "A0": a0,
            "sigma": 1.5,
            "mu_x": 0.0,
            "mu_y": 0.0,
            "coords": (x, y),
        }
        return jnp.sum(forcing(inputs, {"phi": 0.0}))

    value = jax.jit(f)(1.0)
    grad = jax.grad(f)(1.0)

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)


def test_nonlinear_conductivity_jit_vmap_and_grad() -> None:
    phis = jnp.array([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])
    conductivity = NonlinearConductivity()

    def k_fn(phi: jnp.ndarray) -> jnp.ndarray:
        inputs = {"k0": 2.0, "alpha": 0.5}
        return conductivity(inputs, {"phi": phi})

    vmap_out = jax.vmap(k_fn)(phis)
    jit_out = jax.jit(k_fn)(phis[0])

    def g(alpha: jnp.ndarray) -> jnp.ndarray:
        inputs = {"k0": 2.0, "alpha": alpha}
        return jnp.sum(conductivity(inputs, {"phi": phis[0]}))

    grad = jax.grad(g)(0.5)

    assert vmap_out.shape == phis.shape
    assert jnp.isfinite(vmap_out).all()
    assert jnp.allclose(jnp.array([2, 3, 6]), jit_out)
    assert jnp.allclose(jnp.array([10]), grad)


def test_poisson_coercion() -> None:
    model = Poisson2D(
        config={"grid": {"shape": (2, 4), "bounds": [[0, 1], [1, 2]]}},
        forcing={"callable": "gaussian", "inputs_default": {"A0": 0.0}},
        conductivity={"callable": "nonlinear", "inputs_default": {"alpha": 0.25}},
        boundary={
            "callable": "identity",
            "inputs_default": {
                "boundary": (({"type": "dirichlet", "value": 1.1}, {"literally": "whatever"}),),
            },
        },
    )

    assert isinstance(model.config, PoissonConfig)
    assert np.allclose(model.config.grid.spacing, (0.5, 0.25))
    assert isinstance(model.forcing, GaussianForcing)
    assert isinstance(model.conductivity, NonlinearConductivity)
    assert isinstance(model.forcing.inputs_default, DictModel)
    assert isinstance(model.conductivity.inputs_default, DictModel)
    assert model.boundary.inputs_default.boundary[0][1]["literally"] == "whatever"
    assert model.forcing.inputs_default["sigma"] == 1.0
    assert model.conductivity.inputs_default["k0"] == 1.0


def test_poisson_evaluate_and_autodiff() -> None:
    model = Poisson2D(
        config={
            "grid": {"shape": (8, 8), "bounds": ((0, 1), (0, 1))},
            "solver": {"name": "optimistix.Newton", "kwargs": {"rtol": 1.0, "atol": 1e-4}},
            "max_steps": 6,
            "throw": False,
        },
        forcing="gaussian",
    )

    grid = model.config.grid
    x, y = grid.coords

    phi_exact = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    manufactured = Poisson2D(config=model.config, forcing="sinusoid")
    inputs_exact = {
        "conductivity": {"k0": 1.0, "alpha": 0.0},
        "boundary": homogeneous_boundary(ndim=2),
    }
    residual = manufactured.evaluate(inputs_exact, {"phi": phi_exact})["phi_residual"]
    assert jnp.max(jnp.abs(residual)) < 2.5e-1

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
        return jnp.sum(model.evaluate(local_inputs, {"phi": phi})["phi_residual"])

    jax.jit(eval_sum)(inputs["forcing"]["A0"], phi0)
    grad_forcing = jax.grad(lambda a0: eval_sum(a0, phi0))(inputs["forcing"]["A0"])
    grad_phi = jax.grad(lambda p: eval_sum(inputs["forcing"]["A0"], p))(phi0)

    assert jnp.isfinite(grad_forcing)
    assert jnp.isfinite(grad_phi).all()
    assert grad_phi.shape == phi0.shape


def get_laplace_solver() -> Poisson2D:
    return Poisson2D(
        config={
            "grid": UniformGrid(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(12, 12)),
            "solver": {"name": "optimistix.Newton", "kwargs": {"rtol": 1, "atol": 1e-4}},
            "max_steps": 10,
            "initial_guess": lambda coords: jnp.ones_like(coords[0]),
        }
    )


def get_small_poisson(**kwargs) -> Poisson2D:
    return Poisson2D(
        config={
            "grid": UniformGrid(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(6, 6)),
            "solver": {"name": "optimistix.Newton", "kwargs": {"rtol": 1.0, "atol": 1e-4}},
            "max_steps": 5,
            "throw": False,
        },
        **kwargs,
    )


def test_laplace_solve() -> None:
    laplace = get_laplace_solver()
    out = laplace.solve()
    assert jnp.max(jnp.abs(out["phi"])) < 1e-4


def test_laplace_solve_jit_and_grad() -> None:
    laplace = get_laplace_solver()

    @jax.jit
    def solve_sum(a0: jnp.ndarray) -> jnp.ndarray:
        phi = laplace.solve({"forcing": {"const": a0}})["phi"]
        return jnp.sum(phi)

    center = 0.5
    grad = jax.grad(solve_sum)(jnp.array(center))
    eps = 1e-2
    fd = (solve_sum(center + eps) - solve_sum(center - eps)) / (2 * eps)
    assert jnp.allclose(grad, fd, atol=1e-2, rtol=1e-2)


def test_poisson_manufactured_solve(show_plot: bool = False) -> None:
    shape = (12, 12)
    dx_error = (1 / shape[0]) ** 2
    mms = Poisson2D(
        config={
            "grid": {"shape": shape, "bounds": ((0, 1), (0, 1))},
            "max_steps": 15,
            "solver": {"name": "optimistix.Newton", "kwargs": {"rtol": 1e2, "atol": dx_error * 1.5}},
            "initial_guess": lambda coords: jnp.ones_like(coords[0]),
            "throw": False,
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
        opts = {"xlabel": "$x$", "ylabel": "$y$"}
        kwargs = {"shading": "gouraud"}
        approx_spec = {"kind": "pcolor", "data": (x, y, phi), "opts": {**opts, "clim": clim}, "kwargs": kwargs}
        true_spec = {
            "kind": "pcolor",
            "data": (x, y, phi_exact),
            "opts": {**opts, "clim": clim},
            "kwargs": kwargs,
        }
        error_spec = {
            "kind": "pcolor",
            "data": (x, y, error),
            "opts": {**opts, "clim": "auto"},
            "kwargs": {**kwargs, "cmap": "bwr"},
        }
        gridplot([approx_spec, true_spec, error_spec], scheme="dark", shape=(1, 3), sharey="row")
        plt.show()

    assert jnp.max(error) < dx_error * 2


def test_poisson_sample_inputs() -> None:
    fixture_path = Path("tests/fixtures_poisson.yml")
    model = YamlLoader.load(fixture_path)["solver"]

    key = next(gen_keys(1, seed=122))
    sample = model.sample_inputs(key)
    sample_again = model.sample_inputs(key)

    assert isinstance(model.inputs_sampler, PyTreeSampler)
    assert "forcing" in sample
    assert "mu_x" in sample["forcing"]
    assert "A0" in sample["forcing"]
    assert sample["forcing"]["mu_x"] >= 0.4
    assert sample["forcing"]["mu_x"] < 0.6
    assert np.isclose(sample["forcing"]["mu_x"], sample_again["forcing"]["mu_x"])
    assert np.isclose(sample["forcing"]["A0"], sample_again["forcing"]["A0"])
    assert "conductivity" in sample
    assert "k0" in sample["conductivity"]
    assert sample["conductivity"]["k0"].shape == (8, 8)
    assert np.allclose(np.asarray(sample["conductivity"]["k0"]), np.asarray(sample_again["conductivity"]["k0"]))


def test_poisson_outputs_sampler_validation_and_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    model = get_small_poisson(
        outputs_sampler={
            "callable": "near_solution",
            "phi": {
                "callable": "normal",
                "std": 0.1,
                "shape": (6, 6),
            },
        }
    )

    assert isinstance(model.outputs_sampler, NearSolutionSampler)
    assert isinstance(model.outputs_sampler.template["phi"], Distribution)

    with pytest.raises(TypeError):
        get_small_poisson(outputs_sampler={"callable": "near_solution", "phi": 0.1})

    key = jax.random.key(7)
    solution = {"phi": jnp.ones(model.config.grid.shape)}
    sample_a = model.sample_outputs(key, solution=solution)
    sample_b = model.sample_outputs(key, solution=solution)
    assert sample_a["phi"].shape == model.config.grid.shape
    assert jnp.allclose(sample_a["phi"], sample_b["phi"])

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
    assert jnp.isfinite(sample_without_solution["phi"]).all()


def test_poisson_sample_outputs_custom_callable_support() -> None:
    seen: dict[str, object] = {}

    def custom_sampler(key, *, inputs=None, solution=None, bias=0.0):
        del key
        seen["inputs"] = inputs
        seen["solution"] = solution
        return jnp.asarray(solution["phi"]) + bias

    model = get_small_poisson(outputs_sampler={"callable": custom_sampler, "bias": 0.25})
    inputs = {"forcing": {"const": 1.5}}
    solution = {"phi": jnp.zeros(model.config.grid.shape)}

    sample = model.sample_outputs(jax.random.key(9), inputs=inputs, solution=solution)

    assert seen["inputs"] == inputs
    assert seen["solution"] == solution
    assert jnp.allclose(sample["phi"], 0.25)
