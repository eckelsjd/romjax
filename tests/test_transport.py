from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from romjax import YamlLoader
from romjax.pde import (
    BoundarySpec,
    GaussianForcing,
    GridBoundaryInputs,
    IterativeSolver,
    SinusoidForcing,
    UniformGrid,
    homogeneous_boundary,
)
from romjax.plotting import gridplot
from romjax.rng import Distribution, NearSolutionSampler, PyTreeSampler, gen_keys
from romjax.transport import AdvectionDiffusion2D, CubicForcing, PotentialVelocity, QuadraticDiffusion
from romjax.typing import DictModel


def test_gaussian_forcing_jit_and_grad() -> None:
    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([0.0, 1.0, 2.0])
    forcing = GaussianForcing()

    def f(a0: jnp.ndarray) -> jnp.ndarray:
        inputs = {
            "A0": a0,
            "offset": 0.25,
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


def test_cubic_forcing_jit_and_grad() -> None:
    """Cubic forcing supports scalar and field coefficients in JAX transforms."""
    phi = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    forcing = CubicForcing()

    def evaluate(gamma: jax.Array) -> jax.Array:
        return jnp.sum(
            forcing(
                {"q": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "alpha": -1.0, "beta": 0.5, "gamma": gamma},
                {"phi": phi},
            )
        )

    value = jax.jit(evaluate)(-10.0)
    gradient = jax.grad(evaluate)(-10.0)
    expected = jnp.sum(jnp.array([[1.0, 2.0], [3.0, 4.0]]) - phi + 0.5 * phi**2 - 10.0 * phi**3)

    assert jnp.allclose(value, expected)
    assert jnp.allclose(gradient, jnp.sum(phi**3))


def test_cubic_forcing_normalizes_q_rms() -> None:
    """Optional cubic normalization scales only the constant q field to unit RMS."""
    q = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    forcing = CubicForcing(normalize=True)
    normalized = forcing(
        {"q": q, "alpha": 0.0, "beta": 0.0, "gamma": 0.0},
        {"phi": jnp.zeros_like(q)},
    )

    assert jnp.allclose(jnp.sqrt(jnp.mean(normalized**2)), 1.0)


def test_quadratic_diffusion_jit_vmap_and_grad() -> None:
    phis = jnp.array([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])
    diffusion = QuadraticDiffusion()

    def k_fn(phi: jnp.ndarray) -> jnp.ndarray:
        inputs = {"k0": 2.0, "alpha": 0.5}
        return diffusion(inputs, {"phi": phi})

    vmap_out = jax.vmap(k_fn)(phis)
    jit_out = jax.jit(k_fn)(phis[0])

    def g(alpha: jnp.ndarray) -> jnp.ndarray:
        inputs = {"k0": 2.0, "alpha": alpha}
        return jnp.sum(diffusion(inputs, {"phi": phis[0]}))

    grad = jax.grad(g)(0.5)

    assert vmap_out.shape == phis.shape
    assert jnp.isfinite(vmap_out).all()
    assert jnp.allclose(jnp.array([2, 3, 6]), jit_out)
    assert jnp.allclose(jnp.array([10]), grad)


def test_potential_velocity_is_jit_grad_compatible() -> None:
    """Streamfunction velocity has zero discrete divergence and supports JAX transforms."""
    forcing = PotentialVelocity()
    x = jnp.linspace(0.0, 1.0, 9)
    y = jnp.linspace(-1.0, 1.0, 7)
    coords = jnp.meshgrid(x, y, indexing="ij")
    psi = jnp.sin(jnp.pi * coords[0]) * jnp.cos(jnp.pi * coords[1])

    def evaluate(scale: jax.Array) -> jax.Array:
        velocity = forcing({"const": scale * psi, "coords": coords}, {})
        return jnp.sum(velocity**2)

    velocity = forcing({"const": psi, "coords": coords}, {})
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    divergence = forcing._differentiate(velocity[0], dx, axis=0) + forcing._differentiate(
        velocity[1], dy, axis=1
    )

    assert velocity.shape == (2, 9, 7)
    assert jnp.allclose(divergence, 0.0, atol=1e-5)
    assert jnp.isfinite(jax.jit(evaluate)(1.5))
    assert jnp.isfinite(jax.grad(evaluate)(1.5))

    normalized_forcing = PotentialVelocity(normalize=True)
    normalized_velocity = normalized_forcing({"const": psi, "coords": coords}, {})
    speed_rms = jnp.sqrt(jnp.mean(jnp.sum(normalized_velocity**2, axis=0)))
    assert jnp.allclose(speed_rms, 1.0)

    zero_velocity = normalized_forcing({"const": jnp.zeros_like(psi), "coords": coords}, {})
    assert jnp.isfinite(zero_velocity).all()
    assert jnp.all(zero_velocity == 0.0)


def test_potential_velocity_registry_and_shape_validation() -> None:
    """The divergence-free forcing can be configured by registry and rejects mismatched psi fields."""
    model = AdvectionDiffusion2D(
        grid={"shape": (5, 4), "bounds": ((0.0, 1.0), (0.0, 1.0))},
        velocity={"callable": "potential", "normalize": True},
        incompressible=True,
    )
    x, y = model.grid.coords
    psi = jnp.sin(x) * jnp.cos(y)
    velocity = model.velocity({"const": psi, "coords": (x, y)}, {})
    assert velocity.shape == (2, 5, 4)
    assert jnp.allclose(jnp.sqrt(jnp.mean(jnp.sum(velocity**2, axis=0))), 1.0)

    with pytest.raises(ValueError, match="does not match grid shape"):
        model.velocity({"const": jnp.zeros((5, 5)), "coords": (x, y)}, {})


def test_potential_velocity_uses_configured_potential_forcing() -> None:
    """PotentialVelocity forwards runtime inputs to its configured potential forcing."""
    x = jnp.linspace(0.0, 1.0, 5)
    y = jnp.linspace(0.0, 1.0, 4)
    coords = jnp.meshgrid(x, y, indexing="ij")
    potential = SinusoidForcing(inputs_default={"a": 1.0, "b": 2.0, "c": 3.0})
    forcing = PotentialVelocity(potential=potential)

    expected_potential = (
        jnp.sin(jnp.pi * coords[0]) * jnp.sin(jnp.pi * coords[1])
        + 2.0 * jnp.sin(2.0 * jnp.pi * coords[0]) * jnp.sin(jnp.pi * coords[1])
        + 3.0 * jnp.sin(jnp.pi * coords[0]) * jnp.sin(2.0 * jnp.pi * coords[1])
    )
    velocity = forcing({"coords": coords}, {})

    expected = jnp.stack(
        (forcing._differentiate(expected_potential, y[1] - y[0], axis=1),
         -forcing._differentiate(expected_potential, x[1] - x[0], axis=0)),
        axis=0,
    )
    assert jnp.allclose(velocity, expected)


def test_sinusoid_forcing_coefficients() -> None:
    """SinusoidForcing evaluates all three configurable spatial modes."""
    x = jnp.array([[0.25]])
    y = jnp.array([[0.25]])
    forcing = SinusoidForcing()

    value = forcing({"a": 2.0, "b": 3.0, "c": 5.0, "coords": (x, y)}, {})
    expected = 2.0 * 0.5 + (3.0 + 5.0) * jnp.sin(jnp.pi * 0.25)
    assert jnp.allclose(value, expected)


def test_transport_coercion() -> None:
    model = AdvectionDiffusion2D(
        grid={"shape": (2, 4), "bounds": [[0, 1], [1, 2]]},
        forcing={"callable": "gaussian", "inputs_default": {"A0": 0.0}},
        diffusion={"callable": "quadratic", "inputs_default": {"alpha": 0.25}},
        boundary={
            "callable": "identity",
            "inputs_default": {
                "boundary": (({"type": "dirichlet", "value": 1.1}, {"literally": "whatever"}),),
            },
        },
    )

    assert isinstance(model.solver, IterativeSolver)
    assert np.allclose(model.grid.spacing, (0.5, 0.25))
    assert isinstance(model.forcing, GaussianForcing)
    assert isinstance(model.diffusion, QuadraticDiffusion)
    assert isinstance(model.forcing.inputs_default, DictModel)
    assert isinstance(model.diffusion.inputs_default, DictModel)
    assert model.boundary.inputs_default.boundary[0][1]["literally"] == "whatever"
    assert model.forcing.inputs_default["sigma"] == 1.0
    assert model.diffusion.inputs_default["k0"] == 1.0


def test_transport_initial_field_callable_and_runtime_override() -> None:
    """Transport initial fields use the configured callable or a direct runtime field."""
    model = AdvectionDiffusion2D(
        grid=UniformGrid(bounds=((0.0, 2.0), (-1.0, 1.0)), shape=(5, 4)),
        initial={"callable": "constant", "inputs_default": {"const": 2.0}},
    )

    resolved = model._merge_coords({})
    sample = model._initial_field(resolved)
    direct = jnp.arange(20, dtype=float).reshape(5, 4)
    overridden = model._initial_field(model._merge_coords({"initial": {"phi": direct}}))

    assert sample.shape == (5, 4)
    assert jnp.all(sample == 2.0)
    assert jnp.allclose(overridden, direct)

    with pytest.raises(ValueError):
        AdvectionDiffusion2D(grid=model.grid, initial_guess=lambda coords: jnp.ones_like(coords[0]))


def test_transport_evaluate_and_autodiff() -> None:
    model = AdvectionDiffusion2D(
        grid={"shape": (8, 8), "bounds": ((0, 1), (0, 1))},
        solver={
            "solver": {"name": "optimistix.Newton", "kwargs": {"rtol": 1.0, "atol": 1e-4}},
            "max_steps": 6,
            "throw": False,
        },
        forcing="gaussian",
    )

    grid = model.grid
    x, y = grid.coords

    phi_exact = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
    manufactured = AdvectionDiffusion2D(grid=model.grid, solver=model.solver, forcing="sinusoid")
    inputs_exact = {
        "diffusion": {"k0": 1.0, "alpha": 0.0},
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
        "diffusion": {"k0": 1.0, "alpha": 0.0},
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


def test_constant_velocity_advection_matches_linear_manufactured_field() -> None:
    """The conservative centered flux evaluates constant advection exactly for a linear field."""
    model = AdvectionDiffusion2D(
        grid={"shape": (6, 8), "bounds": ((0.0, 1.0), (0.0, 1.0))},
        diffusion={"callable": "constant", "inputs_default": {"const": 0.0}},
    )
    x, y = model.grid.coords
    phi = jnp.asarray(x + 2.0 * y)
    boundary = GridBoundaryInputs(
        boundary=[
            (
                BoundarySpec(type="dirichlet", value=2.0 * y[0, :]),
                BoundarySpec(type="dirichlet", value=1.0 + 2.0 * y[0, :]),
            ),
            (
                BoundarySpec(type="dirichlet", value=x[:, 0]),
                BoundarySpec(type="dirichlet", value=x[:, 0] + 2.0),
            ),
        ]
    )
    residual = model.evaluate(
        {
            "forcing": {"const": 11.0},
            "velocity": {"const": [3.0, 4.0]},
            "boundary": boundary,
        },
        {"phi": phi},
    )["phi_residual"]

    assert jnp.allclose(residual, 0.0, atol=1e-5)


def test_incompressible_constant_velocity_matches_conservative_advection() -> None:
    """The incompressible shortcut agrees with conservative advection for constant velocity."""
    kwargs = {
        "grid": {"shape": (6, 8), "bounds": ((0.0, 1.0), (0.0, 1.0))},
        "diffusion": {"callable": "constant", "inputs_default": {"const": 0.0}},
    }
    conservative = AdvectionDiffusion2D(**kwargs)
    incompressible = AdvectionDiffusion2D(**kwargs, incompressible=True)
    x, y = conservative.grid.coords
    phi = jnp.sin(2.0 * jnp.pi * x) + jnp.cos(2.0 * jnp.pi * y)
    inputs = {
        "forcing": {"const": 0.0},
        "velocity": {"const": [1.5, -0.75]},
        "boundary": GridBoundaryInputs(
            boundary=[
                (
                    BoundarySpec(type="periodic", value=0.0),
                    BoundarySpec(type="periodic", value=0.0),
                ),
                (
                    BoundarySpec(type="periodic", value=0.0),
                    BoundarySpec(type="periodic", value=0.0),
                ),
            ]
        ),
    }

    conservative_residual = conservative.evaluate(inputs, {"phi": phi})["phi_residual"]
    incompressible_residual = incompressible.evaluate(inputs, {"phi": phi})["phi_residual"]
    assert jnp.allclose(conservative_residual, incompressible_residual)


def test_velocity_field_requires_two_components() -> None:
    """Velocity forcing rejects scalar and incorrectly-sized component fields."""
    model = AdvectionDiffusion2D(grid={"shape": (4, 4), "bounds": ((0.0, 1.0), (0.0, 1.0))})
    with pytest.raises(ValueError, match="two components"):
        model.evaluate({"velocity": {"const": 1.0}}, {"phi": jnp.zeros((4, 4))})

    with pytest.raises(ValueError, match="two components"):
        model.evaluate({"velocity": {"const": [1.0, 2.0, 3.0]}}, {"phi": jnp.zeros((4, 4))})


def get_laplace_solver() -> AdvectionDiffusion2D:
    return AdvectionDiffusion2D(
        grid=UniformGrid(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(12, 12)),
        solver={
            "solver": {"name": "optimistix.Newton", "kwargs": {"rtol": 1, "atol": 1e-4}},
            "max_steps": 10,
        },
        initial={"callable": "constant", "inputs_default": {"const": 1.0}},
    )


def get_small_transport(**kwargs) -> AdvectionDiffusion2D:
    return AdvectionDiffusion2D(
        grid=UniformGrid(bounds=((0.0, 1.0), (0.0, 1.0)), shape=(6, 6)),
        solver={
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


def test_transport_manufactured_solve(show_plot: bool = False) -> None:
    shape = (12, 12)
    dx_error = (1 / shape[0]) ** 2
    mms = AdvectionDiffusion2D(
        grid={"shape": shape, "bounds": ((0, 1), (0, 1))},
        solver={
            "solver": {"name": "optimistix.Newton", "kwargs": {"rtol": 1e2, "atol": dx_error * 1.5}},
            "max_steps": 15,
            "throw": False,
        },
        initial={"callable": "constant", "inputs_default": {"const": 1.0}},
        forcing="sinusoid",
    )

    x, y = mms.grid.coords
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


def test_transport_sample_inputs() -> None:
    fixture_path = Path("tests/fixtures_transport.yml")
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
    assert "diffusion" in sample
    assert "k0" in sample["diffusion"]
    assert sample["diffusion"]["k0"].shape == (8, 8)
    assert np.allclose(np.asarray(sample["diffusion"]["k0"]), np.asarray(sample_again["diffusion"]["k0"]))


def test_transport_outputs_sampler_validation_and_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    model = get_small_transport(
        outputs_sampler=NearSolutionSampler(
            phi={"callable": "normal", "std": 0.1, "shape": (6, 6)}
        )
    )

    assert isinstance(model.outputs_sampler, NearSolutionSampler)
    assert isinstance(model.outputs_sampler.template["phi"], Distribution)

    key = jax.random.key(7)
    solution = {"phi": jnp.ones(model.grid.shape)}
    sample_a = model.sample_outputs(key, solution=solution)
    sample_b = model.sample_outputs(key, solution=solution)
    assert sample_a["phi"].shape == model.grid.shape
    assert jnp.allclose(sample_a["phi"], sample_b["phi"])

    calls = {"count": 0}
    solved = {"phi": 2.0 * jnp.ones(model.grid.shape)}

    def solve_spy(self, inputs=None, residuals=None, return_sol=False):
        del inputs, residuals, return_sol
        calls["count"] += 1
        return solved

    monkeypatch.setattr(AdvectionDiffusion2D, "solve", solve_spy)

    provided = {"phi": jnp.ones(model.grid.shape)}
    sample_with_solution = model.sample_outputs(jax.random.key(0), solution=provided)
    assert calls["count"] == 0
    assert not jnp.allclose(sample_with_solution["phi"], solved["phi"])

    sample_without_solution = model.sample_outputs(jax.random.key(0))
    assert calls["count"] == 1
    assert sample_without_solution["phi"].shape == model.grid.shape
    assert jnp.isfinite(sample_without_solution["phi"]).all()


def test_transport_sample_outputs_custom_callable_support() -> None:
    seen: dict[str, object] = {}

    def custom_sampler(key, *, inputs=None, solution=None, bias=0.0):
        del key
        seen["inputs"] = inputs
        seen["solution"] = solution
        return jnp.asarray(solution["phi"]) + bias

    model = get_small_transport(outputs_sampler={"callable": custom_sampler, "bias": 0.25})
    inputs = {"forcing": {"const": 1.5}}
    solution = {"phi": jnp.zeros(model.grid.shape)}

    sample = model.sample_outputs(jax.random.key(9), inputs=inputs, solution=solution)

    assert seen["inputs"] == inputs
    assert seen["solution"] == solution
    assert jnp.allclose(sample["phi"], 0.25)


def test_transport_conditions_are_sampled_separately_from_outputs() -> None:
    model = get_small_transport(
        conditions_sampler=PyTreeSampler(
            template={"phi": {"callable": "normal", "mean": 0.25, "std": 0.0, "shape": (6, 6)}}
        ),
        outputs_sampler=NearSolutionSampler(),
    )
    solution = {"phi": jnp.ones(model.grid.shape)}

    conditions = model.sample_conditions(jax.random.key(4))
    sample = model.sample_outputs(jax.random.key(5), solution=solution, conditions=conditions)

    assert conditions["phi"].shape == model.grid.shape
    assert jnp.allclose(sample["phi"], solution["phi"] + 0.25)
