import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from romjax import YamlLoader
from romjax.graph import FunctionGraph
from romjax.pde import DiffraxSolver
from romjax.rng import NearSolutionSampler, PyTreeSampler
from romjax.vlasov import CosinePerturbation, FVConfig, Vlasov1D1V


def small_vlasov(**kwargs) -> Vlasov1D1V:
    solver = kwargs.pop(
        "solver",
        {
            "solver": {"name": "Euler"},
            "stepsize_controller": {"name": "ConstantStepSize"},
            "t1": 0.02,
            "ts": (0.0, 0.01, 0.02),
            "dt0": 0.005,
            "max_steps": 64,
            "throw": False,
        },
    )
    params = kwargs.pop("params", {"knudsen": float("inf"), "debye": 1.0})
    return Vlasov1D1V(
        grid={"shape": (4, 6), "bounds": ((0.0, 1.0), (-3.0, 3.0))},
        solver=solver,
        params=params,
        **kwargs,
    )


def test_diffrax_solver_config_solves_simple_ode() -> None:
    config = DiffraxSolver(
        solver={"name": "Euler"},
        stepsize_controller={"name": "ConstantStepSize"},
        t1=0.2,
        saveat={"name": "SaveAt", "kwargs": {"ts": [0.0, 0.05, 0.2]}},
        dt0=0.05,
    )
    solution = jax.jit(
        lambda y0: config.diffeqsolve(diffrax.ODETerm(lambda t, y, args: y), y0).ys
    )(jnp.asarray(1.0))

    assert jnp.allclose(config.save_times(), jnp.asarray([0.0, 0.05, 0.2]))
    assert solution.shape == (3,)
    assert jnp.isfinite(solution).all()


def test_vlasov_yaml_loading_and_config_validation() -> None:
    model = YamlLoader.load(
        """
!romx:Vlasov1D1V
grid:
  shape: [4, 6]
  bounds: [[0.0, 1.0], [-3.0, 3.0]]
solver:
  solver: {name: Euler}
  stepsize_controller: {name: ConstantStepSize}
  t1: 0.02
  ts: [0.0, 0.01, 0.02]
  dt0: 0.005
  max_steps: 64
params: {knudsen: .inf, debye: 1.0}
initial:
  callable: cosine
  inputs_default: {alpha: 0.05, k: 6.283185307179586}
fv: {flux: upwind, reconstruction: none, cfl: 0.5}
"""
    )

    assert isinstance(model, Vlasov1D1V)
    assert isinstance(model.initial, CosinePerturbation)
    assert isinstance(model.solver, DiffraxSolver)
    assert isinstance(model.fv, FVConfig)
    assert model.resolve_dof() == 4 * 6 * 3 + 4 * 3


def test_cosine_initial_runtime_generation_and_override() -> None:
    model = small_vlasov()
    generated = model._initial_vdf(model._resolve_inputs({"initial": {"alpha": 0.0}}))
    direct = 0.25 * jnp.ones(model.grid.shape)
    overridden = model._initial_vdf(model._resolve_inputs({"initial": {"alpha": 1.0, "vdf": direct}}))

    assert generated.shape == model.grid.shape
    assert jnp.allclose(overridden, direct)


def test_vlasov_solve_evaluate_and_autodiff() -> None:
    model = small_vlasov()
    solution = model.solve({"initial": {"alpha": 0.05}})
    residual = model.evaluate({"initial": {"alpha": 0.05}}, solution)

    assert solution["fields"]["vdf"].shape == (4, 6, 3)
    assert solution["fields"]["potential"].shape == (4, 3)
    assert residual["fields"]["vdf"].shape == solution["fields"]["vdf"].shape
    assert residual["fields"]["potential"].shape == solution["fields"]["potential"].shape
    assert jnp.isfinite(residual["fields"]["vdf"]).all()
    assert jnp.max(jnp.abs(residual["fields"]["potential"])) < 1e-5

    def solve_sum(alpha: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(model.solve({"initial": {"alpha": alpha}})["fields"]["vdf"])

    value = jax.jit(solve_sum)(jnp.asarray(0.05))
    grad = jax.grad(solve_sum)(jnp.asarray(0.05))
    batched = jax.vmap(solve_sum)(jnp.asarray([0.0, 0.05]))

    assert jnp.isfinite(value)
    assert jnp.isfinite(grad)
    assert batched.shape == (2,)


def test_vlasov_default_solve_is_finite() -> None:
    model = Vlasov1D1V(
        grid={"shape": (4, 6), "bounds": ((0.0, 1.0), (-3.0, 3.0))},
        solver={"t1": 0.01, "ts": (0.0, 0.01), "dt0": 0.005, "max_steps": 64, "throw": False},
        params={"knudsen": 0.5, "debye": 1.0},
    )
    solution = model.solve({"initial": {"alpha": 0.01}})

    assert isinstance(model.solver.solver, diffrax.Tsit5)
    assert solution["fields"]["vdf"].shape == (4, 6, 2)
    assert jnp.isfinite(solution["fields"]["vdf"]).all()


def test_vlasov_single_step_restart_from_last_output() -> None:
    step1 = small_vlasov(
        solver={
            "solver": {"name": "Euler"},
            "stepsize_controller": {"name": "ConstantStepSize"},
            "t0": 0.0,
            "t1": 0.01,
            "ts": (0.0, 0.01),
            "dt0": 0.005,
            "max_steps": 64,
            "throw": False,
        }
    )
    first = step1.solve({"initial": {"alpha": 0.02}})
    restart_vdf = first["fields"]["vdf"][..., -1]

    step2 = small_vlasov(
        initial=None,
        solver={
            "solver": {"name": "Euler"},
            "stepsize_controller": {"name": "ConstantStepSize"},
            "t0": 0.01,
            "t1": 0.02,
            "ts": (0.01, 0.02),
            "dt0": 0.005,
            "max_steps": 64,
            "throw": False,
        },
    )
    second = step2.solve({"initial": {"vdf": restart_vdf}})

    assert first["fields"]["vdf"].shape[-1] == 2
    assert second["fields"]["vdf"].shape[-1] == 2
    assert jnp.allclose(second["fields"]["vdf"][..., 0], restart_vdf)
    assert second["fields"]["potential"].shape == (4, 2)


def test_vlasov_constant_manufactured_residual_is_small() -> None:
    model = small_vlasov(initial=None)
    vdf0 = jnp.ones(model.grid.shape) / (model.grid.bounds[1][1] - model.grid.bounds[1][0])
    vdf = jnp.repeat(vdf0[..., None], 3, axis=-1)
    potential = jnp.zeros((model.grid.shape[0], 3))
    inputs = {"initial": {"vdf": vdf0}}
    residual = model.evaluate(inputs, {"fields": {"vdf": vdf, "potential": potential}})

    assert jnp.max(jnp.abs(residual["fields"]["vdf"])) < 1e-6
    assert jnp.max(jnp.abs(residual["fields"]["potential"])) < 1e-6


def test_vlasov_evaluate_uses_supplied_potential_for_vdf_residual() -> None:
    model = small_vlasov(initial=None)
    x, v = model.grid.coords
    vdf0 = jnp.asarray(1.0 + 0.1 * v)
    vdf = jnp.repeat(vdf0[..., None], 3, axis=-1)
    potential_zero = jnp.zeros((model.grid.shape[0], 3))
    potential_ramp = jnp.repeat(jnp.asarray(x[:, 0])[..., None], 3, axis=-1)
    inputs = {"initial": {"vdf": vdf0}}

    residual_zero = model.evaluate(inputs, {"fields": {"vdf": vdf, "potential": potential_zero}})
    residual_ramp = model.evaluate(inputs, {"fields": {"vdf": vdf, "potential": potential_ramp}})

    assert not jnp.allclose(residual_zero["fields"]["vdf"], residual_ramp["fields"]["vdf"])


def test_vlasov_sampling_and_function_graph_integration() -> None:
    model = small_vlasov(
        inputs_sampler=PyTreeSampler(
            template={
                "params": {"debye": {"callable": "dirac", "value": 1.0}},
                "initial": {"alpha": {"callable": "normal", "mean": 0.05, "std": 0.01, "shape": ()}},
            }
        ),
        outputs_sampler=NearSolutionSampler(
            template={
                "fields": {
                    "vdf": {"callable": "normal", "std": 0.01, "shape": (4, 6, 3)},
                    "potential": {"callable": "normal", "std": 0.01, "shape": (4, 3)},
                }
            },
            scale={"fields": {"vdf": ("max_abs", 0.1), "potential": 1.0}},
        ),
    )
    key = jax.random.key(3)
    inputs = model.sample_inputs(key)
    solution = model.solve(inputs)
    sample = model.sample_outputs(key, inputs=inputs, solution=solution)

    graph = FunctionGraph(edges={"vlasov": model})
    pushed = graph.push_path({"inputs": inputs, "outputs": solution}, path=["vlasov"], start="vlasov_in")
    pulled = graph.push_path(pushed, path=["vlasov"], start="vlasov_out")

    assert "alpha" in inputs["initial"]
    assert sample["fields"]["vdf"].shape == (4, 6, 3)
    assert pulled["outputs"]["fields"]["vdf"].shape == solution["fields"]["vdf"].shape
    assert np.isfinite(np.asarray(pushed["residuals"]["fields"]["vdf"])).all()
