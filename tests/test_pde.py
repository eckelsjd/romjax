from pathlib import Path
from types import SimpleNamespace

import diffrax
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from pydantic import ValidationError

from romjax.compression import SVD
from romjax.graph import Edge, FunctionGraph, Node
from romjax.model import ImplicitModel
from romjax.nn import Affine
from romjax.pde import (
    FORCING_REGISTRY,
    AliveProgressMeter,
    BoundaryType,
    ConstantForcing,
    ImplicitAffine,
    ImplicitIterativeGalerkin,
    IterativeSolver,
    LatentSamplerFactory,
    LinearSolver,
    RandomNewton,
    SumForcing,
    UniformGrid,
    homogeneous_boundary,
)
from romjax.rng import Distribution
from romjax.tree import pytree_merge


def test_sum_forcing_delegates_to_independently_configured_forcings() -> None:
    forcing = SumForcing(
        forcings=(
            ConstantForcing(inputs_default={"const": 2.0}),
            ConstantForcing(inputs_default={"const": 3.0}),
        )
    )

    assert jnp.allclose(forcing({}, {}), 5.0)
    assert FORCING_REGISTRY["sum"] is SumForcing


def test_sum_forcing_validates_nested_registered_specs() -> None:
    forcing = SumForcing.model_validate(
        {
            "forcings": [
                {"name": "constant", "inputs_default": {"const": 1.5}},
                {"callable": "constant", "inputs_default": {"const": 2.5}},
            ]
        }
    )

    assert jnp.allclose(forcing({}, {}), 4.0)


def test_merge_boundary_conditions():
    defaults = homogeneous_boundary(type="dirichlet", value=0.0, ndim=2)

    overrides = {
        "boundary": [
            (
                {"value": jnp.array(1.0)},
                {"value": jnp.array(2.0)},
            ),
            (
                {"type": BoundaryType.neumann, "value": jnp.array(0.5)},
                {"value": jnp.array(3.0)},
            ),
        ]
    }

    merged = pytree_merge(defaults, overrides)

    assert merged["boundary"][0][0]["type"] == BoundaryType.dirichlet
    assert merged["boundary"][1][0]["type"] == BoundaryType.neumann
    assert float(merged["boundary"][0][1]["value"]) == 2.0


def test_grid_boundary_inputs_are_hashable():
    boundary_a = homogeneous_boundary(type="dirichlet", value=0.0, ndim=2)
    boundary_b = homogeneous_boundary(type="dirichlet", value=0.0, ndim=2)

    assert hash(boundary_a) == hash(boundary_b)
    assert {boundary_a: "ok"}[boundary_b] == "ok"


def test_uniform_grid():
    # 1) Specifying bounds and shape and checking that spacing and coords are correct
    grid = UniformGrid(bounds=((0.0, 1.0), (0.0, 2.0)), shape=(2, 4))
    assert grid.spacing == (0.5, 0.5)
    assert grid.coords is not None
    assert grid.coords[0].shape == (2, 4)
    assert isinstance(grid.coords[0], np.ndarray)
    assert jnp.allclose(grid.coords[0][:, 0], jnp.array([0.25, 0.75]))
    assert jnp.allclose(grid.coords[1][0, :], jnp.array([0.25, 0.75, 1.25, 1.75]))

    # 2) Specifying bounds and spacing and checking that shape and coords are correct
    grid = UniformGrid(bounds=((0.0, 1.0), (0.0, 2.0)), spacing=(0.5, 0.5))
    assert grid.shape == (2, 4)
    assert grid.coords is not None
    assert grid.coords[0].shape == (2, 4)

    # 3) Specifying 1d coords and checking the resulting meshgrid (and shape, spacing, and bounds)
    x = jnp.array([0.25, 0.75])
    y = jnp.array([0.25, 0.75, 1.25, 1.75])
    grid = UniformGrid(coords=(x, y))
    assert grid.shape == (2, 4)
    assert grid.coords is not None
    assert jnp.allclose(jnp.array(grid.bounds[0]), jnp.array((0., 1.)))  # cell-centered
    assert jnp.allclose(jnp.array(grid.bounds[1]), jnp.array((0., 2.)))

    # 4) Specifying 2d coords and checking shape, spacing, and bounds
    xg, yg = jnp.meshgrid(x, y, indexing="ij")
    grid = UniformGrid(coords=(xg, yg))
    assert grid.shape == (2, 4)
    assert grid.coords is not None
    assert jnp.allclose(jnp.array(grid.bounds[0]), jnp.array((0., 1.)))
    assert jnp.allclose(jnp.array(grid.bounds[1]), jnp.array((0., 2.)))

    # 5) Making sure we get validation errors for misspecified coords or shape/spacing + bounds
    with pytest.raises(ValueError):
        UniformGrid(bounds=((0.0, 1.0),), shape=(2,), spacing=(0.25,))

    with pytest.raises(ValueError):
        UniformGrid(coords=(jnp.array([0.0, 1.0]), jnp.array([[0.0, 1.0], [2.0, 3.0]])))

    # 6) Make sure we don't serialize big coords array
    d = grid.model_dump()
    assert 'coords' not in d


def test_uniform_grid_accepts_numpy_coords() -> None:
    x = np.array([0.25, 0.75], dtype=np.float32)
    y = np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float32)

    grid = UniformGrid(coords=(x, y))

    assert grid.shape == (2, 4)
    assert grid.spacing == (0.5, 0.5)
    assert isinstance(grid.coords[0], np.ndarray)
    assert np.allclose(np.asarray(grid.bounds[0]), np.asarray((0.0, 1.0)))
    assert np.allclose(np.asarray(grid.bounds[1]), np.asarray((0.0, 2.0)))


def test_implicit_iterative_galerkin_matches_direct_implicit_solve() -> None:

    class TinyNonlinearImplicit(ImplicitModel):
        source: Node = Node(name="implicit_source")
        target: Node = Node(name="implicit_target")
        field_name: str = "u"
        residual_name: str = "r"

        def evaluate(self, inputs, outputs):
            u = jnp.asarray(outputs[self.field_name])
            b = jnp.asarray(inputs["b"])
            return {self.residual_name: u**2 - b}

        def solve(self, inputs, residuals):
            b = jnp.asarray(inputs["b"])
            r = jnp.asarray(residuals[self.residual_name])
            return {self.field_name: jnp.sqrt(b + r)}

    class SourceMapEdge(Edge):
        source: Node = Node(name="galerkin_source")
        target: Node = Node(name="implicit_source")
        scale: float = 2.0
        shift: float = 1.0

        def forward(self, x):
            z = jnp.asarray(x["outputs"])
            return {"inputs": x["inputs"], "outputs": {"u": self.scale * z + self.shift}}

        def backward(self, x):
            u = jnp.asarray(x["outputs"]["u"])
            return {"inputs": x["inputs"], "outputs": (u - self.shift) / self.scale}

    class TargetMapEdge(Edge):
        source: Node = Node(name="implicit_target")
        target: Node = Node(name="galerkin_target")
        scale: float = 3.0
        shift: float = -0.5

        def forward(self, x):
            r = jnp.asarray(x["residuals"]["r"])
            return {"inputs": x["inputs"], "residuals": self.scale * r + self.shift}

        def backward(self, x):
            eta = jnp.asarray(x["residuals"])
            return {"inputs": x["inputs"], "residuals": {"r": (eta - self.shift) / self.scale}}

    graph = FunctionGraph(
        edges={
            "src_map": SourceMapEdge(),
            "implicit": TinyNonlinearImplicit(),
            "tgt_map": TargetMapEdge(),
            "galerkin": ImplicitIterativeGalerkin(
                source="galerkin_source",
                target="galerkin_target",
                path=["src_map", "implicit", "tgt_map"],
            ),
        }
    )

    inputs = {"b": jnp.array([1.0, 1.5]), "solver": {"initial": {"outputs": 0.1 * jnp.ones(2)}}}
    z_true = jnp.array([0.2, 0.4])

    target_payload = graph.push_path(
        {"inputs": inputs, "outputs": z_true},
        path=["src_map", "implicit", "tgt_map"],
        start="galerkin_source",
    )
    eta_target = target_payload["residuals"]

    z_galerkin = graph.push_path(
        {"inputs": inputs, "residuals": eta_target},
        path=["galerkin"],
        start="galerkin_target",
    )["outputs"]

    implicit_residuals = graph.push_path(
        {"inputs": inputs, "residuals": eta_target},
        path=["tgt_map"],
        start="galerkin_target",
    )["residuals"]
    implicit_outputs = graph.push_path(
        {"inputs": inputs, "residuals": implicit_residuals},
        path=["implicit"],
        start="implicit_target",
    )["outputs"]
    z_direct = graph.push_path(
        {"inputs": inputs, "outputs": implicit_outputs},
        path=["src_map"],
        start="implicit_source",
    )["outputs"]

    assert jnp.allclose(z_galerkin, z_direct, atol=1e-6, rtol=1e-6)

    with pytest.raises(ValidationError):
        ImplicitIterativeGalerkin(path=["src_map", "implicit", "tgt_map"], initial_guess=lambda x: x)


def test_implicit_iterative_galerkin_defers_source_sampler_loading(tmp_path: Path) -> None:
    artifact_path = tmp_path / "dataset" / "train" / "galerkin_compression.npz"
    compression = SVD(
        energy_tol=0.9,
        center=False,
        rank=2,
        mean=np.asarray([0.0, 0.0]),
        basis=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        singular_values=np.asarray([2.0, 1.0]),
        minval=np.asarray([-1.0, -2.0]),
        maxval=np.asarray([1.0, 2.0]),
    )
    compression.dump(artifact_path)
    edge = ImplicitIterativeGalerkin(
        source="a",
        target="b",
        name="galerkin",
        path=["ab"],
        compression=artifact_path,
        source_sampler=LatentSamplerFactory(distribution="uniform"),
    )

    assert edge.resolve_rank() == 2
    edge.resolve_source_sampler()
    sample = edge.sample_source(jax.random.key(0))
    assert sample["outputs"].shape == (2,)
    assert jnp.all(sample["outputs"] >= jnp.asarray([-1.0, -2.0]))
    assert jnp.all(sample["outputs"] <= jnp.asarray([1.0, 2.0]))


def test_implicit_affine_residual_inverse_and_sampling(tmp_path: Path) -> None:
    compression = SVD(
        energy_tol=0.9,
        center=False,
        rank=2,
        mean=np.zeros(2),
        basis=np.eye(2),
        singular_values=np.ones(2),
        minval=-np.ones(2),
        maxval=np.ones(2),
        latent_mean=np.zeros(2),
        latent_std=np.ones(2),
    )
    inputs_path = tmp_path / "inputs.npz"
    outputs_path = tmp_path / "outputs.npz"
    compression.dump(inputs_path)
    compression.dump(outputs_path)
    affine = Affine(inputs_rank=2, outputs_rank=2, key=jax.random.key(2), eps=1.0)
    edge = ImplicitAffine(inputs_compression=inputs_path, outputs_compression=outputs_path)
    inputs = jnp.asarray([0.3, -0.4])
    outputs = jnp.asarray([0.5, -0.2])
    runtime_inputs = {"value": inputs, "module": affine}
    output_payload = {"value": outputs}
    residuals = edge.evaluate(runtime_inputs, output_payload)

    assert edge.resolve_inputs_rank() == 2
    assert edge.resolve_outputs_rank() == 2
    assert jnp.allclose(edge.solve(runtime_inputs, residuals)["value"], outputs)
    assert jnp.allclose(
        edge.forward({"inputs": runtime_inputs, "outputs": output_payload})["residuals"]["value"],
        residuals["value"],
    )
    assert edge.sample_inputs(jax.random.key(0))["value"].shape == (2,)
    assert edge.sample_outputs(jax.random.key(1))["value"].shape == (2,)


def test_implicit_affine_scalar_and_nonlinear_jacobian() -> None:
    affine = Affine(inputs_rank=1, outputs_rank=1, key=jax.random.key(3), eps=1.0)
    edge = ImplicitAffine(inputs_rank=1, outputs_rank=1)
    inputs = {"value": jnp.asarray(0.3), "module": affine}
    outputs = {"value": jnp.asarray(0.5)}
    residuals = edge.evaluate(inputs, outputs)

    assert residuals["value"].shape == (1,)
    assert jnp.allclose(edge.solve(inputs, residuals)["value"], outputs["value"])

    nonlinear_affine = Affine(
        inputs_rank=1,
        outputs_rank=1,
        key=jax.random.key(4),
        jacobian_inputs="both",
        eps=1.0,
    )
    nonlinear_inputs = {"value": jnp.asarray(0.3), "module": nonlinear_affine}
    nonlinear_residuals = edge.evaluate(nonlinear_inputs, outputs)
    nonlinear_solution = edge.solve(nonlinear_inputs, nonlinear_residuals)
    assert jnp.allclose(nonlinear_solution["value"], outputs["value"], atol=1e-4)

    def evaluate_scalar(value: jax.Array) -> jax.Array:
        return edge.evaluate({"value": value, "module": affine}, {"value": value})["value"]

    values = jnp.asarray([0.0, 0.5, 1.0])
    assert jax.vmap(evaluate_scalar)(values).shape == (3, 1)
    assert jax.jit(evaluate_scalar)(jnp.asarray(0.2)).shape == (1,)


def test_linear_solver_config_and_implicit_affine_initialization() -> None:
    """Lineax solver configurations support YAML-friendly initial guesses."""
    solver = LinearSolver.model_validate(
        {
            "solver": {"name": "lineax.GMRES", "kwargs": {"rtol": 1e-5, "atol": 1e-6}},
            "initial": {"callable": "constant", "inputs_default": {"const": 0.25}},
        }
    )
    assert isinstance(solver.solver, lx.GMRES)
    assert solver.model_dump()["solver"]["name"] == "lineax.GMRES"

    affine = Affine(inputs_rank=1, outputs_rank=1, key=jax.random.key(22), eps=1.0)
    edge = ImplicitAffine(solver=solver)
    inputs = {"value": jnp.asarray(0.3), "module": affine}
    outputs = {"value": jnp.asarray(0.5)}
    residuals = edge.evaluate(inputs, outputs)

    assert jnp.allclose(edge.solve(inputs, residuals)["value"], outputs["value"], atol=1e-4)
    assert jnp.allclose(
        edge.solve({**inputs, "solver": {"initial": {"const": 0.75}}}, residuals)["value"],
        outputs["value"],
        atol=1e-4,
    )


def test_linear_solver_forwards_initial_guess_as_y0(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper forwards resolved initial fields through Lineax options."""
    seen: dict[str, object] = {}

    def linear_solve_spy(*args, **kwargs):
        seen["options"] = kwargs["options"]
        return SimpleNamespace(value=jnp.asarray([1.0]))

    monkeypatch.setattr(lx, "linear_solve", linear_solve_spy)
    solver = LinearSolver(solver={"name": "lineax.QR"}, options={"keep": 1})
    result = solver.linear_solve(
        lx.MatrixLinearOperator(jnp.eye(1)),
        jnp.asarray([1.0]),
        y0=jnp.asarray([0.25]),
        options={"runtime": 2},
    )

    assert jnp.array_equal(result, jnp.asarray([1.0]))
    assert seen["options"] == {"keep": 1, "runtime": 2, "y0": jnp.asarray([0.25])}


def test_implicit_affine_solver_union_validates_plain_dictionaries() -> None:
    """Pydantic selects the wrapper matching each third-party solver specification."""
    linear = ImplicitAffine.model_validate({"solver": {"solver": {"name": "lineax.QR"}}})
    iterative = ImplicitAffine.model_validate(
        {"solver": {"solver": {"name": "optimistix.Newton", "kwargs": {"rtol": 1.0, "atol": 1e-4}}}}
    )

    assert isinstance(linear.solver, LinearSolver)
    assert isinstance(iterative.solver, IterativeSolver)
    with pytest.raises(ValidationError):
        ImplicitAffine(solver=lx.QR())


def test_implicit_affine_rejects_linear_solver_for_output_jacobian() -> None:
    """Lineax solves are rejected when the affine matrix depends on outputs."""
    affine = Affine(inputs_rank=1, outputs_rank=1, key=jax.random.key(23), jacobian_inputs="both", eps=1.0)
    edge = ImplicitAffine(solver=LinearSolver(solver={"name": "lineax.QR"}))
    inputs = {"value": jnp.asarray(0.3), "module": affine}
    residuals = edge.evaluate(inputs, {"value": jnp.asarray(0.5)})

    with pytest.raises(TypeError, match="Output-dependent"):
        edge.solve(inputs, residuals)


def test_random_newton_runtime_seed_and_relative_perturbations() -> None:
    """Random Newton options produce deterministic, seed-dependent scalar paths."""
    step_size = Distribution(callable="uniform", minval=0.2, maxval=0.8)
    randomized = IterativeSolver(
        solver=RandomNewton(rtol=1.0, atol=1e-6, step_size=step_size),
        max_steps=1,
        throw=False,
    )

    residual = lambda y, _: y - 1.0
    first = randomized.root_find(residual, jnp.asarray(0.0), options={"step_seed": jnp.asarray(7, dtype=jnp.uint32)})
    repeated = randomized.root_find(
        residual, jnp.asarray(0.0), options={"step_seed": jnp.asarray(7, dtype=jnp.uint32)}
    )
    second = randomized.root_find(residual, jnp.asarray(0.0), options={"step_seed": jnp.asarray(8, dtype=jnp.uint32)})

    assert jnp.allclose(first, repeated)
    assert not jnp.allclose(first, second)
    assert jnp.isfinite(
        jax.jit(
            lambda seed: randomized.root_find(residual, jnp.asarray(0.0), options={"step_seed": seed})
        )(jnp.asarray(9, dtype=jnp.uint32))
    )

    direction_solver = IterativeSolver(
        solver=RandomNewton(
            rtol=1.0,
            atol=1e-6,
            step_direction=Distribution.model_validate(2.0),
            step_direction_scale=0.5,
        ),
        max_steps=1,
        throw=False,
    )
    assert jnp.allclose(direction_solver.root_find(residual, jnp.asarray(0.0)), 0.5)

    final_solver = IterativeSolver(
        solver=RandomNewton(
            rtol=1.0,
            atol=1e-6,
            final_perturb=Distribution.model_validate(2.0),
            final_perturb_scale=0.25,
        ),
        max_steps=2,
        throw=False,
    )
    assert jnp.allclose(final_solver.root_find(residual, jnp.asarray(0.0)), 1.25)


def test_random_newton_accepts_yaml_friendly_distribution_config() -> None:
    """Third-party solver specs construct RandomNewton and its distribution mappings."""
    config = IterativeSolver.model_validate(
        {
            "solver": {
                "name": "romjax.RandomNewton",
                "kwargs": {
                    "rtol": 1.0,
                    "atol": 1e-6,
                    "step_size": {"callable": "uniform", "minval": 0.2, "maxval": 0.8},
                },
            }
        }
    )

    assert isinstance(config.solver, RandomNewton)
    assert isinstance(config.solver.step_size, Distribution)
    assert config.model_dump()["solver"]["kwargs"]["step_size"]["callable"] == "uniform"


def test_random_newton_supports_broadcast_distribution_pytrees() -> None:
    """RandomNewton preserves Optimistix PyTree states and validates distribution layouts."""
    target = {"left": jnp.asarray([1.0, 2.0]), "right": jnp.asarray([-3.0])}
    initial = jax.tree.map(jnp.zeros_like, target)

    def residual(state, _):
        return jax.tree.map(lambda value, expected: value - expected, state, target)

    deterministic = IterativeSolver(solver=RandomNewton(rtol=1.0, atol=1e-6), max_steps=1, throw=False)
    solved = deterministic.root_find(residual, initial)
    assert jax.tree.all(jax.tree.map(jnp.allclose, solved, target))

    randomized = IterativeSolver(
        solver=RandomNewton(rtol=1.0, atol=1e-6, step_size=Distribution.model_validate(0.5)),
        max_steps=1,
        throw=False,
    )
    halfway = randomized.root_find(residual, initial)
    assert jax.tree.all(jax.tree.map(lambda value, expected: jnp.allclose(value, 0.5 * expected), halfway, target))

    incompatible = IterativeSolver(
        solver=RandomNewton(
            rtol=1.0,
            atol=1e-6,
            step_direction={"missing": {"callable": "dirac", "value": 1.0}},
        ),
        max_steps=1,
        throw=False,
    )
    with pytest.raises(ValueError, match="state pytree structure"):
        incompatible.root_find(residual, initial)


def test_implicit_affine_passes_runtime_solver_options() -> None:
    """Nonlinear affine solves merge runtime solver options into the root finder."""
    affine = Affine(inputs_rank=1, outputs_rank=1, key=jax.random.key(12), jacobian_inputs="both", eps=1.0)
    edge = ImplicitAffine(
        solver=IterativeSolver(
            solver=RandomNewton(
                rtol=1.0,
                atol=1e-6,
                step_size=Distribution(callable="uniform", minval=0.2, maxval=0.8),
            ),
            max_steps=1,
            throw=False,
        )
    )
    inputs = {"value": jnp.asarray(0.3), "module": affine}
    residuals = edge.evaluate(inputs, {"value": jnp.asarray(0.5)})

    first = edge.solve({**inputs, "solver": {"options": {"step_seed": jnp.asarray(1, dtype=jnp.uint32)}}}, residuals)
    second = edge.solve({**inputs, "solver": {"options": {"step_seed": jnp.asarray(2, dtype=jnp.uint32)}}}, residuals)

    assert not jnp.allclose(first["value"], second["value"])


def test_implicit_affine_collects_additional_input_arrays_deterministically() -> None:
    affine = Affine(inputs_rank=5, outputs_rank=1, key=jax.random.key(8), identity_jac=True)
    inputs = {
        "value": jnp.asarray([0.3]),
        "module": affine,
        "inputs": [{"args": {"z": jnp.asarray([1.0, 2.0]), "a": jnp.asarray([3.0])}}],
        "diffusion": {"alpha": jnp.asarray(0.5)},
    }

    configured = ImplicitAffine(
        additional_inputs=[
            ["inputs", 0, "args"],
            ["diffusion", "alpha"],
        ]
    )
    configured_values, _ = configured._affine_inputs(inputs)
    assert jnp.array_equal(configured_values, jnp.asarray([0.3, 3.0, 1.0, 2.0, 0.5]))

    collect_all = ImplicitAffine(additional_inputs=())
    all_values, _ = collect_all._affine_inputs(inputs)
    assert jnp.array_equal(all_values, jnp.asarray([0.3, 0.5, 3.0, 1.0, 2.0]))

    residuals = configured.evaluate(inputs, {"value": jnp.asarray([0.2])})
    assert configured.solve(inputs, residuals)["value"].shape == (1,)


def test_affine_materializes_ldu_and_log_determinant() -> None:
    affine = Affine(inputs_rank=2, outputs_rank=3, key=jax.random.key(5), eps=1.0)
    matrix, solution = affine.materialize(jnp.ones(2), jnp.ones(3))

    assert matrix.shape == (3, 3)
    assert solution.shape == (3,)
    assert jnp.isfinite(matrix).all()
    payload = {"inputs": {"value": jnp.ones(2)}, "outputs": {"value": jnp.ones(3)}}
    assert jnp.allclose(
        affine.log_determinant(payload),
        jnp.sum(jnp.square(jnp.log(jnp.abs(affine.diagonal(jnp.ones(2)) + affine.eps)))),
    )

    scalar = Affine(inputs_rank=1, outputs_rank=1, key=jax.random.key(6), eps=1.0)
    assert scalar.lower is None
    assert scalar.upper is None
    scalar_matrix, scalar_solution = scalar.materialize(jnp.asarray(0.0), jnp.asarray(0.0))
    assert scalar_matrix.shape == (1, 1)
    assert scalar_solution.shape == (1,)


def test_affine_identity_jacobian_skips_mlps() -> None:
    affine = Affine(inputs_rank=2, outputs_rank=3, key=jax.random.key(7), identity_jac=True)

    assert affine.solution is not None
    assert affine.lower is None
    assert affine.upper is None
    assert affine.diagonal is None

    matrix, solution = affine.materialize(jnp.ones(2), jnp.ones(3))
    assert jnp.array_equal(matrix, jnp.eye(3))
    assert jnp.allclose(solution, affine.solution(jnp.ones(2)))
    assert affine.log_determinant({"not": "used"}) == pytest.approx(0.0)


def test_implicit_rank_fields_take_priority_over_compression() -> None:
    compression = SVD(
        energy_tol=0.9,
        center=False,
        rank=2,
        mean=np.zeros(3),
        basis=np.eye(2, 3),
        singular_values=np.ones(2),
    )

    affine = ImplicitAffine(
        inputs_rank=3,
        outputs_rank=4,
        inputs_compression=compression,
        outputs_compression=compression,
    )
    galerkin = ImplicitIterativeGalerkin(
        source="a",
        target="b",
        name="galerkin",
        path=["ab"],
        rank=5,
        compression=compression,
    )

    assert affine.resolve_inputs_rank() == 3
    assert affine.resolve_outputs_rank() == 4
    assert galerkin.resolve_rank() == 5


def test_alive_progress_meter_is_jit_compatible() -> None:
    solver = diffrax.Euler()
    meter = AliveProgressMeter()

    solution = jax.jit(
        lambda y0: diffrax.diffeqsolve(
            diffrax.ODETerm(lambda t, y, args: y),
            solver=solver,
            t0=0.0,
            t1=0.1,
            dt0=0.05,
            y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.asarray([0.0, 0.05, 0.1])),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=16,
            progress_meter=meter,
        ).ys
    )(jnp.asarray(1.0))

    assert solution.shape == (3,)
    assert jnp.isfinite(solution).all()
    
