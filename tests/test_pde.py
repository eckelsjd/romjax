from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from romjax.compression import SVD
from romjax.graph import Edge, FunctionGraph, Node
from romjax.model import ImplicitModel
from romjax.nn import Affine
from romjax.pde import (
    AliveProgressMeter,
    BoundaryType,
    ImplicitAffine,
    ImplicitIterativeGalerkin,
    LatentSamplerFactory,
    UniformGrid,
    homogeneous_boundary,
)
from romjax.tree import pytree_merge


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

    inputs = {"b": jnp.array([1.0, 1.5]), "initial": {"outputs": 0.1 * jnp.ones(2)}}
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
    
