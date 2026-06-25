from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from romjax.compression import SVD
from romjax.graph import Edge, FunctionGraph, Node
from romjax.model import ImplicitModel
from romjax.pde import (
    AliveProgressMeter,
    BoundaryType,
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
                initial_guess=lambda arr: 0.1 * jnp.ones_like(arr),
            ),
        }
    )

    inputs = {"b": jnp.array([1.0, 1.5])}
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

    assert edge.resolve_latent_dim() == 2
    edge.resolve_source_sampler()
    sample = edge.sample_source(jax.random.key(0))
    assert sample["outputs"].shape == (2,)
    assert jnp.all(sample["outputs"] >= jnp.asarray([-1.0, -2.0]))
    assert jnp.all(sample["outputs"] <= jnp.asarray([1.0, 2.0]))


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
    
