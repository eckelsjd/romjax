"""Tests for configurable Lineax preconditioners."""

from types import SimpleNamespace
from typing import Literal

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from romjax.pde import LinearSolver
from romjax.preconditioner import (
    FivePointBlockJacobi,
    FivePointJacobi,
    FivePointStencil,
    FivePointStencilStructure,
    LinearPreconditioner,
    LinearSystemStructure,
)


def _stencil(shape: tuple[int, int]) -> FivePointStencil:
    ones = jnp.ones(shape)
    zeros = jnp.zeros(shape)
    return FivePointStencil(
        center=5.0 * ones,
        south=zeros.at[1:, :].set(-ones[1:, :]),
        north=zeros.at[:-1, :].set(-ones[:-1, :]),
        west=zeros.at[:, 1:].set(-ones[:, 1:]),
        east=zeros.at[:, :-1].set(-ones[:, :-1]),
    )


def _operator(stencil: FivePointStencil) -> lx.AbstractLinearOperator:
    return lx.FunctionLinearOperator(
        stencil.mv,
        jax.ShapeDtypeStruct(stencil.shape, stencil.center.dtype),
    )


def test_five_point_stencil_recovery_and_jacobi() -> None:
    stencil = _stencil((8, 8))
    operator = _operator(stencil)
    structure = FivePointStencilStructure(shape=stencil.shape)

    recovered = FivePointStencil.from_operator(operator, structure)
    preconditioner = FivePointJacobi().build(operator, jnp.ones(stencil.shape), structure=structure)

    for name in ("center", "south", "north", "west", "east"):
        np.testing.assert_allclose(getattr(recovered, name), getattr(stencil, name))
    np.testing.assert_allclose(preconditioner.mv(stencil.center), jnp.ones(stencil.shape))


def test_block_jacobi_matches_independent_tile_solves() -> None:
    stencil = _stencil((8, 8))
    structure = FivePointStencilStructure(shape=stencil.shape)
    block_shape = (4, 4)
    preconditioner = FivePointBlockJacobi(block_shape=block_shape).build(
        _operator(stencil),
        jnp.ones(stencil.shape),
        structure=structure,
    )
    rhs = jnp.arange(64, dtype=jnp.float32).reshape(stencil.shape) / 64.0

    matrices = stencil.block_diagonal(block_shape)
    packed_rhs = stencil._pack_blocks(rhs, block_shape)
    expected_packed = jax.vmap(jnp.linalg.solve)(matrices, packed_rhs)
    expected = stencil._unpack_blocks(expected_packed, stencil.shape, block_shape)

    np.testing.assert_allclose(preconditioner.mv(rhs), expected, rtol=2.0e-5, atol=2.0e-6)


def test_preconditioner_configuration_round_trip_and_runtime_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solver = LinearSolver.model_validate(
        {"preconditioner": {"name": "five_point_block_jacobi", "block_shape": [2, 2]}}
    )
    dumped = solver.model_dump()["preconditioner"]
    assert dumped == {"name": "five_point_block_jacobi", "block_shape": (2, 2)}

    runtime = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((2, 2), jnp.float32))
    seen: dict[str, object] = {}

    def linear_solve_spy(*args, **kwargs):
        seen["preconditioner"] = kwargs["options"]["preconditioner"]
        return SimpleNamespace(value=jnp.ones((2, 2)))

    monkeypatch.setattr(lx, "linear_solve", linear_solve_spy)
    result = solver.linear_solve(
        runtime,
        jnp.ones((2, 2)),
        options={"preconditioner": runtime},
    )

    assert seen["preconditioner"] is runtime
    np.testing.assert_array_equal(result, jnp.ones((2, 2)))


def test_linear_solver_builds_recipe_from_current_system(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configured recipes receive the current operator, right-hand side, and structure."""
    seen: dict[str, object] = {}

    class RecordingPreconditioner(LinearPreconditioner):
        name: Literal["recording"] = "recording"

        def build(self, operator, vector, *, structure=None):
            seen.update(operator=operator, vector=vector, structure=structure)
            return lx.IdentityLinearOperator(operator.in_structure())

    operator = lx.MatrixLinearOperator(jnp.eye(2))
    vector = jnp.asarray([1.0, 2.0])
    structure = LinearSystemStructure(model="test")
    solver = LinearSolver(solver={"name": "lineax.QR"}, preconditioner=RecordingPreconditioner())

    def linear_solve_spy(*args, **kwargs):
        seen["concrete"] = kwargs["options"]["preconditioner"]
        return SimpleNamespace(value=vector)

    monkeypatch.setattr(lx, "linear_solve", linear_solve_spy)
    solver.linear_solve(operator, vector, structure=structure)

    assert seen["operator"] is operator
    assert seen["vector"] is vector
    assert seen["structure"] is structure
    assert isinstance(seen["concrete"], lx.IdentityLinearOperator)


def test_five_point_preconditioner_validates_structure() -> None:
    stencil = _stencil((6, 6))
    operator = _operator(stencil)
    recipe = FivePointBlockJacobi(block_shape=(4, 4))

    with pytest.raises(TypeError, match="FivePointStencilStructure"):
        recipe.build(operator, jnp.ones(stencil.shape))
    with pytest.raises(ValueError, match="nonperiodic"):
        recipe.build(
            operator,
            jnp.ones(stencil.shape),
            structure=FivePointStencilStructure(shape=stencil.shape, periodic_axes=(True, False)),
        )
    with pytest.raises(ValueError, match="divisible"):
        recipe.build(
            operator,
            jnp.ones(stencil.shape),
            structure=FivePointStencilStructure(shape=stencil.shape),
        )
