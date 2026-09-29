"""Configurable preconditioner construction for Lineax linear systems."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Annotated, Literal

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import PyTree
from pydantic import BeforeValidator, Field, field_validator

from romjax.typing import DictModel, from_registry

__all__ = [
    "LinearSystemStructure",
    "FivePointStencilStructure",
    "LinearPreconditioner",
    "RegisteredLinearPreconditioner",
    "LINEAR_PRECONDITIONER_REGISTRY",
    "FivePointStencil",
    "FivePointJacobi",
    "FivePointBlockJacobi",
]


class LinearSystemStructure(DictModel):
    """Static structural metadata supplied by a model when it assembles a linear system."""


class FivePointStencilStructure(LinearSystemStructure):
    """Describe a two-dimensional array operator with a local five-point stencil.

    :ivar shape: grid shape of both the operator input and output
    :ivar periodic_axes: whether either boundary is periodic along each grid axis
    """

    shape: tuple[int, int]
    periodic_axes: tuple[bool, bool] = (False, False)

    @field_validator("shape")
    @classmethod
    def _positive_shape(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or any(size <= 0 for size in value):
            raise ValueError("shape must contain two positive dimensions")
        return value


class LinearPreconditioner(DictModel, ABC):
    """Recipe that builds a concrete Lineax preconditioner for one linear solve.

    Recipes are static, YAML-friendly configuration. :meth:`build` receives the
    current operator and right-hand side so the resulting preconditioner may vary
    with both model inputs and target residuals.
    """

    name: str

    @abstractmethod
    def build(
        self,
        operator: lx.AbstractLinearOperator,
        vector: PyTree,
        *,
        structure: LinearSystemStructure | None = None,
    ) -> lx.AbstractLinearOperator:
        """Build a concrete preconditioner for ``operator @ x = vector``.

        :param operator: current Lineax operator
        :param vector: current right-hand side
        :param structure: optional model-provided structural metadata
        :return: Lineax operator approximating the inverse of ``operator``
        """
        raise NotImplementedError


class FivePointStencil:
    """Coefficients for a nonperiodic five-point operator on a two-dimensional grid."""

    center: jax.Array
    south: jax.Array
    north: jax.Array
    west: jax.Array
    east: jax.Array

    def __init__(
        self,
        center: jax.Array,
        south: jax.Array,
        north: jax.Array,
        west: jax.Array,
        east: jax.Array,
    ) -> None:
        self.center = center
        self.south = south
        self.north = north
        self.west = west
        self.east = east

    @property
    def shape(self) -> tuple[int, int]:
        """Return the structured grid shape."""
        return self.center.shape

    def mv(self, value: jax.Array) -> jax.Array:
        """Apply the represented stencil to a grid field.

        :param value: grid field
        :return: stencil product
        """
        result = self.center * value
        result = result.at[1:, :].add(self.south[1:, :] * value[:-1, :])
        result = result.at[:-1, :].add(self.north[:-1, :] * value[1:, :])
        result = result.at[:, 1:].add(self.west[:, 1:] * value[:, :-1])
        return result.at[:, :-1].add(self.east[:, :-1] * value[:, 1:])

    @classmethod
    def from_operator(
        cls,
        operator: lx.AbstractLinearOperator,
        structure: FivePointStencilStructure,
    ) -> "FivePointStencil":
        """Recover stencil coefficients with five distance-one coloring probes.

        :param operator: two-dimensional array-valued Lineax operator
        :param structure: declared five-point structure
        :return: recovered stencil coefficients
        """
        if any(structure.periodic_axes):
            raise ValueError("Five-point coloring currently requires nonperiodic boundaries")
        input_structure = operator.in_structure()
        output_structure = operator.out_structure()
        if not isinstance(input_structure, jax.ShapeDtypeStruct) or not isinstance(
            output_structure, jax.ShapeDtypeStruct
        ):
            raise TypeError("Five-point preconditioning requires an array-valued Lineax operator")
        if input_structure.shape != structure.shape or output_structure.shape != structure.shape:
            raise ValueError(
                f"Operator input/output shapes must both equal the declared grid shape {structure.shape}"
            )

        rows, columns = jnp.indices(structure.shape)
        colors = (rows + 2 * columns) % 5
        seeds = jnp.stack([(colors == color).astype(input_structure.dtype) for color in range(5)])
        responses = jax.vmap(operator.mv)(seeds)

        def select(response: jax.Array, color: jax.Array) -> jax.Array:
            return jnp.take_along_axis(response, color[None, ...], axis=0)[0]

        zeros = jnp.zeros(structure.shape, dtype=responses.dtype)
        center = select(responses, colors)
        south = zeros.at[1:, :].set(select(responses[:, 1:, :], colors[:-1, :]))
        north = zeros.at[:-1, :].set(select(responses[:, :-1, :], colors[1:, :]))
        west = zeros.at[:, 1:].set(select(responses[:, :, 1:], colors[:, :-1]))
        east = zeros.at[:, :-1].set(select(responses[:, :, :-1], colors[:, 1:]))
        return cls(center, south, north, west, east)

    def block_diagonal(self, block_shape: tuple[int, int]) -> jax.Array:
        """Assemble dense diagonal blocks for a nonoverlapping tiling.

        :param block_shape: row and column extent of each tile
        :return: batched dense block matrices
        """
        nx, ny = self.shape
        bx, by = block_shape
        if nx % bx or ny % by:
            raise ValueError(f"Grid shape {self.shape} must be divisible by block shape {block_shape}")

        center = self._pack_blocks(self.center, block_shape)
        south = self._pack_blocks(self.south, block_shape)
        north = self._pack_blocks(self.north, block_shape)
        west = self._pack_blocks(self.west, block_shape)
        east = self._pack_blocks(self.east, block_shape)
        block_size = bx * by
        indices = jnp.arange(block_size).reshape(bx, by)
        diagonal = jnp.arange(block_size)
        matrices = jnp.zeros((center.shape[0], block_size, block_size), dtype=center.dtype)
        matrices = matrices.at[:, diagonal, diagonal].set(center)

        south_rows, south_columns = indices[1:, :].ravel(), indices[:-1, :].ravel()
        north_rows, north_columns = indices[:-1, :].ravel(), indices[1:, :].ravel()
        west_rows, west_columns = indices[:, 1:].ravel(), indices[:, :-1].ravel()
        east_rows, east_columns = indices[:, :-1].ravel(), indices[:, 1:].ravel()
        matrices = matrices.at[:, south_rows, south_columns].set(south[:, south_rows])
        matrices = matrices.at[:, north_rows, north_columns].set(north[:, north_rows])
        matrices = matrices.at[:, west_rows, west_columns].set(west[:, west_rows])
        return matrices.at[:, east_rows, east_columns].set(east[:, east_rows])

    @staticmethod
    def _pack_blocks(value: jax.Array, block_shape: tuple[int, int]) -> jax.Array:
        nx, ny = value.shape
        bx, by = block_shape
        return value.reshape(nx // bx, bx, ny // by, by).transpose(0, 2, 1, 3).reshape(-1, bx * by)

    @staticmethod
    def _unpack_blocks(
        value: jax.Array,
        grid_shape: tuple[int, int],
        block_shape: tuple[int, int],
    ) -> jax.Array:
        nx, ny = grid_shape
        bx, by = block_shape
        return value.reshape(nx // bx, ny // by, bx, by).transpose(0, 2, 1, 3).reshape(grid_shape)


class FivePointJacobi(LinearPreconditioner):
    """Point Jacobi preconditioner extracted from a five-point operator."""

    name: Literal["five_point_jacobi"] = "five_point_jacobi"

    def build(
        self,
        operator: lx.AbstractLinearOperator,
        vector: PyTree,
        *,
        structure: LinearSystemStructure | None = None,
    ) -> lx.AbstractLinearOperator:
        """Build the inverse diagonal operator for the current system."""
        del vector
        stencil_structure = _require_five_point_structure(structure)
        stencil = FivePointStencil.from_operator(operator, stencil_structure)
        return lx.DiagonalLinearOperator(jnp.reciprocal(stencil.center))


class FivePointBlockJacobi(LinearPreconditioner):
    """Nonoverlapping block Jacobi preconditioner for a five-point operator.

    :ivar block_shape: row and column extent of each dense diagonal block
    """

    name: Literal["five_point_block_jacobi"] = "five_point_block_jacobi"
    block_shape: tuple[int, int] = Field(default=(4, 4))

    @field_validator("block_shape")
    @classmethod
    def _positive_block_shape(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2 or any(size <= 0 for size in value):
            raise ValueError("block_shape must contain two positive dimensions")
        return value

    def build(
        self,
        operator: lx.AbstractLinearOperator,
        vector: PyTree,
        *,
        structure: LinearSystemStructure | None = None,
    ) -> lx.AbstractLinearOperator:
        """Build and invert all diagonal blocks for the current system."""
        del vector
        stencil_structure = _require_five_point_structure(structure)
        stencil = FivePointStencil.from_operator(operator, stencil_structure)
        inverse = jnp.linalg.inv(stencil.block_diagonal(self.block_shape))
        grid_shape = stencil.shape
        block_shape = self.block_shape

        def apply(value: jax.Array) -> jax.Array:
            packed = stencil._pack_blocks(value, block_shape)
            solved = jnp.einsum("bij,bj->bi", inverse, packed)
            return stencil._unpack_blocks(solved, grid_shape, block_shape)

        return lx.FunctionLinearOperator(apply, operator.in_structure())


def _require_five_point_structure(
    structure: LinearSystemStructure | None,
) -> FivePointStencilStructure:
    if not isinstance(structure, FivePointStencilStructure):
        raise TypeError("Five-point preconditioning requires FivePointStencilStructure metadata")
    return structure


LINEAR_PRECONDITIONER_REGISTRY: dict[str, type[LinearPreconditioner]] = {
    "five_point_jacobi": FivePointJacobi,
    "five_point_block_jacobi": FivePointBlockJacobi,
}

type RegisteredLinearPreconditioner = Annotated[
    LinearPreconditioner,
    BeforeValidator(partial(from_registry, LINEAR_PRECONDITIONER_REGISTRY)),
]
