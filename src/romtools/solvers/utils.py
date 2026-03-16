"""Utilites for PDE-based solvers."""
from typing import Literal

import jax.numpy as jnp
from jax.typing import ArrayLike
from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from romtools.typing import Coordinates, DictModel, PyTree

type BoundaryName = Literal["dirichlet", "neumann", "periodic"]


class BoundarySpec(DictModel):
    """Specify the type and value of a single boundary.
    
    :ivar type: the type of boundary (periodic, dirichlet, or neumann)
    :ivar value: the value of the boundary (periodic~empty, dirichlet~const, neumann~gradient)
    """
    type: BoundaryName
    value: ArrayLike


class GridBoundaryInputs(DictModel):
    """Periodic, neumann, or dirichlet boundaries on uniform grid.
    Each tuple is the left/right boundary conditions for a given dimension.
    """
    boundary: tuple[tuple[BoundarySpec, BoundarySpec], ...]

    @model_validator(mode='after')
    def _check_periodic(self) -> 'GridBoundaryInputs':
        """Make sure both sides are periodic for any dimension with at least one periodic."""
        for left_b, right_b in self.boundary:
            if left_b.type == 'periodic':
                if right_b.type != 'periodic':
                    raise ValueError("Must use matching periodic boundaries")
            
            if right_b.type == 'periodic':
                if left_b.type != 'periodic':
                    raise ValueError("Must use matching periodic boundaries")
        
        return self


def homogeneous_boundary(type: BoundaryName = 'dirichlet', 
                         value: float = 0., 
                         ndim: int = 1
                         ) -> GridBoundaryInputs:
    """Convenience func to use same BC on all boundaries of an N-dim uniform grid.

    Defaults to homogeneous dirichlet BCs.
    
    :param type: the type of boundary condition (periodic, neumann, or dirichlet)
    :param value: the constant value on all boundaries
    :param ndim: the number of dimensions in the grid
    :return: the BoundaryGrid object
    """
    return GridBoundaryInputs(
        boundary=tuple([(BoundarySpec(type=type, value=value), 
                         BoundarySpec(type=type, value=value))
                           for _ in range(ndim)])
    )


def boundary_pass_through(inputs: PyTree) -> PyTree:
    """Simple boundary that uses boundary input params directly (just pass them through)."""
    return inputs


class UniformGrid(DictModel):
    """Uniformly-spaced Cartesian grid (cell-centered). Either provide coords or some consistent 
       combination of shape, spacing, and bounds. If coords is not specified, then you must have
       bounds and only one of shape or spacing. Everything else gets filled in automatically.
       Use matrix 'ij' notation for meshgrid.
    
    :ivar shape: (Nx, ...) the grid shape
    :ivar spacing: (dx, ...) uniform spacing on the grid
    :ivar bounds: (xbounds, ...) the bounds in each dimension
    :ivar coords: (xgrid, ...) with each the same shape as the grid,
                  if 1D grids are passed, will be meshed to ND.
    """
    shape: tuple[PositiveInt, ...] | None = None
    spacing: tuple[PositiveFloat, ...] | None = None
    bounds: tuple[tuple[float, float], ...] | None = None
    coords: Coordinates | None = Field(default=None, exclude=True)  # don't serialize

    @model_validator(mode='after')
    def _coerce_grid(self) -> 'UniformGrid':
        """Ultimately, we need coords to be defined. Also check everything is consistent."""
        spacing_provided = self.spacing is not None and len(self.spacing) > 0
        shape_provided = self.shape is not None and len(self.shape) > 0
        if self.coords is None:
            if self.bounds is None:
                raise ValueError("Can't construct grid without bounds.")
            
            lengths = tuple(b[1] - b[0] for b in self.bounds)
                
            if any([L <= 0 for L in lengths]):
                raise ValueError("Grid bounds must be ordered as (lower, upper).")
            
            # Try to construct from spacing and shape
            if not shape_provided and not spacing_provided:
                raise ValueError("Can't construct grid without either spacing or shape.")
            
            if shape_provided and spacing_provided:
                expected_spacing = tuple(L/Nl for L, Nl in zip(lengths, self.shape))
                spacing_check = all([jnp.allclose(s1, s2) for s1, s2 in zip(expected_spacing, self.spacing)])
                if not spacing_check:
                    raise ValueError("Specified spacing is not consistent with bounds and shape.")
                
            if not shape_provided:
                self.shape = tuple(L/dl for L, dl in zip(lengths, self.spacing))

            if not spacing_provided:
                self.spacing = tuple(L/Nl for L, Nl in zip(lengths, self.shape))

            grids = [jnp.linspace(b[0]+dl/2, b[1]-dl/2, Nl) for b, dl, Nl in 
                     zip(self.bounds, self.spacing, self.shape)]
            self.coords = tuple(jnp.meshgrid(*grids, indexing='ij'))
        
        else:
            if self.coords[0].ndim == 1:
                if not all([arr.ndim == 1 for arr in self.coords]): 
                    raise ValueError("Must have all 1d coord arrays or all N-dim")
                self.coords = tuple(jnp.meshgrid(*self.coords, indexing='ij'))
            
            # Make sure shape, spacing, and bounds are consistent
            ndim = self.coords[0].ndim
            shape = self.coords[0].shape
            if not all([arr.ndim == ndim for arr in self.coords]):
                raise ValueError("All arrays must have same ndim")
            if not all([arr.shape == shape for arr in self.coords]):
                raise ValueError("All arrays must have same shape")
            if not len(self.coords) == ndim:
                raise ValueError("Must have exactly ndim coord arrays")

            bounds = tuple((float(jnp.min(arr)), float(jnp.max(arr))) for arr in self.coords)
            lengths = tuple(b[1] - b[0] for b in bounds)
            spacing = tuple(L / (Nl - 1) if Nl > 1 else 0.0 for L, Nl in zip(lengths, shape))  # cell-centered
            edge_bounds = tuple((b[0] - dl / 2, b[1] + dl / 2) for b, dl in zip(bounds, spacing))

            if self.shape is None:
                self.shape = shape
            else:
                if shape != self.shape:
                    raise ValueError("Specified shape is not consistent with provided coords")

            if self.bounds is None:
                self.bounds = edge_bounds
            else:
                bounds_check = all(
                    [jnp.allclose(jnp.asarray(b1), jnp.asarray(b2)) for b1, b2 in zip(edge_bounds, self.bounds)]
                )
                if not bounds_check:
                    raise ValueError("Specified bounds are not consistent with provided coords")
            
            if self.spacing is None:
                self.spacing = spacing
            else:
                spacing_check = all([jnp.allclose(s1, s2) for s1, s2 in zip(spacing, self.spacing)])
                if not spacing_check:
                    raise ValueError("Specified spacings are not consistent with provided coords")
            
        return self
