"""Utilites for PDE-based solvers."""
from typing import Callable, Literal

import jax
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


def damped_jacobi_step(inputs: PyTree, state: PyTree) -> PyTree:
    """Single damped Jacobi iteration step.

    Expects inputs to provide:
    - residual_fn(phi): residual at current phi
    - diag_fn(phi): diagonal approximation for Jacobian
    - damping: relaxation parameter
    - diag_eps: diagonal stabilization
    """
    phi = state["phi"]
    residual = inputs["residual_fn"](phi)
    diag = inputs["diag_fn"](phi)
    denom = diag + inputs.get("diag_eps", 1e-12)
    phi_next = phi - inputs.get("damping", 1.0) * residual / denom
    return {"phi": phi_next, "residual": residual, "diag": diag}


def fixed_point_solve(step_fn: Callable[[PyTree], PyTree], init_state: PyTree, cfg: PyTree) -> PyTree:
    """Fixed-point iteration with stopping criteria."""
    max_iters = int(getattr(cfg, "max_iters", 100))
    min_iters = int(getattr(cfg, "min_iters", 0))
    tol = float(getattr(cfg, "tol", 1e-6))

    def cond(carry: tuple[jnp.ndarray, PyTree]) -> jnp.ndarray:
        it, state = carry
        res_norm = jnp.linalg.norm(state["residual"])
        return jnp.logical_or(it < min_iters, jnp.logical_and(it < max_iters, res_norm > tol))

    def body(carry: tuple[jnp.ndarray, PyTree]) -> tuple[jnp.ndarray, PyTree]:
        it, state = carry
        return it + 1, step_fn(state)

    it0 = jnp.asarray(0)
    _, state = jax.lax.while_loop(cond, body, (it0, init_state))
    return state


def gmres_solve(op: Callable[[ArrayLike], ArrayLike], rhs: ArrayLike, cfg: PyTree, method_inputs: PyTree) -> ArrayLike:
    """Matrix-free GMRES solve using Arnoldi + least squares."""
    max_iters = int(getattr(cfg, "adjoint_max_iters", 50))
    restart = int(getattr(cfg, "adjoint_restart", max_iters))
    if isinstance(method_inputs, DictModel) and "restart" in method_inputs:
        restart = int(method_inputs["restart"])
    m = max(1, min(max_iters, restart))

    rhs_arr = jnp.asarray(rhs)
    rhs_flat = rhs_arr.reshape(-1)
    n = rhs_flat.shape[0]

    x0 = jnp.zeros_like(rhs_flat)
    r0 = rhs_flat
    beta = jnp.linalg.norm(r0)

    def solve_system() -> ArrayLike:
        v0 = r0 / beta
        V = jnp.zeros((m + 1, n), dtype=rhs_flat.dtype).at[0].set(v0)
        H = jnp.zeros((m + 1, m), dtype=rhs_flat.dtype)

        def arnoldi(j: int, carry: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
            V, H = carry
            w = op(V[j].reshape(rhs_arr.shape)).reshape(-1)

            def orth(i: int, inner: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
                w_local, H_local = inner
                hij = jnp.vdot(V[i], w_local)
                w_local = w_local - hij * V[i]
                H_local = H_local.at[i, j].set(hij)
                return w_local, H_local

            w, H = jax.lax.fori_loop(0, j + 1, orth, (w, H))
            h_next = jnp.linalg.norm(w)
            H = H.at[j + 1, j].set(h_next)
            v_next = jnp.where(h_next > 0, w / h_next, w)
            V = V.at[j + 1].set(v_next)
            return V, H

        V, H = jax.lax.fori_loop(0, m, arnoldi, (V, H))

        e1 = jnp.zeros((m + 1,), dtype=rhs_flat.dtype).at[0].set(beta)
        y, *_ = jnp.linalg.lstsq(H, e1, rcond=None)
        x = x0 + V[:-1].T @ y
        return x.reshape(rhs_arr.shape)

    x = solve_system()
    return jnp.where(beta > 0, x, jnp.zeros_like(rhs_arr))


def adjoint_vjp_solve(
    F: Callable[[ArrayLike, PyTree, PyTree], ArrayLike],
    phi: ArrayLike,
    inputs: PyTree,
    target: PyTree,
    cot_phi: ArrayLike,
    linear_solve: Callable[[Callable[[ArrayLike], ArrayLike], ArrayLike, PyTree, PyTree], ArrayLike],
    cfg: PyTree,
    method_inputs: PyTree,
) -> tuple[PyTree, PyTree]:
    """Solve adjoint system for implicit differentiation."""
    def F_all(phi_: ArrayLike, inputs_: PyTree, target_: PyTree) -> ArrayLike:
        return F(phi_, inputs_, target_)

    _, vjp_all = jax.vjp(F_all, phi, inputs, target)

    def op(v: ArrayLike) -> ArrayLike:
        dphi, _, _ = vjp_all(v)
        return dphi

    lam = linear_solve(op, cot_phi, cfg, method_inputs)
    _, dinputs, dtarget = vjp_all(lam)

    def safe_neg(x: ArrayLike) -> ArrayLike:
        if hasattr(x, "dtype") and x.dtype == jax.dtypes.float0:
            return x
        return -x

    return jax.tree_util.tree_map(safe_neg, dinputs), jax.tree_util.tree_map(safe_neg, dtarget)


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
                spacing_checks = jnp.array(
                    [jnp.allclose(s1, s2, atol=1e-6, rtol=1e-6) for s1, s2 in zip(expected_spacing, self.spacing)]
                )
                if not bool(jnp.all(spacing_checks)):
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
                bounds_checks = jnp.array(
                    [
                        jnp.allclose(jnp.asarray(b1), jnp.asarray(b2), atol=1e-6, rtol=1e-6)
                        for b1, b2 in zip(edge_bounds, self.bounds)
                    ]
                )
                if not bool(jnp.all(bounds_checks)):
                    raise ValueError("Specified bounds are not consistent with provided coords")
            
            if self.spacing is None:
                self.spacing = spacing
            else:
                spacing_checks = jnp.array(
                    [jnp.allclose(s1, s2, atol=1e-6, rtol=1e-6) for s1, s2 in zip(spacing, self.spacing)]
                )
                if not bool(jnp.all(spacing_checks)):
                    raise ValueError("Specified spacings are not consistent with provided coords")
            
        return self
