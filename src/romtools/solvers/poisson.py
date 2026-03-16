from typing import Callable, Literal, Mapping, TypedDict

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from pydantic import Field, PositiveFloat, PositiveInt, ValidationInfo, field_validator

from romtools.model import Model
from romtools.solvers.utils import (
    adjoint_vjp_solve,
    boundary_pass_through,
    damped_jacobi_step,
    fixed_point_solve,
    gmres_solve,
    homogeneous_boundary,
    UniformGrid,
)
from romtools.typing import BoundaryCallable, Coordinates, DictModel, ForcingCallable, PyTree
from romtools.utils import merge_pytrees, to_pytree

type ForcingName = Literal["gaussian", "nonlinear"]
type IterName = Literal["damped_jacobi"]
type AdjointName = Literal["gmres"]
type IterCallable = Callable[[PyTree, PyTree], PyTree]
type AdjointCallable = Callable[[Callable[[ArrayLike], ArrayLike], ArrayLike, PyTree, PyTree], ArrayLike]


class GaussianForcingInputs(DictModel):
    """Inputs for Gaussian forcing function.

    :ivar A0: amplitude
    :ivar sigma: symmetric width of Gaussian
    :ivar mu_x: center of Gaussian in x-direction
    :ivar mu_y: center of Gaussian in y-direction
    :ivar coords: (x,y) coordinates to evaluate at
    """
    A0: ArrayLike = 1.0
    sigma: ArrayLike = 1.0
    mu_x: ArrayLike = 0.0
    mu_y: ArrayLike = 0.0
    coords: Coordinates = (0., 0.)


class NonlinearConductivityInputs(DictModel):
    """Inputs for nonlinear conductivity function.
    
    :ivar k0: the background random field conductivity (2D)
    :ivar alpha: the strength of the nonlinearity
    """
    k0: ArrayLike = 1.0
    alpha: ArrayLike = 1.0


class PoissonInputs(DictModel):
    """Inputs for the Poisson equation.
    
    :ivar forcing: forcing inputs
    :ivar conductivity: conductivity inputs
    :ivar boundary: boundary condition parameters
    """
    forcing: DictModel = Field(default_factory=lambda: GaussianForcingInputs(A0=0.))
    conductivity: DictModel = Field(default_factory=NonlinearConductivityInputs)
    boundary: DictModel = Field(default_factory=lambda: homogeneous_boundary(ndim=2))


class PoissonOutputs(TypedDict):
    """Outputs for the Poisson equation.
    
    :ivar phi: the scalar potential on the grid
    """
    phi: ArrayLike


class PoissonResiduals(TypedDict):
    """Outputs of the Poisson residual.
    
    :ivar phi_residual: the residual of the scalar potential on the grid
    """
    phi_residual: ArrayLike


class PoissonConfig(DictModel):
    """Numerical configs for solving the Poisson PDE on a 2D grid.
    
    :ivar grid: the uniform 2D Cartesian grid
    :ivar solver: iteration and adjoint solver configuration
    """
    grid: UniformGrid
    solver: 'PoissonSolverConfig' = Field(default_factory=lambda: PoissonSolverConfig())

    @field_validator("grid", mode="after")
    @classmethod
    def _check_2d_grid(cls, value: UniformGrid) -> UniformGrid:
        if len(value.shape) != 2:
            raise ValueError("Only 2D grid supported for Poisson")
        
        return value


class DampedJacobiInputs(DictModel):
    """Inputs for damped Jacobi iteration."""
    damping: PositiveFloat = 0.1
    diag_eps: PositiveFloat = 1e-12


class GMRESInputs(DictModel):
    """Inputs for GMRES."""
    restart: PositiveInt = 20


class PoissonSolverConfig(DictModel):
    """Solver configuration for iteration and adjoint solves."""
    iter_method: IterCallable | IterName = "damped_jacobi"
    iter_defaults: DictModel = Field(default_factory=DampedJacobiInputs)
    adjoint_method: AdjointCallable | AdjointName = "gmres"
    adjoint_defaults: DictModel = Field(default_factory=GMRESInputs)

    max_iters: PositiveInt = 200
    min_iters: PositiveInt = 0
    tol: PositiveFloat = 1e-6
    damping: PositiveFloat = 0.1

    adjoint_max_iters: PositiveInt = 200
    adjoint_tol: PositiveFloat = 1e-6
    adjoint_restart: PositiveInt = 20

    @field_validator("iter_method", mode="before")
    @classmethod
    def _coerce_iter_method(cls, value: IterCallable | IterName) -> IterCallable:
        if isinstance(value, str):
            mapping: Mapping[IterName, IterCallable] = {
                "damped_jacobi": damped_jacobi_step,
            }
            if value not in mapping:
                raise ValueError(f"Unknown iteration method: {value!r}")
            return mapping[value]
        if callable(value):
            return value
        raise TypeError("iter_method must be a callable or a supported string literal.")

    @field_validator("adjoint_method", mode="before")
    @classmethod
    def _coerce_adjoint_method(cls, value: AdjointCallable | AdjointName) -> AdjointCallable:
        if isinstance(value, str):
            mapping: Mapping[AdjointName, AdjointCallable] = {
                "gmres": gmres_solve,
            }
            if value not in mapping:
                raise ValueError(f"Unknown adjoint method: {value!r}")
            return mapping[value]
        if callable(value):
            return value
        raise TypeError("adjoint_method must be a callable or a supported string literal.")


def gaussian_forcing(inputs: GaussianForcingInputs, outputs: PoissonOutputs) -> ArrayLike:
    """Symmetric Gaussian bump.

        $f(x,y) = A_0 \exp(-1/(2\sigma) ((x-\mu_x)^2 + (y-\mu_y)^2))$
    
    :param inputs: the input parameters
    :param outputs: the scalar potential on the grid (not used)
    :return: the forcing on the grid
    """
    dx = inputs['coords'][0] - inputs['mu_x']
    dy = inputs['coords'][1] - inputs['mu_y']
    return inputs['A0'] * jnp.exp(-(dx * dx + dy * dy) / (2 * inputs['sigma']))


def nonlinear_conductivity(inputs: NonlinearConductivityInputs, outputs: PoissonOutputs) -> ArrayLike:
    """Nonlinear conductivity.

        $k(x,y) = k_0(1 + \alpha \phi^2)$
    
    :param inputs: the input parameters
    :param outputs: the scalar potential on the grid
    :return: the conductivity on the grid
    """
    phi = outputs['phi']
    return inputs['k0'] * (1 + inputs['alpha'] * (phi * phi))


def _merge_inputs(defaults: DictModel, *overrides: PyTree | None) -> dict:
    merged = to_pytree(defaults)
    for override in overrides:
        if override is None:
            continue
        merged = merge_pytrees(merged, to_pytree(override))
    return merged


class Poisson2D(Model):

    # Required
    config: PoissonConfig

    # Optional/default
    forcing: ForcingCallable = gaussian_forcing
    conductivity: ForcingCallable = nonlinear_conductivity
    boundary: BoundaryCallable = boundary_pass_through

    forcing_defaults: DictModel = Field(
        default_factory=lambda: GaussianForcingInputs(A0=0.),
        description="Default inputs for the forcing function (any PyTree).",
    )
    conductivity_defaults: DictModel = Field(
        default_factory=NonlinearConductivityInputs,
        description="Default inputs for the conductivity function (any PyTree).",
    )
    boundary_defaults: DictModel = Field(
        default_factory=lambda: homogeneous_boundary(ndim=2),
        description="Default inputs for the boundary condition (any PyTree).",
    )
    
    @field_validator("forcing", "conductivity", mode="before")
    @classmethod
    def _coerce_forcing(cls, value: ForcingCallable | ForcingName, info: ValidationInfo) -> ForcingCallable:
        if isinstance(value, str):
            mapping: Mapping[ForcingName, ForcingCallable] = {
                "gaussian": gaussian_forcing,
                "nonlinear": nonlinear_conductivity,
            }
            if value not in mapping:
                raise ValueError(f"Unknown function: {value!r}")
            return mapping[value]
        if callable(value):
            return value
        raise TypeError("forcing and conductivity must be a callable or a supported string literal.")

    @field_validator("forcing_defaults", "conductivity_defaults", mode="before")
    @classmethod
    def _coerce_defaults(cls, value: object, info: ValidationInfo) -> DictModel:
        """Just provides default values for known forcing functions params."""
        if info.field_name == 'forcing_defaults':
            if info.data['forcing'] is gaussian_forcing:
                return GaussianForcingInputs(**value)
        
        if info.field_name == 'conductivity_defaults':
            if info.data['conductivity'] is nonlinear_conductivity:
                return NonlinearConductivityInputs(**value)
        return value

    def evaluate(self, inputs: PoissonInputs, outputs: PoissonOutputs) -> PoissonResiduals:
        """Evalute the Poisson residual on a 2D grid.
        
        :param inputs: params for forcing, conductivity, and boundary conditions
        :param outputs: the scalar potential on a 2D grid
        :return: the scalar residual on the 2D grid
        """
        phi = jnp.asarray(outputs["phi"])
        
        coords = self.config['grid']['coords']
        dx, dy = self.config['grid']['spacing']

        forcing_inputs = _merge_inputs(self.forcing_defaults, {'coords': coords}, inputs.get("forcing"))
        conductivity_inputs = _merge_inputs(self.conductivity_defaults, {'coords': coords}, inputs.get("conductivity"))
        boundary_inputs = _merge_inputs(self.boundary_defaults, {'coords': coords}, inputs.get("boundary"))

        forcing = jnp.asarray(self.forcing(forcing_inputs, outputs))
        conductivity = jnp.asarray(self.conductivity(conductivity_inputs, outputs))

        # 'ij' matrix ordering, so row=x, col=y
        boundary_values = self.boundary(boundary_inputs)
        xbds = boundary_values["boundary"][0]
        ybds = boundary_values["boundary"][1]

        def _ghost_for_side(
            spec: DictModel,
            interior: ArrayLike,
            opposite: ArrayLike,
            interior_k: ArrayLike,
            opposite_k: ArrayLike,
            spacing: ArrayLike,
        ) -> tuple[ArrayLike, ArrayLike]:
            """Get ghost cell values at each of the 4 sides, depending on BC type."""
            b_type = spec["type"]
            if b_type == "periodic":
                return opposite, opposite_k
            if b_type == "dirichlet":
                value = jnp.asarray(spec["value"])
                return 2.0 * value - interior, interior_k
            if b_type == "neumann":
                value = jnp.asarray(spec["value"])
                return interior + spacing * value, interior_k
            raise ValueError(f"Unsupported boundary type: {b_type!r}")

        phi_s, k_s = _ghost_for_side(
            xbds[0], phi[0, :], phi[-1, :], conductivity[0, :], conductivity[-1, :], dx
        )
        phi_n, k_n = _ghost_for_side(
            xbds[1], phi[-1, :], phi[0, :], conductivity[-1, :], conductivity[0, :], dx
        )
        phi_w, k_w = _ghost_for_side(
            ybds[0], phi[:, 0], phi[:, -1], conductivity[:, 0], conductivity[:, -1], dy
        )
        phi_e, k_e = _ghost_for_side(
            ybds[1], phi[:, -1], phi[:, 0], conductivity[:, -1], conductivity[:, 0], dy
        )

        phi_west = jnp.concatenate([phi_w[:, None], phi[:, :-1]], axis=1)
        phi_east = jnp.concatenate([phi[:, 1:], phi_e[:, None]], axis=1)
        phi_south = jnp.concatenate([phi_s[None, :], phi[:-1, :]], axis=0)
        phi_north = jnp.concatenate([phi[1:, :], phi_n[None, :]], axis=0)

        k_west = jnp.concatenate([k_w[:, None], conductivity[:, :-1]], axis=1)
        k_east = jnp.concatenate([conductivity[:, 1:], k_e[:, None]], axis=1)
        k_south = jnp.concatenate([k_s[None, :], conductivity[:-1, :]], axis=0)
        k_north = jnp.concatenate([conductivity[1:, :], k_n[None, :]], axis=0)

        k_face_w = 0.5 * (conductivity + k_west)
        k_face_e = 0.5 * (conductivity + k_east)
        k_face_s = 0.5 * (conductivity + k_south)
        k_face_n = 0.5 * (conductivity + k_north)

        flux_e = k_face_e * (phi_east - phi) / dy  
        flux_w = k_face_w * (phi - phi_west) / dy
        flux_n = k_face_n * (phi_north - phi) / dx
        flux_s = k_face_s * (phi - phi_south) / dx

        phi_residual = (flux_e - flux_w) / dy + (flux_n - flux_s) / dx - forcing

        return {'phi_residual': phi_residual}

    def solve(self, inputs: PoissonInputs, residuals: PoissonResiduals) -> PoissonOutputs:
        """Solve the Poisson equation for a target residual.
        
        :param inputs: params for forcing, conductivity, and boundary conditions
        :param residuals: the target scalar residual on the 2D grid
        :return: the scalar potential solution on the 2D grid
        """
        # TODO: implement a differentiable iterative solver for the Poisson equation.
        # This essentially acts as an "inverse" function for evaluate(), i.e. given some inputs and a target residual,
        # what is the scalar potential "phi" on the grid, i.e. the solution when residuals~0.
        # This should work generally for residuals != 0 as well.
        # It should also be agnostic to the particular iteration scheme, and jax grad should work via an adjoint method
        # rather than differentiating through the iterations.
        # All extra solver-specific configurations should go in Poisson2D.config with appropriate pydantic validation.
        # All "common" PDE-type solver utilities should go in solvers.utils.py.
        solver_cfg = self.config.solver
        iter_method = solver_cfg.iter_method if callable(solver_cfg.iter_method) else damped_jacobi_step
        adjoint_method = solver_cfg.adjoint_method if callable(solver_cfg.adjoint_method) else gmres_solve
        coords = self.config.grid.coords
        dx, dy = self.config.grid.spacing

        boundary_type_map = {"dirichlet": 0, "neumann": 1, "periodic": 2}

        def encode_boundary_types(tree: PyTree) -> PyTree:
            if isinstance(tree, dict):
                encoded = {k: encode_boundary_types(v) for k, v in tree.items()}
                if "type" in encoded and isinstance(encoded["type"], str):
                    encoded["type"] = boundary_type_map[encoded["type"]]
                return encoded
            if isinstance(tree, tuple):
                return tuple(encode_boundary_types(v) for v in tree)
            if isinstance(tree, list):
                return [encode_boundary_types(v) for v in tree]
            return tree

        def compute_residual(phi: ArrayLike, inputs_tree: PyTree, target: ArrayLike) -> ArrayLike:
            forcing_inputs = _merge_inputs(self.forcing_defaults, {"coords": coords}, inputs_tree.get("forcing"))
            conductivity_inputs = _merge_inputs(self.conductivity_defaults, {"coords": coords}, inputs_tree.get("conductivity"))
            boundary_defaults = encode_boundary_types(to_pytree(self.boundary_defaults))
            boundary_inputs = merge_pytrees(boundary_defaults, {"coords": coords})
            if inputs_tree.get("boundary") is not None:
                boundary_inputs = merge_pytrees(boundary_inputs, inputs_tree.get("boundary"))

            boundary_values = self.boundary(boundary_inputs)
            xbds = boundary_values["boundary"][0]
            ybds = boundary_values["boundary"][1]

            def _ghost_for_side(
                spec: DictModel,
                interior: ArrayLike,
                opposite: ArrayLike,
                interior_k: ArrayLike,
                opposite_k: ArrayLike,
                spacing: ArrayLike,
            ) -> tuple[ArrayLike, ArrayLike]:
                b_type = spec["type"]
                if b_type == 2:
                    return opposite, opposite_k
                if b_type == 0:
                    value = jnp.asarray(spec["value"])
                    return 2.0 * value - interior, interior_k
                if b_type == 1:
                    value = jnp.asarray(spec["value"])
                    return interior + spacing * value, interior_k
                raise ValueError(f"Unsupported boundary type: {b_type!r}")
            forcing = jnp.asarray(self.forcing(forcing_inputs, {"phi": phi}))
            conductivity = jnp.asarray(self.conductivity(conductivity_inputs, {"phi": phi}))
            phi_s, k_s = _ghost_for_side(
                xbds[0], phi[0, :], phi[-1, :], conductivity[0, :], conductivity[-1, :], dx
            )
            phi_n, k_n = _ghost_for_side(
                xbds[1], phi[-1, :], phi[0, :], conductivity[-1, :], conductivity[0, :], dx
            )
            phi_w, k_w = _ghost_for_side(
                ybds[0], phi[:, 0], phi[:, -1], conductivity[:, 0], conductivity[:, -1], dy
            )
            phi_e, k_e = _ghost_for_side(
                ybds[1], phi[:, -1], phi[:, 0], conductivity[:, -1], conductivity[:, 0], dy
            )

            phi_west = jnp.concatenate([phi_w[:, None], phi[:, :-1]], axis=1)
            phi_east = jnp.concatenate([phi[:, 1:], phi_e[:, None]], axis=1)
            phi_south = jnp.concatenate([phi_s[None, :], phi[:-1, :]], axis=0)
            phi_north = jnp.concatenate([phi[1:, :], phi_n[None, :]], axis=0)

            k_west = jnp.concatenate([k_w[:, None], conductivity[:, :-1]], axis=1)
            k_east = jnp.concatenate([conductivity[:, 1:], k_e[:, None]], axis=1)
            k_south = jnp.concatenate([k_s[None, :], conductivity[:-1, :]], axis=0)
            k_north = jnp.concatenate([conductivity[1:, :], k_n[None, :]], axis=0)

            k_face_w = 0.5 * (conductivity + k_west)
            k_face_e = 0.5 * (conductivity + k_east)
            k_face_s = 0.5 * (conductivity + k_south)
            k_face_n = 0.5 * (conductivity + k_north)

            flux_e = k_face_e * (phi_east - phi) / dy
            flux_w = k_face_w * (phi - phi_west) / dy
            flux_n = k_face_n * (phi_north - phi) / dx
            flux_s = k_face_s * (phi - phi_south) / dx

            phi_residual = (flux_e - flux_w) / dy + (flux_n - flux_s) / dx - forcing
            return phi_residual - target

        def compute_diag(phi: ArrayLike, inputs_tree: PyTree) -> ArrayLike:
            forcing_inputs = _merge_inputs(self.forcing_defaults, {"coords": coords}, inputs_tree.get("forcing"))
            conductivity_inputs = _merge_inputs(self.conductivity_defaults, {"coords": coords}, inputs_tree.get("conductivity"))
            boundary_defaults = encode_boundary_types(to_pytree(self.boundary_defaults))
            boundary_inputs = merge_pytrees(boundary_defaults, {"coords": coords})
            if inputs_tree.get("boundary") is not None:
                boundary_inputs = merge_pytrees(boundary_inputs, inputs_tree.get("boundary"))

            boundary_values = self.boundary(boundary_inputs)
            xbds = boundary_values["boundary"][0]
            ybds = boundary_values["boundary"][1]

            def _ghost_for_side(
                spec: DictModel,
                interior: ArrayLike,
                opposite: ArrayLike,
                interior_k: ArrayLike,
                opposite_k: ArrayLike,
                spacing: ArrayLike,
            ) -> tuple[ArrayLike, ArrayLike]:
                b_type = spec["type"]
                if b_type == 2:
                    return opposite, opposite_k
                if b_type == 0:
                    value = jnp.asarray(spec["value"])
                    return 2.0 * value - interior, interior_k
                if b_type == 1:
                    value = jnp.asarray(spec["value"])
                    return interior + spacing * value, interior_k
                raise ValueError(f"Unsupported boundary type: {b_type!r}")

            conductivity = jnp.asarray(self.conductivity(conductivity_inputs, {"phi": phi}))
            phi_s, k_s = _ghost_for_side(
                xbds[0], phi[0, :], phi[-1, :], conductivity[0, :], conductivity[-1, :], dx
            )
            phi_n, k_n = _ghost_for_side(
                xbds[1], phi[-1, :], phi[0, :], conductivity[-1, :], conductivity[0, :], dx
            )
            phi_w, k_w = _ghost_for_side(
                ybds[0], phi[:, 0], phi[:, -1], conductivity[:, 0], conductivity[:, -1], dy
            )
            phi_e, k_e = _ghost_for_side(
                ybds[1], phi[:, -1], phi[:, 0], conductivity[:, -1], conductivity[:, 0], dy
            )

            k_west = jnp.concatenate([k_w[:, None], conductivity[:, :-1]], axis=1)
            k_east = jnp.concatenate([conductivity[:, 1:], k_e[:, None]], axis=1)
            k_south = jnp.concatenate([k_s[None, :], conductivity[:-1, :]], axis=0)
            k_north = jnp.concatenate([conductivity[1:, :], k_n[None, :]], axis=0)

            k_face_w = 0.5 * (conductivity + k_west)
            k_face_e = 0.5 * (conductivity + k_east)
            k_face_s = 0.5 * (conductivity + k_south)
            k_face_n = 0.5 * (conductivity + k_north)

            return (k_face_e + k_face_w) / (dy * dy) + (k_face_n + k_face_s) / (dx * dx)

        def solve_impl(inputs_tree: PyTree, residuals_tree: PyTree) -> ArrayLike:
            target = jnp.asarray(residuals_tree["phi_residual"])

            residual_fn = lambda phi: compute_residual(phi, inputs_tree, target)
            diag_fn = lambda phi: compute_diag(phi, inputs_tree)

            phi0 = inputs_tree.get("phi0", jnp.zeros_like(target))
            iter_inputs = merge_pytrees(
                to_pytree(solver_cfg.iter_defaults),
                {"residual_fn": residual_fn, "diag_fn": diag_fn, "damping": solver_cfg.damping},
            )
            init_state = {"phi": phi0, "residual": residual_fn(phi0), "diag": diag_fn(phi0)}
            step_fn = lambda state: iter_method(iter_inputs, state)
            state = fixed_point_solve(step_fn, init_state, solver_cfg)
            return state["phi"]

        @jax.custom_vjp
        def solve_with_adjoint(inputs_tree: PyTree, residuals_tree: PyTree) -> ArrayLike:
            return solve_impl(inputs_tree, residuals_tree)

        def fwd(inputs_tree: PyTree, residuals_tree: PyTree) -> tuple[ArrayLike, tuple[ArrayLike, PyTree, PyTree]]:
            phi = solve_impl(inputs_tree, residuals_tree)
            return phi, (phi, inputs_tree, residuals_tree)

        def bwd(
            res: tuple[ArrayLike, PyTree, PyTree], cot_phi: ArrayLike
        ) -> tuple[PyTree, PyTree]:
            phi, inputs_tree, residuals_tree = res
            target = jnp.asarray(residuals_tree["phi_residual"])

            def F(phi_: ArrayLike, inputs_: PyTree, target_: PyTree) -> ArrayLike:
                return compute_residual(phi_, inputs_, target_)

            method_inputs = to_pytree(solver_cfg.adjoint_defaults)
            dinputs, dtarget = adjoint_vjp_solve(
                F,
                phi,
                inputs_tree,
                target,
                cot_phi,
                adjoint_method,
                solver_cfg,
                method_inputs,
            )
            return dinputs, {"phi_residual": dtarget}

        solve_with_adjoint.defvjp(fwd, bwd)
        inputs_tree = encode_boundary_types(to_pytree(inputs))
        residuals_tree = to_pytree(residuals)
        phi = solve_with_adjoint(inputs_tree, residuals_tree)
        return {"phi": phi}
    
