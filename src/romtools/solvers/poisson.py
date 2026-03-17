from typing import Literal, Mapping, TypedDict, Any

import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from jax.typing import ArrayLike
from pydantic import (
    Field, 
    PositiveFloat, 
    PositiveInt, 
    ValidationInfo, 
    field_validator, 
    model_validator,
)

from romtools.model import Model
from romtools.solvers.utils import (
    boundary_pass_through,
    homogeneous_boundary,
    UniformGrid,
)
from romtools.typing import (
    BoundaryCallable, 
    Coordinates, 
    DictModel, 
    ForcingCallable, 
    PyTree,
    OptxObject,
)
from romtools.utils import merge_pytrees, to_pytree


type ForcingName = Literal["gaussian", "nonlinear"]


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

    See Optimistix docs for `root_find()` method.
    
    :ivar grid: the uniform 2D Cartesian grid
    :ivar solver: Optimistix nonlinear root finding solver (name+opts or instance), default is Newton
    :ivar options: runtime options for the nonlinear solver
    :ivar max_steps: maximum number of solver steps
    :ivar adjoint: Optimistix adjoint method
    :ivar throw: whether to throw failures as errors (default True)
    """
    grid: UniformGrid
    solver: OptxObject = Field(default_factory=dict(name='Newton', opts={'rtol': 1e-3, 'atol': 1e-6}))
    adjoint: OptxObject = Field(default_factory=dict(name='ImplicitAdjoint'))
    options: dict[str, Any] = Field(default_factory=dict)
    max_steps: PositiveInt = 256
    throw: bool = True

    @field_validator("grid", mode="after")
    @classmethod
    def _check_2d_grid(cls, value: UniformGrid) -> UniformGrid:
        if len(value.shape) != 2:
            raise ValueError("Only 2D grid supported for Poisson")
        
        return value


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


# def _split_boundary_tree(tree: PyTree) -> tuple[PyTree, PyTree]:
#     """Split boundary tree into (types, values), keeping values as JAX arrays."""
#     if isinstance(tree, dict) and "type" in tree and "value" in tree:
#         return tree["type"], jnp.asarray(tree["value"])
#     if isinstance(tree, dict):
#         types: dict = {}
#         values: dict = {}
#         for key, val in tree.items():
#             t_child, v_child = _split_boundary_tree(val)
#             types[key] = t_child
#             values[key] = v_child
#         return types, values
#     if isinstance(tree, tuple):
#         types_list = []
#         values_list = []
#         for val in tree:
#             t_child, v_child = _split_boundary_tree(val)
#             types_list.append(t_child)
#             values_list.append(v_child)
#         return tuple(types_list), tuple(values_list)
#     if isinstance(tree, list):
#         types_list = []
#         values_list = []
#         for val in tree:
#             t_child, v_child = _split_boundary_tree(val)
#             types_list.append(t_child)
#             values_list.append(v_child)
#         return types_list, values_list
#     return tree, tree


def _poisson_residual_and_diag(
    phi: ArrayLike,
    forcing: ArrayLike,
    conductivity: ArrayLike,
    boundary_types: PyTree,
    boundary_values: PyTree,
    dx: ArrayLike,
    dy: ArrayLike,
    *,
    compute_diag: bool = True,
) -> tuple[ArrayLike, ArrayLike | None]:
    """Compute residual (and optional diagonal) for 2D Poisson."""
    boundary_types = boundary_types["boundary"]
    boundary_values = boundary_values["boundary"]
    xbds_types = boundary_types[0]
    ybds_types = boundary_types[1]
    xbds_values = boundary_values[0]
    ybds_values = boundary_values[1]

    def _ghost_for_side(
        b_type: str,
        value: ArrayLike,
        interior: ArrayLike,
        opposite: ArrayLike,
        interior_k: ArrayLike,
        opposite_k: ArrayLike,
        spacing: ArrayLike,
    ) -> tuple[ArrayLike, ArrayLike]:
        if b_type == "periodic":
            return opposite, opposite_k
        if b_type == "dirichlet":
            value = jnp.asarray(value)
            return 2.0 * value - interior, interior_k
        if b_type == "neumann":
            value = jnp.asarray(value)
            return interior + spacing * value, interior_k
        raise ValueError(f"Unsupported boundary type: {b_type!r}")

    phi_s, k_s = _ghost_for_side(
        xbds_types[0], xbds_values[0], phi[0, :], phi[-1, :], conductivity[0, :], conductivity[-1, :], dx
    )
    phi_n, k_n = _ghost_for_side(
        xbds_types[1], xbds_values[1], phi[-1, :], phi[0, :], conductivity[-1, :], conductivity[0, :], dx
    )
    phi_w, k_w = _ghost_for_side(
        ybds_types[0], ybds_values[0], phi[:, 0], phi[:, -1], conductivity[:, 0], conductivity[:, -1], dy
    )
    phi_e, k_e = _ghost_for_side(
        ybds_types[1], ybds_values[1], phi[:, -1], phi[:, 0], conductivity[:, -1], conductivity[:, 0], dy
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

    residual = (flux_e - flux_w) / dy + (flux_n - flux_s) / dx - forcing
    if not compute_diag:
        return residual, None

    diag = (k_face_e + k_face_w) / (dy * dy) + (k_face_n + k_face_s) / (dx * dx)
    return residual, diag


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

        boundary_tree = to_pytree(self.boundary(boundary_inputs))
        boundary_types, boundary_values = _split_boundary_tree(boundary_tree)

        phi_residual, _ = _poisson_residual_and_diag(
            phi,
            forcing,
            conductivity,
            boundary_types,
            boundary_values,
            dx,
            dy,
            compute_diag=False,
        )

        return {'phi_residual': phi_residual}

    def solve(self, inputs: PoissonInputs, residuals: PoissonResiduals) -> PoissonOutputs:
        """Solve the Poisson equation for a target residual.
        
        :param inputs: params for forcing, conductivity, and boundary conditions
        :param residuals: the target scalar residual on the 2D grid
        :return: the scalar potential solution on the 2D grid
        """
        solver_cfg = self.config.solver
        coords = self.config.grid.coords
        dx, dy = self.config.grid.spacing

        inputs_tree = to_pytree(inputs)
        residuals_tree = to_pytree(residuals)
        target = jnp.asarray(residuals_tree["phi_residual"])

        forcing_inputs = _merge_inputs(self.forcing_defaults, {"coords": coords}, inputs_tree.get("forcing"))
        conductivity_inputs = _merge_inputs(self.conductivity_defaults, {"coords": coords}, inputs_tree.get("conductivity"))
        boundary_inputs = _merge_inputs(self.boundary_defaults, {"coords": coords}, inputs_tree.get("boundary"))

        boundary_tree = to_pytree(self.boundary(boundary_inputs))
        boundary_types, boundary_values = _split_boundary_tree(boundary_tree)

        args = {
            "forcing": forcing_inputs,
            "conductivity": conductivity_inputs,
            "boundary": boundary_values,
        }

        phi0 = inputs_tree.get("phi0", jnp.zeros_like(target))
        root_solver = _build_root_solver(solver_cfg)
        adjoint_solver = _build_linear_solver(solver_cfg)
        adjoint = optx.ImplicitAdjoint(linear_solver=adjoint_solver)
        options = solver_cfg.solver_options or None

        def _forcing_cond(phi: ArrayLike, args_: PyTree) -> tuple[ArrayLike, ArrayLike]:
            forcing = jnp.asarray(self.forcing(args_["forcing"], {"phi": phi}))
            conductivity = jnp.asarray(self.conductivity(args_["conductivity"], {"phi": phi}))
            return forcing, conductivity

        def residual_fn(phi: ArrayLike, args_: PyTree) -> ArrayLike:
            forcing, conductivity = _forcing_cond(phi, args_)
            residual, _ = _poisson_residual_and_diag(
                phi,
                forcing,
                conductivity,
                boundary_types,
                args_["boundary"],
                dx,
                dy,
                compute_diag=False,
            )
            return residual - target

        def fixed_point_fn(phi: ArrayLike, args_: PyTree) -> ArrayLike:
            forcing, conductivity = _forcing_cond(phi, args_)
            residual, diag = _poisson_residual_and_diag(
                phi,
                forcing,
                conductivity,
                boundary_types,
                args_["boundary"],
                dx,
                dy,
                compute_diag=True,
            )
            denom = diag + solver_cfg.diag_eps
            return phi - solver_cfg.damping * (residual - target) / denom

        if isinstance(root_solver, optx.AbstractFixedPointSolver):
            fn = fixed_point_fn
        else:
            fn = residual_fn

        solution = optx.root_find(
            fn,
            solver=root_solver,
            y0=phi0,
            args=args,
            options=options,
            max_steps=solver_cfg.max_steps,
            adjoint=adjoint,
        )
        return {"phi": solution.value}
    
