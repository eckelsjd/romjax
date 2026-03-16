from typing import Literal, Mapping, TypedDict

import jax.numpy as jnp
from jax.typing import ArrayLike
from pydantic import Field, ValidationInfo, field_validator

from romtools.model import Model
from romtools.solvers.utils import UniformGrid, boundary_pass_through, homogeneous_boundary
from romtools.typing import BoundaryCallable, Coordinates, DictModel, ForcingCallable, PyTree
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
    
    :ivar grid: the uniform 2D Cartesian grid
    """
    grid: UniformGrid

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
        pass
    
