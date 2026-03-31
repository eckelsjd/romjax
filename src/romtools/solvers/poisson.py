from typing import Literal, Mapping, TypedDict, Any

import jax.numpy as jnp
import optimistix as optx
from jax.typing import ArrayLike
from pydantic import (
    Field,
    PositiveInt, 
    ValidationInfo, 
    field_validator,
)

from romtools.model import Model
from romtools.solvers.utils import (
    boundary_pass_through,
    homogeneous_boundary,
    UniformGrid,
    BoundarySpec,
    BoundaryType
)
from romtools.typing import (
    BoundaryCallable, 
    Coordinates, 
    DictModel, 
    ForcingCallable, 
    InitialCallable,
    PyTree,
    IterativeSolver,
    AdjointMethod
)
from romtools.utils import merge_pytrees, to_pytree


type ForcingName = Literal["gaussian", "nonlinear", "sinusoid", "constant"]


class SinusoidForcingInputs(DictModel):
    """Method of manufactured solutions sinusoid forcing."""
    coords: Coordinates = (0., 0.)


class ConstantForcingInputs(DictModel):
    """Constant forcing."""
    const: ArrayLike = 0.0


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
    forcing: DictModel = Field(default_factory=lambda: ConstantForcingInputs(const=0.0))
    conductivity: DictModel = Field(default_factory=lambda: ConstantForcingInputs(const=1.0))
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


def zero_initial_guess(coords: Coordinates) -> ArrayLike:
    return jnp.zeros_like(coords[0])


class PoissonConfig(DictModel):
    """Numerical configs for solving the Poisson PDE on a 2D grid.

    See Optimistix docs for `root_find()` method.
    
    :ivar grid: the uniform 2D Cartesian grid
    :ivar solver: Optimistix nonlinear root finding solver (name+opts or instance), default is Newton
    :ivar initial_guess: array matching the shape of the grid, or a callable that takes coords and generates the array
    :ivar options: runtime options for the nonlinear solver
    :ivar max_steps: maximum number of solver steps
    :ivar adjoint: Optimistix adjoint method
    :ivar throw: whether to throw failures as errors (default True)
    """
    grid: UniformGrid
    solver: IterativeSolver = Field(default_factory=lambda: dict(name='Newton', opts={'rtol': 1e-2, 'atol': 1e-4}),
                                    validate_default=True)
    initial_guess: ArrayLike = Field(default_factory=lambda: zero_initial_guess, exclude=True, validate_default=True)
    options: dict[str, Any] = Field(default_factory=dict)
    max_steps: PositiveInt = 100
    adjoint: AdjointMethod = Field(default_factory=lambda: dict(name='ImplicitAdjoint'), validate_default=True)
    throw: bool = True

    @field_validator("grid", mode="after")
    @classmethod
    def _check_2d_grid(cls, value: UniformGrid) -> UniformGrid:
        if len(value.shape) != 2:
            raise ValueError("Only 2D grid supported for Poisson")
        
        return value
    
    @field_validator("initial_guess", mode="before")
    @classmethod
    def _coerce_initial_guess(cls, value: ArrayLike | InitialCallable, info: ValidationInfo) -> ArrayLike:
        if callable(value):
            return value(info.data["grid"].coords)

        return value


def gaussian_forcing(inputs: GaussianForcingInputs, outputs: PoissonOutputs) -> ArrayLike:
    r"""Symmetric Gaussian bump.

        $f(x,y) = A_0 \exp(-1/(2\sigma) ((x-\mu_x)^2 + (y-\mu_y)^2))$
    
    :param inputs: the input parameters
    :param outputs: the scalar potential on the grid (not used)
    :return: the forcing on the grid
    """
    dx = inputs['coords'][0] - inputs['mu_x']
    dy = inputs['coords'][1] - inputs['mu_y']
    return inputs['A0'] * jnp.exp(-(dx * dx + dy * dy) / (2 * inputs['sigma']))


def constant_forcing(inputs: ConstantForcingInputs, outputs: PoissonOutputs) -> ArrayLike:
    """Just a constant forcing (inputs/outputs not used)."""
    return inputs['const']


def sinusoid_forcing(inputs: SinusoidForcingInputs, outputs: PoissonOutputs) -> ArrayLike:
    r"""Sinusoid forcing. Used in method of manufactured solutions.
    
        $f(x,y) = -2 \pi^2 \sin{\pi x}\sin{\pi y}$

    :param inputs: just uses (x,y) coords
    :param outputs: the scalar potential on grid (not used)
    :return: the forcing on the grid.
    """
    return -2 * jnp.pi**2 * jnp.sin(jnp.pi * inputs['coords'][0]) * jnp.sin(jnp.pi * inputs['coords'][1])


def nonlinear_conductivity(inputs: NonlinearConductivityInputs, outputs: PoissonOutputs) -> ArrayLike:
    r"""Nonlinear conductivity.

        $k(x,y) = k_0(1 + \alpha \phi^2)$
    
    :param inputs: the input parameters
    :param outputs: the scalar potential on the grid
    :return: the conductivity on the grid
    """
    phi = outputs['phi']
    return inputs['k0'] * (1 + inputs['alpha'] * (phi * phi))


class Poisson2D(Model):

    # Required
    config: PoissonConfig

    # Optional/default
    forcing: ForcingCallable = constant_forcing
    conductivity: ForcingCallable = constant_forcing
    boundary: BoundaryCallable = boundary_pass_through

    forcing_defaults: DictModel = Field(
        default_factory=lambda: ConstantForcingInputs(const=0.0),
        description="Default inputs for the forcing function (any PyTree).",
    )
    conductivity_defaults: DictModel = Field(
        default_factory=lambda: ConstantForcingInputs(const=1.0),
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
                "constant": constant_forcing,
                "sinusoid": sinusoid_forcing,
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
            if info.data['forcing'] is constant_forcing:
                return ConstantForcingInputs(**value)
            if info.data['forcing'] is sinusoid_forcing:
                return SinusoidForcingInputs(**value)
        
        if info.field_name == 'conductivity_defaults':
            if info.data['conductivity'] is nonlinear_conductivity:
                return NonlinearConductivityInputs(**value)
            if info.data['conductivity'] is constant_forcing:
                return ConstantForcingInputs(**value)
            
        return value
    
    def _merge_inputs(self, inputs: PoissonInputs) -> PoissonInputs:
        """Merge incoming inputs with default values and coords. Also converts all to pytrees for jax.
        Arrays are just moved around by reference, so computational graph is not broken.
        """
        def _merge_defaults(defaults: DictModel, *overrides: PyTree | None) -> dict:
            merged = to_pytree(defaults)
            for override in overrides:
                if override is None:
                    continue
                merged = merge_pytrees(merged, to_pytree(override))
            return merged
        
        coords = {'coords': self.config['grid']['coords']}
        forcing_inputs = _merge_defaults(self.forcing_defaults, coords, inputs.get("forcing"))
        conductivity_inputs = _merge_defaults(self.conductivity_defaults, coords, inputs.get("conductivity"))
        boundary_inputs = self.boundary(_merge_defaults(self.boundary_defaults, coords, inputs.get("boundary")))
        # boundary is assumed constant, so compute once up front (if applicable)

        return {'forcing': forcing_inputs, 'conductivity': conductivity_inputs, 'boundary': boundary_inputs}
    
    def _compute_residual(self, inputs: PoissonInputs, outputs: PoissonOutputs) -> PoissonResiduals:
        """Helper to compute the finite volume residual on the grid. Used for forward and backward directions."""
        phi = jnp.asarray(outputs["phi"])
        dx, dy = self.config['grid']['spacing']
        forcing = jnp.asarray(self.forcing(inputs['forcing'], outputs))
        conductivity = jnp.broadcast_to(self.conductivity(inputs['conductivity'], outputs), phi.shape)
        xbds, ybds = inputs['boundary']['boundary']  # constant

        def _ghost_for_side(
            spec: BoundarySpec,
            interior: ArrayLike,
            opposite: ArrayLike,
            interior_k: ArrayLike,
            opposite_k: ArrayLike,
            spacing: ArrayLike,
        ) -> tuple[ArrayLike, ArrayLike]:
            """Helper to get correct ghost cell values depending on BC."""
            b_type = spec["type"]
            if b_type == BoundaryType.periodic:
                return opposite, opposite_k
            if b_type == BoundaryType.dirichlet:
                value = jnp.asarray(spec["value"])
                return 2.0 * value - interior, interior_k
            if b_type == BoundaryType.neumann:
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

        return {"phi_residual": phi_residual}

    def evaluate(self, inputs: PoissonInputs, outputs: PoissonOutputs) -> PoissonResiduals:
        """Evalute the Poisson residual on a 2D grid.
        
        :param inputs: params for forcing, conductivity, and boundary conditions
        :param outputs: the scalar potential on a 2D grid
        :return: the scalar residual on the 2D grid
        """
        return self._compute_residual(self._merge_inputs(inputs), outputs)

    def solve(
            self, 
            inputs: PoissonInputs | None = None, 
            residuals: PoissonResiduals | None = None,
            return_sol: bool = False
        ) -> PoissonOutputs | optx.Solution:
        """Solve the Poisson equation for a target residual.
        
        :param inputs: params for forcing, conductivity, and boundary conditions (use defaults if None)
        :param residuals: the target scalar residual on the 2D grid (defaults to zeros with same shape as grid)
        :param return_sol: return the full Solution object (default False)
        :return: the scalar potential solution on the 2D grid
        """
        inputs = {} if inputs is None else inputs
        residuals = {} if residuals is None else residuals
        if "phi_residual" in residuals:
            target = jnp.asarray(residuals["phi_residual"])
        else:
            target = jnp.zeros_like(self.config.grid.coords[0])
        merged_inputs = self._merge_inputs(inputs)
        args = {'inputs': merged_inputs, 'target': target}

        def residual_fn(phi: ArrayLike, args: PyTree) -> ArrayLike:
            residual = self._compute_residual(args['inputs'], {'phi': phi})
            return residual['phi_residual'] - args['target']

        solution = optx.root_find(
            residual_fn,
            solver=self.config.solver,
            y0=jnp.asarray(self.config.initial_guess),
            args=args,
            options=self.config.options,
            max_steps=self.config.max_steps,
            adjoint=self.config.adjoint,
            throw=self.config.throw,
        )

        ret = solution if return_sol else {"phi": solution.value} 
        return ret
    