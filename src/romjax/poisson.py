"""Example 2D Poisson solver."""
import functools
from collections.abc import Mapping
from typing import Annotated, Callable, Literal, TypedDict

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import ArrayLike, Key, PyTree
from pydantic import BeforeValidator, ConfigDict, Field, field_validator

from romjax.graph import Node
from romjax.model import ImplicitModel, ImplicitSampleable
from romjax.pde import (
    BoundarySpec,
    BoundaryType,
    ConstantInitialize,
    Coordinates,
    ForcingCallable,
    IterativeSolver,
    RegisteredInitialize,
    UniformGrid,
    homogeneous_boundary,
)
from romjax.rng import SamplerCallable
from romjax.tree import to_pytree
from romjax.typing import DictModel, from_registry

__all__ = ["Poisson2D"]


class PoissonInputs(TypedDict):
    """Inputs for the Poisson equation.
    
    :ivar forcing: forcing inputs
    :ivar conductivity: conductivity inputs
    :ivar boundary: boundary condition parameters
    """
    forcing: dict
    conductivity: dict
    boundary: dict


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


class GaussianForcing(ForcingCallable):
    class Inputs(DictModel):
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
    
    def callable(inputs: Inputs, outputs: PoissonOutputs) -> ArrayLike:
        r"""Symmetric Gaussian bump.

            $f(x,y) = A_0 \exp(-1/(2\sigma) ((x-\mu_x)^2 + (y-\mu_y)^2))$
        
        :param inputs: the input parameters
        :param outputs: the scalar potential on the grid (not used)
        :return: the forcing on the grid
        """
        dx = inputs['coords'][0] - inputs['mu_x']
        dy = inputs['coords'][1] - inputs['mu_y']
        return inputs['A0'] * jnp.exp(-(dx * dx + dy * dy) / (2 * inputs['sigma']))


class NonlinearConductivity(ForcingCallable):

    amplitude: Callable[[ArrayLike], ArrayLike] | Literal["exp"] | None = None

    class Inputs(DictModel):
        """Inputs for nonlinear conductivity function.
    
        :ivar k0: the background random field conductivity (2D)
        :ivar alpha: the strength of the nonlinearity
        """
        k0: ArrayLike = 1.0
        alpha: ArrayLike = 1.0
    
    @field_validator("amplitude", mode="before")
    @classmethod
    def _validate_amplitude(cls, amplitude):
        if amplitude is None:
            amplitude = lambda x: x
        if amplitude == 'exp':
            amplitude = jnp.exp
        
        return amplitude
    
    def callable(self, inputs: Inputs, outputs: PoissonOutputs) -> ArrayLike:
        r"""Nonlinear conductivity.

            $k(x,y) = amp(k_0) * (1 + \alpha \phi^2)$
        
        :param inputs: the input parameters
        :param outputs: the scalar potential on the grid
        :param amplitude: function to apply to k0 
        :return: the conductivity on the grid
        """
        phi = next(iter(outputs.values()))
        return self.amplitude(inputs['k0']) * (1 + inputs['alpha'] * (phi * phi))


class ConstantForcing(ForcingCallable):

    class Inputs(DictModel):
        const: ArrayLike = 0.0
    
    def callable(self, inputs: Inputs, outputs: PoissonOutputs) -> ArrayLike:
        """Just a constant forcing (inputs/outputs not used)."""
        return inputs['const']


class SinusoidForcing(ForcingCallable):

    class Inputs(DictModel):
        coords: Coordinates = (0., 0.)
    
    def callable(self, inputs: Inputs, outputs: PoissonOutputs) -> ArrayLike:
        r"""Sinusoid forcing. Used in method of manufactured solutions.
    
            $f(x,y) = -2 \pi^2 \sin{\pi x}\sin{\pi y}$

        :param inputs: just uses (x,y) coords
        :param outputs: the scalar potential on grid (not used)
        :return: the forcing on the grid.
        """
        return -2 * jnp.pi**2 * jnp.sin(jnp.pi * inputs['coords'][0]) * jnp.sin(jnp.pi * inputs['coords'][1])


class IdentityInputs(ForcingCallable):

    def callable(self, inputs, outputs):
        """Simple boundary that uses boundary input params directly (just pass them through)."""
        return inputs
    

_forcing_registry = {
    "gaussian": GaussianForcing,
    "nonlinear": NonlinearConductivity,
    "sinusoid": SinusoidForcing,
    "constant": ConstantForcing,
    "identity": IdentityInputs
}


type PoissonForcing = Annotated[ForcingCallable, BeforeValidator(functools.partial(from_registry, _forcing_registry))]


class Poisson2D(ImplicitModel, ImplicitSampleable):
    model_config = ConfigDict(extra='forbid')

    grid: UniformGrid  # Required

    solver: IterativeSolver = Field(default_factory=IterativeSolver)
    initial_guess: RegisteredInitialize = Field(default_factory=ConstantInitialize)

    # To satisfy criteria for being a graph edge
    source: Node = Node(name="poisson_in")
    target: Node = Node(name="poisson_out")

    field_name: str = "phi"
    residual_name: str = "phi_residual"

    forcing: PoissonForcing = Field(default_factory=ConstantForcing)
    conductivity: PoissonForcing = Field(default_factory=lambda: ConstantForcing(inputs_default=dict(const=1.0)))
    boundary: PoissonForcing = Field(
        default_factory=lambda: IdentityInputs(inputs_default=homogeneous_boundary(ndim=2))
    )

    inputs_sampler: SamplerCallable | None = None
    outputs_sampler: SamplerCallable | None = None

    @field_validator("grid", mode="after")
    @classmethod
    def _check_2d_grid(cls, value: UniformGrid) -> UniformGrid:
        if len(value.shape) != 2:
            raise ValueError("Only 2D grid supported for Poisson")
        
        return value

    def _jax_coords(self) -> tuple[jax.Array, ...]:
        """Return grid coordinates as JAX arrays for numerical routines."""
        return tuple(jnp.asarray(coord) for coord in self.grid.coords)
    
    def _merge_coords(self, inputs: PoissonInputs, coords: Coordinates | None = None) -> PoissonInputs:
        """Merge grid coords into incoming inputs."""
        inputs = to_pytree(inputs)
        for name in ("forcing", "conductivity", "boundary"):
            inputs.setdefault(name, {})
        coords = {"coords": self._jax_coords() if coords is None else coords}
        for k in inputs:
            inputs[k].update(coords)
        
        return inputs
    
    def _compute_residual(self, inputs: PoissonInputs, outputs: PoissonOutputs) -> PoissonResiduals:
        """Helper to compute the finite volume residual on the grid. Used for forward and backward directions."""
        phi = jnp.asarray(outputs[self.field_name])
        dx, dy = self.grid['spacing']
        forcing = jnp.asarray(self.forcing(inputs['forcing'], outputs))
        conductivity = jnp.broadcast_to(self.conductivity(inputs['conductivity'], outputs), phi.shape)
        xbds, ybds = self.boundary(inputs['boundary'], outputs)['boundary']

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

        return {self.residual_name: phi_residual}

    def evaluate(self, inputs: PoissonInputs, outputs: PoissonOutputs) -> PoissonResiduals:
        """Evalute the Poisson residual on a 2D grid.
        
        :param inputs: params for forcing, conductivity, and boundary conditions
        :param outputs: the scalar potential on a 2D grid
        :return: the scalar residual on the 2D grid
        """
        coords = self._jax_coords()
        return self._compute_residual(self._merge_coords(inputs, coords), outputs)

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
        grid_coords = self._jax_coords()
        if self.residual_name in residuals:
            target = jnp.asarray(residuals[self.residual_name])
        else:
            target = jnp.zeros_like(grid_coords[0])
        args = {'inputs': self._merge_coords(inputs, grid_coords), 'target': target}

        def residual_fn(phi: ArrayLike, args: PyTree) -> ArrayLike:
            residual = self._compute_residual(args['inputs'], {self.field_name: phi})
            return residual[self.residual_name] - args['target']
        
        y0 = jnp.asarray(self.initial_guess(grid_coords))
        solution = self.solver.root_find(residual_fn, y0, args, return_sol=return_sol)

        ret = solution if return_sol else {self.field_name: solution} 
        return ret
    
    def sample_inputs(self, key: Key) -> PoissonInputs:
        """Produce one sample of inputs for the given key."""
        if self.inputs_sampler is not None:
            return self.inputs_sampler(key)
        return {}
    
    def sample_outputs(
        self, 
        key: Key, 
        inputs: PoissonInputs | None = None, 
        solution: PoissonOutputs | None = None
    ) -> PoissonOutputs:
        """
        Produce one sample of outputs for the given key.
        
        :param key: the random key
        :param inputs: optionally condition on inputs
        :param solution: for efficiency, optionally condition on the precomputed solution of solve(inputs)=0
        :return: the outputs sample
        """
        if self.outputs_sampler is None:
            return {}

        if solution is None:
            solution = self.solve(inputs)

        sample = self.outputs_sampler(key, inputs=inputs, solution=solution)
        if isinstance(sample, Mapping):
            return {self.field_name: jnp.asarray(sample[self.field_name])}
        return {self.field_name: jnp.asarray(sample)}
    
    def resolve_dof(self) -> int:
        return self.grid.coords[0].shape[0] * self.grid.coords[0].shape[1]  # Nx x Ny
