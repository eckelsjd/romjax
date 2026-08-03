"""Finite-volume 2D advection-diffusion solver:  $nabla dot (phi v) - nabla dot (k nabla phi) = f(x,y)."""
import functools
from collections.abc import Mapping
from typing import Annotated, Any, Callable, Literal, TypedDict

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import ArrayLike, Key, PyTree
from pydantic import BeforeValidator, ConfigDict, Field, field_validator

from romjax.graph import Node
from romjax.model import ImplicitModel, ImplicitSampleable
from romjax.pde import (
    FORCING_REGISTRY,
    BoundarySpec,
    BoundaryType,
    ConstantForcing,
    Coordinates,
    ForcingCallable,
    IdentityInputs,
    IterativeSolver,
    UniformGrid,
    homogeneous_boundary,
)
from romjax.rng import SamplerCallable
from romjax.tree import to_pytree
from romjax.typing import DictModel, from_registry

__all__ = [
    "AdvectionDiffusion2D",
]


class AdvectionDiffusionInputs(TypedDict, total=False):
    """Inputs for the two-dimensional advection-diffusion equation.
    
    :ivar forcing: forcing inputs
    :ivar diffusion: diffusion inputs
    :ivar velocity: velocity-field inputs
    :ivar boundary: boundary condition parameters
    :ivar initial: initial-field parameters or a direct ``phi`` field
    """
    forcing: dict
    diffusion: dict
    velocity: dict
    boundary: dict
    initial: dict[str, Any]


class AdvectionDiffusionOutputs(TypedDict):
    """Outputs for the advection-diffusion equation.
    
    :ivar phi: the scalar potential on the grid
    """
    phi: ArrayLike


class AdvectionDiffusionResiduals(TypedDict):
    """Outputs of the advection-diffusion residual.
    
    :ivar phi_residual: the residual of the scalar potential on the grid
    """
    phi_residual: ArrayLike


class CubicForcing(ForcingCallable):
    """State-dependent cubic forcing for reaction-diffusion problems."""

    normalize: bool = False

    class Inputs(DictModel):
        """Inputs for the cubic forcing function.

        :ivar q: constant or spatial volumetric source term
        :ivar alpha: linear state coefficient
        :ivar beta: quadratic state coefficient
        :ivar gamma: cubic state coefficient
        """

        q: ArrayLike = 0.0
        alpha: ArrayLike = -1.0
        beta: ArrayLike = 0.0
        gamma: ArrayLike = -1.0

    def callable(self, inputs: Inputs, outputs: AdvectionDiffusionOutputs) -> ArrayLike:
        r"""Evaluate a broadcastable cubic state-dependent forcing.

        .. math::

            f(\phi) = q + \alpha\phi + \beta\phi^2 + \gamma\phi^3.

        :param inputs: volumetric and reaction coefficients
        :param outputs: scalar potential containing ``phi``
        :return: forcing field broadcastable to ``phi``; when ``normalize`` is true,
            the ``q`` field is scaled to unit RMS before the reaction terms are added
        """
        phi = jnp.asarray(outputs["phi"])
        q = jnp.asarray(inputs["q"])
        if self.normalize:
            q_rms = jnp.sqrt(jnp.mean(jnp.square(q)))
            q = jnp.where(q_rms > 0.0, q / q_rms, q)
        return (
            q
            + jnp.asarray(inputs["alpha"]) * phi
            + jnp.asarray(inputs["beta"]) * phi**2
            + jnp.asarray(inputs["gamma"]) * phi**3
        )


class QuadraticDiffusion(ForcingCallable):

    amplitude: Callable[[ArrayLike], ArrayLike] | Literal["exp"] | None = None

    class Inputs(DictModel):
        """Inputs for quadratic diffusion function.
    
        :ivar k0: the background diffusion field (2D)
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
    
    def callable(self, inputs: Inputs, outputs: AdvectionDiffusionOutputs) -> ArrayLike:
        r"""Evaluate quadratic state-dependent diffusion.

            $D(x,y) = amp(k_0) * (1 + \alpha \phi^2)$
        
        :param inputs: the input parameters
        :param outputs: the scalar potential on the grid
        :param amplitude: function to apply to k0 
        :return: the diffusion field on the grid
        """
        phi = next(iter(outputs.values()))
        return self.amplitude(inputs['k0']) * (1 + inputs['alpha'] * (phi * phi))


class ConstantCoordinateVelocity(ForcingCallable):
    """Construct a 2D velocity where vx = ax + by and vy = cx + dy, i.e. constant in either coordinate x or y."""

    class Inputs(DictModel):

        const_vx: ArrayLike | list[Any] | tuple[Any, ...] = (0.0, 0.0)
        const_vy: ArrayLike | list[Any] | tuple[Any, ...] = (0.0, 0.0)
        coords: Coordinates = (0.0, 0.0)

    def callable(self, inputs: Inputs, outputs: AdvectionDiffusionOutputs) -> jax.Array:
        del outputs
        x, y = (jnp.asarray(coord) for coord in inputs['coords'])
        const_vx = jnp.asarray(inputs['const_vx'])
        const_vy = jnp.asarray(inputs['const_vy'])

        vx = const_vx[0] * x + const_vx[1] * y
        vy = const_vy[0] * x + const_vy[1] * y
        return jnp.stack((vx, vy), axis=0)


class PotentialVelocity(ForcingCallable):
    r"""Construct a 2D divergence-free velocity from a streamfunction potential field.

    The returned components use the streamfunction convention

    .. math::

        v_x = \partial_y \psi, \qquad v_y = -\partial_x \psi.

    :ivar psi: scalar streamfunction field on the solver grid
    """

    normalize: bool = False

    class Inputs(DictModel):
        """Inputs for the streamfunction velocity forcing."""

        psi: ArrayLike | list[Any] | tuple[Any, ...] = 0.0
        coords: Coordinates = (0.0, 0.0)

    @staticmethod
    def _differentiate(field: jax.Array, spacing: ArrayLike, axis: int) -> jax.Array:
        """Differentiate a field with centered interior and one-sided edge stencils."""
        if field.shape[axis] < 2:
            raise ValueError("Streamfunction field must have at least two cells per dimension.")

        if axis == 0:
            lower = (field[1:2, :] - field[0:1, :]) / spacing
            interior = (field[2:, :] - field[:-2, :]) / (2.0 * spacing)
            upper = (field[-1:, :] - field[-2:-1, :]) / spacing
            return jnp.concatenate((lower, interior, upper), axis=0)

        lower = (field[:, 1:2] - field[:, 0:1]) / spacing
        interior = (field[:, 2:] - field[:, :-2]) / (2.0 * spacing)
        upper = (field[:, -1:] - field[:, -2:-1]) / spacing
        return jnp.concatenate((lower, interior, upper), axis=1)

    @staticmethod
    def _grid_spacing(coords: Coordinates) -> tuple[ArrayLike, ArrayLike]:
        """Extract two-dimensional grid spacing from one- or two-dimensional coordinates."""
        x, y = (jnp.asarray(coord) for coord in coords)
        if x.ndim == 1 and y.ndim == 1:
            return x[1] - x[0], y[1] - y[0]
        if x.ndim == 2 and y.ndim == 2:
            return x[1, 0] - x[0, 0], y[0, 1] - y[0, 0]
        raise ValueError("Streamfunction coordinates must both be one-dimensional or both be two-dimensional.")

    def callable(self, inputs: Inputs, outputs: AdvectionDiffusionOutputs) -> jax.Array:
        """Evaluate the components-first velocity field generated by ``psi``.

        :param inputs: streamfunction field and solver coordinates
        :param outputs: current transport outputs, unused
        :return: velocity with shape ``(2, nx, ny)``
        """
        del outputs
        psi = jnp.asarray(inputs["psi"])
        if psi.ndim != 2:
            raise ValueError("Streamfunction psi must be a two-dimensional scalar field.")

        coords = inputs["coords"]
        dx, dy = self._grid_spacing(coords)
        x, y = (jnp.asarray(coord) for coord in coords)
        expected_shape = (x.shape[0], y.shape[0]) if x.ndim == 1 else x.shape
        if psi.shape != expected_shape:
            raise ValueError(f"Streamfunction psi shape {psi.shape} does not match grid shape {expected_shape}.")

        dpsi_dx = self._differentiate(psi, dx, axis=0)
        dpsi_dy = self._differentiate(psi, dy, axis=1)
        velocity = jnp.stack((dpsi_dy, -dpsi_dx), axis=0)
        if self.normalize:
            speed_rms = jnp.sqrt(jnp.mean(jnp.sum(jnp.square(velocity), axis=0)))
            velocity = jnp.where(speed_rms > 0.0, velocity / speed_rms, velocity)
        return velocity


_forcing_registry = {
    **FORCING_REGISTRY,
    "cubic": CubicForcing,
    "quadratic": QuadraticDiffusion,
    "potential": PotentialVelocity,
    "const_coord": ConstantCoordinateVelocity,
}


type AdvectionDiffusionForcing = Annotated[
    ForcingCallable, BeforeValidator(functools.partial(from_registry, _forcing_registry))
]


class AdvectionDiffusion2D(ImplicitModel, ImplicitSampleable):
    model_config = ConfigDict(extra='forbid')

    grid: UniformGrid  # Required

    solver: IterativeSolver = Field(default_factory=IterativeSolver)

    # To satisfy criteria for being a graph edge
    source: Node = Node(name="advection_diffusion_in")
    target: Node = Node(name="advection_diffusion_out")

    field_name: str = "phi"
    residual_name: str = "phi_residual"

    forcing: AdvectionDiffusionForcing = Field(default_factory=ConstantForcing)
    diffusion: AdvectionDiffusionForcing = Field(
        default_factory=lambda: ConstantForcing(inputs_default=dict(const=1.0))
    )
    velocity: AdvectionDiffusionForcing = Field(
        default_factory=lambda: ConstantForcing(inputs_default=dict(const=(0.0, 0.0)))
    )
    boundary: AdvectionDiffusionForcing = Field(
        default_factory=lambda: IdentityInputs(inputs_default=homogeneous_boundary(ndim=2))
    )
    initial: AdvectionDiffusionForcing = Field(default_factory=ConstantForcing)

    incompressible: bool = False

    inputs_sampler: SamplerCallable | None = None
    outputs_sampler: SamplerCallable | None = None

    @field_validator("grid", mode="after")
    @classmethod
    def _check_2d_grid(cls, value: UniformGrid) -> UniformGrid:
        if len(value.shape) != 2:
            raise ValueError("Only 2D grid supported for advection-diffusion")
        
        return value

    def _jax_coords(self) -> tuple[jax.Array, ...]:
        """Return grid coordinates as JAX arrays for numerical routines."""
        return tuple(jnp.asarray(coord) for coord in self.grid.coords)
    
    def _merge_coords(
        self, inputs: AdvectionDiffusionInputs, coords: Coordinates | None = None
    ) -> AdvectionDiffusionInputs:
        """Merge grid coords into incoming inputs."""
        inputs = to_pytree(inputs)
        for name in ("forcing", "diffusion", "velocity", "boundary", "initial"):
            inputs.setdefault(name, {})
        coords = {"coords": self._jax_coords() if coords is None else coords}
        for k in inputs:
            inputs[k].update(coords)
        
        return inputs

    def _initial_field(self, inputs: AdvectionDiffusionInputs) -> jax.Array:
        """Resolve the initial field from runtime inputs or the configured callable.

        :param inputs: resolved advection-diffusion inputs, including grid coordinates
        :return: initial scalar field on the grid
        """
        initial_inputs = inputs.get("initial", {})
        if self.field_name in initial_inputs:
            initial = initial_inputs[self.field_name]
        else:
            initial = self.initial(initial_inputs, {})
        return jnp.broadcast_to(jnp.asarray(initial), inputs["initial"]["coords"][0].shape)
    
    def _compute_residual(
        self, inputs: AdvectionDiffusionInputs, outputs: AdvectionDiffusionOutputs
    ) -> AdvectionDiffusionResiduals:
        """Compute the finite-volume residual on the grid."""
        phi = jnp.asarray(outputs[self.field_name])
        dx, dy = self.grid["spacing"]
        forcing = jnp.asarray(self.forcing(inputs["forcing"], outputs))
        diffusion = jnp.broadcast_to(self.diffusion(inputs["diffusion"], outputs), phi.shape)
        velocity = self._velocity_field(inputs["velocity"], outputs, phi.shape)
        xbds, ybds = self.boundary(inputs["boundary"], outputs)["boundary"]

        def _ghost_for_side(
            spec: BoundarySpec,
            interior: ArrayLike,
            opposite: ArrayLike,
            interior_diffusion: ArrayLike,
            opposite_diffusion: ArrayLike,
            spacing: ArrayLike,
        ) -> tuple[ArrayLike, ArrayLike]:
            """Return scalar and diffusion ghost-cell values for one boundary."""
            b_type = spec["type"]
            if isinstance(b_type, str):
                b_type = BoundaryType[b_type]
            if b_type == BoundaryType.periodic:
                return opposite, opposite_diffusion
            if b_type == BoundaryType.dirichlet:
                value = jnp.asarray(spec["value"])
                return 2.0 * value - interior, interior_diffusion
            if b_type == BoundaryType.neumann:
                value = jnp.asarray(spec["value"])
                return interior + spacing * value, interior_diffusion
            raise ValueError(f"Unsupported boundary type: {b_type!r}")

        def _velocity_ghost_for_side(
            spec: BoundarySpec, interior: jax.Array, opposite: jax.Array
        ) -> jax.Array:
            """Return velocity ghost values using periodic or zero-gradient extension."""
            b_type = spec["type"]
            if isinstance(b_type, str):
                b_type = BoundaryType[b_type]
            if b_type == BoundaryType.periodic:
                return opposite
            if b_type in (BoundaryType.dirichlet, BoundaryType.neumann):
                return interior
            raise ValueError(f"Unsupported boundary type: {b_type!r}")

        phi_s, diffusion_s = _ghost_for_side(
            xbds[0], phi[0, :], phi[-1, :], diffusion[0, :], diffusion[-1, :], dx
        )
        phi_n, diffusion_n = _ghost_for_side(
            xbds[1], phi[-1, :], phi[0, :], diffusion[-1, :], diffusion[0, :], dx
        )
        phi_w, diffusion_w = _ghost_for_side(
            ybds[0], phi[:, 0], phi[:, -1], diffusion[:, 0], diffusion[:, -1], dy
        )
        phi_e, diffusion_e = _ghost_for_side(
            ybds[1], phi[:, -1], phi[:, 0], diffusion[:, -1], diffusion[:, 0], dy
        )

        velocity_s = _velocity_ghost_for_side(xbds[0], velocity[:, 0, :], velocity[:, -1, :])
        velocity_n = _velocity_ghost_for_side(xbds[1], velocity[:, -1, :], velocity[:, 0, :])
        velocity_w = _velocity_ghost_for_side(ybds[0], velocity[:, :, 0], velocity[:, :, -1])
        velocity_e = _velocity_ghost_for_side(ybds[1], velocity[:, :, -1], velocity[:, :, 0])

        phi_west = jnp.concatenate([phi_w[:, None], phi[:, :-1]], axis=1)
        phi_east = jnp.concatenate([phi[:, 1:], phi_e[:, None]], axis=1)
        phi_south = jnp.concatenate([phi_s[None, :], phi[:-1, :]], axis=0)
        phi_north = jnp.concatenate([phi[1:, :], phi_n[None, :]], axis=0)

        diffusion_west = jnp.concatenate([diffusion_w[:, None], diffusion[:, :-1]], axis=1)
        diffusion_east = jnp.concatenate([diffusion[:, 1:], diffusion_e[:, None]], axis=1)
        diffusion_south = jnp.concatenate([diffusion_s[None, :], diffusion[:-1, :]], axis=0)
        diffusion_north = jnp.concatenate([diffusion[1:, :], diffusion_n[None, :]], axis=0)

        velocity_west = jnp.concatenate([velocity_w[:, :, None], velocity[:, :, :-1]], axis=2)
        velocity_east = jnp.concatenate([velocity[:, :, 1:], velocity_e[:, :, None]], axis=2)
        velocity_south = jnp.concatenate([velocity_s[:, None, :], velocity[:, :-1, :]], axis=1)
        velocity_north = jnp.concatenate([velocity[:, 1:, :], velocity_n[:, None, :]], axis=1)

        diffusion_face_w = 0.5 * (diffusion + diffusion_west)
        diffusion_face_e = 0.5 * (diffusion + diffusion_east)
        diffusion_face_s = 0.5 * (diffusion + diffusion_south)
        diffusion_face_n = 0.5 * (diffusion + diffusion_north)

        diffusion_flux_e = diffusion_face_e * (phi_east - phi) / dy
        diffusion_flux_w = diffusion_face_w * (phi - phi_west) / dy
        diffusion_flux_n = diffusion_face_n * (phi_north - phi) / dx
        diffusion_flux_s = diffusion_face_s * (phi - phi_south) / dx
        diffusion_divergence = (diffusion_flux_e - diffusion_flux_w) / dy + (
            diffusion_flux_n - diffusion_flux_s
        ) / dx

        if self.incompressible:
            advection_divergence = velocity[0] * (phi_north - phi_south) / (2.0 * dx) + velocity[1] * (
                phi_east - phi_west
            ) / (2.0 * dy)
        else:
            phi_face_e = 0.5 * (phi + phi_east)
            phi_face_w = 0.5 * (phi + phi_west)
            phi_face_n = 0.5 * (phi + phi_north)
            phi_face_s = 0.5 * (phi + phi_south)
            velocity_face_e = 0.5 * (velocity + velocity_east)
            velocity_face_w = 0.5 * (velocity + velocity_west)
            velocity_face_n = 0.5 * (velocity + velocity_north)
            velocity_face_s = 0.5 * (velocity + velocity_south)
            advection_flux_e = phi_face_e * velocity_face_e[1]
            advection_flux_w = phi_face_w * velocity_face_w[1]
            advection_flux_n = phi_face_n * velocity_face_n[0]
            advection_flux_s = phi_face_s * velocity_face_s[0]
            advection_divergence = (advection_flux_e - advection_flux_w) / dy + (
                advection_flux_n - advection_flux_s
            ) / dx

        phi_residual = advection_divergence - diffusion_divergence - forcing
        return {self.residual_name: phi_residual}

    def _velocity_field(
        self, inputs: dict, outputs: AdvectionDiffusionOutputs, field_shape: tuple[int, int]
    ) -> jax.Array:
        """Validate and broadcast a components-first velocity field."""
        velocity = jnp.asarray(self.velocity(inputs, outputs))
        if velocity.ndim == 0 or velocity.shape[0] != 2:
            raise ValueError(
                "Velocity forcing must return two components with shape (2,) or (2, nx, ny)."
            )
        if velocity.ndim == 1:
            velocity = velocity[:, None, None]
        try:
            return jnp.broadcast_to(velocity, (2, *field_shape))
        except ValueError as exc:
            expected_shape = f"(2, {field_shape[0]}, {field_shape[1]})"
            raise ValueError(
                f"Velocity forcing shape {velocity.shape} is not broadcastable to {expected_shape}."
            ) from exc

    def evaluate(
        self, inputs: AdvectionDiffusionInputs, outputs: AdvectionDiffusionOutputs
    ) -> AdvectionDiffusionResiduals:
        """Evaluate the advection-diffusion residual on a 2D grid.
        
        :param inputs: parameters for forcing, diffusion, velocity, and boundary conditions
        :param outputs: the scalar field on a 2D grid
        :return: the scalar residual on the 2D grid
        """
        coords = self._jax_coords()
        return self._compute_residual(self._merge_coords(inputs, coords), outputs)

    def solve(
        self, 
        inputs: AdvectionDiffusionInputs | None = None,
        residuals: AdvectionDiffusionResiduals | None = None,
        return_sol: bool = False
    ) -> AdvectionDiffusionOutputs | optx.Solution:
        """Solve the advection-diffusion equation for a target residual.
        
        :param inputs: parameters for forcing, diffusion, velocity, and boundary conditions
            (use defaults if None)
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
        
        y0 = self._initial_field(args['inputs'])
        solution = self.solver.root_find(residual_fn, y0, args, return_sol=return_sol)

        ret = solution if return_sol else {self.field_name: solution} 
        return ret
    
    def sample_inputs(self, key: Key) -> AdvectionDiffusionInputs:
        """Produce one sample of inputs for the given key."""
        if self.inputs_sampler is not None:
            return self.inputs_sampler(key)
        return {}
    
    def sample_outputs(
        self, 
        key: Key, 
        inputs: AdvectionDiffusionInputs | None = None,
        solution: AdvectionDiffusionOutputs | None = None
    ) -> AdvectionDiffusionOutputs:
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
