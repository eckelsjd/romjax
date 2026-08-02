"""Finite-volume 1D1V Vlasov-BGK-Poisson solver."""

import functools
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypedDict, get_origin

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key
from loguru import logger
from pydantic import BeforeValidator, ConfigDict, Field, PositiveFloat, field_validator

from romjax.graph import Node
from romjax.model import ImplicitModel, ImplicitSampleable
from romjax.pde import (
    FORCING_REGISTRY,
    BoundarySpec,
    BoundaryType,
    Coordinates,
    DiffraxSolver,
    ForcingCallable,
    GridBoundaryInputs,
    IdentityInputs,
    UniformGrid,
    homogeneous_boundary,
)
from romjax.rng import SamplerCallable
from romjax.tree import pytree_merge, to_pytree
from romjax.typing import DictModel, from_registry

__all__ = ["Vlasov1D1V"]


class VlasovParams(TypedDict):
    """Scalar dimensionless parameters for the Vlasov equation."""

    knudsen: ArrayLike
    debye: ArrayLike


class VlasovFields(TypedDict):
    """Field values for the Vlasov equation."""

    vdf: ArrayLike
    potential: ArrayLike


class VlasovInputs(TypedDict, total=False):
    """Inputs for the Vlasov 1D1V equation.

    :param params: scalar parameters including ``knudsen`` and ``debye``
    :param initial: initial-condition parameters or direct initial fields
    :param boundary: boundary-condition parameters for ``vdf`` and ``potential``
    """

    params: VlasovParams
    initial: dict[str, Any]
    boundary: dict[str, GridBoundaryInputs]


class VlasovOutputs(TypedDict):
    """Outputs for the Vlasov 1D1V equation.

    :param fields: trajectory fields with ``vdf`` shaped ``(Nx, Nv, Nt)`` and
        ``potential`` shaped ``(Nx, Nt)``
    """

    fields: VlasovFields


class VlasovResiduals(TypedDict):
    """Residual fields for the Vlasov equation."""

    fields: VlasovFields


class FVConfig(DictModel):
    """Finite-volume flux configuration.

    :param flux: numerical flux, either ``"upwind"`` or ``"lax_friedrichs"``
    :param reconstruction: interface reconstruction. ``"muscl"`` is accepted and
        currently uses limited piecewise-linear states for the periodic x-flux.
    :param limiter: slope limiter used by MUSCL reconstruction
    :param cfl: CFL factor used to choose the default ``dt0``
    :param dt_min: lower bound for CFL-derived step sizes
    :param dt_max: optional upper bound for CFL-derived step sizes
    :param enforce_cfl: whether to derive ``dt0`` from the current state when not configured
    """

    flux: Literal["upwind", "lax_friedrichs"] = "upwind"
    reconstruction: Literal["none", "muscl"] = "muscl"
    limiter: Literal["minmod", "van_leer", "none"] = "van_leer"
    cfl: PositiveFloat = 0.8
    dt_min: PositiveFloat = 1e-6
    dt_max: PositiveFloat | None = None
    enforce_cfl: bool = True


def spatial_boundary_with_neumann_velocity(
    type: str | BoundaryType = "dirichlet",
    value: float = 0.0,
) -> GridBoundaryInputs:
    """Specify a spatial 1D boundary while using homogeneous Neumann velocity boundaries.

    :param type: spatial boundary type
    :param value: spatial boundary value
    :return: two-dimensional phase-space boundary specification
    """
    return GridBoundaryInputs(
        boundary=[
            (BoundarySpec(type=type, value=value), BoundarySpec(type=type, value=value)),
            (BoundarySpec(type="neumann", value=0.0), BoundarySpec(type="neumann", value=0.0)),
        ]
    )


class CosinePerturbation(ForcingCallable):
    """Cosine perturbation of a Maxwellian velocity distribution."""

    class Inputs(DictModel):
        """Inputs for the cosine perturbation function.

        :param alpha: perturbation amplitude
        :param k: perturbation wavenumber
        :param n0: background density
        :param u0: background bulk velocity
        :param T0: background temperature
        :param coords: ``(x, v)`` phase-space grid coordinates
        """

        alpha: ArrayLike = 1.0
        k: ArrayLike = 1.0
        n0: ArrayLike = 1.0
        u0: ArrayLike = 0.0
        T0: ArrayLike = 1.0
        coords: Coordinates = (0.0, 0.0)

    def callable(self, inputs: Inputs, outputs: VlasovOutputs) -> ArrayLike:
        r"""Evaluate ``f(x,v,0) = (1 + alpha cos(kx)) M(v; n0, u0, T0)``.

        :param inputs: perturbation parameters
        :param outputs: Vlasov fields, unused
        :return: initial velocity distribution function
        """
        del outputs
        x, v = inputs["coords"]
        temperature = jnp.maximum(jnp.asarray(inputs["T0"]), 1e-12)
        perturbation = 1.0 + jnp.asarray(inputs["alpha"]) * jnp.cos(jnp.asarray(inputs["k"]) * x)
        gaussian = (
            jnp.asarray(inputs["n0"])
            / jnp.sqrt(2.0 * jnp.pi * temperature)
            * jnp.exp(-((v - jnp.asarray(inputs["u0"])) ** 2) / (2.0 * temperature))
        )
        return perturbation * gaussian


_forcing_registry = {
    **FORCING_REGISTRY,
    "cosine": CosinePerturbation,
}


type VlasovForcing = Annotated[ForcingCallable, BeforeValidator(functools.partial(from_registry, _forcing_registry))]


def _boundary_type(spec: Mapping[str, Any] | BoundarySpec) -> BoundaryType:
    """Return the enum value from a boundary spec."""
    return BoundaryType(spec["type"])


def _boundary_value(spec: Mapping[str, Any] | BoundarySpec) -> jax.Array:
    """Return the boundary value as a JAX scalar/array."""
    return jnp.asarray(spec["value"])


class Vlasov1D1V(ImplicitModel, ImplicitSampleable):
    """Finite-volume 1D1V Vlasov-BGK-Poisson implicit model.

    The output trajectory stores saved times on the last axis. ``solve`` integrates
    only the distribution function and derives the potential by solving Poisson's
    equation at every saved time.
    """

    model_config = ConfigDict(extra="forbid")

    grid: UniformGrid
    solver: DiffraxSolver = Field(default_factory=DiffraxSolver)
    fv: FVConfig = Field(default_factory=FVConfig)
    residual: Literal["backward_euler", "forward_euler", "central_fd"] = "backward_euler"

    source: Node = Node(name="vlasov_in")
    target: Node = Node(name="vlasov_out")

    params: VlasovParams = Field(default_factory=lambda: {"knudsen": 0.1, "debye": 1.0})
    initial: VlasovForcing | None = Field(default_factory=CosinePerturbation)
    boundary: VlasovForcing = Field(
        default_factory=lambda: IdentityInputs(
            inputs_default={
                "vdf": spatial_boundary_with_neumann_velocity(type="periodic", value=0.0),
                "potential": homogeneous_boundary(type="dirichlet", value=0.0, ndim=1),
            }
        )
    )

    density_floor: PositiveFloat = 1e-12
    temperature_floor: PositiveFloat = 1e-12
    inputs_sampler: SamplerCallable | None = None
    outputs_sampler: SamplerCallable | None = None

    @field_validator("grid", mode="after")
    @classmethod
    def _check_2d_grid(cls, value: UniformGrid) -> UniformGrid:
        """Validate that the grid is a 1D1V phase-space grid."""
        if len(value.shape) != 2:
            raise ValueError("Only 2D (1D1V) grid supported for Vlasov.")
        return value

    def _jax_coords(self) -> tuple[jax.Array, ...]:
        """Return grid coordinates as JAX arrays for numerical routines.

        :return: ``(x, v)`` coordinates
        """
        return tuple(jnp.asarray(coord) for coord in self.grid.coords)

    def _save_times(self) -> jax.Array:
        """Return solver save times as a JAX array."""
        return self.solver.save_times()

    def _resolve_inputs(self, inputs: VlasovInputs | None, coords: Coordinates | None = None) -> dict[str, Any]:
        """Merge configured defaults, runtime overrides, and grid coordinates.

        :param inputs: runtime input overrides
        :param coords: optional coordinate override
        :return: merged input tree
        """
        merged = pytree_merge(
            {"params": self.params, "initial": {}, "boundary": {}},
            to_pytree({} if inputs is None else inputs),
        )
        grid_coords = self._jax_coords() if coords is None else coords
        merged.setdefault("initial", {})
        merged.setdefault("boundary", {})
        merged["initial"].setdefault("coords", grid_coords)
        merged["boundary"].setdefault("coords", grid_coords)
        return merged

    def _boundary_inputs(self, inputs: dict[str, Any], outputs: VlasovOutputs | None = None) -> dict[str, Any]:
        """Evaluate the configured boundary callable.

        :param inputs: resolved inputs
        :param outputs: optional current outputs
        :return: boundary tree with ``vdf`` and ``potential`` entries
        """
        boundary = self.boundary(inputs.get("boundary", {}), {} if outputs is None else outputs)
        if "vdf" not in boundary or "potential" not in boundary:
            raise ValueError("Vlasov boundary inputs must define both 'vdf' and 'potential'.")
        
        return {k: GridBoundaryInputs.model_validate(boundary[k]) for k in VlasovFields.__annotations__}

    def _initial_vdf(self, inputs: dict[str, Any]) -> jax.Array:
        """Resolve the initial velocity distribution function.

        :param inputs: resolved inputs
        :return: ``(Nx, Nv)`` initial VDF
        """
        initial_inputs = inputs.get("initial", {})
        if "vdf" in initial_inputs:
            return jnp.asarray(initial_inputs["vdf"])
        if self.initial is None:
            raise ValueError("Vlasov initial condition must be configured or passed as inputs['initial']['vdf'].")
        return jnp.asarray(self.initial(initial_inputs, {"fields": {}}))

    def _potential_boundary_pair(self, boundary: dict[str, Any]) -> tuple[BoundarySpec, BoundarySpec]:
        """Return the left/right potential boundary pair."""
        return boundary["potential"]["boundary"][0]

    def _vdf_boundary_pairs(self, boundary: dict[str, Any]) -> list[tuple[BoundarySpec, BoundarySpec]]:
        """Return phase-space VDF boundary pairs."""
        return boundary["vdf"]["boundary"]

    def _spatial_ghosts(
        self,
        values: jax.Array,
        pair: tuple[BoundarySpec, BoundarySpec],
        dx: ArrayLike,
    ) -> tuple[jax.Array, jax.Array]:
        """Return left/right spatial ghost values for a one-dimensional field."""
        left, right = pair
        left_type = _boundary_type(left)
        right_type = _boundary_type(right)
        if left_type == BoundaryType.periodic and right_type == BoundaryType.periodic:
            return values[-1], values[0]

        left_value = _boundary_value(left)
        right_value = _boundary_value(right)
        if left_type == BoundaryType.dirichlet:
            ghost_left = 2.0 * left_value - values[0]
        elif left_type == BoundaryType.neumann:
            ghost_left = values[0] - jnp.asarray(dx) * left_value
        else:
            raise ValueError(f"Unsupported left boundary type {left_type!r}.")

        if right_type == BoundaryType.dirichlet:
            ghost_right = 2.0 * right_value - values[-1]
        elif right_type == BoundaryType.neumann:
            ghost_right = values[-1] + jnp.asarray(dx) * right_value
        else:
            raise ValueError(f"Unsupported right boundary type {right_type!r}.")
        return ghost_left, ghost_right

    def _velocity_ghosted(self, f: jax.Array, pair: tuple[BoundarySpec, BoundarySpec], dv: ArrayLike) -> jax.Array:
        """Return a VDF array padded by one ghost cell in velocity."""
        left, right = pair
        left_type = _boundary_type(left)
        right_type = _boundary_type(right)
        if left_type == BoundaryType.periodic and right_type == BoundaryType.periodic:
            left_ghost = f[:, -1:]
            right_ghost = f[:, :1]
        else:
            if left_type == BoundaryType.dirichlet:
                left_ghost = 2.0 * _boundary_value(left) - f[:, :1]
            elif left_type == BoundaryType.neumann:
                left_ghost = f[:, :1] - jnp.asarray(dv) * _boundary_value(left)
            else:
                raise ValueError(f"Unsupported velocity left boundary type {left_type!r}.")

            if right_type == BoundaryType.dirichlet:
                right_ghost = 2.0 * _boundary_value(right) - f[:, -1:]
            elif right_type == BoundaryType.neumann:
                right_ghost = f[:, -1:] + jnp.asarray(dv) * _boundary_value(right)
            else:
                raise ValueError(f"Unsupported velocity right boundary type {right_type!r}.")
        return jnp.concatenate([left_ghost, f, right_ghost], axis=1)

    def _density_moments(self, f: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Compute density, bulk velocity, and temperature moments.

        :param f: VDF shaped ``(Nx, Nv)``
        :return: ``(n, u, T)`` arrays shaped ``(Nx,)``
        """
        _, v = self._jax_coords()
        dv = self.grid.spacing[1]
        density = self._density(f)
        momentum = jnp.sum(v * f, axis=1) * dv
        velocity = momentum / density
        temperature_raw = jnp.sum(((v - velocity[:, None]) ** 2) * f, axis=1) * dv / density
        temperature = jnp.maximum(temperature_raw, self.temperature_floor)
        return density, velocity, temperature

    def _density(self, f: jax.Array) -> jax.Array:
        """Compute only the density moment of a VDF."""
        dv = self.grid.spacing[1]
        return jnp.maximum(jnp.sum(f, axis=1) * dv, self.density_floor)

    def _maxwellian(self, f: jax.Array, moments: tuple[jax.Array, jax.Array, jax.Array] | None = None) -> jax.Array:
        """Return the local Maxwellian equilibrium for ``f``."""
        _, v = self._jax_coords()
        density, velocity, temperature = self._density_moments(f) if moments is None else moments
        prefactor = density[:, None] / jnp.sqrt(2.0 * jnp.pi * temperature[:, None])
        exponent = -((v - velocity[:, None]) ** 2) / (2.0 * temperature[:, None])
        return prefactor * jnp.exp(exponent)

    def _dirichlet_tdma_poisson(
        self,
        source: jax.Array,
        debye: ArrayLike,
        pair: tuple[BoundarySpec, BoundarySpec],
    ) -> jax.Array:
        """Solve a one-dimensional Dirichlet Poisson problem using TDMA."""
        nx = self.grid.shape[0]
        dx = self.grid.spacing[0]
        if nx < 3:
            left, right = pair
            return jnp.asarray([_boundary_value(left), _boundary_value(right)])

        scale = jnp.asarray(debye) ** 2 / (dx * dx)
        left_value = _boundary_value(pair[0])
        right_value = _boundary_value(pair[1])
        rhs = source[1:-1]
        rhs = rhs.at[0].add(-scale * left_value)
        rhs = rhs.at[-1].add(-scale * right_value)
        nsys = nx - 2
        lower = scale * jnp.ones((nsys - 1,))
        diag = -2.0 * scale * jnp.ones((nsys,))
        upper = scale * jnp.ones((nsys - 1,))

        def forward_step(
            carry: tuple[jax.Array, jax.Array],
            row: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
            prev_cprime, prev_dprime = carry
            lower_i, diag_i, rhs_i, upper_i = row
            denom = diag_i - lower_i * prev_cprime
            cprime = jnp.where(denom != 0.0, upper_i / denom, 0.0)
            dprime = (rhs_i - lower_i * prev_dprime) / denom
            return (cprime, dprime), (cprime, dprime)

        first_cprime = jnp.where(nsys > 1, upper[0] / diag[0], 0.0)
        first_dprime = rhs[0] / diag[0]
        if nsys == 1:
            interior = first_dprime[None]
        else:
            upper_tail = jnp.concatenate([upper[1:], jnp.zeros((1,), dtype=upper.dtype)])
            rows = (lower, diag[1:], rhs[1:], upper_tail)
            (_, _), scanned = jax.lax.scan(forward_step, (first_cprime, first_dprime), rows)
            cprime_all = jnp.concatenate([first_cprime[None], scanned[0]])
            dprime_all = jnp.concatenate([first_dprime[None], scanned[1]])

            def backward_step(carry: jax.Array, row: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
                cprime_i, dprime_i = row
                value = dprime_i - cprime_i * carry
                return value, value

            _, rev_values = jax.lax.scan(
                backward_step,
                dprime_all[-1],
                (cprime_all[:-1][::-1], dprime_all[:-1][::-1]),
            )
            interior = jnp.concatenate([rev_values[::-1], dprime_all[-1:]])
        return jnp.concatenate([left_value[None], interior, right_value[None]])

    def _periodic_poisson(self, source: jax.Array, debye: ArrayLike) -> jax.Array:
        """Solve periodic Poisson with a zero-mean gauge using FFT."""
        nx = self.grid.shape[0]
        dx = self.grid.spacing[0]
        source = source - jnp.mean(source)
        source_hat = jnp.fft.fft(source)
        modes = jnp.arange(nx)
        eigenvalues = -4.0 * (jnp.sin(jnp.pi * modes / nx) ** 2) / (dx * dx)
        denom = (jnp.asarray(debye) ** 2) * eigenvalues
        phi_hat = jnp.where(modes == 0, 0.0 + 0.0j, source_hat / denom)
        return jnp.real(jnp.fft.ifft(phi_hat))

    def _dense_poisson(self, source: jax.Array, debye: ArrayLike, pair: tuple[BoundarySpec, BoundarySpec]) -> jax.Array:
        """Solve mixed non-periodic Poisson BCs with a dense fallback."""
        nx = self.grid.shape[0]
        dx = self.grid.spacing[0]
        scale = jnp.asarray(debye) ** 2 / (dx * dx)
        matrix = jnp.zeros((nx, nx), dtype=jnp.result_type(source, debye))
        rhs = source
        left, right = pair

        left_type = _boundary_type(left)
        if left_type == BoundaryType.dirichlet:
            matrix = matrix.at[0, 0].set(1.0)
            rhs = rhs.at[0].set(_boundary_value(left))
        elif left_type == BoundaryType.neumann:
            matrix = matrix.at[0, 0].set(-1.0 / dx)
            matrix = matrix.at[0, 1].set(1.0 / dx)
            rhs = rhs.at[0].set(_boundary_value(left))
        else:
            raise ValueError(f"Unsupported potential left boundary type {left_type!r}.")

        right_type = _boundary_type(right)
        if right_type == BoundaryType.dirichlet:
            matrix = matrix.at[-1, -1].set(1.0)
            rhs = rhs.at[-1].set(_boundary_value(right))
        elif right_type == BoundaryType.neumann:
            matrix = matrix.at[-1, -2].set(-1.0 / dx)
            matrix = matrix.at[-1, -1].set(1.0 / dx)
            rhs = rhs.at[-1].set(_boundary_value(right))
        else:
            raise ValueError(f"Unsupported potential right boundary type {right_type!r}.")

        diag = -2.0 * scale * jnp.ones((nx - 2,))
        offdiag = scale * jnp.ones((nx - 3,))
        matrix = matrix.at[jnp.arange(1, nx - 1), jnp.arange(1, nx - 1)].set(diag)
        matrix = matrix.at[jnp.arange(1, nx - 2), jnp.arange(2, nx - 1)].set(offdiag)
        matrix = matrix.at[jnp.arange(2, nx - 1), jnp.arange(1, nx - 2)].set(offdiag)
        return jnp.linalg.solve(matrix, rhs)

    def _solve_poisson(
        self,
        density: jax.Array,
        params: VlasovParams,
        boundary: dict[str, Any],
        target_residual: ArrayLike | None = None,
    ) -> jax.Array:
        """Solve the Vlasov Poisson equation for a density field."""
        residual_target = 0.0 if target_residual is None else jnp.asarray(target_residual)
        source = density - 1.0 + residual_target
        debye = params["debye"]
        pair = self._potential_boundary_pair(boundary)
        left_type = _boundary_type(pair[0])
        right_type = _boundary_type(pair[1])
        if left_type == BoundaryType.periodic and right_type == BoundaryType.periodic:
            return self._periodic_poisson(source, debye)
        if left_type == BoundaryType.dirichlet and right_type == BoundaryType.dirichlet:
            return self._dirichlet_tdma_poisson(source, debye, pair)
        return self._dense_poisson(source, debye, pair)

    def _poisson_residual(
        self,
        potential: jax.Array,
        density: jax.Array,
        params: VlasovParams,
        boundary: dict[str, Any],
    ) -> jax.Array:
        """Evaluate the discrete Poisson residual."""
        dx = self.grid.spacing[0]
        pair = self._potential_boundary_pair(boundary)
        left_type = _boundary_type(pair[0])
        right_type = _boundary_type(pair[1])
        if left_type == BoundaryType.periodic and right_type == BoundaryType.periodic:
            west = jnp.roll(potential, 1)
            east = jnp.roll(potential, -1)
            laplace = (west - 2.0 * potential + east) / (dx * dx)
            return jnp.asarray(params["debye"]) ** 2 * laplace - (density - 1.0)

        residual = jnp.zeros_like(potential)
        if left_type == BoundaryType.dirichlet:
            residual = residual.at[0].set(potential[0] - _boundary_value(pair[0]))
        elif left_type == BoundaryType.neumann:
            residual = residual.at[0].set((potential[1] - potential[0]) / dx - _boundary_value(pair[0]))
        else:
            raise ValueError(f"Unsupported potential left boundary type {left_type!r}.")

        if right_type == BoundaryType.dirichlet:
            residual = residual.at[-1].set(potential[-1] - _boundary_value(pair[1]))
        elif right_type == BoundaryType.neumann:
            residual = residual.at[-1].set((potential[-1] - potential[-2]) / dx - _boundary_value(pair[1]))
        else:
            raise ValueError(f"Unsupported potential right boundary type {right_type!r}.")

        laplace = (potential[:-2] - 2.0 * potential[1:-1] + potential[2:]) / (dx * dx)
        interior = jnp.asarray(params["debye"]) ** 2 * laplace - (density[1:-1] - 1.0)
        return residual.at[1:-1].set(interior)

    def _potential_gradient(self, potential: jax.Array, boundary: dict[str, Any]) -> jax.Array:
        """Compute ``dphi/dx`` using BC-aware finite differences."""
        dx = self.grid.spacing[0]
        pair = self._potential_boundary_pair(boundary)
        left_type = _boundary_type(pair[0])
        right_type = _boundary_type(pair[1])
        if left_type == BoundaryType.periodic and right_type == BoundaryType.periodic:
            return (jnp.roll(potential, -1) - jnp.roll(potential, 1)) / (2.0 * dx)

        grad = jnp.zeros_like(potential)
        grad = grad.at[1:-1].set((potential[2:] - potential[:-2]) / (2.0 * dx))
        grad = grad.at[0].set((potential[1] - potential[0]) / dx)
        grad = grad.at[-1].set((potential[-1] - potential[-2]) / dx)
        return grad

    def _limited_slope(self, left_delta: jax.Array, right_delta: jax.Array) -> jax.Array:
        """Return limited slopes for MUSCL reconstruction."""
        if self.fv.limiter == "none":
            return 0.5 * (left_delta + right_delta)
        same_sign = left_delta * right_delta > 0.0
        if self.fv.limiter == "minmod":
            limited = jnp.sign(left_delta) * jnp.minimum(jnp.abs(left_delta), jnp.abs(right_delta))
            return jnp.where(same_sign, limited, 0.0)
        numerator = 2.0 * left_delta * right_delta
        denominator = left_delta + right_delta
        return jnp.where(same_sign, numerator / jnp.where(denominator == 0.0, 1.0, denominator), 0.0)

    def _x_interface_states(
        self,
        f: jax.Array,
        vdf_boundary: list[tuple[BoundarySpec, BoundarySpec]],
    ) -> tuple[jax.Array, jax.Array]:
        """Return left/right states at x interfaces shaped ``(Nx, Nv)``."""
        pair = vdf_boundary[0]
        left_type = _boundary_type(pair[0])
        right_type = _boundary_type(pair[1])
        if (
            left_type != BoundaryType.periodic
            or right_type != BoundaryType.periodic
            or self.fv.reconstruction == "none"
        ):
            return f, jnp.roll(f, -1, axis=0)

        left_delta = f - jnp.roll(f, 1, axis=0)
        right_delta = jnp.roll(f, -1, axis=0) - f
        slopes = self._limited_slope(left_delta, right_delta)
        next_slopes = jnp.roll(slopes, -1, axis=0)
        return f + 0.5 * slopes, jnp.roll(f, -1, axis=0) - 0.5 * next_slopes

    def _spatial_flux_divergence(
        self,
        f: jax.Array,
        vdf_boundary: list[tuple[BoundarySpec, BoundarySpec]],
    ) -> jax.Array:
        """Compute spatial advection flux divergence."""
        _, v = self._jax_coords()
        dx = self.grid.spacing[0]
        left_state, right_state = self._x_interface_states(f, vdf_boundary)
        if self.fv.flux == "upwind":
            flux_right = v * jnp.where(v >= 0.0, left_state, right_state)
        else:
            speed = jnp.abs(v)
            flux_right = 0.5 * v * (left_state + right_state) - 0.5 * speed * (right_state - left_state)
        flux_left = jnp.roll(flux_right, 1, axis=0)
        return (flux_right - flux_left) / dx

    def _velocity_flux_divergence(
        self,
        f: jax.Array,
        acceleration: jax.Array,
        vdf_boundary: list[tuple[BoundarySpec, BoundarySpec]],
    ) -> jax.Array:
        """Compute velocity-space advection flux divergence."""
        dv = self.grid.spacing[1]
        f_ghosted = self._velocity_ghosted(f, vdf_boundary[1], dv)
        left_state = f_ghosted[:, :-1]
        right_state = f_ghosted[:, 1:]
        acceleration = acceleration[:, None]
        if self.fv.flux == "upwind":
            flux = acceleration * jnp.where(acceleration >= 0.0, left_state, right_state)
        else:
            speed = jnp.abs(acceleration)
            flux = 0.5 * acceleration * (left_state + right_state) - 0.5 * speed * (right_state - left_state)
        return (flux[:, 1:] - flux[:, :-1]) / dv

    def _advection_rhs_from_potential(
        self,
        f: jax.Array,
        potential: jax.Array,
        boundary: dict[str, Any],
    ) -> jax.Array:
        """Evaluate the explicit phase-space advection RHS using a provided potential."""
        acceleration = self._potential_gradient(potential, boundary)
        vdf_boundary = self._vdf_boundary_pairs(boundary)
        spatial_div = self._spatial_flux_divergence(f, vdf_boundary)
        velocity_div = self._velocity_flux_divergence(f, acceleration, vdf_boundary)
        return -spatial_div - velocity_div

    def _advection_rhs(
        self,
        _t: ArrayLike,
        f: jax.Array,
        params: VlasovParams,
        boundary: dict[str, Any],
        potential_residual: ArrayLike | None = None,
    ) -> jax.Array:
        """Evaluate the explicit advection RHS, deriving potential from Poisson."""
        density = self._density(f)
        potential = self._solve_poisson(density, params, boundary, potential_residual)
        return self._advection_rhs_from_potential(f, potential, boundary)

    def _collision_rhs(self, _t: ArrayLike, f: jax.Array, params: VlasovParams) -> jax.Array:
        """Evaluate the BGK collision RHS."""
        tau = jnp.asarray(params["knudsen"])
        moments = self._density_moments(f)
        collision = (self._maxwellian(f, moments) - f) / tau
        return jnp.where(jnp.isfinite(tau) & (tau > 0.0), collision, 0.0)

    def _rhs(
        self,
        t: ArrayLike,
        f: jax.Array,
        params: VlasovParams,
        boundary: dict[str, Any],
        potential_residual: ArrayLike | None = None,
    ) -> jax.Array:
        """Evaluate the unsplit method-of-lines RHS for single-term solvers."""
        advection = self._advection_rhs(t, f, params, boundary, potential_residual)
        collision = self._collision_rhs(t, f, params)
        return advection + collision

    def _stable_dt(self, f: jax.Array, params: VlasovParams, boundary: dict[str, Any]) -> jax.Array:
        """Compute a CFL-limited step size for the current state."""
        _, v = self._jax_coords()
        dx, dv = self.grid.spacing
        density = self._density(f)
        potential = self._solve_poisson(density, params, boundary)
        acceleration = self._potential_gradient(potential, boundary)
        rate = jnp.max(jnp.abs(v)) / dx + jnp.max(jnp.abs(acceleration)) / dv
        dt = self.fv.cfl / jnp.maximum(rate, 1e-12)
        dt = jnp.maximum(dt, self.fv.dt_min)
        if self.fv.dt_max is not None:
            dt = jnp.minimum(dt, self.fv.dt_max)
        return dt

    def _interp_saved(self, t: ArrayLike, values: jax.Array, times: jax.Array) -> jax.Array:
        """Linearly interpolate arrays whose last axis is saved time."""
        nt = values.shape[-1]
        if nt == 1:
            return values[..., 0]
        idx = jnp.clip(jnp.searchsorted(times, t, side="right") - 1, 0, nt - 2)
        t0 = times[idx]
        t1 = times[idx + 1]
        weight = (jnp.asarray(t) - t0) / jnp.maximum(t1 - t0, 1e-12)
        return (1.0 - weight) * values[..., idx] + weight * values[..., idx + 1]

    def _saved_potential(
        self,
        f_trajectory: jax.Array,
        output_times: jax.Array,
        params: VlasovParams,
        boundary: dict[str, Any],
        potential_residuals: jax.Array | None = None,
        residual_times: jax.Array | None = None,
    ) -> jax.Array:
        """Compute potential at saved times for a VDF trajectory."""
        f_by_time = jnp.moveaxis(f_trajectory, -1, 0)
        if potential_residuals is None:
            potential_residuals_by_time = jnp.zeros((f_by_time.shape[0], self.grid.shape[0]), dtype=f_by_time.dtype)
        else:
            residual_times = output_times if residual_times is None else residual_times
            potential_residuals_by_time = jax.vmap(
                lambda t: self._interp_saved(t, potential_residuals, residual_times)
            )(output_times)

        def solve_one(f: jax.Array, target: jax.Array) -> jax.Array:
            density = self._density(f)
            return self._solve_poisson(density, params, boundary, target)

        return jnp.moveaxis(jax.vmap(solve_one)(f_by_time, potential_residuals_by_time), 0, -1)

    def _check_output_shapes(self, vdf: jax.Array, potential: jax.Array) -> None:
        """Validate output trajectory shapes."""
        nx, nv = self.grid.shape
        if vdf.ndim != 3 or vdf.shape[:2] != (nx, nv):
            raise ValueError(f"Vlasov vdf trajectory must have shape (Nx, Nv, Nt); got {vdf.shape}.")
        if potential.shape != (nx, vdf.shape[-1]):
            raise ValueError(f"Vlasov potential trajectory must have shape (Nx, Nt); got {potential.shape}.")

    def _solver_uses_split_terms(self) -> bool:
        """Return whether the configured diffrax solver expects explicit/implicit terms."""
        term_structure = getattr(self.solver.solver, "term_structure", None)
        return get_origin(term_structure) is diffrax.MultiTerm

    def evaluate(self, inputs: VlasovInputs | None, outputs: VlasovOutputs) -> VlasovResiduals:
        """Evaluate the saved-grid Vlasov residual on a 1D1V grid.

        :param inputs: boundary, initial-condition, and scalar parameter overrides
        :param outputs: VDF and potential trajectories
        :return: residual trajectories with matching shapes
        """
        resolved = self._resolve_inputs(inputs)
        fields = outputs["fields"]
        vdf = jnp.asarray(fields["vdf"])
        potential = jnp.asarray(fields["potential"])
        self._check_output_shapes(vdf, potential)

        boundary = self._boundary_inputs(resolved, outputs)
        params = resolved["params"]
        times = self._save_times()
        if times.shape[0] != vdf.shape[-1]:
            raise ValueError(
                f"Number of solver save times ({times.shape[0]}) must match trajectory Nt ({vdf.shape[-1]})."
            )

        initial = self._initial_vdf(resolved)
        density_by_time = jax.vmap(self._density)(jnp.moveaxis(vdf, -1, 0))
        potential_residual = jnp.moveaxis(
            jax.vmap(lambda phi, density: self._poisson_residual(phi, density, params, boundary))(
                jnp.moveaxis(potential, -1, 0),
                density_by_time,
            ),
            0,
            -1,
        )

        rhs_by_time = jnp.moveaxis(
            jax.vmap(
                lambda f, phi: self._advection_rhs_from_potential(f, phi, boundary)
                + self._collision_rhs(0.0, f, params)
            )(
                jnp.moveaxis(vdf, -1, 0),
                jnp.moveaxis(potential, -1, 0),
            ),
            0,
            -1,
        )
        dt = jnp.diff(times)
        if self.residual == "backward_euler":
            temporal = (vdf[..., 1:] - vdf[..., :-1]) / dt.reshape((1, 1, -1))
            vdf_residual = jnp.concatenate(
                [(vdf[..., 0] - initial)[..., None], temporal - rhs_by_time[..., 1:]],
                axis=-1,
            )
        elif self.residual == "forward_euler":
            temporal = (vdf[..., 1:] - vdf[..., :-1]) / dt.reshape((1, 1, -1))
            interior = temporal - rhs_by_time[..., :-1]
            vdf_residual = jnp.concatenate([interior, (vdf[..., -1] - vdf[..., -2])[..., None]], axis=-1)
        else:
            centered_dt = (times[2:] - times[:-2]).reshape((1, 1, -1))
            centered = (vdf[..., 2:] - vdf[..., :-2]) / centered_dt
            middle = centered - rhs_by_time[..., 1:-1]
            first = (vdf[..., 0] - initial)[..., None]
            last = ((vdf[..., -1] - vdf[..., -2]) / dt[-1] - rhs_by_time[..., -1])[..., None]
            vdf_residual = jnp.concatenate([first, middle, last], axis=-1)

        return {"fields": {"vdf": vdf_residual, "potential": potential_residual}}

    def solve(
        self,
        inputs: VlasovInputs | None = None,
        residuals: VlasovResiduals | None = None,
        return_sol: bool = False,
        **kwargs
    ) -> VlasovOutputs | tuple[VlasovOutputs, diffrax.Solution]:
        """Solve the Vlasov equation for a target residual trajectory.

        :param inputs: runtime input overrides
        :param residuals: target residuals, defaulting to zero
        :param return_sol: return the raw diffrax solution when true
        :param kwargs: everything else passed to diffeqsolve (e.g. for restarting controller state or event handling)
        :return: Vlasov output trajectory or diffrax solution
        """
        resolved = self._resolve_inputs(inputs)
        boundary = self._boundary_inputs(resolved)
        params = resolved["params"]
        times = self._save_times()
        y0 = self._initial_vdf(resolved)
        residual_fields = {} if residuals is None else residuals.get("fields", {})
        vdf_target = jnp.asarray(
            residual_fields.get("vdf", jnp.zeros((*self.grid.shape, times.shape[0]), dtype=y0.dtype))
        )
        potential_target = jnp.asarray(
            residual_fields.get("potential", jnp.zeros((self.grid.shape[0], times.shape[0]), dtype=y0.dtype))
        )

        def explicit_rhs(t: ArrayLike, y: jax.Array, args: dict[str, Any]) -> jax.Array:
            vdf_residual_t = self._interp_saved(t, args["vdf_target"], args["times"])
            potential_residual_t = self._interp_saved(t, args["potential_target"], args["times"])
            advection = self._advection_rhs(t, y, args["params"], args["boundary"], potential_residual_t)
            return advection + vdf_residual_t

        def implicit_rhs(t: ArrayLike, y: jax.Array, args: dict[str, Any]) -> jax.Array:
            return self._collision_rhs(t, y, args["params"])

        def combined_rhs(t: ArrayLike, y: jax.Array, args: dict[str, Any]) -> jax.Array:
            return explicit_rhs(t, y, args) + implicit_rhs(t, y, args)

        dt0 = self.solver.dt0
        if dt0 is None and self.fv.enforce_cfl:
            dt0 = self._stable_dt(y0, params, boundary)
            if not isinstance(dt0, jax.core.Tracer):
                logger.debug(f"Vlasov chose CFL-based dt0: {dt0}")
        elif dt0 is None:
            dt0 = jnp.minimum((times[-1] - times[0]) / jnp.maximum(times.shape[0] - 1, 1), self.fv.dt_max or jnp.inf)

        terms = (
            diffrax.MultiTerm(diffrax.ODETerm(explicit_rhs), diffrax.ODETerm(implicit_rhs))
            if self._solver_uses_split_terms()
            else diffrax.ODETerm(combined_rhs)
        )

        solution = self.solver.diffeqsolve(
            terms,
            y0, 
            args={
                "params": params,
                "boundary": boundary,
                "times": times,
                "vdf_target": vdf_target,
                "potential_target": potential_target,
            }, 
            dt0=dt0,
            **kwargs
        )

        vdf = jnp.moveaxis(solution.ys, 0, -1)
        output_times = jnp.asarray(solution.ts)
        potential = self._saved_potential(vdf, output_times, params, boundary, potential_target, times)
        res = {"fields": {"vdf": vdf, "potential": potential}}

        return (res, solution) if return_sol else res

    def sample_inputs(self, key: Key) -> VlasovInputs:
        """Produce one sample of inputs for the given key."""
        if self.inputs_sampler is not None:
            return self.inputs_sampler(key)
        return {}

    def sample_outputs(
        self,
        key: Key,
        inputs: VlasovInputs | None = None,
        solution: VlasovOutputs | None = None,
    ) -> VlasovOutputs:
        """Produce one sample of outputs for the given key."""
        if self.outputs_sampler is None:
            return {}
        if solution is None:
            solution = self.solve(inputs)
        sample = self.outputs_sampler(key, inputs=inputs, solution=solution)
        if isinstance(sample, Mapping):
            return {"fields": sample["fields"]} if "fields" in sample else {"fields": sample}
        return {"fields": {"vdf": jnp.asarray(sample)}}

    def resolve_dof(self) -> int:
        """Return the trajectory degrees of freedom implied by the save-time grid."""
        num_steps = int(self._save_times().shape[0])
        nx, nv = self.grid.shape
        return nx * nv * num_steps + nx * num_steps
