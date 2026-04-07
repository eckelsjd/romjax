"""Example 2D Poisson solver."""
from typing import Literal, Mapping, TypedDict, Any, Iterable, Callable
from os import PathLike
import copy

import jax.numpy as jnp
import optimistix as optx
from jaxtyping import ArrayLike, PyTree, Key
from pydantic import Field, PositiveInt, ValidationInfo, field_validator
import jax

from romjax.pde import Coordinates, homogeneous_boundary, UniformGrid, BoundarySpec, BoundaryType
from romjax.typing import IterativeSolver, AdjointMethod, DictModel, ImplicitModel
from romjax.utils import merge_pytrees, to_pytree
from romjax.random import BatchSampler, Distribution, parametric_sampler


type ForcingName = Literal["gaussian", "nonlinear", "sinusoid", "constant"]
type SamplerName = Literal["parametric"]
type ForcingCallable = Callable[[PyTree, PyTree], ArrayLike]
type BoundaryCallable = Callable[[PyTree], PyTree]
type InitialCallable = Callable[[Coordinates], ArrayLike]


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


def const_initial_guess(const: float) -> InitialCallable:
    return lambda coords: const * jnp.ones_like(coords[0])


def boundary_pass_through(inputs: PyTree) -> PyTree:
    """Simple boundary that uses boundary input params directly (just pass them through)."""
    return inputs


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


def darcy_field(
    key: Key, 
    nsamples: int = 1, 
    bounds: tuple[tuple[int, int], tuple[int, int]] = ((0, 1), (0, 1)),
    shape: tuple[int, int] = (16, 16),
    nemytskii: tuple[float, float] = (3.0, 12.0),
    cov_diag: float = 9.0
) -> ArrayLike:
    """Sample darcy flow conductivity field according to Sec 6.2 of Kovachki 2022.

    https://arxiv.org/abs/2108.08481

    :param key: the random key
    :param nsamples: number of random fields to generate
    :param bounds: 2d bounds of grid
    :param shape: 2d shape of grid
    :param nemytskii: the result of the nemytskii pushforward, term 0 if field < 0, term 1 if field > 0
    :param cov_diag: additional amount to add to cov diagonal on top of Laplacian eigenvalues
    :return: Array of shape [nsamples, W, H] giving the random field samples
    """
    (x0, x1), (y0, y1) = bounds
    nx, ny = shape
    x = jnp.linspace(x0, x1, nx)
    y = jnp.linspace(y0, y1, ny)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")

    kx = jnp.arange(nx)
    ky = jnp.arange(ny)

    scale_x = jnp.where(kx == 0, 1.0, jnp.sqrt(2.0))
    scale_y = jnp.where(ky == 0, 1.0, jnp.sqrt(2.0))

    lx = x1 - x0
    ly = y1 - y0
    x_hat = (xx[:, 0] - x0) / lx
    y_hat = (yy[0, :] - y0) / ly
    phi_x = jnp.cos(jnp.pi * x_hat[:, None] * kx[None, :]) * scale_x[None, :]
    phi_y = jnp.cos(jnp.pi * y_hat[:, None] * ky[None, :]) * scale_y[None, :]

    lap_eigs = (jnp.pi ** 2) * ((kx[:, None] / lx) ** 2 + (ky[None, :] / ly) ** 2)
    cov_eigs = (lap_eigs + cov_diag) ** -2
    sqrt_cov = jnp.sqrt(cov_eigs)

    coeffs = jax.random.normal(key, (nsamples, nx, ny)) * sqrt_cov[None, :, :]
    gauss_field = jnp.einsum("ik,jl,bkl->bij", phi_x, phi_y, coeffs)

    low, high = nemytskii
    return jnp.where(gauss_field >= 0.0, high, low)


class PoissonConfig(DictModel):
    """Numerical configs for solving the Poisson PDE on a 2D grid.

    See Optimistix docs for `root_find()` method.
    
    :ivar grid: the uniform 2D Cartesian grid
    :ivar solver: Optimistix nonlinear root finding solver (name+opts or instance), default is Newton
    :ivar initial_guess: callable that takes coords and generates an initial guess on the grid
    :ivar options: runtime options for the nonlinear solver
    :ivar max_steps: maximum number of solver steps
    :ivar adjoint: Optimistix adjoint method
    :ivar throw: whether to throw failures as errors (default True)
    """
    grid: UniformGrid
    solver: IterativeSolver = Field(default_factory=lambda: dict(name='Newton', opts={'rtol': 1e-2, 'atol': 1e-4}),
                                    validate_default=True)
    initial_guess: InitialCallable = Field(default_factory=lambda: const_initial_guess(0.0), exclude=True)
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


class Poisson2D(ImplicitModel):

    # Required (and static once set)
    config: PoissonConfig

    # Optional/default (and optionally variable online)
    forcing: ForcingCallable = constant_forcing
    conductivity: ForcingCallable = constant_forcing
    boundary: BoundaryCallable = boundary_pass_through

    forcing_sampler: BatchSampler | None = None
    conductivity_sampler: BatchSampler | None = None
    boundary_sampler: BatchSampler | None = None

    forcing_sampler_opts: dict[str, Any] = Field(default_factory=dict)
    conductivity_sampler_opts: dict[str, Any] = Field(default_factory=dict)
    boundary_sampler_opts: dict[str, Any] = Field(default_factory=dict)

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
                raise ValueError(f"Unknown forcing function: {value!r}")
            return mapping[value]
        if callable(value):
            return value
        raise TypeError("forcing and conductivity must be a callable or a supported string literal.")

    @field_validator("forcing_sampler", "conductivity_sampler", "boundary_sampler", mode="before")
    @classmethod
    def _coerce_sampler(cls, value: BatchSampler | SamplerName | None) -> BatchSampler | None:
        if isinstance(value, str):
            mapping: Mapping[SamplerName, BatchSampler] = {
                "parametric": parametric_sampler
            }
            if value not in mapping:
                raise ValueError(f"Unknown sampler: {value!r}")
            return mapping[value]
        if value is None:
            return value
        if callable(value):
            return value
        raise TypeError("samplers must be either a valid name, callable, or none.")
    
    @field_validator("forcing_sampler_opts", "conductivity_sampler_opts", "boundary_sampler_opts", mode="after")
    @classmethod
    def _coerce_sampler_opts(cls, value: dict[str, Any], info: ValidationInfo) -> dict[str, Any]:
        """Just provides validation for known sampler function params."""
        check_flag = False
        for s in ['forcing', 'conductivity', 'boundary']:
            s_flag = info.field_name == f"{s}_sampler_opts" and info.data[f"{s}_sampler"] is parametric_sampler
            check_flag = check_flag or s_flag

        # Validate distributions for parametric sampling
        if check_flag:
            for k in list(value.keys()):
                if k not in ['write_summary', 'format', 'prefix']:
                    if not isinstance(value[k], Mapping):
                        raise TypeError(f"Extra parametric sampler opts must be a Distribution-like mapping.")
                    value[k] = Distribution(**value[k])
        
        return value
    
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
            y0=jnp.asarray(self.config.initial_guess(self.config.grid.coords)),
            args=args,
            options=self.config.options,
            max_steps=self.config.max_steps,
            adjoint=self.config.adjoint,
            throw=self.config.throw,
        )

        ret = solution if return_sol else {"phi": solution.value} 
        return ret
    
    def sample_inputs(self, keys: Iterable[Key], paths: Iterable[str | PathLike]) -> None:
        def _sample(keys, sampler, opts, prefix):
            if sampler is not None:
                opts = copy.copy(opts)
                opts['prefix'] = (f"{p}_" if (p := opts.get("prefix")) is not None else "") + prefix
                sampler(keys, paths, **opts)
        
        if len(keys) > 0 and len(paths) > 0:
            forcing_keys, conductivity_keys, boundary_keys = zip(*[jax.random.split(k, 3) for k in keys])
            
            _sample(forcing_keys, self.forcing_sampler, self.forcing_sampler_opts, 'forcing')
            _sample(conductivity_keys, self.conductivity_sampler, self.conductivity_sampler_opts, 'conductivity')
            _sample(boundary_keys, self.boundary_sampler, self.boundary_sampler_opts, 'boundary')
