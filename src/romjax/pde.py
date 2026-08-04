"""Utilites for PDE-based solvers."""
from collections.abc import Mapping
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, cast

import diffrax
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
from alive_progress import alive_bar
from diffrax._progress_meter import _progress_meter_manager
from jaxtyping import ArrayLike, Key, PyTree
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)

from romjax.compression import Compression
from romjax.graph import CompositeEdge, EdgePatch
from romjax.model import ImplicitModel, ImplicitSampleable, SourceSampleable
from romjax.nn import Affine
from romjax.rng import PyTreeSampler, SamplerCallable, SolverSampler
from romjax.tree import TreePath, pytree_merge, set_subtree
from romjax.typing import CallableModel, DictModel, ThirdPartyType, from_registry, require_type

__all__ = ['Coordinates', 'BoundaryType', 'BoundarySpec', 'GridBoundaryInputs', 'homogeneous_boundary', 'UniformGrid',
           'ForcingCallable', 'RegisteredForcing', 'FORCING_REGISTRY', 'IdentityInputs', 'ConstantForcing',
           'GaussianForcing', 'SinusoidForcing', 'IterativeSolver',
           'LatentSamplerFactory', 'ImplicitAffine', 'ImplicitIterativeGalerkin', 'DiffraxSolver', 'AliveProgressMeter']


type Coordinates = tuple[ArrayLike, ...] | ArrayLike
type LinearSolver = Annotated[
    ThirdPartyType(default_modules=lx.__name__),
    AfterValidator(partial(require_type, lx.AbstractLinearSolver)),
]


class ForcingCallable(CallableModel):

    class Inputs(DictModel):
        pass

    class Outputs(DictModel):
        pass

    inputs_default: DictModel = Field(default_factory=dict)
    outputs_default: DictModel = Field(default_factory=dict)

    @field_validator("inputs_default", "outputs_default", mode="before")
    @classmethod
    def _apply_defaults(cls, value: None | DictModel, info) -> DictModel:
        """Initialize inputs and outputs defaults with special Inputs and Outputs schemas."""
        schema = cls.Inputs if info.field_name == "inputs_default" else cls.Outputs
        return schema.model_validate(value)
    
    @field_serializer("inputs_default", "outputs_default")
    def _dump_defaults(self, value):
        return value.model_dump()

    def __call__(self, inputs: PyTree, outputs: PyTree) -> ArrayLike:
        return super().__call__(
            pytree_merge(self.inputs_default.model_dump(), inputs), 
            pytree_merge(self.outputs_default.model_dump(), outputs)
        )


class IdentityInputs(ForcingCallable):

    def callable(self, inputs, outputs):
        """Simple boundary that uses boundary input params directly (just pass them through)."""
        return inputs


class ConstantForcing(ForcingCallable):
    """Return a constant scalar, vector, or broadcastable field."""

    class Inputs(DictModel):
        # ``Any`` preserves YAML-friendly Python sequences such as ``[vx, vy]``;
        # the numerical path converts the value to a JAX array when evaluating it.
        const: ArrayLike | list[Any] | tuple[Any, ...] = 0.0

    def callable(self, inputs: Inputs, outputs: PyTree) -> ArrayLike:
        """Return the configured constant value.

        :param inputs: constant forcing parameters
        :param outputs: current model outputs, unused
        :return: constant scalar or field
        """
        del outputs
        return inputs["const"]


class GaussianForcing(ForcingCallable):
    """Return a two-dimensional Gaussian bump with an optional offset."""

    class Inputs(DictModel):
        A0: ArrayLike = 1.0
        offset: ArrayLike = 0.0
        sigma: ArrayLike = 1.0
        mu_x: ArrayLike = 0.0
        mu_y: ArrayLike = 0.0
        coords: Coordinates = (0.0, 0.0)

    def callable(self, inputs: Inputs, outputs: PyTree) -> ArrayLike:
        r"""Evaluate ``offset + A0 exp(-((x-mu_x)^2 + (y-mu_y)^2)/(2 sigma))``.

        :param inputs: Gaussian parameters and coordinates
        :param outputs: current model outputs, unused
        :return: Gaussian field
        """
        del outputs
        dx = inputs["coords"][0] - inputs["mu_x"]
        dy = inputs["coords"][1] - inputs["mu_y"]
        return inputs["offset"] + inputs["A0"] * jnp.exp(-(dx * dx + dy * dy) / (2 * inputs["sigma"]))


class SinusoidForcing(ForcingCallable):
    """Return the manufactured-solution sinusoidal forcing field."""

    class Inputs(DictModel):
        coords: Coordinates = (0.0, 0.0)

    def callable(self, inputs: Inputs, outputs: PyTree) -> ArrayLike:
        """Evaluate ``2 pi^2 sin(pi x) sin(pi y)``.

        :param inputs: coordinates
        :param outputs: current model outputs, unused
        :return: sinusoidal field
        """
        del outputs
        return 2 * jnp.pi**2 * jnp.sin(jnp.pi * inputs["coords"][0]) * jnp.sin(
            jnp.pi * inputs["coords"][1]
        )


FORCING_REGISTRY = {
    "identity": IdentityInputs,
    "constant": ConstantForcing,
    "gaussian": GaussianForcing,
    "sinusoid": SinusoidForcing,
}

type RegisteredForcing = Annotated[
    ForcingCallable,
    BeforeValidator(partial(from_registry, FORCING_REGISTRY)),
]


class BoundaryType(IntEnum):
    dirichlet = 1
    neumann = 2
    periodic = 3


class BoundarySpec(DictModel):
    """Specify the type and value of a single boundary.
    
    :ivar type: the type of boundary (periodic, dirichlet, or neumann)
    :ivar value: the value of the boundary (periodic~empty, dirichlet~const, neumann~gradient)
    """
    type: BoundaryType
    value: ArrayLike

    @field_validator('type', mode='before')
    @classmethod
    def _coerce_boundary_type(cls, value: str | BoundaryType) -> BoundaryType:
        dct = {i.name: i.value for i in BoundaryType}
        if isinstance(value, str) and value in dct:
            return dct[value]

        return value


class GridBoundaryInputs(DictModel):
    """Periodic, neumann, or dirichlet boundaries on uniform grid.
    Each tuple is the left/right boundary conditions for a given dimension.
    """
    boundary: list[tuple[BoundarySpec, BoundarySpec]]

    @staticmethod
    def _hashable_value(value: Any) -> Any:
        """Convert nested boundary data into a deterministic hashable structure."""
        if isinstance(value, DictModel):
            value = value.model_dump(mode="python")

        if isinstance(value, dict):
            return tuple(
                sorted((key, GridBoundaryInputs._hashable_value(item)) for key, item in value.items())
            )

        if isinstance(value, list | tuple):
            return tuple(GridBoundaryInputs._hashable_value(item) for item in value)

        if isinstance(value, np.ndarray | jax.Array):
            array = np.asarray(value)
            return ("array", array.dtype.str, array.shape, array.tobytes())

        return value

    def __hash__(self) -> int:
        return hash(self._hashable_value(self.model_dump(mode="python")))

    def __eq__(self, other) -> bool:
        if isinstance(other, GridBoundaryInputs):
            self_value = self._hashable_value(self.model_dump(mode="python"))
            other_value = self._hashable_value(other.model_dump(mode="python"))
            return self_value == other_value
        return False

    @model_validator(mode='after')
    def _check_periodic(self) -> 'GridBoundaryInputs':
        """Make sure both sides are periodic for any dimension with at least one periodic."""
        for left_b, right_b in self.boundary:
            if left_b.type == BoundaryType.periodic:
                if right_b.type != BoundaryType.periodic:
                    raise ValueError("Must use matching periodic boundaries")
            
            if right_b.type == BoundaryType.periodic:
                if left_b.type != BoundaryType.periodic:
                    raise ValueError("Must use matching periodic boundaries")
        
        return self


def homogeneous_boundary(type: str | BoundaryType = 'dirichlet', 
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
        boundary=[(BoundarySpec(type=type, value=value), BoundarySpec(type=type, value=value)) for _ in range(ndim)]
    )


class UniformGrid(DictModel):
    """
    Uniformly-spaced Cartesian grid (cell-centered). Either provide coords or some consistent 
    combination of shape, spacing, and bounds. If coords is not specified, then you must have
    bounds and only one of shape or spacing. Everything else gets filled in automatically.
    Use matrix 'ij' notation for meshgrid.
    
    :ivar shape: (Nx, ...) the grid shape
    :ivar spacing: (dx, ...) uniform spacing on the grid
    :ivar bounds: (xbounds, ...) the bounds in each dimension
    :ivar coords: (xgrid, ...) with each the same shape as the grid,
                  if 1D grids are passed, will be meshed to ND.
    """

    model_config = ConfigDict(validate_assignment=False)

    shape: tuple[PositiveInt, ...] | None = None
    spacing: tuple[PositiveFloat, ...] | None = None
    bounds: tuple[tuple[float, float], ...] | None = None
    coords: Coordinates | None = Field(default=None, exclude=True)  # don't serialize

    @model_validator(mode='after')
    def _coerce_grid(self) -> 'UniformGrid':
        """Ultimately, we need coords to be defined. Also check everything is consistent."""
        def _as_numpy(value: Any) -> np.ndarray:
            return np.asarray(value)

        spacing_provided = self.spacing is not None and len(self.spacing) > 0
        shape_provided = self.shape is not None and len(self.shape) > 0
        if self.coords is None:
            if self.bounds is None:
                raise ValueError("Can't construct grid without bounds.")

            bounds = tuple(tuple(float(v) for v in bound) for bound in self.bounds)
            lengths = tuple(b[1] - b[0] for b in bounds)

            if any(L <= 0 for L in lengths):
                raise ValueError("Grid bounds must be ordered as (lower, upper).")

            # Try to construct from spacing and shape
            if not shape_provided and not spacing_provided:
                raise ValueError("Can't construct grid without either spacing or shape.")

            if shape_provided and spacing_provided:
                expected_spacing = tuple(L/Nl for L, Nl in zip(lengths, self.shape))
                spacing_checks = np.array(
                    [np.allclose(s1, s2, atol=1e-6, rtol=1e-6) for s1, s2 in zip(expected_spacing, self.spacing)]
                )
                if not bool(np.all(spacing_checks)):
                    raise ValueError("Specified spacing is not consistent with bounds and shape.")
                
            if not shape_provided:
                inferred_shape = tuple(int(np.rint(L / dl)) for L, dl in zip(lengths, self.spacing))
                if not np.allclose(
                    tuple(L / dl for L, dl in zip(lengths, self.spacing)),
                    inferred_shape,
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise ValueError("Specified spacing is not consistent with bounds and an integer grid shape.")
                self.shape = inferred_shape

            if not spacing_provided:
                self.spacing = tuple(L/Nl for L, Nl in zip(lengths, self.shape))

            grids = [
                np.linspace(b[0] + dl / 2, b[1] - dl / 2, Nl)
                for b, dl, Nl in zip(bounds, self.spacing, self.shape)
            ]
            self.coords = tuple(np.asarray(arr) for arr in np.meshgrid(*grids, indexing='ij'))
        
        else:
            coords = tuple(_as_numpy(arr) for arr in self.coords)

            if coords[0].ndim == 1:
                if not all(arr.ndim == 1 for arr in coords):
                    raise ValueError("Must have all 1d coord arrays or all N-dim")
                coords = tuple(np.asarray(arr) for arr in np.meshgrid(*coords, indexing='ij'))
            
            # Make sure shape, spacing, and bounds are consistent
            ndim = coords[0].ndim
            shape = coords[0].shape
            if not all(arr.ndim == ndim for arr in coords):
                raise ValueError("All arrays must have same ndim")
            if not all(arr.shape == shape for arr in coords):
                raise ValueError("All arrays must have same shape")
            if not len(coords) == ndim:
                raise ValueError("Must have exactly ndim coord arrays")

            bounds = tuple((float(np.min(arr)), float(np.max(arr))) for arr in coords)
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
                bounds_checks = np.array(
                    [
                        np.allclose(np.asarray(b1), np.asarray(b2), atol=1e-6, rtol=1e-6)
                        for b1, b2 in zip(edge_bounds, self.bounds)
                    ]
                )
                if not bool(np.all(bounds_checks)):
                    raise ValueError("Specified bounds are not consistent with provided coords")
            
            if self.spacing is None:
                self.spacing = spacing
            else:
                spacing_checks = np.array(
                    [np.allclose(s1, s2, atol=1e-6, rtol=1e-6) for s1, s2 in zip(spacing, self.spacing)]
                )
                if not bool(np.all(spacing_checks)):
                    raise ValueError("Specified spacings are not consistent with provided coords")
            self.coords = tuple(np.asarray(arr) for arr in coords)
            
        return self


type AbstractIterativeSolver = Annotated[
    ThirdPartyType(default_modules="optimistix"), 
    AfterValidator(partial(require_type, optx.AbstractIterativeSolver))
]
type AbstractAdjoint = Annotated[
    ThirdPartyType(default_modules="optimistix"), 
    AfterValidator(partial(require_type, optx.AbstractAdjoint))
]

type DiffraxObject = ThirdPartyType(default_modules="diffrax")


class IterativeSolver(DictModel):
    """Configuration for optimistix iterative solvers. Only root find supported.
    
    :ivar solver: Optimistix nonlinear root finding solver (name+kwargs or instance), default is Newton
    :ivar options: runtime options for the nonlinear solver
    :ivar max_steps: maximum number of solver steps
    :ivar adjoint: Optimistix adjoint method
    :ivar throw: whether to throw failures as errors (default True)
    """
    solver: AbstractIterativeSolver = Field(
        default_factory=lambda: dict(name='optimistix.Newton', kwargs={'rtol': 1e-2, 'atol': 1e-4}), 
        validate_default=True
    )
    options: dict[str, Any] = Field(default_factory=dict)
    max_steps: PositiveInt = 100
    adjoint: AbstractAdjoint = Field(
        default_factory=lambda: dict(name='optimistix.ImplicitAdjoint'), 
        validate_default=True
    )
    throw: bool = False

    def root_find(
        self,
        fn: Callable[[ArrayLike, Any], ArrayLike], 
        y0: ArrayLike,
        args: Any | None = None,
        return_sol: bool = False
    ) -> ArrayLike | optx.Solution:
        """Small wrapper around optimistix root find.

        See Optimistix docs for `root_find()` method.
        
        :param fn: the objective function to find the root of, callable as `fn(y_k, Any) -> y_(k+1)`
        :param y0: the initial guess
        :param args: extra arguments for the objective function
        :param return_sol: whether to return the solution object or just the result (default)
        :return: the solution object or the result
        """
        solution = optx.root_find(
            fn,
            solver=self.solver,
            y0=y0,
            args=args,
            options=self.options,
            max_steps=self.max_steps,
            adjoint=self.adjoint,
            throw=self.throw
        )
        return solution if return_sol else solution.value


class DiffraxSolver(DictModel):
    """Configuration wrapper for :mod:`diffrax` ODE solves.

    The `ts` and `num_save` options are for convenience. You can also manually specify any `saveat` config.

    :param solver: diffrax solver instance or module spec
    :param stepsize_controller: diffrax controller
    :param adjoint: diffrax adjoint, default ``RecursiveCheckpointAdjoint``
    :param progress_meter: for showing solution progress
    :param saveat: optional explicit ``diffrax.SaveAt`` object
    :param t0: initial integration time
    :param t1: final integration time
    :param dt0: initial step size. If omitted, Vlasov computes a CFL-limited value.
    :param ts: saved times. If omitted, ``num_save`` evenly spaced times are used.
    :param num_save: number of evenly spaced saved times when ``ts`` is omitted
    :param max_steps: maximum diffrax internal steps
    :param throw: whether diffrax should raise on solver failure
    """

    solver: DiffraxObject = Field(default_factory=lambda: {"name": "Tsit5"}, validate_default=True)
    stepsize_controller: DiffraxObject = Field(
        default_factory=lambda: {"name": "ConstantStepSize"},
        validate_default=True,
    )
    adjoint: DiffraxObject = Field(
        default_factory=lambda: {"name": "RecursiveCheckpointAdjoint"},
        validate_default=True,
    )
    progress_meter: DiffraxObject = Field(
        default_factory=lambda: {"name": "NoProgressMeter"},
        validate_default=True,
    )
    saveat: DiffraxObject | None = None
    t0: float = 0.0
    t1: float = 1.0
    dt0: PositiveFloat | None = None
    ts: tuple[float, ...] | None = None
    num_save: PositiveInt = 2
    max_steps: PositiveInt = 4096
    throw: bool = True

    @field_validator("ts", mode="before")
    @classmethod
    def _coerce_ts(cls, value: Any) -> tuple[float, ...] | None:
        """Coerce saved times to a serializable tuple."""
        if value is None:
            return None
        return tuple(float(t) for t in value)

    def save_times(self) -> jax.Array:
        """Return the saved-time grid used by ``evaluate`` and default ``SaveAt``.

        :return: one-dimensional JAX array of saved times
        """
        if self.saveat is not None:
            saveat_times = self._saveat_times(self.saveat)
            if saveat_times is not None:
                return saveat_times
        if self.ts is not None:
            return jnp.asarray(self.ts)
        return jnp.linspace(self.t0, self.t1, self.num_save)

    def _saveat_times(self, saveat: diffrax.SaveAt) -> jax.Array | None:
        """Extract statically configured saved times from a ``diffrax.SaveAt`` object.

        ``SaveAt(steps=True)`` and ``SaveAt(dense=True)`` do not define a compact
        fixed time grid ahead of the solve, so those cases intentionally fall back
        to ``ts``/``num_save``.
        """

        def _subsaveat_times(subsaveat: Any) -> list[jax.Array]:
            if isinstance(subsaveat, dict):
                return [part for value in subsaveat.values() for part in _subsaveat_times(value)]
            if isinstance(subsaveat, tuple | list):
                return [part for value in subsaveat for part in _subsaveat_times(value)]
            if not hasattr(subsaveat, "ts"):
                return []

            parts = []
            if bool(getattr(subsaveat, "t0", False)):
                parts.append(jnp.asarray([self.t0]))
            if (ts := getattr(subsaveat, "ts", None)) is not None:
                parts.append(jnp.ravel(jnp.asarray(ts)))
            if bool(getattr(subsaveat, "t1", False)):
                parts.append(jnp.asarray([self.t1]))
            return parts

        parts = _subsaveat_times(saveat.subs)
        if not parts:
            return None
        return jnp.unique(jnp.concatenate(parts))

    def save_at(self) -> diffrax.SaveAt:
        """Return the diffrax save configuration.

        :return: configured or default ``diffrax.SaveAt``
        """
        if self.saveat is not None:
            return self.saveat
        return diffrax.SaveAt(ts=self.save_times())
    
    def diffeqsolve(
        self,
        terms: PyTree,
        y0: PyTree,
        args: Any | None = None,
        dt0: float | None = None,
        **kwargs
    ) -> PyTree | diffrax.Solution:
        """Small wrapper around diffrax diffeqsolve.

        See Diffrax docs for `diffeqsolve()` method.
        
        :param terms: the ODE terms
        :param y0: the initial conditions
        :param args: extra arguments for the ode terms
        :param dt0: the initial time step (overrides default config)
        :param kwargs: everthing else passed directly to diffeqsolve (basically just event and solver/controller state)
        :return: the solution object or the result
        """
        solution = diffrax.diffeqsolve(
            terms,
            solver=self.solver,
            t0=float(self.t0),
            t1=float(self.t1),
            dt0=self.dt0 if dt0 is None else dt0,
            y0=y0,
            args=args,
            saveat=self.save_at(),
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            max_steps=self.max_steps,
            throw=self.throw,
            progress_meter=self.progress_meter,
            **kwargs
        )
        return solution 


class _AliveProgressMeterState(eqx.Module):
    """Internal JAX-compatible state for :class:`AliveProgressMeter`."""

    progress: jax.Array
    meter_idx: Any


class AliveProgressMeter(diffrax.AbstractProgressMeter[_AliveProgressMeterState]):
    """Progress meter for ``diffrax`` solves backed by :func:`alive_progress.alive_bar`."""

    minimum_increase: float = 0.02

    @staticmethod
    def _init_bar() -> list[Any]:
        """Initialise and enter an ``alive_bar`` context."""
        ctx = alive_bar(1, manual=True)
        bar = ctx.__enter__()
        bar(0.0)
        return [ctx, bar, 0.0]

    @staticmethod
    def _step_bar(bar_state: list[Any], progress: jax.Array | np.ndarray | float) -> None:
        """Advance the underlying ``alive_bar`` to the supplied solve progress."""
        if eqx.is_array(progress):
            # May not be an array when called with `JAX_DISABLE_JIT=1`
            progress = cast(jax.Array | np.ndarray, progress)
            progress = cast(float, progress.item())
        else:
            progress = cast(float, progress)
        bar_state[2] = progress
        bar_state[1](progress)

    @staticmethod
    def _close_bar(bar_state: list[Any]) -> None:
        """Close the underlying ``alive_bar`` context."""
        if bar_state[2] != 1.0:
            bar_state[1](1.0)
        bar_state[0].__exit__(None, None, None)

    def init(self) -> _AliveProgressMeterState:
        """Initialise the progress meter state."""
        meter_idx = _progress_meter_manager.init(self._init_bar)
        return _AliveProgressMeterState(progress=jnp.array(0.0), meter_idx=meter_idx)

    def step(
        self,
        state: _AliveProgressMeterState,
        progress: jax.Array | np.ndarray | float,
    ) -> _AliveProgressMeterState:
        """Advance the progress bar to the supplied solve progress."""
        pred = eqxi.unvmap_all((progress - state.progress > self.minimum_increase) | (progress == 1))

        next_progress, meter_idx = jax.lax.cond(
            eqxi.nonbatchable(pred),
            lambda _idx: (
                progress,
                _progress_meter_manager.step(self._step_bar, progress, _idx),
            ),
            lambda _idx: (state.progress, _idx),
            state.meter_idx,
        )

        return _AliveProgressMeterState(progress=next_progress, meter_idx=meter_idx)

    def close(self, state: _AliveProgressMeterState) -> None:
        """Close the underlying ``alive_bar`` context."""
        _progress_meter_manager.close(self._close_bar, state.meter_idx)
    

def _default_latent_sampler(
    compression: Compression, 
    *, 
    path: TreePath = ("outputs",),
    distribution: Literal["uniform", "normal"] = "normal",
) -> SamplerCallable:
    """Build a uniform or normal latent sampler under the requested pytree path."""
    minval, maxval = compression.latent_bounds()
    latent_normal = compression.latent_normal()
    latent_size = compression.latent_size()

    if distribution == "uniform":
        if minval is None or maxval is None:
            raise ValueError("Uniform latent sampling requires compression latent bounds.")

        sampler = {
            "callable": "uniform",
            "shape": [latent_size],
            "minval": jnp.asarray(minval).tolist(),
            "maxval": jnp.asarray(maxval).tolist(),
        }     
    
    elif distribution == "normal":
        if latent_normal is None:
            raise ValueError("Normal latent sampling requires compression (mean, std)")
        
        mean, std = latent_normal
        sampler = {
            "callable": "normal",
            "shape": [latent_size],
            "mean": jnp.asarray(mean).tolist(),
            "std": jnp.asarray(std).tolist(),
        }
    
    else:
        raise ValueError(f"Latent sampler distribution '{distribution}' not recognized.")
    
    template = set_subtree(None, path, sampler)
    return PyTreeSampler(**template)


class LatentSamplerFactory(CallableModel):
    """Factory for building a source sampler from latent size and latent bounds."""

    callable: Callable[[Compression], SamplerCallable] = _default_latent_sampler


class ImplicitAffine(ImplicitModel, ImplicitSampleable):
    """Invertible input-conditioned affine residual model in latent space.

    $f(u;b) = A(b)u + c(b)$ for inputs $b$ and outputs $u$.

    Inputs are a mapping with latent ``values`` and a runtime
    :class:`~romjax.nn.Affine` under ``call_args``. Compression artifacts
    provide ranks and default latent samplers without becoming part of the
    numerical JAX path.
    """

    solver: LinearSolver = Field(default_factory=lambda: lx.AutoLinearSolver(well_posed=True))
    inputs_rank: PositiveInt | None = None
    outputs_rank: PositiveInt | None = None
    inputs_compression: Path | str | Compression | None = None
    outputs_compression: Path | str | Compression | None = None
    inputs_sampler: LatentSamplerFactory | SamplerCallable | None = Field(
        default_factory=lambda: LatentSamplerFactory(
            callable=partial(_default_latent_sampler, path=("values",))
        )
    )
    conditions_sampler: SamplerCallable | None = None
    outputs_sampler: LatentSamplerFactory | SamplerCallable | None = Field(default_factory=LatentSamplerFactory)
    _resolved_inputs_compression: Compression | None = PrivateAttr(default=None)
    _resolved_outputs_compression: Compression | None = PrivateAttr(default=None)
    _resolved_inputs_sampler: SamplerCallable | None = PrivateAttr(default=None)
    _resolved_outputs_sampler: SamplerCallable | None = PrivateAttr(default=None)

    def _resolve_compression(self, artifact: Path | str | Compression | None, cache_name: str) -> Compression | None:
        cached = getattr(self, cache_name)
        if cached is not None:
            return cached
        if isinstance(artifact, Compression):
            object.__setattr__(self, cache_name, artifact)
            return artifact
        if isinstance(artifact, str | Path) and Path(artifact).exists():
            compression = Compression.load(Path(artifact))
            object.__setattr__(self, cache_name, compression)
            return compression
        return None

    def resolve_inputs_compression(self) -> Compression | None:
        """Resolve the input compression artifact."""
        return self._resolve_compression(self.inputs_compression, "_resolved_inputs_compression")

    def resolve_outputs_compression(self) -> Compression | None:
        """Resolve the output compression artifact."""
        return self._resolve_compression(self.outputs_compression, "_resolved_outputs_compression")

    def resolve_inputs_rank(self) -> int | None:
        """Resolve the input rank from explicit configuration or compression."""
        if self.inputs_rank is not None:
            return int(self.inputs_rank)
        compression = self.resolve_inputs_compression()
        return None if compression is None or compression.latent_size() is None else int(compression.latent_size())

    def resolve_outputs_rank(self) -> int | None:
        """Resolve the output rank from explicit configuration or compression."""
        if self.outputs_rank is not None:
            return int(self.outputs_rank)
        compression = self.resolve_outputs_compression()
        return None if compression is None or compression.latent_size() is None else int(compression.latent_size())

    def _resolve_sampler(
        self,
        sampler: LatentSamplerFactory | SamplerCallable | None,
        compression: Compression | None,
        cache_name: str,
    ) -> SamplerCallable | None:
        cached = getattr(self, cache_name)
        if cached is not None:
            return cached
        if isinstance(sampler, SamplerCallable):
            object.__setattr__(self, cache_name, sampler)
            return sampler
        if sampler is not None and compression is not None:
            resolved = sampler(compression)
            object.__setattr__(self, cache_name, resolved)
            return resolved
        return None

    def _affine_inputs(self, inputs: PyTree) -> tuple[ArrayLike, Affine]:
        """Extract latent inputs and the runtime affine module from the input PyTree."""
        if not isinstance(inputs, Mapping) or "values" not in inputs or "call_args" not in inputs:
            raise TypeError("ImplicitAffine inputs must contain 'values' and runtime 'call_args'.")
        return jnp.asarray(inputs["values"]), self._affine(inputs["call_args"])

    def _affine(self, call_args: Any) -> Affine:
        if isinstance(call_args, Affine):
            return call_args
        if isinstance(call_args, Mapping) and isinstance(call_args.get("affine"), Affine):
            return call_args["affine"]
        raise TypeError("ImplicitAffine requires an Affine instance in edge payload call_args.")

    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate ``A(inputs) @ outputs + c(inputs)``.

        :param inputs: input payload containing latent ``values`` and ``call_args``
        :param outputs: output latent vector
        :return: residual latent vector
        """
        values, affine = self._affine_inputs(inputs)
        matrix, offset = affine.materialize(values)
        return matrix @ jnp.asarray(outputs) + offset

    def solve(self, inputs: PyTree, residuals: PyTree) -> PyTree:
        """Solve the affine residual equation for output latent coordinates."""
        values, affine = self._affine_inputs(inputs)
        matrix, offset = affine.materialize(values)
        return lx.linear_solve(
            lx.MatrixLinearOperator(matrix),
            jnp.asarray(residuals) - offset,
            solver=self.solver,
        ).value

    def sample_inputs(self, key: Key) -> PyTree:
        """Sample an input latent vector."""
        sampler = self._resolve_sampler(
            self.inputs_sampler, self.resolve_inputs_compression(), "_resolved_inputs_sampler"
        )
        if sampler is None:
            raise ValueError("ImplicitAffine input sampler could not be resolved.")
        return sampler.sample(key) if hasattr(sampler, "sample") else sampler(key)

    def sample_conditions(self, key: Key) -> PyTree | None:
        """Produce one optional output-condition sample for the given key."""
        if self.conditions_sampler is not None:
            return self.conditions_sampler(key)
        return None

    def sample_outputs(
        self,
        key: Key,
        inputs: PyTree | None = None,
        solution: PyTree | None = None,
        conditions: PyTree | None = None,
    ) -> PyTree:
        """Sample an output latent vector, optionally conditioned on a solution."""
        sampler = self._resolve_sampler(
            self.outputs_sampler, self.resolve_outputs_compression(), "_resolved_outputs_sampler"
        )
        if sampler is None:
            raise ValueError("ImplicitAffine output sampler could not be resolved.")
        sampler_kwargs = {"inputs": inputs, "solution": solution}
        if conditions is not None:
            sampler_kwargs["conditions"] = conditions
        if isinstance(sampler, SolverSampler):
            sampler_kwargs["solve"] = self.solve
        return sampler(key, **sampler_kwargs)


class ImplicitIterativeGalerkin(CompositeEdge, SourceSampleable):
    """Galerkin ROM that solves any `ImplicitModel` via an iterative solver in latent space."""

    solver: IterativeSolver = Field(default_factory=IterativeSolver)
    initial: RegisteredForcing = Field(default_factory=ConstantForcing)
    source_sampler: LatentSamplerFactory | SamplerCallable | None = Field(default_factory=LatentSamplerFactory)
    rank: PositiveInt | None = None
    compression: Path | str | Compression | None = None
    _resolved_source_sampler: SamplerCallable | None = PrivateAttr(default=None)
    _resolved_compression: Compression | None = PrivateAttr(default=None)

    def resolve_compression(self) -> Compression | None:
        """Resolve the compression artifact from a preloaded object or a file path."""
        if self._resolved_compression is not None:
            return self._resolved_compression
        
        artifact = self.compression
        if isinstance(artifact, Compression):
            object.__setattr__(self, "_resolved_compression", artifact)
            return artifact
        
        if isinstance(artifact, (str, Path)):
            artifact_path = Path(artifact)
            if artifact_path.exists():
                compression = Compression.load(artifact_path)
                object.__setattr__(self, "_resolved_compression", compression)
                return compression
            
        return None

    def resolve_rank(self) -> int | None:
        """Resolve the rank from explicit configuration or compression."""
        if self.rank is not None:
            return int(self.rank)
        compression = self.resolve_compression()
        rank = None if compression is None else compression.latent_size()
        return None if rank is None else int(rank)

    def resolve_source_sampler(self) -> SamplerCallable | None:
        """Resolve the source sampler from explicit configuration or a compression artifact."""
        if self._resolved_source_sampler is not None:
            return self._resolved_source_sampler

        sampler = self.source_sampler
        if isinstance(sampler, SamplerCallable):
            object.__setattr__(self, "_resolved_source_sampler", sampler)
            return sampler

        compression = self.resolve_compression()
        if compression is None:
            return None

        if sampler is not None:
            sampler = sampler(compression)
            object.__setattr__(self, "_resolved_source_sampler", sampler)
            return sampler

        return None

    # Override default composite edge behavior by solving in latent space directly
    def backward_aux(
        self, 
        x: PyTree, 
        aux: PyTree | None = None,
        edge_payload_patches: EdgePatch | None = None, 
        composite_stack: tuple[str, ...] = ()
    ) -> tuple[PyTree, PyTree | None]:
        """Solve in latent space (with optional aux data)."""
        
        def residual_fn(z: ArrayLike, args: PyTree, aux, edge_payload_patches, composite_stack) -> ArrayLike:
            """Root find residual function, with `z` as the latent coordinates."""
            payload = {"outputs": z}
            if (inputs := args.get("inputs", None)) is not None:
                payload["inputs"] = inputs

            result, aux = self.forward_aux(payload, aux, edge_payload_patches, composite_stack)

            return result["residuals"] - args["residuals"]
        
        initial_inputs = {}
        if isinstance(x.get("inputs"), Mapping):
            initial_inputs = x["inputs"].get("initial", {})

        if isinstance(initial_inputs, Mapping) and "outputs" in initial_inputs:
            initial = jnp.asarray(initial_inputs["outputs"])
        else:
            initial = jnp.asarray(self.initial(initial_inputs, {}))
        initial = jnp.broadcast_to(initial, jnp.asarray(x["residuals"]).shape)

        solution = self.solver.root_find(
            lambda z, args: residual_fn(z, args, aux, edge_payload_patches, composite_stack), 
            initial,
            x, 
            return_sol=False
        )

        ret = {"outputs": solution}

        # Pass inputs through
        if (inputs := x.get("inputs", None)) is not None:
            ret["inputs"] = inputs

        return ret, aux

    def sample_source(self, key: Key) -> PyTree:
        sampler = self._resolved_source_sampler
        if sampler is not None:
            if hasattr(sampler, "sample"):
                return sampler.sample(key)
            return sampler(key)
        else:
            raise ValueError("Source sampler has not been resolved yet.")
