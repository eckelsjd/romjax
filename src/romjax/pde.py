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
import jax.tree_util as jtu
import lineax as lx
import numpy as np
import optimistix as optx
from alive_progress import alive_bar
from diffrax._progress_meter import _progress_meter_manager
from equinox.internal import ω
from jaxtyping import ArrayLike, Key, PyTree
from optimistix._solver.newton_chord import _AbstractNewtonChord
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
from romjax.rng import Distribution, DistributionPyTree, PyTreeSampler, SamplerCallable, validate_distribution_pytree
from romjax.tree import TreePath, coerce_tree_paths, get_subtree, pytree_merge, set_subtree
from romjax.typing import CallableModel, DictModel, ThirdPartyType, from_registry, require_type

__all__ = ['Coordinates', 'BoundaryType', 'BoundarySpec', 'GridBoundaryInputs', 'homogeneous_boundary', 'UniformGrid',
           'ForcingCallable', 'RegisteredForcing', 'FORCING_REGISTRY', 'IdentityInputs', 'ConstantForcing',
           'GaussianForcing', 'SinusoidForcing', 'SumForcing', 'RandomNewton', 'IterativeSolver', 'LinearSolver',
           'LatentSamplerFactory', 'ImplicitAffine', 'ImplicitIterativeGalerkin', 'DiffraxSolver', 'AliveProgressMeter']


type Coordinates = tuple[ArrayLike, ...] | ArrayLike
type AbstractLinearSolver = Annotated[
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
    """Return a configurable combination of two-dimensional sinusoidal modes."""

    class Inputs(DictModel):
        """Inputs for the sinusoidal forcing function.

        :ivar a: coefficient of ``sin(pi x) sin(pi y)``
        :ivar b: coefficient of ``sin(2 pi x) sin(pi y)``
        :ivar c: coefficient of ``sin(pi x) sin(2 pi y)``
        :ivar coords: spatial coordinates
        """

        a: ArrayLike = 2.0 * jnp.pi**2
        b: ArrayLike = 0.0
        c: ArrayLike = 0.0
        coords: Coordinates = (0.0, 0.0)

    def callable(self, inputs: Inputs, outputs: PyTree) -> ArrayLike:
        r"""Evaluate the configurable sinusoidal forcing field.

        .. math::

            f(x, y) = a\sin(\pi x)\sin(\pi y)
                + b\sin(2\pi x)\sin(\pi y)
                + c\sin(\pi x)\sin(2\pi y).

        :param inputs: sinusoidal coefficients and coordinates
        :param outputs: current model outputs, unused
        :return: sinusoidal field
        """
        del outputs
        x, y = (jnp.asarray(coord) for coord in inputs["coords"])
        return (
            inputs["a"] * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
            + inputs["b"] * jnp.sin(2.0 * jnp.pi * x) * jnp.sin(jnp.pi * y)
            + inputs["c"] * jnp.sin(jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y)
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


class SumForcing(ForcingCallable):
    """Evaluate and add a sequence of independently configured forcings.

    Each nested forcing receives the same runtime ``inputs`` and ``outputs``.
    Its own ``inputs_default`` and ``outputs_default`` are applied by its
    :class:`ForcingCallable` implementation before evaluation.
    """

    forcings: tuple[RegisteredForcing, ...] = Field(min_length=1)

    def callable(self, inputs: PyTree, outputs: PyTree) -> ArrayLike:
        """Return the sum of all nested forcing evaluations.

        :param inputs: runtime inputs shared by all nested forcings
        :param outputs: runtime outputs shared by all nested forcings
        :return: sum of the nested forcing values
        """
        return sum((forcing(inputs, outputs) for forcing in self.forcings), jnp.asarray(0.0))


FORCING_REGISTRY["sum"] = SumForcing


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


class RandomNewton(_AbstractNewtonChord):
    """Newton root finder with reproducible stochastic update perturbations.

    Distribution definitions are static solver configuration, while ``step_seed``
    and the relative scales may be supplied through Optimistix ``options`` at solve
    time. This keeps random trajectories configurable through JAX values without
    rebuilding the solver object.

    :param rtol: relative termination tolerance
    :param atol: absolute termination tolerance
    :param step_size: optional distribution for the multiplicative Newton step size
    :param step_direction: optional distribution for an additive update direction
    :param step_direction_scale: optional direction magnitude relative to the Newton update
    :param final_perturb: optional distribution added to the terminated iterate
    :param final_perturb_scale: optional final perturbation magnitude relative to the final iterate
    :param step_seed: default seed, overridden by ``options["step_seed"]``
    """

    step_size: DistributionPyTree | None = eqx.field(static=True, default=None)
    step_direction: DistributionPyTree | None = eqx.field(static=True, default=None)
    step_direction_scale: PyTree | None = eqx.field(static=True, default=None)
    final_perturb: DistributionPyTree | None = eqx.field(static=True, default=None)
    final_perturb_scale: PyTree | None = eqx.field(static=True, default=None)
    step_seed: int = eqx.field(static=True, default=0)

    _is_newton = True

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], ArrayLike] = optx.max_norm,
        kappa: float = 1e-2,
        linear_solver: lx.AbstractLinearSolver | None = None,
        cauchy_termination: bool = True,
        step_seed: int = 0,
        step_size: DistributionPyTree | None = None,
        step_direction: DistributionPyTree | None = None,
        step_direction_scale: PyTree | None = None,
        final_perturb: DistributionPyTree | None = None,
        final_perturb_scale: PyTree | None = None,
    ) -> None:
        """Initialize the deterministic Newton settings and random distributions."""
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.kappa = kappa
        self.linear_solver = lx.AutoLinearSolver(well_posed=None) if linear_solver is None else linear_solver
        self.cauchy_termination = cauchy_termination
        self.step_seed = step_seed
        self.step_size = self._coerce_distribution(step_size)
        self.step_direction = self._coerce_distribution(step_direction)
        self.step_direction_scale = step_direction_scale
        self.final_perturb = self._coerce_distribution(final_perturb)
        self.final_perturb_scale = final_perturb_scale

    @staticmethod
    def _coerce_distribution(value: DistributionPyTree | None) -> DistributionPyTree | None:
        """Validate a scalar distribution or a distribution pytree specification."""
        return None if value is None else validate_distribution_pytree(value)

    def _keys(self, options: dict[str, Any]) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return independent random streams for update size, direction, and final noise."""
        seed = jnp.asarray(options.get("step_seed", self.step_seed), dtype=jnp.uint32)
        return tuple(jax.random.split(jax.random.key(seed), 3))

    @staticmethod
    def _is_tree_spec(value: Any) -> bool:
        """Return whether a value is an explicit container rather than one broadcast scale."""
        return isinstance(value, Mapping | tuple | list)

    @staticmethod
    def _broadcast_tree(template: PyTree, reference: PyTree) -> PyTree:
        """Broadcast a scalar distribution or scale specification over a state pytree."""
        if not RandomNewton._is_tree_spec(template) or isinstance(template, Distribution):
            return jax.tree.map(lambda _: template, reference)
        try:
            return jax.tree.map(lambda value, _: value, template, reference)
        except (TypeError, ValueError) as exc:
            raise ValueError("RandomNewton configuration does not match the state pytree structure.") from exc

    @staticmethod
    def _sample_tree(template: DistributionPyTree, key: jax.Array) -> PyTree:
        """Sample a validated distribution pytree with independent keys per leaf."""
        leaves, treedef = jax.tree.flatten(template, is_leaf=lambda value: isinstance(value, Distribution))
        if not all(isinstance(value, Distribution) for value in leaves):
            raise TypeError("RandomNewton distributions must contain only Distribution leaves.")
        keys = jax.random.split(key, len(leaves))
        return jax.tree.unflatten(treedef, [value.sample(subkey) for value, subkey in zip(leaves, keys)])

    def _distribution_for_state(self, distribution: DistributionPyTree, state: PyTree) -> DistributionPyTree:
        """Expand a distribution specification to match a state pytree and validate leaf broadcasting."""
        template = self._broadcast_tree(distribution, state)
        sampled = self._sample_tree(template, jax.random.key(0))
        try:
            jax.tree.map(
                lambda sample, value: jnp.broadcast_shapes(jnp.shape(sample), jnp.shape(value)), sampled, state
            )
        except (TypeError, ValueError) as exc:
            message = "RandomNewton distribution samples must be broadcast-compatible with the state pytree."
            raise ValueError(message) from exc
        return template

    def _relative_noise(self, noise: PyTree, reference: PyTree, scale: PyTree | None) -> PyTree:
        """Scale noise globally or leafwise to a relative reference magnitude."""
        if scale is None:
            return noise
        if not self._is_tree_spec(scale):
            noise_norm = jnp.asarray(self.norm(noise))
            reference_norm = jnp.asarray(self.norm(reference))
            factor = jnp.where(noise_norm > 0, jnp.asarray(scale) * reference_norm / noise_norm, 0.0)
            return jax.tree.map(lambda value: value * factor, noise)

        scale_tree = self._broadcast_tree(scale, reference)

        def scale_leaf(noise_leaf: ArrayLike, reference_leaf: ArrayLike, scale_leaf: ArrayLike) -> jax.Array:
            noise_norm = jnp.asarray(self.norm(noise_leaf))
            reference_norm = jnp.asarray(self.norm(reference_leaf))
            factor = jnp.where(noise_norm > 0, jnp.asarray(scale_leaf) * reference_norm / noise_norm, 0.0)
            return jnp.asarray(noise_leaf) * factor

        return jax.tree.map(scale_leaf, noise, reference, scale_tree)

    def init(
        self,
        fn: Callable,
        y: PyTree,
        args: PyTree,
        options: dict[str, Any],
        f_struct: PyTree[jax.ShapeDtypeStruct],
        aux_struct: PyTree[jax.ShapeDtypeStruct],
        tags: frozenset[object],
    ) -> Any:
        """Initialize the parent Newton state after validating random distribution shapes."""
        for distribution in (self.step_size, self.step_direction, self.final_perturb):
            if distribution is not None:
                self._distribution_for_state(distribution, y)
        return super().init(fn, y, args, options, f_struct, aux_struct, tags)

    def step(
        self,
        fn: Callable,
        y: PyTree,
        args: PyTree,
        options: dict[str, Any],
        state: Any,
        tags: frozenset[object],
    ) -> tuple[PyTree, Any, Any]:
        """Take one Newton step with optional randomized magnitude and direction."""
        _, deterministic_state, aux = super().step(fn, y, args, options, state, tags)
        deterministic_diff = deterministic_state.diff
        size_key, direction_key, _ = self._keys(options)
        step = deterministic_state.step - 1

        alpha = jax.tree.map(lambda _: jnp.asarray(1.0), deterministic_diff)

        direction = jax.tree.map(jnp.zeros_like, deterministic_diff)
        if self.step_direction is not None:
            template = self._distribution_for_state(self.step_direction, deterministic_diff)
            sampled = self._sample_tree(template, jax.random.fold_in(direction_key, step))
            direction_scale = options.get("step_direction_scale", self.step_direction_scale)
            direction = self._relative_noise(sampled, deterministic_diff, direction_scale)

        if self.step_size is not None:
            template = self._distribution_for_state(self.step_size, deterministic_diff)
            alpha = self._sample_tree(template, jax.random.fold_in(size_key, step))
        random_diff = (alpha**ω * (deterministic_diff**ω + direction**ω)).ω
        new_y = (y**ω - random_diff**ω).ω
        lower = options.get("lower")
        upper = options.get("upper")
        if lower is not None:
            new_y = jtu.tree_map(lambda value, bound: jnp.clip(value, min=bound), new_y, lower)
        if upper is not None:
            new_y = jtu.tree_map(lambda value, bound: jnp.clip(value, max=bound), new_y, upper)
        random_diff = (y**ω - new_y**ω).ω

        scale = (self.atol + self.rtol * ω(new_y).call(jnp.abs)).ω
        with jax.numpy_dtype_promotion("standard"):
            diffsize = self.norm((random_diff**ω / scale**ω).ω)
        random_state = eqx.tree_at(
            lambda value: (value.diff, value.diffsize),
            deterministic_state,
            (random_diff, jnp.asarray(diffsize, dtype=deterministic_state.diffsize.dtype)),
        )
        return new_y, random_state, aux

    def postprocess(
        self,
        fn: Callable,
        y: ArrayLike,
        aux: Any,
        args: PyTree,
        options: dict[str, Any],
        state: Any,
        tags: frozenset[object],
        result: Any,
    ) -> tuple[PyTree, Any, dict[str, Any]]:
        """Optionally perturb the final iterate after Optimistix terminates."""
        del fn, args, tags, result
        if self.final_perturb is None:
            return y, aux, {}
        _, _, perturb_key = self._keys(options)
        template = self._distribution_for_state(self.final_perturb, y)
        sampled = self._sample_tree(template, jax.random.fold_in(perturb_key, state.step))
        perturb_scale = options.get("final_perturb_scale", self.final_perturb_scale)
        perturbation = self._relative_noise(sampled, y, perturb_scale)
        return (y**ω + perturbation**ω).ω, aux, {}


class IterativeSolver(DictModel):
    """Configuration for optimistix iterative solvers. Only root find supported.
    
    :ivar solver: Optimistix nonlinear root finding solver (name+kwargs or instance), default is Newton
    :ivar initial: configured initial guess forcing callable
    :ivar options: runtime options for the nonlinear solver
    :ivar max_steps: maximum number of solver steps
    :ivar adjoint: Optimistix adjoint method
    :ivar throw: whether to throw failures as errors (default True)
    """
    solver: AbstractIterativeSolver = Field(
        default_factory=lambda: dict(name='optimistix.Newton', kwargs={'rtol': 1e-2, 'atol': 1e-4}), 
        validate_default=True
    )
    initial: RegisteredForcing = Field(default_factory=ConstantForcing)
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
        options: Mapping[str, Any] | None = None,
        return_sol: bool = False
    ) -> ArrayLike | optx.Solution:
        """Small wrapper around optimistix root find.

        See Optimistix docs for `root_find()` method.
        
        :param fn: the objective function to find the root of, callable as `fn(y_k, Any) -> y_(k+1)`
        :param y0: the initial guess
        :param args: extra arguments for the objective function
        :param options: runtime solver options merged over configured options
        :param return_sol: whether to return the solution object or just the result (default)
        :return: the solution object or the result
        """
        runtime_options = pytree_merge(self.options, options or {})
        solution = optx.root_find(
            fn,
            solver=self.solver,
            y0=y0,
            args=args,
            options=runtime_options,
            max_steps=self.max_steps,
            adjoint=self.adjoint,
            throw=self.throw
        )
        return solution if return_sol else solution.value


class LinearSolver(DictModel):
    """Configuration wrapper for Lineax linear solves.

    :ivar solver: Lineax linear solver (name+kwargs or instance)
    :ivar initial: configured initial guess forcing callable, passed as ``options["y0"]``
    :ivar options: runtime options for the linear solver
    :ivar throw: whether Lineax should raise on solver failure
    """

    solver: AbstractLinearSolver = Field(
        default_factory=lambda: dict(name="lineax.AutoLinearSolver", kwargs={"well_posed": True}),
        validate_default=True,
    )
    initial: RegisteredForcing = Field(default_factory=ConstantForcing)
    options: dict[str, Any] = Field(default_factory=dict)
    throw: bool = False

    def linear_solve(
        self,
        operator: lx.AbstractLinearOperator,
        vector: PyTree,
        *,
        y0: PyTree | None = None,
        options: Mapping[str, Any] | None = None,
        state: PyTree | None = None,
        return_sol: bool = False,
    ) -> PyTree | lx.Solution:
        """Solve a linear system with configured Lineax settings.

        :param operator: linear operator in ``A @ y = b``
        :param vector: right-hand-side vector ``b``
        :param y0: optional initial guess forwarded as ``options["y0"]``
        :param options: runtime solver options merged over configured options
        :param state: optional reusable Lineax solver state
        :param return_sol: whether to return the Lineax solution object
        :return: the solved value or complete Lineax solution
        """
        runtime_options = pytree_merge(self.options, options or {})
        if y0 is not None:
            runtime_options["y0"] = y0
        kwargs: dict[str, Any] = {
            "solver": self.solver,
            "options": runtime_options,
            "throw": self.throw,
        }
        if state is not None:
            kwargs["state"] = state
        solution = lx.linear_solve(operator, vector, **kwargs)
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


class AffineInitial(ForcingCallable):
    """
    Return an initial root find guess u0 = g(b) + H^-1(b, g(b)) r, 
    associated with the Affine model F(b,u)=H(b,u)(u-g(b)).
    """

    solver: AbstractLinearSolver = Field(default_factory=lambda: lx.AutoLinearSolver(well_posed=True))

    def callable(self, inputs: PyTree, outputs: PyTree) -> ArrayLike:
        del outputs
        inputs, module, residuals = inputs["inputs"], inputs["module"], inputs["residuals"]
        matrix, solution = module.materialize(inputs, outputs="solution")

        u0 = lx.linear_solve(
            lx.MatrixLinearOperator(matrix),
            residuals,
            solver=self.solver,
        ).value + solution

        return u0


FORCING_REGISTRY["affine"] = AffineInitial


class ImplicitAffine(ImplicitModel, ImplicitSampleable, SourceSampleable):
    """Invertible input-conditioned affine residual model.

    ``Affine`` supplies ``f(b, u) = H(b, u) @ (u - g(b))``. Inputs use the
    payload ``{"value": ..., "module": Affine(...)}``; outputs and residuals
    use ``{"value": ...}``.

    ``additional_inputs`` appends flattened array leaves from configured
    input subtrees to ``inputs["value"]``. ``None`` disables this behavior;
    an empty tuple collects all array leaves except the reserved ``value`` and
    ``module`` payloads.

    Implicit sampling is for sampling inputs/outputs in latent space.
    Source sampling is for sampling residuals in latent space.
    """

    solver: LinearSolver | IterativeSolver | None = None
    inputs_rank: PositiveInt | None = None
    outputs_rank: PositiveInt | None = None
    additional_inputs: tuple[TreePath, ...] | None = None
    inputs_compression: Path | str | Compression | None = None
    outputs_compression: Path | str | Compression | None = None
    residuals_compression: Path | str | Compression | None = None
    inputs_sampler: LatentSamplerFactory | SamplerCallable | None = Field(
        default_factory=lambda: LatentSamplerFactory(
            callable=partial(_default_latent_sampler, path=("value",))
        )
    )
    conditions_sampler: SamplerCallable | None = None
    outputs_sampler: LatentSamplerFactory | SamplerCallable | None = Field(
        default_factory=lambda: LatentSamplerFactory(
            callable=partial(_default_latent_sampler, path=("value",))
        )
    )
    residuals_sampler: LatentSamplerFactory | SamplerCallable | None = Field(
        default_factory=lambda: LatentSamplerFactory(
            callable=partial(_default_latent_sampler, path=("residuals", "value",))
        )
    )
    _resolved_inputs_compression: Compression | None = PrivateAttr(default=None)
    _resolved_outputs_compression: Compression | None = PrivateAttr(default=None)
    _resolved_residuals_compression: Compression | None = PrivateAttr(default=None)
    _resolved_inputs_sampler: SamplerCallable | None = PrivateAttr(default=None)
    _resolved_outputs_sampler: SamplerCallable | None = PrivateAttr(default=None)
    _resolved_residuals_sampler: SamplerCallable | None = PrivateAttr(default=None)

    @field_validator("additional_inputs", mode="before")
    @classmethod
    def _coerce_additional_inputs(cls, value: Any) -> Any:
        """Normalize configured paths while preserving ``()`` as the all-arrays sentinel."""
        if value is None or (isinstance(value, tuple | list) and len(value) == 0):
            return tuple(value)
        return tuple(coerce_tree_paths(value))

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

    def resolve_inputs_compression(self) -> Compression | None:
        """Resolve the input compression artifact."""
        return self._resolve_compression(self.inputs_compression, "_resolved_inputs_compression")

    def resolve_outputs_compression(self) -> Compression | None:
        """Resolve the output compression artifact."""
        return self._resolve_compression(self.outputs_compression, "_resolved_outputs_compression")

    def resolve_residuals_compression(self) -> Compression | None:
        """Resolve the output compression artifact."""
        return self._resolve_compression(self.residuals_compression, "_resolved_residuals_compression")

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

    def resolve_inputs_sampler(self) -> SamplerCallable | None:
        """Resolve the inputs sampler from explicit configuration or a compression artifact."""
        return self._resolve_sampler(
            self.inputs_sampler, self.resolve_inputs_compression(), "_resolved_inputs_sampler"
        )

    def resolve_outputs_sampler(self) -> SamplerCallable | None:
        """Resolve the outputs sampler from explicit configuration or a compression artifact."""
        return self._resolve_sampler(
            self.outputs_sampler, self.resolve_outputs_compression(), "_resolved_outputs_sampler"
        )

    def resolve_source_sampler(self) -> SamplerCallable | None:
        """Resolve the residuals sampler from explicit configuration or a compression artifact."""
        return self._resolve_sampler(
            self.residuals_sampler, self.resolve_residuals_compression(), "_resolved_residuals_sampler"
        )

    def _affine_inputs(self, inputs: PyTree) -> tuple[ArrayLike, Affine]:
        """Extract the augmented input value and runtime affine module."""
        if not isinstance(inputs, Mapping) or "module" not in inputs:
            raise TypeError("ImplicitAffine inputs must contain 'module'.")
        if not isinstance(inputs["module"], Affine):
            raise TypeError("ImplicitAffine input 'module' must be an Affine instance.")

        value = jnp.asarray(inputs["value"]).reshape(-1) if "value" in inputs else None

        if self.additional_inputs is None:
            if value is None:
                raise ValueError("Must pass in 'value' or additional_inputs to Affine.")
            
            return value, inputs["module"]

        paths = self.additional_inputs
        if paths == ():
            paths = ((),)

        leaves: list[ArrayLike] = []
        for path in paths:
            subtree = get_subtree(inputs, path)
            if subtree is None:
                raise KeyError(f"ImplicitAffine additional input path not found: {path!r}.")
            subtree_leaves = jax.tree_util.tree_flatten_with_path(subtree)[0]
            for relative_path, leaf in subtree_leaves:
                full_path = path + self._path_tokens(relative_path)
                if self._is_special_input_path(full_path) or not eqx.is_array_like(leaf):
                    continue
                leaves.append(jnp.asarray(leaf).reshape(-1))

        if not leaves:
            if value is None:
                raise ValueError("Must pass in 'value' or additional_inputs to Affine.")
            return value, inputs["module"]

        leaves = (value, *leaves) if value is not None else tuple(leaves)

        return jnp.concatenate(leaves), inputs["module"]

    @staticmethod
    def _path_tokens(path: tuple[Any, ...]) -> TreePath:
        """Convert JAX key-path entries to the project's string/integer path tokens."""
        tokens: list[str | int] = []
        for entry in path:
            if hasattr(entry, "key"):
                tokens.append(entry.key)
            elif hasattr(entry, "idx"):
                tokens.append(entry.idx)
            else:
                tokens.append(str(entry))
        return tuple(tokens)

    @staticmethod
    def _is_special_input_path(path: TreePath) -> bool:
        """Return whether a path belongs to the reserved value or affine module payloads."""
        return (
            path[:1] in (("value",), ("module",))
            or path[:2] in (("inputs", "value"), ("inputs", "module"))
        )

    @staticmethod
    def _value(payload: PyTree, name: str) -> ArrayLike:
        """Extract a canonical one-dimensional value from a payload."""
        if not isinstance(payload, Mapping) or "value" not in payload:
            raise TypeError(f"ImplicitAffine {name} must contain {{'value': ...}}.")
        return jnp.asarray(payload["value"]).reshape(-1)

    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate ``H(inputs, outputs) @ (outputs - solution(inputs))``."""
        values, affine = self._affine_inputs(inputs)
        output_values = self._value(outputs, "outputs")
        matrix, solution = affine.materialize(values, output_values)
        return {"value": matrix @ (output_values - solution)}

    def solve(self, inputs: PyTree, residuals: PyTree) -> PyTree:
        """Solve the affine residual equation for output coordinates."""
        values, affine = self._affine_inputs(inputs)
        residual_values = self._value(residuals, "residuals")
        solver_inputs = inputs.get("solver", {})

        def initial_for(solver: LinearSolver | IterativeSolver) -> jax.Array:
            """Resolve the configured initial field for either solver wrapper."""
            initial_inputs = dict(solver_inputs.get("initial", {}))
            if isinstance(solver.initial, AffineInitial):
                initial_inputs["inputs"] = values
                initial_inputs["module"] = affine
                initial_inputs["residuals"] = residual_values
            return jnp.broadcast_to(solver.initial(initial_inputs, {}), residual_values.shape)

        # Preserve the no-configuration shortcut, but honor an explicitly selected solver.
        if affine.identity_jac is True and self.solver is None:
            _, solution = affine.materialize(values)
            return {"value": solution + residual_values}

        if isinstance(self.solver, LinearSolver) or (
            self.solver is None and affine.jacobian_inputs == "inputs"
        ):
            if affine.jacobian_inputs in ("outputs", "both"):
                raise TypeError("Output-dependent ImplicitAffine Jacobians require an IterativeSolver.")
            matrix, solution = affine.materialize(values)
            operator = lx.MatrixLinearOperator(matrix)
            if isinstance(self.solver, LinearSolver):
                output_values = self.solver.linear_solve(
                    operator,
                    residual_values,
                    y0=initial_for(self.solver),
                    options=solver_inputs.get("options"),
                    state=solver_inputs.get("state"),
                )
            else:
                output_values = lx.linear_solve(
                    operator,
                    residual_values,
                    solver=lx.AutoLinearSolver(well_posed=True),
                ).value
            return {"value": output_values + solution}

        # Nonlinear solve if H(b, u) is output-dependent, or explicitly requested.
        solver = self.solver
        if solver is None:
            solver = IterativeSolver()
        if not isinstance(solver, IterativeSolver):
            raise TypeError("Output-dependent ImplicitAffine Jacobians require an IterativeSolver.")

        initial = initial_for(solver)

        def root_residual(output_values: ArrayLike, args: PyTree) -> ArrayLike:
            return self.evaluate(args["inputs"], {"value": output_values})["value"] - args["residuals"]["value"]

        return {
            "value": solver.root_find(
                root_residual,
                initial,
                {"inputs": inputs, "residuals": residuals},
                options=solver_inputs.get("options"),
            )
        }

    def sample_inputs(self, key: Key) -> PyTree:
        """Sample an input value payload."""
        sampler = self.resolve_inputs_sampler()

        if sampler is None:
            return None
        
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
        """Sample an output value payload, inputs/solution/conditions not used."""
        del inputs, solution, conditions

        sampler = self.resolve_outputs_sampler()

        if sampler is None:
            raise ValueError("ImplicitAffine output sampler could not be resolved.")

        return sampler.sample(key) if hasattr(sampler, "sample") else sampler(key)

    def sample_source(self, key: Key) -> PyTree:
        """Use source sampler for sampling residuals in latent space."""
        sampler = self.resolve_source_sampler()

        if sampler is None:
            raise ValueError("ImplicitAffine source sampler could not be resolved.")

        return sampler.sample(key) if hasattr(sampler, "sample") else sampler(key)


class ImplicitIterativeGalerkin(CompositeEdge, SourceSampleable):
    """Galerkin ROM that solves any `ImplicitModel` via an iterative solver in latent space."""

    solver: IterativeSolver = Field(default_factory=IterativeSolver)
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
        residual_payload = x["residuals"]
        nested_latent = isinstance(residual_payload, Mapping) and "latent" in residual_payload
        target_residual = residual_payload["latent"] if nested_latent else residual_payload

        def residual_fn(z: ArrayLike, args: PyTree, aux, edge_payload_patches, composite_stack) -> ArrayLike:
            """Root find residual function, with `z` as the latent coordinates."""
            payload = {"outputs": {"latent": z} if nested_latent else z}
            if (inputs := args.get("inputs", None)) is not None:
                payload["inputs"] = inputs

            result, aux = self.forward_aux(payload, aux, edge_payload_patches, composite_stack)

            result_residual = result["residuals"]
            if nested_latent:
                result_residual = result_residual["latent"]
            args_residual = args["residuals"]
            if nested_latent:
                args_residual = args_residual["latent"]
            return result_residual - args_residual
        
        solver_inputs = {}
        if isinstance(x.get("inputs"), Mapping):
            solver_inputs = x["inputs"].get("solver", {})
        initial_inputs = solver_inputs.get("initial", {})

        if isinstance(initial_inputs, Mapping) and "outputs" in initial_inputs:
            initial = jnp.asarray(initial_inputs["outputs"])
        else:
            initial = jnp.asarray(self.solver.initial(initial_inputs, {}))
        initial = jnp.broadcast_to(initial, jnp.asarray(target_residual).shape)

        solution = self.solver.root_find(
            lambda z, args: residual_fn(z, args, aux, edge_payload_patches, composite_stack), 
            initial,
            x,
            options=solver_inputs.get("options"),
            return_sol=False
        )

        ret = {"outputs": {"latent": solution} if nested_latent else solution}

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
