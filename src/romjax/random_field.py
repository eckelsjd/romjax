"""Random field samplers for reproducible PDE inputs."""
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key
from pydantic import Field, PositiveFloat, PositiveInt, field_validator, model_validator

from romjax.typing import DictModel

__all__ = ["GaussianWavePacketConfig", "KLEConfig", "darcy", "gaussian_wave_packets", "kle"]


type Coordinates = tuple[ArrayLike, ...] | ArrayLike


class KLEConfig(DictModel):
    r"""Configuration for a truncated cosine-basis Karhunen-Loeve sampler.

    The field is sampled on a uniform, cell-centered tensor-product grid as

    .. math::

        f(x) = \mu(x) + \sum_k \sqrt{\lambda_k} \xi_k \phi_k(x),

    with the obvious separable extension to multiple dimensions. The cosine basis is
    weighted by a smooth spectrum controlled by ``correlation_lengths`` and
    ``spectral_decay``.

    :param bounds: rectangular domain bounds ``((x0, x1), ...)``
    :param shape: output grid shape ``(n0, ...)``
    :param truncation: number of retained cosine modes along each axis
    :param correlation_lengths: smoothness controls for the spectrum along each axis
    :param variance: target average marginal variance across the grid
    :param spectral_decay: exponent controlling modal energy decay
    :param mean: scalar or array-like mean field broadcastable to ``shape``
    :param nsamples: number of fields to draw. ``1`` returns a single 2D field.
    :param random_override: use these samples of N(0,1) rather than the provided key (default: ignored).
                            essentially just to check convergence for truncation
    :param weight: optional weighting field to reduce noise near boundaries
    :param weight_opts: options to pass to weight function
    """

    bounds: tuple[tuple[Any, Any], ...] | tuple[Any, Any] = (0.0, 1.0)
    shape: tuple[PositiveInt, ...] | PositiveInt = 16
    truncation: tuple[PositiveInt, ...] | PositiveInt | None = None
    correlation_lengths: tuple[PositiveFloat, ...] | PositiveFloat = 0.2
    variance: PositiveFloat = 1.0
    spectral_decay: PositiveFloat = 2.0
    mean: ArrayLike = 0.0
    nsamples: PositiveInt = 1
    random_override: ArrayLike | None = None
    weight: Callable[[Coordinates], ArrayLike] | Literal["smooth"] | None = None
    weight_opts: dict = Field(default_factory=dict)

    @field_validator("weight", mode="before")
    @classmethod
    def _validate_weight(cls, weight):
        if isinstance(weight, str):
            if weight == "smooth":
                weight = _smooth_ramp
            else:
                raise ValueError(f"Unknown weighting function: {weight}")
        return weight

    @model_validator(mode="before")
    @classmethod
    def _normalize_1d_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        default_bounds = cls.model_fields["bounds"].default

        def _is_interval(value: Any) -> bool:
            return isinstance(value, tuple | list) and len(value) == 2 and not isinstance(value[0], tuple | list)

        def _sequence_ndim(value: Any) -> int | None:
            return len(value) if isinstance(value, tuple | list) else None

        inferred_ndims = [
            ndim for ndim in (
                _sequence_ndim(normalized.get("shape")),
                _sequence_ndim(normalized.get("truncation")),
                _sequence_ndim(normalized.get("correlation_lengths")),
            )
            if ndim is not None
        ]
        target_ndim = max(inferred_ndims, default=1)

        bounds = normalized.get("bounds", default_bounds)
        if _is_interval(bounds):
            normalized["bounds"] = tuple(tuple(bounds) for _ in range(target_ndim))
        elif isinstance(bounds, tuple | list):
            normalized["bounds"] = tuple(tuple(bound) for bound in bounds)

        for name in ("shape", "truncation", "correlation_lengths"):
            value = normalized.get(name, cls.model_fields[name].default)
            if value is not None and not isinstance(value, tuple | list):
                normalized[name] = (value,) * target_ndim

        return normalized

    @model_validator(mode="after")
    def _validate_config(self) -> "KLEConfig":
        ndim = len(self.bounds)
        if ndim not in (1, 2, 3):
            raise ValueError("KLEConfig currently supports only 1D, 2D, or 3D bounds.")

        if len(self.shape) != ndim:
            raise ValueError("shape must match the number of bounds.")

        if self.truncation is None:
            self.truncation = self.shape
        elif len(self.truncation) != ndim:
            raise ValueError("truncation must match the number of bounds.")

        if len(self.correlation_lengths) != ndim:
            raise ValueError("correlation_lengths must match the number of bounds.")

        for lower, upper in self.bounds:
            # Bounds inferred from coordinates may be tracers while the sampler is
            # being staged by ``jax.jit``. Defer their runtime ordering to the
            # numerical path, while retaining validation for ordinary configs.
            if isinstance(lower, jax.core.Tracer) or isinstance(upper, jax.core.Tracer):
                continue
            if upper <= lower:
                raise ValueError("Grid bounds must be ordered as (lower, upper).")

        if any(mode > size for mode, size in zip(self.truncation, self.shape)):
            raise ValueError("truncation must not exceed the output grid shape.")

        return self


class GaussianWavePacketConfig(DictModel):
    r"""Configuration for a finite Gaussian wave-packet random-field basis.

    Each packet contributes independent sine and cosine components of the form

    .. math::

        g_p(x)\cos(2\pi k_p\cdot(x-c_p)/L),\qquad
        g_p(x)\sin(2\pi k_p\cdot(x-c_p)/L),

    where ``g_p`` is a separable Gaussian envelope. This provides localized smooth
    perturbations when all wavenumbers are zero and localized oscillatory modes
    otherwise.

    :param bounds: rectangular domain bounds ``((x0, x1), ...)``
    :param shape: output grid shape ``(n0, ...)``
    :param centers: one packet center per row; defaults to the domain midpoint
    :param variances: Gaussian variances, one scalar/vector per packet
    :param wavenumbers: dimensionless sine/cosine wavevectors, one per packet
    :param amplitudes: optional relative packet amplitudes before global variance scaling
    :param variance: target average marginal variance across the grid
    :param mean: scalar or array-like mean field broadcastable to ``shape``
    :param nsamples: number of fields to draw
    :param random_override: standard-normal coefficients with shape
        ``(nsamples, n_packets, 2)`` for deterministic experiments
    :param weight: optional weighting field to reduce noise near boundaries
    :param weight_opts: options passed to ``weight``
    """

    bounds: tuple[tuple[float, float], ...] | tuple[float, float] = (0.0, 1.0)
    shape: tuple[PositiveInt, ...] | PositiveInt = 16
    centers: tuple[tuple[float, ...], ...] | tuple[float, ...] | None = None
    variances: tuple[tuple[PositiveFloat, ...], ...] | tuple[PositiveFloat, ...] | PositiveFloat = 0.05
    wavenumbers: tuple[tuple[float, ...], ...] | tuple[float, ...] | float = 0.0
    amplitudes: tuple[float, ...] | None = None
    variance: PositiveFloat = 1.0
    mean: ArrayLike = 0.0
    nsamples: PositiveInt = 1
    random_override: ArrayLike | None = None
    weight: Callable[[Coordinates], ArrayLike] | Literal["smooth"] | None = None
    weight_opts: dict = Field(default_factory=dict)

    @field_validator("weight", mode="before")
    @classmethod
    def _validate_weight(cls, weight):
        if isinstance(weight, str):
            if weight == "smooth":
                weight = _smooth_ramp
            else:
                raise ValueError(f"Unknown weighting function: {weight}")
        return weight

    @model_validator(mode="before")
    @classmethod
    def _normalize_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        bounds = normalized.get("bounds", cls.model_fields["bounds"].default)
        if isinstance(bounds, tuple | list) and len(bounds) == 2 and not isinstance(bounds[0], tuple | list):
            shape = normalized.get("shape", cls.model_fields["shape"].default)
            ndim = len(shape) if isinstance(shape, tuple | list) else 1
            normalized["bounds"] = tuple(tuple(bounds) for _ in range(ndim))
        elif isinstance(bounds, tuple | list):
            normalized["bounds"] = tuple(tuple(bound) for bound in bounds)

        ndim = len(normalized["bounds"])
        shape = normalized.get("shape", cls.model_fields["shape"].default)
        if not isinstance(shape, tuple | list):
            normalized["shape"] = (shape,) * ndim

        def _as_packet_rows(value: Any, default: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
            if value is None:
                return (default,)
            if not isinstance(value, tuple | list):
                return ((value,) * ndim,)
            if len(value) == 0:
                return ()
            if not isinstance(value[0], tuple | list):
                if len(value) == ndim:
                    return (tuple(value),)
                if ndim == 1:
                    return tuple((item,) for item in value)
                raise ValueError("Packet vectors must match the field dimensionality.")
            return tuple(tuple(row) for row in value)

        lower_upper = normalized["bounds"]
        midpoint = tuple((lower + upper) / 2.0 for lower, upper in lower_upper)
        normalized["centers"] = _as_packet_rows(normalized.get("centers"), midpoint)
        normalized["variances"] = _as_packet_rows(normalized.get("variances", 0.05), (0.05,) * ndim)
        normalized["wavenumbers"] = _as_packet_rows(normalized.get("wavenumbers", 0.0), (0.0,) * ndim)
        return normalized

    @model_validator(mode="after")
    def _validate_config(self) -> "GaussianWavePacketConfig":
        ndim = len(self.bounds)
        if ndim not in (1, 2, 3):
            raise ValueError("GaussianWavePacketConfig supports only 1D, 2D, or 3D bounds.")
        if len(self.shape) != ndim:
            raise ValueError("shape must match the number of bounds.")
        if not self.centers:
            raise ValueError("At least one Gaussian wave packet is required.")
        if len(self.variances) != len(self.centers) or len(self.wavenumbers) != len(self.centers):
            raise ValueError("centers, variances, and wavenumbers must contain the same number of packets.")
        if self.amplitudes is not None and len(self.amplitudes) != len(self.centers):
            raise ValueError("amplitudes must contain one value per packet.")
        for lower, upper in self.bounds:
            if upper <= lower:
                raise ValueError("Grid bounds must be ordered as (lower, upper).")
        vector_options = (("centers", self.centers), ("variances", self.variances), ("wavenumbers", self.wavenumbers))
        for name, vectors in vector_options:
            if any(len(vector) != ndim for vector in vectors):
                raise ValueError(f"Each {name} vector must match the field dimensionality.")
        return self


def _cell_centered_axis(lower: float, upper: float, npts: int) -> ArrayLike:
    """Construct cell-centered coordinates on a bounded interval."""
    spacing = (upper - lower) / npts
    return jnp.linspace(lower + spacing / 2.0, upper - spacing / 2.0, npts)


def _cosine_basis(axis: ArrayLike, lower: float, upper: float, nmodes: int) -> ArrayLike:
    """Evaluate the normalized cosine basis on one axis."""
    length = upper - lower
    modes = jnp.arange(nmodes)
    normalized_axis = (axis - lower) / length
    scale = jnp.where(modes == 0, 1.0, jnp.sqrt(2.0))
    return jnp.cos(jnp.pi * normalized_axis[:, None] * modes[None, :]) * scale[None, :]


def _smooth_ramp(
    coords: Coordinates,
    low: float = 0.0,
    high: float = 1.0,
    length_abs: float | tuple[float, ...] | None = None,
    length_rel: float | tuple[float, ...] = 0.1
) -> ArrayLike:
    """Return a smooth boundary-to-interior weighting field on a tensor-product grid."""
    if len(coords) == 0:
        raise ValueError("coords must contain at least one axis.")

    ndim = len(coords)
    length = length_abs if length_abs is not None else length_rel
    if isinstance(length, tuple | list):
        if len(length) != ndim:
            raise ValueError("Ramp lengths must be a scalar or match the number of coordinate axes.")
        ramp_lengths = tuple(float(axis_length) for axis_length in length)
    else:
        ramp_lengths = (float(length),) * ndim

    weight = jnp.ones_like(coords[0], dtype=jnp.result_type(coords[0], low, high))
    for axis, axis_length in zip(coords, ramp_lengths):
        axis_min = jnp.min(axis)
        axis_max = jnp.max(axis)
        axis_span = axis_max - axis_min
        effective_length = axis_length if length_abs is not None else axis_length * axis_span
        dist_to_boundary = jnp.minimum(axis - axis_min, axis_max - axis)
        if axis_length <= 0.0:
            axis_weight = jnp.ones_like(axis, dtype=weight.dtype)
        else:
            t = jnp.clip(dist_to_boundary / effective_length, 0.0, 1.0)
            axis_weight = t * t * (3.0 - 2.0 * t)
        weight = weight * axis_weight

    return low + (high - low) * weight


def kle(key: Key, **config: KLEConfig) -> ArrayLike:
    r"""Sample a scalar 1D, 2D, or 3D random field from a truncated KLE on a uniform grid.

    This callable is designed to be used directly in :class:`romjax.rng.Distribution`,
    for example from YAML via ``distribution: !!python/name:romjax.random_field.kle``.

    The spectrum is parameterized as

    .. math::

        \lambda_{ij} \propto
        \left(1 + (\pi \ell_x i / L_x)^2 + (\pi \ell_y j / L_y)^2\right)^{-p},

    where the correlation lengths and mode indices are applied along each axis and
    ``spectral_decay = p``.
    The proportionality constant is chosen so that the average marginal variance on the
    sampled grid is approximately ``variance``. In tests, this seems to mean the spread
    can be expected to be about +/-4*sqrt(variance), but depends on truncation.

    :param key: JAX random key
    :param config: see KLEConfig
    :return: ``shape`` when ``nsamples == 1``, otherwise ``(nsamples, *shape)``
    """
    cfg = KLEConfig(**config)
    ndim = len(cfg.bounds)

    # Some annoying things:
    # - typically always avoid solving the exact eigenvalue problem -- exp kernel simplifies nicely to cosine basis
    # - the assumption of cosine basis is very common, but technically only correct for neumann bcs
    # - these are essentially the assumed eigenfunctions, then the raw_eigs are the corresponding eigenvalues
    # - the proportionality const for eigs is different wherever you look (pi^2, avg_var, etc.)

    if ndim == 1:
        (x0, x1), = cfg.bounds
        nx, = cfg.shape
        mx, = cfg.truncation
        ell_x, = cfg.correlation_lengths
        lx = x1 - x0

        x = _cell_centered_axis(x0, x1, nx)
        phi_x = _cosine_basis(x, x0, x1, mx)
        kx = jnp.arange(mx)
        raw_eigs = (1.0 + (jnp.pi * ell_x * kx / lx) ** 2) ** (-cfg.spectral_decay)
        pointwise_var = jnp.einsum("ik,k->i", phi_x**2, raw_eigs)
        avg_var = jnp.mean(pointwise_var)
        scale = jnp.where(avg_var > 0.0, cfg.variance / avg_var, 0.0)
        sqrt_cov = jnp.sqrt(scale * raw_eigs)

        coeff_shape = (cfg.nsamples, mx)
        coeffs = cfg.random_override if cfg.random_override is not None else jax.random.normal(key, coeff_shape)
        coeffs = coeffs * sqrt_cov[jnp.newaxis, :]
        samples = jnp.einsum("ik,bk->bi", phi_x, coeffs)
        coords = (x,)
    elif ndim == 2:
        (x0, x1), (y0, y1) = cfg.bounds
        nx, ny = cfg.shape
        mx, my = cfg.truncation
        ell_x, ell_y = cfg.correlation_lengths
        lx = x1 - x0
        ly = y1 - y0

        x = _cell_centered_axis(x0, x1, nx)
        y = _cell_centered_axis(y0, y1, ny)
        phi_x = _cosine_basis(x, x0, x1, mx)
        phi_y = _cosine_basis(y, y0, y1, my)
        kx = jnp.arange(mx)
        ky = jnp.arange(my)
        raw_eigs = (
            1.0
            + (jnp.pi * ell_x * kx / lx)[:, None] ** 2
            + (jnp.pi * ell_y * ky / ly)[None, :] ** 2
        ) ** (-cfg.spectral_decay)
        pointwise_var = jnp.einsum("ik,jl,kl->ij", phi_x**2, phi_y**2, raw_eigs)
        avg_var = jnp.mean(pointwise_var)
        scale = jnp.where(avg_var > 0.0, cfg.variance / avg_var, 0.0)
        sqrt_cov = jnp.sqrt(scale * raw_eigs)

        coeff_shape = (cfg.nsamples, mx, my)
        coeffs = cfg.random_override if cfg.random_override is not None else jax.random.normal(key, coeff_shape)
        coeffs = coeffs * sqrt_cov[jnp.newaxis, :, :]
        samples = jnp.einsum("ik,jl,bkl->bij", phi_x, phi_y, coeffs)
        coords = jnp.meshgrid(x, y, indexing="ij")
    else:
        (x0, x1), (y0, y1), (z0, z1) = cfg.bounds
        nx, ny, nz = cfg.shape
        mx, my, mz = cfg.truncation
        ell_x, ell_y, ell_z = cfg.correlation_lengths
        lx = x1 - x0
        ly = y1 - y0
        lz = z1 - z0

        x = _cell_centered_axis(x0, x1, nx)
        y = _cell_centered_axis(y0, y1, ny)
        z = _cell_centered_axis(z0, z1, nz)
        phi_x = _cosine_basis(x, x0, x1, mx)
        phi_y = _cosine_basis(y, y0, y1, my)
        phi_z = _cosine_basis(z, z0, z1, mz)
        kx = jnp.arange(mx)
        ky = jnp.arange(my)
        kz = jnp.arange(mz)
        raw_eigs = (
            1.0
            + (jnp.pi * ell_x * kx / lx)[:, None, None] ** 2
            + (jnp.pi * ell_y * ky / ly)[None, :, None] ** 2
            + (jnp.pi * ell_z * kz / lz)[None, None, :] ** 2
        ) ** (-cfg.spectral_decay)
        pointwise_var = jnp.einsum("ik,jl,mn,kln->ijm", phi_x**2, phi_y**2, phi_z**2, raw_eigs)
        avg_var = jnp.mean(pointwise_var)
        scale = jnp.where(avg_var > 0.0, cfg.variance / avg_var, 0.0)
        sqrt_cov = jnp.sqrt(scale * raw_eigs)

        coeff_shape = (cfg.nsamples, mx, my, mz)
        coeffs = cfg.random_override if cfg.random_override is not None else jax.random.normal(key, coeff_shape)
        coeffs = coeffs * sqrt_cov[jnp.newaxis, :, :, :]
        samples = jnp.einsum("ik,jl,mn,bkln->bijm", phi_x, phi_y, phi_z, coeffs)
        coords = jnp.meshgrid(x, y, z, indexing="ij")

    samples = samples + jnp.asarray(cfg.mean)

    # Scale the samples by a weighting matrix (e.g. to make 0 near boundaries)
    if cfg.weight is not None:
        samples = cfg.weight(coords, **cfg.weight_opts)[jnp.newaxis, ...] * samples

    if cfg.nsamples == 1:
        return samples[0]
    return samples


def gaussian_wave_packets(key: Key, **config: GaussianWavePacketConfig) -> ArrayLike:
    r"""Sample a 1D, 2D, or 3D random field from Gaussian wave packets.

    The finite basis contains sine and cosine modulations for each configured
    Gaussian envelope. Coefficients are independent standard-normal variates and
    are normalized so that the average marginal variance is ``variance``.

    :param key: JAX random key
    :param config: see :class:`GaussianWavePacketConfig`
    :return: ``shape`` when ``nsamples == 1``, otherwise ``(nsamples, *shape)``
    """
    cfg = GaussianWavePacketConfig(**config)
    axes = tuple(
        _cell_centered_axis(lower, upper, npts)
        for (lower, upper), npts in zip(cfg.bounds, cfg.shape)
    )
    coords = (axes[0],) if len(axes) == 1 else tuple(jnp.meshgrid(*axes, indexing="ij"))
    packet_count = len(cfg.centers)
    packet_shape = (packet_count,) + tuple(cfg.shape)
    envelope = jnp.ones(packet_shape)
    phase = jnp.zeros(packet_shape)

    for axis, (lower, upper), center, packet_variance, packet_wavenumber in zip(
        coords,
        cfg.bounds,
        zip(*cfg.centers),
        zip(*cfg.variances),
        zip(*cfg.wavenumbers),
    ):
        centers = jnp.asarray(center).reshape((packet_count,) + (1,) * len(cfg.shape))
        variances = jnp.asarray(packet_variance).reshape((packet_count,) + (1,) * len(cfg.shape))
        wavenumbers = jnp.asarray(packet_wavenumber).reshape((packet_count,) + (1,) * len(cfg.shape))
        axis_values = jnp.asarray(axis)[jnp.newaxis, ...]
        envelope = envelope * jnp.exp(-0.5 * (axis_values - centers) ** 2 / variances)
        phase = phase + 2.0 * jnp.pi * wavenumbers * (axis_values - centers) / (upper - lower)

    amplitudes = jnp.ones(packet_count) if cfg.amplitudes is None else jnp.asarray(cfg.amplitudes)
    amplitudes = amplitudes.reshape((packet_count,) + (1,) * len(cfg.shape))
    basis = jnp.stack((envelope * jnp.cos(phase), envelope * jnp.sin(phase)), axis=1)
    basis = amplitudes[:, jnp.newaxis, ...] * basis
    pointwise_variance = jnp.sum(basis**2, axis=(0, 1))
    scale = jnp.where(jnp.mean(pointwise_variance) > 0.0, cfg.variance / jnp.mean(pointwise_variance), 0.0)

    coeff_shape = (cfg.nsamples, packet_count, 2)
    if cfg.random_override is None:
        sample_indices = jnp.arange(cfg.nsamples)
        coefficients = jax.vmap(
            lambda sample_index: jax.random.normal(jax.random.fold_in(key, sample_index), (packet_count, 2))
        )(sample_indices)
    else:
        coefficients = jnp.asarray(cfg.random_override)
        if coefficients.shape != coeff_shape:
            raise ValueError(f"random_override must have shape {coeff_shape}, got {coefficients.shape}.")
    samples = jnp.sqrt(scale) * jnp.einsum("bpc,pc...->b...", coefficients, basis)
    samples = samples + jnp.asarray(cfg.mean)

    if cfg.weight is not None:
        samples = cfg.weight(coords, **cfg.weight_opts)[jnp.newaxis, ...] * samples

    return samples[0] if cfg.nsamples == 1 else samples


def darcy(
    key: Key, 
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
    shape: tuple[int, int] = (16, 16),
    nemytskii: tuple[float, float] = (3.0, 12.0),
    cov_diag: float = 9.0,
    nsamples: int = 1
) -> ArrayLike:
    """Sample darcy flow conductivity field according to Sec 6.2 of Kovachki 2022.

    https://arxiv.org/abs/2108.08481

    :param key: the random key
    :param bounds: 2d bounds of grid
    :param shape: 2d shape of grid
    :param nemytskii: the result of the nemytskii pushforward, term 0 if field < 0, term 1 if field > 0
    :param cov_diag: additional amount to add to cov diagonal on top of Laplacian eigenvalues
    :param nsamples: number of random fields to generate
    :return: Array of shape [nsamples, W, H] giving the random field samples
    """
    (x0, x1), (y0, y1) = bounds
    nx, ny = shape
    lx = x1 - x0
    ly = y1 - y0

    x = _cell_centered_axis(x0, x1, nx)
    y = _cell_centered_axis(y0, y1, ny)
    phi_x = _cosine_basis(x, x0, x1, nx)
    phi_y = _cosine_basis(y, y0, y1, ny)

    kx = jnp.arange(nx)
    ky = jnp.arange(ny)

    lap_eigs = (jnp.pi ** 2) * ((kx[:, None] / lx) ** 2 + (ky[None, :] / ly) ** 2)
    cov_eigs = (lap_eigs + cov_diag) ** -2
    sqrt_cov = jnp.sqrt(cov_eigs)

    if nsamples == 1:
        coeffs = jax.random.normal(key, (nx, ny)) * sqrt_cov
        gauss_field = jnp.einsum("ik,jl,kl->ij", phi_x, phi_y, coeffs)
    else:
        coeffs = jax.random.normal(key, (nsamples, nx, ny)) * sqrt_cov[None, :, :]
        gauss_field = jnp.einsum("ik,jl,bkl->bij", phi_x, phi_y, coeffs)

    low, high = nemytskii
    return jnp.where(gauss_field >= 0.0, high, low)
