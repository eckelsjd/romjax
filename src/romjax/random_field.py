"""Random field samplers for reproducible PDE inputs."""

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key
from pydantic import PositiveFloat, PositiveInt, model_validator

from romjax.typing import DictModel

__all__ = ["KLEConfig", "kle", "darcy"]


class KLEConfig(DictModel):
    r"""Configuration for a truncated 2D cosine-basis Karhunen-Loeve sampler.

    The field is sampled on a uniform, cell-centered grid as

    .. math::

        f(x, y) = \mu(x, y) + \sum_{i=0}^{m_x-1} \sum_{j=0}^{m_y-1}
        \sqrt{\lambda_{ij}} \xi_{ij} \phi_i(x) \phi_j(y),

    where ``xi_ij`` are iid standard normal coefficients and the separable cosine basis
    is weighted by a smooth spectrum controlled by ``correlation_lengths`` and
    ``spectral_decay``.

    :param bounds: 2D rectangular domain bounds ``((x0, x1), (y0, y1))``
    :param shape: output grid shape ``(nx, ny)``
    :param truncation: number of retained cosine modes along each axis
    :param correlation_lengths: smoothness controls for the x/y spectrum
    :param variance: target average marginal variance across the grid
    :param spectral_decay: exponent controlling modal energy decay
    :param mean: scalar or array-like mean field broadcastable to ``shape``
    :param nsamples: number of fields to draw. ``1`` returns a single 2D field.
    """

    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0))
    shape: tuple[PositiveInt, PositiveInt] = (16, 16)
    truncation: tuple[PositiveInt, PositiveInt] | None = None
    correlation_lengths: tuple[PositiveFloat, PositiveFloat] = (0.2, 0.2)
    variance: float = 1.0
    spectral_decay: PositiveFloat = 2.0
    mean: ArrayLike = 0.0
    nsamples: PositiveInt = 1

    @model_validator(mode="after")
    def _validate_config(self) -> "KLEConfig":
        if self.truncation is None:
            self.truncation = self.shape

        for lower, upper in self.bounds:
            if upper <= lower:
                raise ValueError("Grid bounds must be ordered as (lower, upper).")

        if any(mode > size for mode, size in zip(self.truncation, self.shape)):
            raise ValueError("truncation must not exceed the output grid shape.")

        if self.variance < 0.0:
            raise ValueError("variance must be nonnegative.")

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


def kle(
    key: Key,
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0)),
    shape: tuple[int, int] = (16, 16),
    truncation: tuple[int, int] | None = None,
    correlation_lengths: tuple[float, float] = (0.2, 0.2),
    variance: float = 1.0,
    spectral_decay: float = 2.0,
    mean: ArrayLike = 0.0,
    nsamples: int = 1,
    random_override: ArrayLike | None = None,
) -> ArrayLike:
    r"""Sample a scalar 2D random field from a truncated KLE on a uniform grid.

    This callable is designed to be used directly in :class:`romjax.rng.Distribution`,
    for example from YAML via ``distribution: !!python/name:romjax.random_field.kle``.

    The spectrum is parameterized as

    .. math::

        \lambda_{ij} \propto
        \left(1 + (\pi \ell_x i / L_x)^2 + (\pi \ell_y j / L_y)^2\right)^{-p},

    where ``correlation_lengths = (ell_x, ell_y)`` and ``spectral_decay = p``.
    The proportionality constant is chosen so that the average marginal variance on the
    sampled grid is approximately ``variance``. In tests, this seems to mean the spread
    can be expected to be about +/-4*sqrt(variance), but depends on truncation.

    :param key: JAX random key
    :param bounds: 2D rectangular domain bounds ``((x0, x1), (y0, y1))``
    :param shape: output grid shape ``(nx, ny)``
    :param truncation: retained cosine modes along each axis. Defaults to ``shape``.
    :param correlation_lengths: smoothness controls for the x/y spectrum
    :param variance: target average marginal variance across the grid
    :param spectral_decay: exponent controlling modal energy decay
    :param mean: scalar or array-like mean field broadcastable to ``shape``
    :param nsamples: number of fields to draw. ``1`` returns a single 2D field.
    :param random_override: use these samples of N(0,1) rather than the provided key (default: ignored).
                            essentially just to check convergence for truncation
    :return: ``(nx, ny)`` when ``nsamples == 1``, otherwise ``(nsamples, nx, ny)``
    """
    config = KLEConfig(
        bounds=bounds,
        shape=shape,
        truncation=truncation,
        correlation_lengths=correlation_lengths,
        variance=variance,
        spectral_decay=spectral_decay,
        mean=mean,
        nsamples=nsamples,
    )

    (x0, x1), (y0, y1) = config.bounds
    nx, ny = config.shape
    mx, my = config.truncation
    lx = x1 - x0
    ly = y1 - y0

    x = _cell_centered_axis(x0, x1, nx)
    y = _cell_centered_axis(y0, y1, ny)
    phi_x = _cosine_basis(x, x0, x1, mx)
    phi_y = _cosine_basis(y, y0, y1, my)

    kx = jnp.arange(mx)
    ky = jnp.arange(my)
    ell_x, ell_y = config.correlation_lengths
    raw_eigs = (
        1.0
        + (jnp.pi * ell_x * kx / lx)[:, None] ** 2
        + (jnp.pi * ell_y * ky / ly)[None, :] ** 2
    ) ** (-config.spectral_decay)

    # Some annoying things:
    # - typically always avoid solving the exact eigenvalue problem -- exp kernel simplifies nicely to cosine basis
    # - the assumption of cosine basis is very common, but technically only correct for neumann bcs
    # - these are essentially the assumed eigenfunctions, then the raw_eigs are the corresponding eigenvalues
    # - the proportionality const for eigs is different wherever you look (pi^2, avg_var, etc.)

    pointwise_var = jnp.einsum("ik,jl,kl->ij", phi_x**2, phi_y**2, raw_eigs)
    avg_var = jnp.mean(pointwise_var)
    scale = jnp.where(avg_var > 0.0, config.variance / avg_var, 0.0)
    sqrt_cov = jnp.sqrt(scale * raw_eigs)

    coeffs = random_override if random_override is not None else jax.random.normal(key, (config.nsamples, mx, my))
    coeffs =  coeffs * sqrt_cov[None, :, :]
    samples = jnp.einsum("ik,jl,bkl->bij", phi_x, phi_y, coeffs)
    samples = samples + jnp.asarray(config.mean)

    if config.nsamples == 1:
        return samples[0]
    return samples


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
