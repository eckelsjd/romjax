"""Tests for random field samplers."""

import jax
import numpy as np
import pytest

from romjax.random_field import KLEConfig, kle
from romjax.rng import Distribution, parametric_sampler


def test_kle_supports_1d_bounds() -> None:
    key = jax.random.key(19)
    opts = dict(
        bounds=((0.0, 2.0),),
        shape=(6,),
        truncation=(4,),
        correlation_lengths=(0.3,),
        variance=0.5,
        spectral_decay=2.5,
        mean=1.25,
    )

    sample_a = kle(key, **opts)
    sample_b = kle(key, **opts)
    batched = kle(key, nsamples=3, **opts)

    assert sample_a.shape == (6,)
    assert batched.shape == (3, 6)
    assert np.allclose(np.asarray(sample_a), np.asarray(sample_b))
    assert np.allclose(np.asarray(sample_a), np.asarray(batched[0]))


def test_kle_supports_scalar_style_1d_inputs_and_defaults() -> None:
    key = jax.random.key(23)

    default_sample = kle(key)
    sample = kle(
        key,
        bounds=(0.0, 2.0),
        shape=6,
        truncation=4,
        correlation_lengths=0.3,
        variance=0.5,
        spectral_decay=2.5,
        mean=1.25,
    )

    assert default_sample.shape == (16,)
    assert sample.shape == (6,)


def test_kle_deterministic_and_shapes() -> None:
    key = jax.random.key(7)
    opts = dict(
        bounds=((0.0, 2.0), (-1.0, 1.0)),
        shape=(6, 5),
        truncation=(4, 3),
        correlation_lengths=(0.3, 0.2),
        variance=0.5,
        spectral_decay=2.5,
        mean=1.25,
    )

    sample_a = kle(key, **opts)
    sample_b = kle(key, **opts)
    batched = kle(key, nsamples=3, **opts)

    assert sample_a.shape == (6, 5)
    assert batched.shape == (3, 6, 5)
    assert np.allclose(np.asarray(sample_a), np.asarray(sample_b))
    assert np.allclose(np.asarray(sample_a), np.asarray(batched[0]))


def test_kle_variance_scaling() -> None:
    samples = kle(
        jax.random.key(11),
        shape=(8, 8),
        truncation=(4, 4),
        correlation_lengths=(0.2, 0.2),
        variance=0.75,
        mean=2.0,
        nsamples=512,
    )
    centered = np.asarray(samples) - 2.0
    empirical_variance = centered.var(axis=0).mean()
    empirical_mean = np.asarray(samples).mean()

    assert np.isclose(empirical_mean, 2.0, atol=0.1)
    assert np.isclose(empirical_variance, 0.75, atol=0.15)


def test_kle_parametric_sampler_integration() -> None:
    key = jax.random.key(3)
    sample = parametric_sampler(
        key,
        conductivity={
            "distribution": kle,
            "shape": (7, 9),
            "truncation": (3, 4),
            "correlation_lengths": (0.15, 0.25),
            "variance": 0.2,
            "mean": 1.0,
        },
    )
    expected = kle(
        jax.random.split(key, 1)[0],
        shape=(7, 9),
        truncation=(3, 4),
        correlation_lengths=(0.15, 0.25),
        variance=0.2,
        mean=1.0,
    )

    assert sample["conductivity"].shape == (7, 9)
    assert np.allclose(np.asarray(sample["conductivity"]), np.asarray(expected))


def test_kle_distribution_validation() -> None:
    with pytest.raises(ValueError):
        KLEConfig(shape=(4, 4), truncation=(5, 2))

    with pytest.raises(ValueError):
        Distribution(distribution=kle, shape=(4, 4), variance=-1.0).sample(jax.random.key(0))
