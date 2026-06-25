"""Tests for random field samplers."""

import jax
import numpy as np
import pytest

from romjax.random_field import KLEConfig, kle
from romjax.rng import Distribution, PyTreeSampler


def test_kle_supports_1d_and_2d_shapes_and_determinism() -> None:
    key_1d = jax.random.key(19)
    opts_1d = dict(
        bounds=((0.0, 2.0),),
        shape=(6,),
        truncation=(4,),
        correlation_lengths=(0.3,),
        variance=0.5,
        spectral_decay=2.5,
        mean=1.25,
    )
    sample_1d = kle(key_1d, **opts_1d)
    batched_1d = kle(key_1d, nsamples=3, **opts_1d)
    scalar_style_1d = kle(
        jax.random.key(23),
        bounds=(0.0, 2.0),
        shape=6,
        truncation=4,
        correlation_lengths=0.3,
        variance=0.5,
        spectral_decay=2.5,
        mean=1.25,
    )

    key_2d = jax.random.key(7)
    opts_2d = dict(
        bounds=((0.0, 2.0), (-1.0, 1.0)),
        shape=(5, 4),
        truncation=(3, 2),
        correlation_lengths=(0.3, 0.2),
        variance=0.5,
        spectral_decay=2.5,
        mean=1.25,
    )
    sample_2d = kle(key_2d, **opts_2d)
    batched_2d = kle(key_2d, nsamples=3, **opts_2d)

    assert kle(jax.random.key(0)).shape == (16,)
    assert sample_1d.shape == (6,)
    assert batched_1d.shape == (3, 6)
    assert np.allclose(np.asarray(sample_1d), np.asarray(batched_1d[0]))
    assert scalar_style_1d.shape == (6,)
    assert sample_2d.shape == (5, 4)
    assert batched_2d.shape == (3, 5, 4)
    assert np.allclose(np.asarray(sample_2d), np.asarray(batched_2d[0]))


def test_kle_supports_3d_shapes_and_determinism() -> None:
    key = jax.random.key(29)
    opts = dict(
        bounds=((0.0, 1.0), (-1.0, 1.0), (0.0, 0.5)),
        shape=(4, 5, 3),
        truncation=(3, 2, 2),
        correlation_lengths=(0.2, 0.3, 0.1),
        variance=0.25,
        spectral_decay=2.5,
        mean=0.5,
    )
    sample = kle(key, **opts)
    batched = kle(key, nsamples=2, **opts)

    assert sample.shape == (4, 5, 3)
    assert batched.shape == (2, 4, 5, 3)
    assert np.allclose(np.asarray(sample), np.asarray(batched[0]))


def test_kle_variance_scaling() -> None:
    samples = kle(
        jax.random.key(11),
        shape=(6, 6),
        truncation=(3, 3),
        correlation_lengths=(0.2, 0.2),
        variance=0.75,
        mean=2.0,
        nsamples=128,
    )
    centered = np.asarray(samples) - 2.0
    empirical_variance = centered.var(axis=0).mean()
    empirical_mean = np.asarray(samples).mean()

    assert np.isclose(empirical_mean, 2.0, atol=0.1)
    assert np.isclose(empirical_variance, 0.75, atol=0.15)


def test_kle_parametric_sampler_integration() -> None:
    key = jax.random.key(3)
    sample = PyTreeSampler(
        template={
            "conductivity": {
                "callable": kle,
                "shape": (5, 6),
                "truncation": (2, 3),
                "correlation_lengths": (0.15, 0.25),
                "variance": 0.2,
                "mean": 1.0,
            },
        }
    ).sample(key)
    expected = kle(
        jax.random.split(key, 1)[0],
        shape=(5, 6),
        truncation=(2, 3),
        correlation_lengths=(0.15, 0.25),
        variance=0.2,
        mean=1.0,
    )

    assert sample["conductivity"].shape == (5, 6)
    assert np.allclose(np.asarray(sample["conductivity"]), np.asarray(expected))


def test_kle_distribution_validation() -> None:
    with pytest.raises(ValueError):
        KLEConfig(shape=(4, 4), truncation=(5, 2))

    with pytest.raises(ValueError):
        Distribution(callable=kle, shape=(4, 4), variance=-1.0).sample(jax.random.key(0))
