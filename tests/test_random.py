from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from romjax.rng import parametric_sampler, Distribution, gen_keys


def test_distribution():
    key = jax.random.key(0)
    uniform = Distribution(distribution="uniform", minval=-1.0, maxval=2.0, shape=(2, 3))
    samples_a = uniform.sample(key)
    samples_b = uniform.sample(key)

    assert samples_a.shape == (2, 3)
    assert jnp.all(samples_a >= -1.0)
    assert jnp.all(samples_a < 2.0)
    assert jnp.allclose(samples_a, samples_b)

    with pytest.raises(ValueError):
        Distribution(distribution="not_a_distribution")
    with pytest.raises(TypeError):
        Distribution(distribution=123)

    key = jax.random.key(1)
    normal = Distribution(distribution="normal", mean=1.5, std=2.0, shape=(4,))
    normal_samples = normal.sample(key)
    expected = jax.random.normal(key, shape=(4,)) * 2.0 + 1.5

    assert normal_samples.shape == (4,)
    assert jnp.allclose(normal_samples, expected)

    def custom_dist(key: jax.Array, shape=(3,), scale=1.0):
        return jax.random.uniform(key, shape=shape) * scale

    custom = Distribution(distribution=custom_dist, shape=(5,), scale=3.0)
    custom_samples = custom.sample(jax.random.key(2))
    assert custom_samples.shape == (5,)
    assert jnp.all(custom_samples >= 0.0)
    assert jnp.all(custom_samples < 3.0)


def test_gen_keys(tmp_path):
    pairs = list(gen_keys(2, path=tmp_path, seed=7))
    keys, paths = zip(*pairs)

    assert (tmp_path / "romjax.txt").exists()
    assert (tmp_path / "seed_7").exists()
    assert len(keys) == len(paths) == 2
    assert all((tmp_path / "seed_7" / f"sample_{i}").exists() for i in range(2))
    assert all(Path(p).parent == tmp_path / "seed_7" for p in paths)

    pairs_again = list(gen_keys(2, path=tmp_path, seed=7, skip="existing"))
    assert len(pairs_again) == 0
    assert all((tmp_path / "seed_7" / f"sample_{i}").exists() for i in range(2))

    pairs_more = list(gen_keys(4, path=tmp_path, seed=7, skip="existing"))
    keys_more, paths_more = zip(*pairs_more)
    assert len(keys_more) == len(paths_more) == 2
    assert {p.name for p in paths_more} == {"sample_2", "sample_3"}
    assert all((tmp_path / "seed_7" / f"sample_{i}").exists() for i in range(4))

    base_key = jax.random.key(7)
    expected_keys = [jax.random.fold_in(base_key, i) for i in [2, 3]]
    assert all(jnp.all(k == e) for k, e in zip(keys_more, expected_keys))

    pairs_new = list(gen_keys(1, path=tmp_path, seed=8))
    keys_new, paths_new = zip(*pairs_new)
    assert len(keys_new) == len(paths_new) == 1
    assert (tmp_path / "seed_8").exists()
    assert (tmp_path / "seed_8" / "sample_0").exists()


def test_parametric_sampler():
    keys = list(gen_keys(2, seed=21))

    for key in keys:
        sample = parametric_sampler(
            key,
            a={"distribution": "uniform", "minval": 0.0, "maxval": 1.0, "shape": (2,)},
            b={"distribution": "normal", "mean": 1.0, "std": 0.5, "shape": (3,)},
        )

        assert set(sample.keys()) == {"a", "b"}
        assert sample["a"].shape == (2,)
        assert sample["b"].shape == (3,)

        subkeys = jax.random.split(key, 2)
        expected_a = jax.random.uniform(subkeys[0], minval=0.0, maxval=1.0, shape=(2,))
        expected_b = jax.random.normal(subkeys[1], shape=(3,)) * 0.5 + 1.0

        assert np.allclose(np.asarray(sample["a"]), np.asarray(expected_a))
        assert np.allclose(np.asarray(sample["b"]), np.asarray(expected_b))

    scalar_key = next(gen_keys(1, seed=22))
    scalar = parametric_sampler(
        scalar_key,
        x={"distribution": "uniform", "minval": 0.0, "maxval": 1.0, "shape": ()},
        y={"distribution": "normal", "mean": 2.0, "std": 1.0, "shape": ()},
    )

    subkeys = jax.random.split(scalar_key, 2)
    expected_x = jax.random.uniform(subkeys[0], minval=0.0, maxval=1.0, shape=())
    expected_y = jax.random.normal(subkeys[1], shape=()) * 1.0 + 2.0

    assert np.isclose(float(scalar["x"]), float(expected_x))
    assert np.isclose(float(scalar["y"]), float(expected_y))
