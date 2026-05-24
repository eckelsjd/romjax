"""Tests for rng and random field sampling."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from romjax.rng import Distribution, NearSolutionSampler, PyTreeSampler, gen_keys


def test_distribution() -> None:
    key = jax.random.key(0)
    uniform = Distribution(callable="uniform", minval=-1.0, maxval=2.0, shape=(2, 3))
    samples_a = uniform.sample(key)
    samples_b = uniform.sample(key)

    assert samples_a.shape == (2, 3)
    assert jnp.all(samples_a >= -1.0)
    assert jnp.all(samples_a < 2.0)
    assert jnp.allclose(samples_a, samples_b)

    with pytest.raises(KeyError):
        Distribution(callable="not_a_distribution")
    with pytest.raises(Exception):
        Distribution(callable=123)

    key = jax.random.key(1)
    normal = Distribution(callable="normal", mean=1.5, std=2.0, shape=(4,))
    normal_samples = normal.sample(key)
    expected = jax.random.normal(key, shape=(4,)) * 2.0 + 1.5

    assert normal_samples.shape == (4,)
    assert jnp.allclose(normal_samples, expected)

    def custom_dist(key: jax.Array, shape=(3,), scale=1.0):
        return jax.random.uniform(key, shape=shape) * scale

    custom = Distribution(callable=custom_dist, shape=(5,), scale=3.0)
    custom_samples = custom.sample(jax.random.key(2))
    assert custom_samples.shape == (5,)
    assert jnp.all(custom_samples >= 0.0)
    assert jnp.all(custom_samples < 3.0)


def test_gen_keys(tmp_path: Path) -> None:
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


def test_pytree_sampler() -> None:
    key = jax.random.key(5)
    sampler = PyTreeSampler(
        template={
            "phi": {"callable": "normal", "mean": 1.0, "std": 0.25, "shape": (2, 3)},
            "aux": (
                {"callable": "uniform", "minval": -1.0, "maxval": 1.0, "shape": ()},
            ),
        }
    )
    sample = sampler.sample(key)
    aux_key, phi_key = jax.random.split(key, 2)
    expected_aux = jax.random.uniform(aux_key, minval=-1.0, maxval=1.0, shape=())
    expected_phi = jax.random.normal(phi_key, shape=(2, 3)) * 0.25 + 1.0

    assert isinstance(sampler.template["phi"], Distribution)
    assert sample["phi"].shape == (2, 3)
    assert np.allclose(np.asarray(sample["phi"]), np.asarray(expected_phi))
    assert np.isclose(float(sample["aux"][0]), float(expected_aux))

    passthrough_sampler = PyTreeSampler(template={"phi": 1.0})
    passthrough_sample = passthrough_sampler.sample(key)
    assert passthrough_sample["phi"] == 1.0


def test_pytree_sampler_allows_inline_distribution_kwargs() -> None:
    sampler = PyTreeSampler(x={"callable": "uniform", "shape": ()})
    sample = sampler.sample(jax.random.key(0))

    assert np.asarray(sample["x"]).shape == ()


def test_pytree_sampler_preserves_non_distribution_leaves() -> None:
    sampler = PyTreeSampler(
        template={
            "phi": {"callable": "normal", "mean": 0.0, "std": 1.0, "shape": (2,)},
            "meta": {"label": "keep-me", "step": 7},
        }
    )

    sample = sampler.sample(jax.random.key(1))

    assert sample["meta"] == {"label": "keep-me", "step": 7}
    assert sample["phi"].shape == (2,)


def test_near_solution_sampler_with_noise_wrapper() -> None:
    key = jax.random.key(13)
    solution = {
        "phi": jnp.ones((3, 4)),
        "stats": {"mean": jnp.asarray(2.0)},
    }
    sampler = NearSolutionSampler(
        template={
            "phi": {"callable": "normal", "std": 0.2, "shape": (3, 4)},
            "stats": {"mean": {"callable": "normal", "std": 0.5, "shape": ()}},
        },
        scale={"phi": 0.5, "stats": {"mean": 2.0}},
    )
    sample = sampler.sample(key, solution=solution)
    noise_keys = jax.random.split(key, 2)
    expected_phi = solution["phi"] + 0.5 * (jax.random.normal(noise_keys[0], shape=(3, 4)) * 0.2)
    expected_mean = solution["stats"]["mean"] + 2.0 * (jax.random.normal(noise_keys[1], shape=()) * 0.5)

    assert np.allclose(np.asarray(sample["phi"]), np.asarray(expected_phi))
    assert np.isclose(float(sample["stats"]["mean"]), float(expected_mean))


def test_near_solution_sampler_broadcast_relative_scale_spec() -> None:
    key = jax.random.key(17)
    solution = {
        "phi": jnp.full((2, 2), 4.0),
        "psi": jnp.asarray([3.0, 1.0]),
    }
    sampler = NearSolutionSampler(
        template={
            "phi": {"callable": "normal", "std": 1.0, "shape": (2, 2)},
            "psi": {"callable": "normal", "std": 1.0, "shape": (2,)},
        },
        scale=("max_abs", 0.1),
    )
    sample = sampler.sample(key, solution=solution)
    phi_key, psi_key = jax.random.split(key, 2)
    expected_phi = solution["phi"] + 0.4 * jax.random.normal(phi_key, shape=(2, 2))
    expected_psi = solution["psi"] + 0.3 * jax.random.normal(psi_key, shape=(2,))

    assert np.allclose(np.asarray(sample["phi"]), np.asarray(expected_phi))
    assert np.allclose(np.asarray(sample["psi"]), np.asarray(expected_psi))


def test_near_solution_sampler_per_leaf_relative_scale_specs() -> None:
    key = jax.random.key(23)
    solution = {
        "phi": jnp.full((2, 2), 4.0),
        "stats": {"mean": jnp.asarray(3.0)},
    }
    sampler = NearSolutionSampler(
        template={
            "phi": {"callable": "normal", "std": 0.5, "shape": (2, 2)},
            "stats": {"mean": {"callable": "normal", "std": 1.0, "shape": ()}},
        },
        scale={"phi": ("rms", 0.25), "stats": {"mean": (jnp.mean, 0.1)}},
    )
    sample = sampler.sample(key, solution=solution)
    phi_key, mean_key = jax.random.split(key, 2)
    expected_phi_scale = jnp.sqrt(jnp.mean(jnp.square(solution["phi"]))) * 0.25
    expected_mean_scale = jnp.mean(solution["stats"]["mean"]) * 0.1
    expected_phi = solution["phi"] + expected_phi_scale * (jax.random.normal(phi_key, shape=(2, 2)) * 0.5)
    expected_mean = solution["stats"]["mean"] + expected_mean_scale * jax.random.normal(mean_key, shape=())

    assert np.allclose(np.asarray(sample["phi"]), np.asarray(expected_phi))
    assert np.isclose(float(sample["stats"]["mean"]), float(expected_mean))
