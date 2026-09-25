from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pydantic import TypeAdapter

from romjax.compression import SVD, Compression, SplitLinearCompression
from romjax.nn import LinearProjection, SplitLinearProjection
from romjax.rng import CompressionSampler
from romjax.train import BatchLoader, TerminationConfig, Train, resolve_orbax_params


def _sample_pytree() -> list[dict[str, dict[str, jnp.ndarray]]]:
    return [
        {"state": {"x": jnp.asarray([1.0, 2.0], dtype=jnp.float32)}},
        {"state": {"x": jnp.asarray([2.0, 0.0], dtype=jnp.float32)}},
        {"state": {"x": jnp.asarray([3.0, 1.0], dtype=jnp.float32)}},
    ]


def test_svd_fit_compress_reconstruct() -> None:
    samples = _sample_pytree()
    compression = SVD(rank=2, center=True).fit(samples)

    latent = compression.compress(samples[0])
    reconstructed = compression.reconstruct(latent)
    bounds = compression.latent_bounds()

    assert compression.latent_size() == 2
    assert latent.shape == (2,)
    assert reconstructed["state"]["x"].shape == (2,)
    assert jnp.allclose(reconstructed["state"]["x"], samples[0]["state"]["x"])
    assert bounds is not None
    assert bounds[0].shape == (2,)
    assert bounds[1].shape == (2,)
    assert jnp.all(bounds[0] <= bounds[1])
    assert compression.latent_covariance is not None
    latent_samples = np.asarray([compression.compress(sample) for sample in samples])
    np.testing.assert_allclose(compression.latent_covariance, np.cov(latent_samples.T), atol=1e-7)


def test_compression_registry_and_round_trip(tmp_path: Path) -> None:
    compression = Compression._from_registry({"kind": "svd", "rank": 1, "center": False}).fit(_sample_pytree())
    artifact_path = tmp_path / "compression.h5"

    compression.dump(artifact_path)
    reloaded = Compression.load(artifact_path)

    assert isinstance(reloaded, SVD)
    assert reloaded.rank == 1
    assert reloaded.template is not None
    assert reloaded.latent_size() == 1


def test_svd_h5_round_trip_preserves_all_fields_and_templates(tmp_path: Path) -> None:
    template = {
        "state": {"x": jax.ShapeDtypeStruct((2,), jnp.float32), "index": 3},
        "label": "static",
        "items": [jax.ShapeDtypeStruct((), jnp.int32), None],
        "pair": ("yes", False),
    }
    orbax_template = {
        "coordinate transform": {"call_args": None},
        "residual transform": "coordinate transform",
    }
    compression = SVD(
        energy_tol=0.9,
        center=False,
        rank=1,
        mean=np.asarray([1.0, 2.0]),
        basis=np.asarray([[0.5, 0.25]]),
        singular_values=np.asarray([3.0, 1.0]),
        minval=np.asarray([-2.0]),
        maxval=np.asarray([2.0]),
        latent_mean=np.asarray([0.25]),
        latent_std=np.asarray([0.75]),
        latent_covariance=np.asarray([[0.5625]]),
        template=template,
        orbax_template=orbax_template,
    )
    artifact_path = compression.dump(tmp_path / "compression.h5")

    reloaded = Compression.load(artifact_path)

    assert isinstance(reloaded, SVD)
    assert reloaded.energy_tol == compression.energy_tol
    assert reloaded.center is compression.center
    assert reloaded.rank == compression.rank
    for field in (
        "mean", "basis", "singular_values", "minval", "maxval", "latent_mean", "latent_std", "latent_covariance"
    ):
        np.testing.assert_array_equal(getattr(reloaded, field), getattr(compression, field))
    assert isinstance(reloaded.template["state"]["x"], jax.ShapeDtypeStruct)
    assert reloaded.template["state"]["x"] == jax.ShapeDtypeStruct((2,), jnp.float32)
    assert reloaded.template["state"]["index"] == 3
    assert reloaded.template["label"] == "static"
    assert reloaded.template["items"][0] == jax.ShapeDtypeStruct((), jnp.int32)
    assert reloaded.template["items"][1] is None
    assert reloaded.template["pair"] == ("yes", False)
    assert reloaded.orbax_template == orbax_template
    with h5py.File(artifact_path, "r") as artifact:
        assert "basis" in artifact
        assert "mean" in artifact
        assert "payload" not in artifact


def test_compression_sampler_unpacks_and_reconstructs(tmp_path: Path) -> None:
    compression = SVD(rank=2, center=False).fit(_sample_pytree())
    artifact_path = compression.dump(tmp_path / "compression.h5")
    template = {"latent": jax.ShapeDtypeStruct((2,), jnp.float32)}
    sampler = CompressionSampler(compression=artifact_path, distribution="uniform", template=template)

    sampler.resolve_sampler()
    sample = sampler.sample(jax.random.key(3))
    assert sample["latent"].shape == (2,)

    reconstructed = CompressionSampler(compression=artifact_path, reconstruct=True).sample(jax.random.key(3))
    assert reconstructed["state"]["x"].shape == (2,)


def test_compression_sampler_normal_uses_joint_covariance_by_default() -> None:
    compression = SplitLinearCompression(
        encoder_b=np.asarray([[1.0, 0.0]]), encoder_u=np.asarray([[0.0, 1.0]]),
        decoder_b=np.asarray([[1.0, 0.0]]), decoder_u=np.asarray([[0.0, 1.0]]),
        input_size=2, b_latent=1, u_latent=1, b_output=1, u_output=1,
        latent_mean=np.asarray([1.0, -2.0]), latent_std=np.asarray([2.0, 3.0]),
        latent_covariance=np.asarray([[4.0, 3.0], [3.0, 9.0]]),
    )
    key = jax.random.key(8)

    joint = CompressionSampler(compression=compression).sample(key)
    marginal = CompressionSampler(compression=compression, marginal=True).sample(key)

    np.testing.assert_allclose(
        joint,
        jax.random.multivariate_normal(key, compression.latent_mean, compression.latent_covariance, method="svd"),
    )
    expected_marginal = jax.random.normal(key, (2,)) * compression.latent_std + compression.latent_mean
    np.testing.assert_allclose(marginal, expected_marginal)


def test_split_linear_compression_round_trip_and_artifact(tmp_path: Path) -> None:
    template = {
        "inputs": {"b": jax.ShapeDtypeStruct((1,), jnp.float32)},
        "outputs": {"u": jax.ShapeDtypeStruct((1,), jnp.float32)},
    }
    projection = SplitLinearProjection(
        encoder_b=jnp.asarray([[1.0, 0.0]]),
        encoder_u=jnp.asarray([[0.0, 1.0]]),
        decoder_b=jnp.asarray([[1.0, 0.0]]),
        decoder_u=jnp.asarray([[0.0, 1.0]]),
    )
    compression = SplitLinearCompression(
        encoder_b=np.asarray(projection.encoder_b), encoder_u=np.asarray(projection.encoder_u),
        decoder_b=np.asarray(projection.decoder_b), decoder_u=np.asarray(projection.decoder_u),
        input_size=2, b_latent=1, u_latent=1, b_output=1, u_output=1,
        minval=np.asarray([-1.0, -1.0]), maxval=np.asarray([1.0, 1.0]),
        latent_mean=np.zeros(2), latent_std=np.ones(2), latent_covariance=np.eye(2), template=template,
    )
    sample = {"inputs": {"b": jnp.asarray([0.25])}, "outputs": {"u": jnp.asarray([-0.5])}}
    assert jax.tree.all(jax.tree.map(jnp.allclose, compression.reconstruct(compression.compress(sample)), sample))

    reloaded = Compression.load(compression.dump(tmp_path / "split.h5"))
    assert isinstance(reloaded, SplitLinearCompression)
    assert reloaded.reconstruct(reloaded.compress(sample))["outputs"]["u"].shape == (1,)


def test_split_linear_compression_fits_configured_train() -> None:
    samples = [
        {"inputs": {"b": jnp.asarray([0.0])}, "outputs": {"u": jnp.asarray([1.0])}},
        {"inputs": {"b": jnp.asarray([1.0])}, "outputs": {"u": jnp.asarray([0.0])}},
    ]
    projection = SplitLinearProjection(
        input_size=2, b_latent=1, u_latent=1, b_output=1, u_output=1, key=jax.random.key(4)
    )
    train = Train(
        loss=lambda _params, _batch: jnp.asarray(0.0),
        init_params=projection,
        optimizer=optax.sgd(0.01),
        termination=TerminationConfig(max_steps=2),
    )

    compression = SplitLinearCompression(train=train, show_progress=False).fit(samples)

    assert compression.template is not None
    assert compression.latent_size() == 2
    assert compression.latent_covariance is not None
    assert compression.latent_covariance.shape == (2, 2)
    assert compression.reconstruct(compression.compress(samples[0]))["inputs"]["b"].shape == (1,)


def test_split_linear_compression_train_mapping_adds_defaults_and_diagnostics(monkeypatch) -> None:
    """Mapping configs supply reconstruction training defaults and split-only options."""
    samples = [
        {"inputs": {"b": jnp.asarray([1.0])}, "outputs": {"u": jnp.asarray([0.0])}},
        {"inputs": {"b": jnp.asarray([0.0])}, "outputs": {"u": jnp.asarray([1.0])}},
    ]
    projection = SplitLinearProjection(
        encoder_b=jnp.asarray([[2.0, 0.0]]),
        encoder_u=jnp.asarray([[0.0, 2.0]]),
        decoder_b=jnp.asarray([[2.0, 0.0]]),
        decoder_u=jnp.asarray([[0.0, 2.0]]),
    )
    compression = SplitLinearCompression(
        train={
            "init_params": projection,
            "optimizer": optax.sgd(0.01),
            "termination": {"max_steps": 1},
            "orthogonal_reg": 0.5,
            "test": True,
        },
        show_progress=False,
    )
    assert isinstance(compression.train, Train)
    assert callable(compression.train.loss)
    assert isinstance(compression.train.dataloader, BatchLoader)
    assert compression.orthogonal_reg == 0.5
    assert compression.test is True

    captured = {}

    def fake_train(routine):
        captured["routine"] = routine
        return routine.init_params

    monkeypatch.setattr(Train, "__call__", fake_train)
    compression.fit(samples)

    routine = captured["routine"]
    vectors = routine.dataloader.data
    mse = jnp.mean(jnp.square(jax.vmap(projection.reconstruct)(jax.vmap(projection.reduce)(vectors)) - vectors))
    # All four matrices are non-orthogonal, so each contributes to the penalty.
    assert jnp.allclose(routine.loss(projection, vectors) - mse, 18.0)
    assert routine.diagnostics.test_interval == 1
    assert jnp.allclose(routine.test(projection), 3.0)


def test_compression_rejects_npz_artifacts(tmp_path: Path) -> None:
    compression = SVD(rank=1)

    with pytest.raises(ValueError, match="Unsupported compression artifact path"):
        compression.dump(tmp_path / "compression.npz")
    with pytest.raises(ValueError, match="Unsupported compression artifact path"):
        Compression.load(tmp_path / "compression.npz")


def test_compression_type_adapter_accepts_registry_dict() -> None:
    compression = TypeAdapter(Compression).validate_python({"energy_tol": 0.99})

    assert isinstance(compression, SVD)
    assert compression.energy_tol == 0.99


def test_svd_requires_rank_or_energy_tol() -> None:
    with pytest.raises(ValueError):
        SVD()


def test_svd_orbax_checkpoint_matches_nested_compare_template(tmp_path):
    samples = [
        {"outputs": jnp.array([0.0, 1.0, 2.0, 3.0])},
        {"outputs": jnp.array([1.0, 1.5, 2.5, 4.0])},
        {"outputs": jnp.array([2.0, 3.0, 4.0, 6.0])},
    ]
    orbax_template = {
        "coordinate transform": {"call_args": None},
        "residual transform": "coordinate transform",
    }

    compression = SVD(rank=2, orbax_template=orbax_template).fit(samples)
    assert compression.orbax_template == orbax_template

    checkpoint_dir = tmp_path / "compression"
    compression.save_orbax(checkpoint_dir)

    params_template = {
        "coordinate transform": {"call_args": LinearProjection(matrix=jnp.zeros((2, 4)), bias=jnp.zeros(4))},
        "residual transform": None,
    }
    params = resolve_orbax_params(checkpoint_dir, params_template)

    projection = params["coordinate transform"]["call_args"]
    assert isinstance(projection, LinearProjection)
    np.testing.assert_allclose(projection.matrix, compression.basis)
    np.testing.assert_allclose(projection.bias, compression.mean)
    assert params["residual transform"] is None
    assert "matrix" not in params
