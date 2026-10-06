from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from pydantic import TypeAdapter

from romjax.compression import SVD, BlockLinearCompression, Compression
from romjax.graph import FunctionGraph
from romjax.nn import BlockLinearAutoencoder, LinearProjection
from romjax.pde import ImplicitAffine
from romjax.rng import CompressionSampler, PyTreeSampler
from romjax.train import BatchLoader, TerminationConfig, Train, resolve_orbax_params
from romjax.tree import TreeRef


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
    compression = BlockLinearCompression(
        input_sizes=(1, 1), latent_sizes=(1, 1),
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


def _conditionable_block_compression(*, ignore_nan: bool = True) -> BlockLinearCompression:
    template = {
        "inputs": {"x": jax.ShapeDtypeStruct((1,), jnp.float32)},
        "outputs": {"y": jax.ShapeDtypeStruct((1,), jnp.float32)},
    }
    blocks = ((np.ones((1, 1)), None), (None, np.ones((1, 1))))
    return BlockLinearCompression(
        encoder_blocks=blocks,
        decoder_blocks=blocks,
        input_sizes=(1, 1),
        latent_sizes=(1, 1),
        diagonal=True,
        ignore_nan=ignore_nan,
        minval=np.asarray([-1.0, -2.0]),
        maxval=np.asarray([1.0, 2.0]),
        latent_mean=np.asarray([1.0, -2.0]),
        latent_std=np.asarray([2.0, 3.0]),
        latent_covariance=np.asarray([[4.0, 2.0], [2.0, 9.0]]),
        template=template,
    )


def test_compression_condition_indices_support_exact_indices_and_blocks() -> None:
    compression = _conditionable_block_compression()

    assert SVD(rank=2).condition_indices(indices=[1, 0]) == (0, 1)
    assert compression.condition_indices(indices=[1]) == (1,)
    assert compression.condition_indices(blocks=[0]) == (0,)
    with pytest.raises(ValueError, match="exactly one"):
        compression.condition_indices(indices=[0], blocks=[1])
    with pytest.raises(ValueError, match="unique"):
        compression.condition_indices(blocks=[0, 0])
    with pytest.raises(ValueError, match="outside"):
        compression.condition_indices(indices=[2])


def test_compression_sampler_conditions_joint_normal_with_missing_nan_fill() -> None:
    compression = _conditionable_block_compression()
    key = jax.random.key(9)
    sampler = CompressionSampler(
        compression=compression,
        distribution="normal",
        conditioning={"blocks": [0], "missing": "nan"},
    )

    sample = sampler.sample(key, inputs={"x": jnp.asarray([1.5], dtype=jnp.float32), "unused": 7})
    conditional_mean = jnp.asarray([-2.0 + 2.0 / 4.0 * (1.5 - 1.0)])
    conditional_covariance = jnp.asarray([[9.0 - 2.0 * 2.0 / 4.0]])
    expected_free = jax.random.multivariate_normal(key, conditional_mean, conditional_covariance, method="svd")

    np.testing.assert_allclose(sample, jnp.asarray([1.5, expected_free[0]]), rtol=1e-6, atol=1e-6)

    marginal = CompressionSampler(
        compression=compression,
        distribution="normal",
        marginal=True,
        conditioning={"blocks": [0], "missing": "nan"},
    ).sample(key, inputs={"x": jnp.asarray([1.5], dtype=jnp.float32)})
    expected_marginal = jax.random.normal(key, (1,)) * 3.0 - 2.0
    np.testing.assert_allclose(marginal, jnp.asarray([1.5, expected_marginal[0]]))


def test_compression_sampler_conditions_singular_joint_normal() -> None:
    compression = _conditionable_block_compression().model_copy(
        update={
            "latent_mean": np.zeros(2),
            "latent_std": np.ones(2),
            "latent_covariance": np.ones((2, 2)),
        }
    )
    sampler = CompressionSampler(
        compression=compression,
        distribution="normal",
        conditioning={"blocks": [0], "missing": "nan"},
    )

    sample = sampler.sample(jax.random.key(3), inputs={"x": jnp.asarray([0.75], dtype=jnp.float32)})

    np.testing.assert_allclose(sample, jnp.asarray([0.75, 0.75]), rtol=1e-5, atol=1e-5)


def test_compression_sampler_uniform_condition_overlay_and_missing_error() -> None:
    compression = _conditionable_block_compression()
    default_sampler = CompressionSampler(
        compression=compression,
        distribution="uniform",
        conditioning={"blocks": [0]},
    )
    with pytest.raises(ValueError, match="outputs.y"):
        default_sampler.sample(jax.random.key(1), inputs={"x": jnp.asarray([0.25], dtype=jnp.float32)})

    sampler = CompressionSampler(
        compression=compression,
        distribution="uniform",
        conditioning={"indices": [0], "missing": "nan"},
    )
    sample = sampler.sample(
        jax.random.key(2),
        inputs={"x": jnp.asarray([0.25], dtype=jnp.float32)},
        conditions={"inputs": {"x": jnp.asarray([3.0], dtype=jnp.float32)}},
    )

    assert sample[0] == 3.0
    assert -2.0 <= sample[1] < 2.0
    with pytest.raises(ValueError, match="rooted at inputs/outputs"):
        sampler.sample(jax.random.key(2), conditions={"x": jnp.asarray([1.0])})


def test_compression_sampler_rejects_artifact_conditioning_and_handles_all_fixed() -> None:
    compression = _conditionable_block_compression()
    with pytest.raises(ValueError, match="does not support conditioning"):
        CompressionSampler(
            compression=compression,
            distribution="artifact",
            conditioning={"indices": [0]},
        ).resolve_sampler()

    sampler = CompressionSampler(
        compression=compression,
        distribution="uniform",
        conditioning={"indices": [0, 1]},
    )
    kwargs = {
        "inputs": {"x": jnp.asarray([0.25], dtype=jnp.float32)},
        "solution": {"y": jnp.asarray([-0.5], dtype=jnp.float32)},
    }
    np.testing.assert_allclose(sampler.sample(jax.random.key(1), **kwargs), jnp.asarray([0.25, -0.5]))
    np.testing.assert_allclose(sampler.sample(jax.random.key(2), **kwargs), jnp.asarray([0.25, -0.5]))


def test_block_linear_compression_persists_ignore_nan(tmp_path: Path) -> None:
    compression = _conditionable_block_compression()

    reloaded = Compression.load(compression.dump(tmp_path / "ignore_nan.h5"))

    assert isinstance(reloaded, BlockLinearCompression)
    assert reloaded.ignore_nan is True
    np.testing.assert_allclose(
        reloaded.compress(
            {
                "inputs": {"x": jnp.asarray([2.0], dtype=jnp.float32)},
                "outputs": {"y": jnp.asarray([jnp.nan], dtype=jnp.float32)},
            }
        ),
        jnp.asarray([2.0, 0.0]),
    )


def test_block_linear_compression_round_trip_and_artifact(tmp_path: Path) -> None:
    template = {
        "inputs": {"b": jax.ShapeDtypeStruct((1,), jnp.float32)},
        "outputs": {"u": jax.ShapeDtypeStruct((1,), jnp.float32)},
    }
    projection = BlockLinearAutoencoder(
        input_sizes=[1, 1], latent_sizes=[1, 1], diagonal=True,
        encoder_blocks=((jnp.ones((1, 1)), None), (None, jnp.ones((1, 1)))),
        decoder_blocks=((jnp.ones((1, 1)), None), (None, jnp.ones((1, 1)))),
    )
    compression = BlockLinearCompression(
        encoder_blocks=tuple(
            tuple(None if block is None else np.asarray(block) for block in row)
            for row in projection.encoder_blocks
        ),
        decoder_blocks=tuple(
            tuple(None if block is None else np.asarray(block) for block in row)
            for row in projection.decoder_blocks
        ),
        input_sizes=projection.input_sizes, latent_sizes=projection.latent_sizes, diagonal=True,
        minval=np.asarray([-1.0, -1.0]), maxval=np.asarray([1.0, 1.0]),
        latent_mean=np.zeros(2), latent_std=np.ones(2), latent_covariance=np.eye(2), template=template,
    )
    sample = {"inputs": {"b": jnp.asarray([0.25])}, "outputs": {"u": jnp.asarray([-0.5])}}
    assert jax.tree.all(jax.tree.map(jnp.allclose, compression.reconstruct(compression.compress(sample)), sample))

    reloaded = Compression.load(compression.dump(tmp_path / "block.h5"))
    assert isinstance(reloaded, BlockLinearCompression)
    assert reloaded.encoder_blocks[0][1] is None
    assert reloaded.reconstruct(reloaded.compress(sample))["outputs"]["u"].shape == (1,)


def test_block_linear_compression_fits_configured_train() -> None:
    samples = [
        {"inputs": {"b": jnp.asarray([0.0])}, "outputs": {"u": jnp.asarray([1.0])}},
        {"inputs": {"b": jnp.asarray([1.0])}, "outputs": {"u": jnp.asarray([0.0])}},
    ]
    projection = BlockLinearAutoencoder(input_sizes=[1, 1], latent_sizes=[1, 1], key=jax.random.key(4))
    train = Train(
        loss=lambda _params, _batch: jnp.asarray(0.0),
        init_params=projection,
        optimizer=optax.sgd(0.01),
        termination=TerminationConfig(max_steps=2),
    )

    compression = BlockLinearCompression(train=train, show_progress=False).fit(samples)

    assert compression.template is not None
    assert compression.latent_size() == 2
    assert compression.latent_covariance is not None
    assert compression.latent_covariance.shape == (2, 2)
    assert compression.reconstruct(compression.compress(samples[0]))["inputs"]["b"].shape == (1,)


def test_block_linear_compression_fits_diagonal_centered_pods(tmp_path: Path) -> None:
    vectors = jnp.asarray(
        [
            [0.0, 1.0, 2.0, 1.0, -1.0],
            [1.0, 2.0, 0.0, 2.0, 1.0],
            [2.0, 0.0, 1.0, -1.0, 2.0],
            [3.0, 1.0, -1.0, 0.0, 1.0],
        ]
    )
    samples = [{"first": row[:2], "second": row[2:]} for row in vectors]
    initial = BlockLinearAutoencoder(
        input_sizes=[2, 3], latent_sizes=[1, 2], key=jax.random.key(2),
        bias=jnp.zeros(5), diagonal=True,
    )

    compression = BlockLinearCompression(
        fit_mode="pod", init_params=initial, show_progress=False,
    ).fit(samples)
    projection = compression._projection()

    expected_chunks = []
    for chunk, rank in ((vectors[:, :2], 1), (vectors[:, 2:], 2)):
        pod = SVD(rank=rank, center=True).fit(list(chunk))
        basis = jnp.asarray(pod.basis)
        mean = jnp.asarray(pod.mean)
        expected_chunks.append((chunk - mean) @ basis.T @ basis + mean)
    expected = jnp.concatenate(expected_chunks, axis=1)
    actual = projection.reconstruct(projection.reduce(vectors))

    assert compression.diagonal is True
    assert compression.encoder_blocks[0][1] is None
    assert compression.decoder_blocks[1][0] is None
    np.testing.assert_allclose(compression.bias, jnp.mean(vectors, axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    reloaded = Compression.load(compression.dump(tmp_path / "pod_blocks.h5"))
    np.testing.assert_allclose(reloaded.bias, compression.bias)
    np.testing.assert_allclose(reloaded.encoder_blocks[1][1], compression.encoder_blocks[1][1])


def test_block_linear_compression_fits_joint_uncentered_pod() -> None:
    vectors = jnp.asarray(
        [[1.0, 0.0, 2.0, -1.0], [0.0, 2.0, 1.0, 1.0], [2.0, 1.0, 0.0, 3.0]]
    )
    initial = BlockLinearAutoencoder(
        input_sizes=[2, 2], latent_sizes=[1, 1], key=jax.random.key(3), bias=None,
    )
    compression = BlockLinearCompression(fit_mode="pod", init_params=initial).fit(list(vectors))
    projection = compression._projection()
    pod = SVD(rank=2, center=False).fit(list(vectors))

    basis = jnp.asarray(pod.basis)
    expected = vectors @ basis.T @ basis
    actual = projection.reconstruct(projection.reduce(vectors))
    assert compression.bias is None
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(projection.decoder_blocks[i][j], projection.encoder_blocks[j][i].T)


def test_block_linear_compression_pod_initializes_train(monkeypatch) -> None:
    vectors = jnp.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
    initial = BlockLinearAutoencoder(
        input_sizes=[1, 1], latent_sizes=[1, 1], key=jax.random.key(7), bias=jnp.zeros(2),
    )
    train = Train(
        loss=lambda _params, _batch: jnp.asarray(0.0), init_params=initial,
        optimizer=optax.sgd(0.01), termination=TerminationConfig(max_steps=1),
    )
    captured = {}

    def fake_train(routine):
        captured["params"] = routine.init_params
        return routine.init_params

    monkeypatch.setattr(Train, "__call__", fake_train)
    compression = BlockLinearCompression(
        fit_mode="pod_then_train", train=train, show_progress=False,
    ).fit(list(vectors))

    assert captured["params"] is not initial
    np.testing.assert_allclose(captured["params"].bias, jnp.mean(vectors, axis=0))
    np.testing.assert_allclose(compression.bias, jnp.mean(vectors, axis=0))


def test_block_linear_compression_pod_resolves_sampler_tree_refs() -> None:
    graph = FunctionGraph(
        edges={"model": ImplicitAffine(source="a", target="b", inputs_size=2, outputs_size=2)}
    )
    sampler = PyTreeSampler(
        template={
            "name": "BlockLinearAutoencoder",
            "kwargs": {
                "input_sizes": [
                    TreeRef(path=("edges", "model", "inputs_size")),
                    TreeRef(path=("edges", "model", "outputs_size")),
                ],
                "latent_sizes": [1, 1],
                "diagonal": True,
            },
        }
    )
    samples = [jnp.arange(4.0), jnp.arange(4.0) + 1.0]

    compression = BlockLinearCompression(
        fit_mode="pod", init_params=sampler, init_seed=9, graph=graph,
    ).fit(samples)

    assert compression.input_sizes == (2, 2)
    assert compression.latent_sizes == (1, 1)


def test_block_linear_compression_pod_rejects_rank_shortfall_and_conflicts() -> None:
    initial = BlockLinearAutoencoder(input_sizes=[2], latent_sizes=[2], key=jax.random.key(1))
    with pytest.raises(ValueError, match="POD rank 2 exceeds"):
        BlockLinearCompression(fit_mode="pod", init_params=initial).fit([jnp.ones(2)])

    train = Train(
        loss=lambda _params, _batch: jnp.asarray(0.0), init_params=initial,
        optimizer=optax.sgd(0.01), termination=TerminationConfig(max_steps=1),
    )
    with pytest.raises(ValueError, match="accepts top-level init_params"):
        BlockLinearCompression(fit_mode="pod", init_params=initial, train=train).fit([jnp.ones(2), jnp.zeros(2)])


def test_block_linear_compression_train_mapping_adds_defaults_and_diagnostics(monkeypatch) -> None:
    """Mapping configs supply reconstruction training defaults and block-only options."""
    samples = [
        {"inputs": {"b": jnp.asarray([1.0])}, "outputs": {"u": jnp.asarray([0.0])}},
        {"inputs": {"b": jnp.asarray([0.0])}, "outputs": {"u": jnp.asarray([1.0])}},
    ]
    projection = BlockLinearAutoencoder(
        input_sizes=[1, 1], latent_sizes=[1, 1], diagonal=True,
        encoder_blocks=((jnp.asarray([[2.0]]), None), (None, jnp.asarray([[2.0]]))),
        decoder_blocks=((jnp.asarray([[2.0]]), None), (None, jnp.asarray([[2.0]]))),
    )
    compression = BlockLinearCompression(
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
    # The two encoder rows each have a Gram diagonal of four.
    assert jnp.allclose(routine.loss(projection, vectors) - mse, 9.0)
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

    block = TypeAdapter(Compression).validate_python(
        {"kind": "block_linear", "input_sizes": [1, 1], "latent_sizes": [1, 1]}
    )
    assert isinstance(block, BlockLinearCompression)


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
