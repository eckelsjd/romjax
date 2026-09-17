from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import TypeAdapter

from romjax.compression import SVD, Compression
from romjax.nn import LinearProjection
from romjax.train import resolve_orbax_params


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
        template=template,
        orbax_template=orbax_template,
    )
    artifact_path = compression.dump(tmp_path / "compression.h5")

    reloaded = Compression.load(artifact_path)

    assert isinstance(reloaded, SVD)
    assert reloaded.energy_tol == compression.energy_tol
    assert reloaded.center is compression.center
    assert reloaded.rank == compression.rank
    for field in ("mean", "basis", "singular_values", "minval", "maxval", "latent_mean", "latent_std"):
        np.testing.assert_array_equal(getattr(reloaded, field), getattr(compression, field))
    assert isinstance(reloaded.template["state"]["x"], jax.ShapeDtypeStruct)
    assert reloaded.template["state"]["x"] == jax.ShapeDtypeStruct((2,), jnp.float32)
    assert reloaded.template["state"]["index"] == 3
    assert reloaded.template["label"] == "static"
    assert reloaded.template["items"][0] == jax.ShapeDtypeStruct((), jnp.int32)
    assert reloaded.template["items"][1] is None
    assert reloaded.template["pair"] == ("yes", False)
    assert reloaded.orbax_template == orbax_template


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
