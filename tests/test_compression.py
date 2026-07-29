from pathlib import Path

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
    artifact_path = tmp_path / "compression.npz"

    compression.dump(artifact_path)
    reloaded = Compression.load(artifact_path)

    assert isinstance(reloaded, SVD)
    assert reloaded.rank == 1
    assert reloaded.template is not None
    assert reloaded.latent_size() == 1


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
