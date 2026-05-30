import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from romjax.compression import SVD, Compression
from romjax.data_gen import (
    DataGeneration,
    DataLoader,
    GenDataConfig,
    GenLatent,
    GenSource,
    LoadImplicitModel,
    LoadSource,
)
from romjax.graph import FunctionGraph
from romjax.model import Edge, ImplicitSampleable, SourceSampleable
from romjax.utils import save_h5


class _ToySampleableEdge(Edge, ImplicitSampleable):
    shift: float = 1.0

    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def sample_inputs(self, key):
        return {"x": jnp.asarray([jax.random.uniform(key)], dtype=jnp.float32)}

    def solve(self, inputs):
        return {"y": inputs["x"] + self.shift}

    def evaluate(self, inputs, outputs):
        return {"residual": outputs["y"] - self.solve(inputs)["y"]}

    def sample_outputs(self, key, inputs=None, solution=None):
        del key
        if solution is not None:
            return solution
        assert inputs is not None
        return self.solve(inputs)


def _get_graph():
    graph = FunctionGraph(
        edges={
            "low": _ToySampleableEdge(source="inputs", target="low", shift=1.0),
            "high": _ToySampleableEdge(source="inputs", target="high", shift=2.0),
        }
    )
    return graph


def _write_poisson_dataset(
    root: Path,
    dataset_name: str = "train/poisson",
    sample_count: int = 2,
    outputs_per_input: int = 1,
) -> None:
    samples = (
        (np.asarray([10.0, 0.0], dtype=np.float32), np.asarray([-1.0, 0.0], dtype=np.float32)),
        (np.asarray([0.0, 1.0], dtype=np.float32), np.asarray([0.0, -1.0], dtype=np.float32)),
    )
    for sample_idx in range(sample_count):
        phi, residual = samples[sample_idx % len(samples)]
        input_dir = root / dataset_name / "seed_0" / f"sample_{sample_idx}"
        input_dir.mkdir(parents=True, exist_ok=True)
        save_h5({"x": np.asarray(sample_idx, dtype=np.float32)}, input_dir / "input.h5", mode="w")
        save_h5({"phi": phi}, input_dir / "solution.h5", mode="w")
        save_h5({"phi_residual": residual}, input_dir / "solution_residual.h5", mode="w")
        for output_idx in range(outputs_per_input):
            output_dir = input_dir / "seed_0" / f"sample_{output_idx}"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_h5({"phi": phi + output_idx}, output_dir / "output.h5", mode="w")
            save_h5({"phi_residual": residual}, output_dir / "residual.h5", mode="w")


class _ToySourceEdge(Edge, SourceSampleable):
    bias: float = 0.0

    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def sample_source(self, key):
        return {
            "state": {
                "x": jnp.asarray(jax.random.uniform(key) + self.bias, dtype=jnp.float32),
                "meta": jnp.asarray([self.bias], dtype=jnp.float32),
            }
        }


class _DummyCompression(Compression):
    scale: float = 1.0
    rank: int = 1
    template: dict | None = None

    def fit(self, samples):
        return type(self)(scale=self.scale, rank=self.rank, template=samples[0] if samples else None)

    def compress(self, sample):
        return jnp.asarray([self.scale], dtype=jnp.float32)

    def reconstruct(self, latent):
        return self.template if self.template is not None else latent

    def latent_size(self):
        return int(self.rank)

    def latent_bounds(self):
        return jnp.asarray([0.0], dtype=jnp.float32), jnp.asarray([1.0], dtype=jnp.float32)
    
    def latent_normal(self):
        pass


class _TrackingLoadSource(LoadSource):
    select_calls: int = 0

    def select_epoch_refs(self, refs, indices):
        self.select_calls += 1
        return super().select_epoch_refs(refs, indices)


def _write_source_dataset(root: Path, dataset_name: str = "train/source", sample_count: int = 6) -> None:
    for sample_idx in range(sample_count):
        sample_dir = root / dataset_name / "seed_0" / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_h5({"state": {"x": np.asarray([sample_idx], dtype=np.float32)}}, sample_dir / "source.h5", mode="w")


@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_data(tmp_path, batch_size):
    config = DataGeneration(
        root=tmp_path,
        datasets={
            "train": {
                "low": {
                    "input_samples": 2,
                    "outputs_per_input": 1,
                    "input_seed": 3,
                    "output_seed": 5,
                    "batch_size": batch_size,
                },
                "high": {
                    "input_samples": 2,
                    "outputs_per_input": 1,
                    "input_seed": 3,
                    "output_seed": 5,
                    "batch_size": batch_size,
                },
            },
            "validation": {
                "low": {
                    "input_samples": 1,
                    "outputs_per_input": 1,
                    "input_seed": 7,
                    "output_seed": 11,
                    "batch_size": batch_size,
                },
                "high": {
                    "input_samples": 1,
                    "outputs_per_input": 1,
                    "input_seed": 7,
                    "output_seed": 11,
                    "batch_size": batch_size,
                },
            },
        },
        graph=_get_graph(),
    )
    config.run()

    for dataset_name, seed, input_samples in (("train", 3, 2), ("validation", 7, 1)):
        for edge_name in ("low", "high"):
            edge_dir = tmp_path / dataset_name / edge_name
            assert edge_dir.exists()
            assert (edge_dir / "romjax.txt").exists()
            seed_dir = edge_dir / f"seed_{seed}"
            assert seed_dir.exists()
            for sample_idx in range(input_samples):
                sample_dir = seed_dir / f"sample_{sample_idx}"
                assert (sample_dir / "input.h5").exists()
                assert (sample_dir / "solution.h5").exists()
                assert (sample_dir / "solution_residual.h5").exists()
                output_dir = sample_dir / "seed_5" if dataset_name == "train" else sample_dir / "seed_11"
                assert (output_dir / "sample_0" / "output.h5").exists()
                assert (output_dir / "sample_0" / "residual.h5").exists()


class _CustomDataset(GenDataConfig):
    marker: str
    last_call: tuple[str, str, str] | None = None

    def generate(self, path, format=None, write_policy=None):
        file_path = path / f"{self.marker}.txt"
        path.mkdir(parents=True, exist_ok=True)
        file_path.write_text(self.marker)
        self.last_call = (str(path), str(format), str(write_policy))


def test_custom_dataset_config_leaf(tmp_path):
    custom = _CustomDataset(marker="ok")
    config = DataGeneration(
        root=tmp_path,
        datasets={"custom": custom},
        format="h5",
        write_policy="overwrite",
    )

    config.run()

    assert (tmp_path / "custom" / "ok.txt").exists()
    assert custom.last_call == (str(tmp_path / "custom"), "h5", "overwrite")


def test_source_data_config_types(tmp_path):
    graph = FunctionGraph(edges={"source": _ToySourceEdge(source="noise", target="source", bias=2.0)})
    generation = DataGeneration(
        root=tmp_path,
        datasets={"source": {"samples": 2, "seed": 5}},
        graph=graph,
    )
    loader = DataLoader(
        root=tmp_path,
        datasets={"source": {"kind": "source", "batch_size": 1, "max_epochs": 1}},
    )

    assert isinstance(generation.datasets["source"], GenSource)
    assert isinstance(loader.datasets["source"], LoadSource)


def test_generate_galerkin_compression_from_poisson_data(tmp_path: Path) -> None:
    _write_poisson_dataset(tmp_path)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "train": {
                "poisson": {
                    "kind": "implicit",
                    "batch_size": 1,
                    "load_solution": False,
                }
            }
        },
    )
    artifact_path = tmp_path / "train" / "compression" / "galerkin_compression.npz"
    generator = GenLatent(
        loader=loader,
        gather_paths=[("poisson", "outputs", "phi")],
        compression=_DummyCompression(scale=2.0, rank=1),
        filename="galerkin_compression.npz",
    )

    generator.generate(tmp_path / "train" / "compression", format="h5", write_policy="overwrite")
    compression = Compression.load(artifact_path)
    manifest = json.loads((artifact_path.with_suffix(".manifest.json")).read_text())
    assert isinstance(compression, _DummyCompression)
    assert compression.rank == 1
    assert compression.scale == 2.0
    assert compression.template is not None
    assert manifest == {
        "latent_size": 1,
        "latent_bounds": [[0.0], [1.0]],
        "latent_normal": None,
    }


def test_data_loader_respects_max_samples_per_epoch(tmp_path: Path) -> None:
    _write_poisson_dataset(tmp_path, sample_count=5)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "train": {
                "poisson": {
                    "kind": "implicit",
                    "batch_size": 2,
                    "max_samples": 3,
                    "max_epochs": 1,
                    "load_solution": False,
                }
            }
        },
    )

    batches = list(loader)

    assert len(batches) == 2
    assert sum(batch["poisson"]["inputs"]["x"].shape[0] for batch in batches) == 3
    assert sum(batch["poisson"]["outputs"]["phi"].shape[0] for batch in batches) == 3
    assert len(np.unique(np.asarray(jnp.concatenate([batch["poisson"]["inputs"]["x"] 
                                                     for batch in batches]).reshape(-1)))) == 3


def test_load_implicit_model_respects_global_input_and_output_caps(tmp_path: Path) -> None:
    _write_poisson_dataset(tmp_path, sample_count=4, outputs_per_input=3)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "train": {
                "poisson": {
                    "kind": "implicit",
                    "batch_size": 2,
                    "max_samples": 5,
                    "max_input_samples": 2,
                    "max_outputs_per_input": 2,
                    "load_solution": False,
                    "shuffle_seed": 0,
                    "max_epochs": 1,
                }
            }
        },
    )

    batches = list(loader)
    inputs = jnp.concatenate([batch["poisson"]["inputs"]["x"] for batch in batches]).reshape(-1)

    assert sum(batch["poisson"]["outputs"]["phi"].shape[0] for batch in batches) == 4
    assert len(np.unique(np.asarray(inputs))) == 2
    assert all(count <= 2 for count in np.unique(np.asarray(inputs), return_counts=True)[1])


def test_loader_selects_epoch_pool_once_per_dataset(tmp_path: Path) -> None:
    _write_source_dataset(tmp_path, sample_count=6)
    source_cfg = _TrackingLoadSource(
        batch_size=2,
        max_samples=3,
        max_epochs=2,
        shuffle_seed=0,
    )
    loader = DataLoader(root=tmp_path, datasets={"train": {"source": source_cfg}})

    batches = list(loader)

    assert source_cfg.select_calls == 1
    assert sum(batch["source"]["state"]["x"].shape[0] for batch in batches) == 6


def test_generate_svd_galerkin_compression_with_template_cache(tmp_path: Path) -> None:
    _write_poisson_dataset(tmp_path)
    for sample_idx in range(2):
        output_dir = (
            tmp_path
            / "train"
            / "poisson"
            / "seed_0"
            / f"sample_{sample_idx}"
            / "seed_0"
            / "sample_0"
        )
        save_h5(
            {
                "phi": np.asarray([sample_idx, sample_idx + 1], dtype=np.float32),
                "psi": np.asarray([2.0, 3.0], dtype=np.float32),
            },
            output_dir / "output.h5",
                mode="w",
            )

    loader = DataLoader(
        root=tmp_path,
        datasets={
            "train": {
                "poisson": {
                    "kind": "implicit",
                    "batch_size": 1,
                    "load_solution": False,
                }
            }
        },
    )
    artifact_path = tmp_path / "train" / "compression" / "galerkin_compression.npz"
    generator = GenLatent(
        loader=loader,
        gather_paths=[("poisson", "outputs", "phi"), ("poisson", "outputs", "psi")],
        compression=SVD(rank=1, center=False),
        filename="galerkin_compression.npz",
    )

    generator.generate(tmp_path / "train" / "compression", format="h5", write_policy="overwrite")
    compression = Compression.load(artifact_path)
    assert isinstance(compression, SVD)
    assert compression.rank == 1
    assert compression.latent_size() == 1
    assert compression.template is not None
    reconstructed = compression.reconstruct(compression.compress(compression.template))
    assert reconstructed["poisson"]["outputs"]["phi"].shape == (2,)
    assert reconstructed["poisson"]["outputs"]["psi"].shape == (2,)


def test_data_loader_supports_nested_dataset_pytree(tmp_path):
    nested_root = tmp_path / "nested"
    for dataset_name, value in (("alpha", 1.0), ("beta", 2.0)):
        split = "train" if dataset_name == "alpha" else "validation"
        input_dir = nested_root / split / dataset_name / "seed_0" / "sample_0"
        input_dir.mkdir(parents=True, exist_ok=True)

        save_h5({"x": jnp.asarray([value], dtype=jnp.float32)}, input_dir / "input.h5", mode="w")
        save_h5({"y": jnp.asarray([value + 1.0], dtype=jnp.float32)}, input_dir / "solution.h5", mode="w")
        save_h5({"r": jnp.asarray([1.0], dtype=jnp.float32)}, input_dir / "solution_residual.h5", mode="w")
        output_dir = input_dir / "seed_1" / "sample_0"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_h5({"y": jnp.asarray([value + 2.0], dtype=jnp.float32)}, output_dir / "output.h5", mode="w")
        save_h5({"r": jnp.asarray([0.0], dtype=jnp.float32)}, output_dir / "residual.h5", mode="w")

    loader = DataLoader(
        root=nested_root,
        datasets={
            "train": {"alpha": {"kind": "implicit", "batch_size": 1, "max_epochs": 1, "load_solution": False}},
            "validation": {
                "beta": {"kind": "implicit", "batch_size": 1, "max_epochs": 1, "load_solution": False}
            },
        },
    )

    batch = next(loader)

    assert set(batch) == {"alpha", "beta"}
    assert batch["alpha"]["inputs"]["x"].shape == (1, 1)
    assert batch["alpha"]["outputs"]["y"].shape == (1, 1)
    assert batch["beta"]["inputs"]["x"].shape == (1, 1)
    with pytest.raises(StopIteration):
        next(loader)


def test_data_loader_mixed_generated_implicit_and_source_datasets(tmp_path):
    graph = FunctionGraph(
        edges={
            "toy": _ToySampleableEdge(source="inputs", target="toy", shift=1.5),
            "source": _ToySourceEdge(source="noise", target="source", bias=0.25),
        }
    )
    generation = DataGeneration(
        root=tmp_path,
        datasets={
            "toy": {
                "input_samples": 2,
                "outputs_per_input": 2,
                "input_seed": 3,
                "output_seed": 5,
                "batch_size": 2,
            },
            "source": {
                "samples": 3,
                "seed": 7,
                "batch_size": 2,
            },
        },
        graph=graph,
    )

    assert generation.run() == 0

    loader = DataLoader(
        root=tmp_path,
        datasets={
            "toy": {"kind": "implicit", "batch_size": 3, "max_epochs": 1},
            "source": {"kind": "source", "batch_size": 2, "max_epochs": 1},
        },
    )

    batch = next(loader)

    assert set(batch) == {"toy", "source"}
    assert batch["toy"]["inputs"]["x"].shape == (3, 1)
    assert batch["toy"]["outputs"]["y"].shape == (3, 1)
    assert batch["toy"]["residuals"]["residual"].shape == (3, 1)
    assert batch["source"]["state"]["x"].shape == (2,)
    assert batch["source"]["state"]["meta"].shape == (2, 1)


def test_load_implicit_model_includes_solution_sample_by_default(tmp_path):
    input_dir = tmp_path / "toy" / "seed_0" / "sample_0"
    input_dir.mkdir(parents=True, exist_ok=True)
    save_h5({"x": jnp.asarray([1.0], dtype=jnp.float32)}, input_dir / "input.h5", mode="w")
    save_h5({"y": jnp.asarray([2.0], dtype=jnp.float32)}, input_dir / "solution.h5", mode="w")
    save_h5({"r": jnp.asarray([-1.0], dtype=jnp.float32)}, input_dir / "solution_residual.h5", mode="w")

    output_dir = input_dir / "seed_0" / "sample_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_h5({"y": jnp.asarray([3.0], dtype=jnp.float32)}, output_dir / "output.h5", mode="w")
    save_h5({"r": jnp.asarray([0.5], dtype=jnp.float32)}, output_dir / "residual.h5", mode="w")

    refs = LoadImplicitModel(batch_size=4).discover_sample_refs(tmp_path / "toy")
    assert [sample_kind for _, _, sample_kind in refs] == ["solution", "output"]

    batch = LoadImplicitModel(batch_size=4).load_batch(refs)
    assert batch["inputs"]["x"].shape == (2, 1)
    assert batch["outputs"]["y"][:, 0].tolist() == [2.0, 3.0]
    assert batch["residuals"]["r"][:, 0].tolist() == [-1.0, 0.5]


def test_load_implicit_model_can_disable_solution_sample_loading(tmp_path):
    input_dir = tmp_path / "toy" / "seed_0" / "sample_0"
    input_dir.mkdir(parents=True, exist_ok=True)
    save_h5({"x": jnp.asarray([1.0], dtype=jnp.float32)}, input_dir / "input.h5", mode="w")
    save_h5({"y": jnp.asarray([2.0], dtype=jnp.float32)}, input_dir / "solution.h5", mode="w")
    save_h5({"r": jnp.asarray([-1.0], dtype=jnp.float32)}, input_dir / "solution_residual.h5", mode="w")

    output_dir = input_dir / "seed_0" / "sample_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_h5({"y": jnp.asarray([3.0], dtype=jnp.float32)}, output_dir / "output.h5", mode="w")
    save_h5({"r": jnp.asarray([0.5], dtype=jnp.float32)}, output_dir / "residual.h5", mode="w")

    refs = LoadImplicitModel(batch_size=4, load_solution=False).discover_sample_refs(tmp_path / "toy")
    assert [sample_kind for _, _, sample_kind in refs] == ["output"]
