import jax
import jax.numpy as jnp
import pytest

from romjax.data_gen import DataGeneration, DataLoader, GenDataConfig, GenSource, LoadSource
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


def test_data_loader_supports_nested_dataset_pytree(tmp_path):
    nested_root = tmp_path / "nested"
    for dataset_name, value in (("alpha", 1.0), ("beta", 2.0)):
        split = "train" if dataset_name == "alpha" else "validation"
        input_dir = nested_root / split / dataset_name / "seed_0" / "sample_0"
        input_dir.mkdir(parents=True, exist_ok=True)

        save_h5({"x": jnp.asarray([value], dtype=jnp.float32)}, input_dir / "input.h5", mode="w")
        save_h5({"y": jnp.asarray([value + 1.0], dtype=jnp.float32)}, input_dir / "solution.h5", mode="w")
        output_dir = input_dir / "seed_1" / "sample_0"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_h5({"y": jnp.asarray([value + 2.0], dtype=jnp.float32)}, output_dir / "output.h5", mode="w")
        save_h5({"r": jnp.asarray([0.0], dtype=jnp.float32)}, output_dir / "residual.h5", mode="w")

    loader = DataLoader(
        root=nested_root,
        datasets={
            "train": {"alpha": {"kind": "implicit", "batch_size": 1, "max_epochs": 1}},
            "validation": {"beta": {"kind": "implicit", "batch_size": 1, "max_epochs": 1}},
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
