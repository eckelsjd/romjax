import jax
import jax.numpy as jnp
import pytest

from romjax.data_gen import DataGeneration, DatasetConfig
from romjax.graph import FunctionGraph
from romjax.model import Edge, ImplicitSampleable


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


class _CustomDataset(DatasetConfig):
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
