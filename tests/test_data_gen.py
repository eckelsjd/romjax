import json
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import BeforeValidator

from romjax.compression import SVD, BlockLinearCompression, Compression
from romjax.data_gen import (
    DataGeneration,
    DataLoader,
    GenDataBenchmark,
    GenDataConfig,
    GenLatent,
    GenNorm,
    GenSource,
    GenSymlink,
    LoadImplicitModel,
    LoadSource,
)
from romjax.execution import ExecutionError, ExecutorConfig
from romjax.graph import FunctionGraph
from romjax.model import Edge, ImplicitSampleable, SourceSampleable
from romjax.norm import NormTree
from romjax.rng import CompressionSampler
from romjax.routine import RoutineError
from romjax.typing import from_yaml
from romjax.utils import load_h5, save_h5


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


class _RandomOutputEdge(Edge, ImplicitSampleable):
    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def sample_inputs(self, key):
        return {"x": jax.random.uniform(key)}

    def sample_outputs(self, key, inputs=None, solution=None):
        del inputs, solution
        return {"y": jax.random.uniform(key)}


class _NoneInputEdge(_RandomOutputEdge):
    def sample_inputs(self, key):
        del key
        return None


class _ConditionedSampleableEdge(Edge, ImplicitSampleable):
    """Small edge used to exercise persisted per-output conditions."""

    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def sample_inputs(self, key):
        del key
        return {"x": jnp.asarray([1.0], dtype=jnp.float32)}

    def sample_conditions(self, key):
        del key
        return {"x": jnp.asarray([4.0], dtype=jnp.float32)}

    def solve(self, inputs):
        return {"y": inputs["x"] + 1.0}

    def evaluate(self, inputs, outputs):
        return {"residual": outputs["y"] - self.solve(inputs)["y"]}

    def sample_outputs(self, key, inputs=None, solution=None, conditions=None):
        del key, solution
        assert inputs is not None
        assert conditions is not None
        augmented_inputs = {**inputs, **conditions}
        return self.solve(augmented_inputs)


class _CompressionSampleableEdge(Edge, ImplicitSampleable):
    """Exercise CompressionSampler through the standard data-generation contract."""

    outputs_sampler: Any

    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def sample_inputs(self, key):
        del key
        return {"x": jnp.asarray([0.25], dtype=jnp.float32)}

    def solve(self, inputs):
        del inputs
        raise AssertionError("skip_solve must not evaluate the baseline solution")

    def sample_outputs(self, key, inputs=None, solution=None, conditions=None):
        sample = self.outputs_sampler(
            key,
            inputs=inputs,
            solution=solution,
            conditions=conditions,
        )
        return sample["outputs"]


def _get_graph():
    graph = FunctionGraph(
        edges={
            "low": _ToySampleableEdge(source="inputs", target="low", shift=1.0),
            "high": _ToySampleableEdge(source="inputs", target="high", shift=2.0),
        }
    )
    return graph


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    ("conditions", "expected_inputs", "has_conditions"),
    [
        ("ignore", 1.0, False),
        ("include", 1.0, True),
        ("merge", 4.0, False),
    ],
)
def test_generate_implicit_persists_and_loads_conditions(
    tmp_path, batch_size, conditions, expected_inputs, has_conditions
):
    graph = FunctionGraph(
        edges={"conditioned": _ConditionedSampleableEdge(source="inputs", target="conditioned")}
    )
    generation = DataGeneration(
        root=tmp_path,
        datasets={
            "train": {
                "conditioned": {
                    "input_samples": 2,
                    "outputs_per_input": 2,
                    "input_seed": 3,
                    "output_seed": 5,
                    "batch_size": batch_size,
                }
            }
        },
        graph=graph,
    )

    assert generation.run() == 0

    edge_dir = tmp_path / "train" / "conditioned"
    refs = LoadImplicitModel().discover_sample_refs(edge_dir)
    output_refs = [ref for ref in refs if ref[2] == "output"]
    assert len(output_refs) == 4
    for input_path, output_path, _ in output_refs:
        assert (output_path / "conditions.h5").exists()
        sample = LoadImplicitModel(conditions=conditions).load_sample((input_path, output_path, "output"))
        assert np.allclose(sample["inputs"]["x"], expected_inputs)
        assert ("conditions" in sample) is has_conditions
        if has_conditions:
            assert np.allclose(sample["conditions"]["x"], 4.0)
        assert np.allclose(sample["outputs"]["y"], 5.0)


def test_load_implicit_model_conditions_default_is_include():
    """Retain separate condition payloads unless a loading mode is requested."""
    assert LoadImplicitModel().conditions == "include"


@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_implicit_conditions_compression_without_solution(tmp_path, batch_size) -> None:
    template = {
        "inputs": {"x": jax.ShapeDtypeStruct((1,), jnp.float32)},
        "outputs": {"y": jax.ShapeDtypeStruct((1,), jnp.float32)},
    }
    blocks = ((np.ones((1, 1)), None), (None, np.ones((1, 1))))
    compression = BlockLinearCompression(
        encoder_blocks=blocks,
        decoder_blocks=blocks,
        input_sizes=(1, 1),
        latent_sizes=(1, 1),
        diagonal=True,
        ignore_nan=True,
        minval=-np.ones(2),
        maxval=np.ones(2),
        latent_mean=np.zeros(2),
        latent_std=np.ones(2),
        latent_covariance=np.eye(2),
        template=template,
    )
    edge = _CompressionSampleableEdge(
        source="inputs",
        target="outputs",
        outputs_sampler=CompressionSampler(
            compression=compression,
            distribution="uniform",
            reconstruct=True,
            conditioning={"blocks": [0], "missing": "nan"},
        ),
    )
    generation = DataGeneration(
        root=tmp_path,
        datasets={
            "conditioned": {
                "input_samples": 2,
                "outputs_per_input": 2,
                "input_seed": 0,
                "output_seed": 1,
                "batch_size": batch_size,
                "skip_solve": True,
                "skip_evaluate": True,
            }
        },
        graph=FunctionGraph(edges={"conditioned": edge}),
    )

    assert generation.run() == 0
    output_paths = sorted((tmp_path / "conditioned").glob("seed_0/sample_*/seed_1/sample_*/output.h5"))
    assert len(output_paths) == 4
    for output_path in output_paths:
        output = load_h5({}, output_path, jax=True)
        assert output["y"].shape == (1,)
        assert jnp.isfinite(output["y"]).all()


@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_implicit_mixes_output_keys_with_input_keys(tmp_path, batch_size):
    graph = FunctionGraph(edges={"random": _RandomOutputEdge(source="inputs", target="random")})

    generation = DataGeneration(
        root=tmp_path / "mixed",
        datasets={
            "random": {
                "input_samples": 2,
                "outputs_per_input": 1,
                "input_seed": 3,
                "output_seed": 5,
                "batch_size": batch_size,
            }
        },
        graph=graph,
    )
    assert generation.run() == 0

    mixed_values = [
        load_h5({}, tmp_path / "mixed" / "random" / "seed_3" / f"sample_{i}" / "seed_5" / "sample_0" / "output.h5")["y"]
        for i in range(2)
    ]
    assert not np.array_equal(mixed_values[0], mixed_values[1])

    legacy_generation = DataGeneration(
        root=tmp_path / "legacy",
        datasets={
            "random": {
                "input_samples": 2,
                "outputs_per_input": 1,
                "input_seed": 3,
                "output_seed": 5,
                "mix_output_seed": False,
                "batch_size": batch_size,
            }
        },
        graph=graph,
    )
    assert legacy_generation.run() == 0

    legacy_values = [
        load_h5(
            {}, tmp_path / "legacy" / "random" / "seed_3" / f"sample_{i}" / "seed_5" / "sample_0" / "output.h5"
        )["y"]
        for i in range(2)
    ]
    assert np.array_equal(legacy_values[0], legacy_values[1])


@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_and_load_implicit_none_inputs(tmp_path, batch_size):
    graph = FunctionGraph(edges={"random": _NoneInputEdge(source="inputs", target="random")})
    generation = DataGeneration(
        root=tmp_path,
        datasets={
            "random": {
                "input_samples": 2,
                "outputs_per_input": 1,
                "input_seed": 3,
                "output_seed": 5,
                "batch_size": batch_size,
            }
        },
        graph=graph,
    )

    assert generation.run() == 0

    edge_dir = tmp_path / "random"
    input_dirs = sorted(edge_dir.glob("seed_3/sample_*"))
    assert len(input_dirs) == 2
    assert all(not (input_dir / "input.h5").exists() for input_dir in input_dirs)
    assert all((input_dir / "seed_5/sample_0/output.h5").exists() for input_dir in input_dirs)

    refs = LoadImplicitModel().discover_sample_refs(edge_dir)
    assert len(refs) == 2
    assert all("inputs" not in LoadImplicitModel().load_sample(ref) for ref in refs)


def _write_transport_dataset(
    root: Path,
    dataset_name: str = "train/transport",
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


class _FailingImplicitEdge(Edge, ImplicitSampleable):
    shift: float = 0.1
    fail_stage: str = "solve"
    solve_threshold: float = 0.5
    evaluate_threshold: float = 1.5
    output_offset: float = 1.0

    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def sample_inputs(self, key):
        return {"x": jnp.asarray([jax.random.uniform(key)], dtype=jnp.float32)}

    def solve(self, inputs):
        x_value = float(np.asarray(inputs["x"]).reshape(-1)[0])
        if self.fail_stage == "solve" and x_value > self.solve_threshold:
            raise RuntimeError("solve failure")
        return {"y": inputs["x"] + self.shift}

    def evaluate(self, inputs, outputs):
        output_value = float(np.asarray(outputs["y"]).reshape(-1)[0])
        if self.fail_stage == "output_evaluate" and output_value > self.evaluate_threshold:
            raise RuntimeError("evaluate failure")
        return {"residual": outputs["y"] - (inputs["x"] + self.shift)}

    def sample_outputs(self, key, inputs=None, solution=None):
        del key
        if solution is not None:
            if self.fail_stage == "output_evaluate":
                return {"y": solution["y"] + jnp.asarray([self.output_offset], dtype=jnp.float32)}
            return solution
        assert inputs is not None
        return {"y": inputs["x"] + self.shift}


class _FailingSourceEdge(Edge, SourceSampleable):
    fail_threshold: float = 0.5

    def forward(self, x):
        return x

    def backward(self, x):
        return x

    def sample_source(self, key):
        value = jax.random.uniform(key)
        if float(np.asarray(value)) > self.fail_threshold:
            raise RuntimeError("source failure")
        return {
            "state": {
                "x": jnp.asarray(value, dtype=jnp.float32),
                "meta": jnp.asarray([0.0], dtype=jnp.float32),
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


def _write_source_dataset(root: Path, dataset_name: str = "source", sample_count: int = 6) -> None:
    for sample_idx in range(sample_count):
        sample_dir = root / dataset_name / "seed_0" / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_h5({"state": {"x": np.asarray([sample_idx], dtype=np.float32)}}, sample_dir / "source.h5", mode="w")


def _write_implicit_dataset(
    root: Path,
    dataset_name: str = "toy",
    sample_count: int = 2,
    output_width: int = 1,
) -> None:
    for sample_idx in range(sample_count):
        input_dir = root / dataset_name / "seed_0" / f"sample_{sample_idx}"
        input_dir.mkdir(parents=True, exist_ok=True)
        save_h5({"x": np.asarray([sample_idx], dtype=np.float32)}, input_dir / "input.h5", mode="w")
        save_h5({"y": np.asarray([sample_idx + 1], dtype=np.float32)}, input_dir / "solution.h5", mode="w")
        save_h5({"r": np.asarray([0.0], dtype=np.float32)}, input_dir / "solution_residual.h5", mode="w")

        output_dir = input_dir / "seed_0" / "sample_0"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_h5(
            {"y": np.arange(output_width, dtype=np.float32) + sample_idx},
            output_dir / "output.h5",
            mode="w",
        )
        save_h5({"r": np.asarray([0.0], dtype=np.float32)}, output_dir / "residual.h5", mode="w")


def _failed_sample_dirs(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob(".romjax_failed"))


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


def test_generate_implicit_throw_false_serial_marks_failed_input_and_skips_loader(tmp_path, monkeypatch):
    monkeypatch.setattr("romjax.data_gen.eqx.filter_jit", lambda fn: fn)
    debug_messages: list[str] = []
    monkeypatch.setattr("romjax.data_gen.logger.debug", lambda message: debug_messages.append(str(message)))

    graph = FunctionGraph(
        edges={
            "low": _FailingImplicitEdge(source="inputs", target="low", fail_stage="solve"),
        }
    )
    generation = DataGeneration(
        root=tmp_path,
        datasets={
            "train": {
                "low": {
                    "input_samples": 2,
                    "outputs_per_input": 1,
                    "input_seed": 0,
                    "output_seed": 0,
                    "batch_size": 1,
                    "throw": False,
                }
            }
        },
        graph=graph,
    )

    generation.run()

    edge_dir = tmp_path / "train" / "low"
    failed_dirs = _failed_sample_dirs(edge_dir)
    assert len(failed_dirs) == 1

    failed_dir = failed_dirs[0]
    failure_log = failed_dir / "failure.log"
    assert failure_log.exists()
    failure_text = failure_log.read_text(encoding="utf-8")
    assert "solve failure" in failure_text
    assert "Traceback" in failure_text
    assert any("1 failed cases" in message for message in debug_messages)

    assert (failed_dir / "input.h5").exists()
    assert not (failed_dir / "solution.h5").exists()

    refs = LoadImplicitModel().discover_sample_refs(edge_dir)
    assert len(refs) == 2
    assert all(input_path != failed_dir for input_path, _, _ in refs)


def test_generate_implicit_throw_true_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr("romjax.data_gen.eqx.filter_jit", lambda fn: fn)

    graph = FunctionGraph(
        edges={
            "low": _FailingImplicitEdge(source="inputs", target="low", fail_stage="solve"),
        }
    )
    generation = DataGeneration(
        root=tmp_path,
        datasets={
            "train": {
                "low": {
                    "input_samples": 2,
                    "outputs_per_input": 1,
                    "input_seed": 0,
                    "output_seed": 0,
                    "batch_size": 1,
                    "throw": True,
                }
            }
        },
        graph=graph,
    )

    with pytest.raises(RuntimeError, match="solve failure"):
        generation.run()


def test_generate_implicit_throw_false_batch_marks_failed_output_and_skips_loader(tmp_path, monkeypatch):
    monkeypatch.setattr("romjax.data_gen.eqx.filter_jit", lambda fn: fn)
    debug_messages: list[str] = []
    monkeypatch.setattr("romjax.data_gen.logger.debug", lambda message: debug_messages.append(str(message)))

    graph = FunctionGraph(
        edges={
            "low": _FailingImplicitEdge(source="inputs", target="low", fail_stage="output_evaluate"),
        }
    )
    generation = DataGeneration(
        root=tmp_path,
        datasets={
            "train": {
                "low": {
                    "input_samples": 2,
                    "outputs_per_input": 1,
                    "input_seed": 0,
                    "output_seed": 0,
                    "batch_size": 2,
                    "throw": False,
                }
            }
        },
        graph=graph,
    )

    generation.run()

    edge_dir = tmp_path / "train" / "low"
    failed_dirs = _failed_sample_dirs(edge_dir)
    assert len(failed_dirs) == 1

    failed_dir = failed_dirs[0]
    failure_log = failed_dir / "failure.log"
    assert failure_log.exists()
    failure_text = failure_log.read_text(encoding="utf-8")
    assert "evaluate failure" in failure_text
    assert "Traceback" in failure_text
    assert any("1 failed cases" in message for message in debug_messages)

    assert failed_dir.parent.parent.name.startswith("sample_")
    refs = LoadImplicitModel().discover_sample_refs(edge_dir)
    assert len(refs) == 3
    assert all(output_path != failed_dir for _, output_path, _ in refs)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_source_throw_false_skips_failed_samples(tmp_path, monkeypatch, batch_size):
    monkeypatch.setattr("romjax.data_gen.eqx.filter_jit", lambda fn: fn)
    debug_messages: list[str] = []
    monkeypatch.setattr("romjax.data_gen.logger.debug", lambda message: debug_messages.append(str(message)))

    graph = FunctionGraph(edges={"source": _FailingSourceEdge(source="noise", target="source")})
    generation = DataGeneration(
        root=tmp_path,
        datasets={
            "source": {
                "samples": 2,
                "seed": 0,
                "batch_size": batch_size,
                "throw": False,
            }
        },
        graph=graph,
    )

    generation.run()

    source_dir = tmp_path / "source"
    failed_dirs = _failed_sample_dirs(source_dir)
    assert len(failed_dirs) == 1

    failed_dir = failed_dirs[0]
    failure_log = failed_dir / "failure.log"
    assert failure_log.exists()
    failure_text = failure_log.read_text(encoding="utf-8")
    assert "source failure" in failure_text
    assert "Traceback" in failure_text
    assert any("1 failed cases" in message for message in debug_messages)

    loader = DataLoader(root=tmp_path)
    assert isinstance(loader.datasets["source"], LoadSource)
    refs = loader.datasets["source"].discover_sample_refs(source_dir)
    assert len(refs) == 1
    assert refs[0] != failed_dir


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


def test_symlink_dataset_config_resolves_source_from_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "existing" / "dataset"
    source.mkdir(parents=True)
    (source / "sample.h5").touch()

    generation = DataGeneration(
        root=tmp_path / "generated",
        datasets={"linked": {"symlink": "existing/dataset"}},
    )

    assert isinstance(generation.datasets["linked"], GenSymlink)
    assert generation.run() == 0

    destination = tmp_path / "generated" / "linked"
    assert destination.is_symlink()
    assert destination.resolve() == source
    assert destination.readlink() == Path("../existing/dataset")


def test_symlink_dataset_config_supports_absolute_artifact_path(tmp_path):
    source = tmp_path / "existing.h5"
    source.touch()
    destination = tmp_path / "generated" / "linked.h5"

    GenSymlink(symlink=source).generate(destination, format="h5", write_policy="overwrite")

    assert destination.is_symlink()
    assert destination.readlink() == source


def test_symlink_dataset_config_requires_existing_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "generated" / "linked"

    with pytest.raises(RoutineError, match=str(tmp_path / "missing")):
        GenSymlink(symlink="missing").generate(destination, format="h5", write_policy="overwrite")


def test_symlink_dataset_config_write_policies(tmp_path):
    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    first_source.mkdir()
    second_source.mkdir()
    destination = tmp_path / "linked"
    destination.symlink_to(first_source, target_is_directory=True)
    generator = GenSymlink(symlink=second_source)

    generator.generate(destination, format="h5", write_policy="reuse")
    assert destination.resolve() == first_source

    with pytest.raises(RoutineError, match="policy='error'"):
        generator.generate(destination, format="h5", write_policy="error")

    generator.generate(destination, format="h5", write_policy="overwrite")
    assert destination.resolve() == second_source


@pytest.mark.parametrize("destination_kind", ["file", "directory"])
def test_symlink_dataset_config_does_not_overwrite_real_destination(tmp_path, destination_kind):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    if destination_kind == "file":
        destination.write_text("keep", encoding="utf-8")
    else:
        destination.mkdir()

    with pytest.raises(RoutineError, match="Refusing to overwrite non-symlink"):
        GenSymlink(symlink=source).generate(destination, format="h5", write_policy="overwrite")

    assert destination.exists()
    assert not destination.is_symlink()
    if destination_kind == "file":
        assert destination.read_text(encoding="utf-8") == "keep"


def test_symlink_dataset_config_handles_dangling_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "linked"
    destination.symlink_to(tmp_path / "missing", target_is_directory=True)
    generator = GenSymlink(symlink=source)

    with pytest.raises(RoutineError, match="policy='error'"):
        generator.generate(destination, format="h5", write_policy="error")

    generator.generate(destination, format="h5", write_policy="overwrite")
    assert destination.resolve() == source


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
    assert loader.datasets["source"].stack_batch is False


def test_data_generation_expands_base_overrides_cartesian_product(tmp_path: Path) -> None:
    graph = FunctionGraph(edges={"source": _ToySourceEdge(source="noise", target="source")})
    generation = DataGeneration(
        root=tmp_path,
        base={"source": {"samples": 1, "seed": 0}},
        overrides=[
            {
                "name": "inputs",
                "cases": [
                    {"name": "1", "value": {"source": {"samples": 1}}},
                    {"name": "2", "value": {"source": {"samples": 2}}},
                ],
            },
            {
                "name": "outputs",
                "cases": [
                    {"name": "0", "value": {"source": {"seed": 0}}},
                    {"name": "1", "value": {"source": {"seed": 1}}},
                ],
            },
        ],
        graph=graph,
    )

    generation.run()

    for samples in (1, 2):
        for seed in (0, 1):
            source_root = tmp_path / f"inputs={samples}" / f"outputs={seed}" / "source"
            assert (source_root / f"seed_{seed}").exists()
            assert (source_root / f"seed_{seed}" / "sample_0" / "source.h5").exists()


@pytest.mark.parametrize("kind", ["implicit", "source"])
def test_data_generation_benchmark_caches_total_and_amortized_timings(tmp_path: Path, kind: str) -> None:
    if kind == "implicit":
        graph = FunctionGraph(edges={"sample": _ToySampleableEdge(source="inputs", target="sample")})
        generator = {
            "kind": "implicit",
            "graph": graph,
            "input_samples": 1,
            "outputs_per_input": 0,
            "input_seed": 3,
            "output_seed": 5,
            "batch_size": 2,
        }
    else:
        graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
        generator = {"kind": "source", "graph": graph, "samples": 1, "seed": 3, "batch_size": 2}

    benchmark = GenDataBenchmark(
        generator=generator,
        samples=6,
        repeat=2,
        warmup=1,
        hardware="test",
    )
    benchmark.generate(tmp_path / "sample", format="h5", write_policy="overwrite")

    entries = load_h5({}, tmp_path / "sample" / "timings.h5", jax=False)
    assert len(entries) == 1 and next(iter(entries)).startswith("test.serial.")
    cached = next(iter(entries.values()))
    assert cached["total"].shape == (2,)
    assert np.all(cached["total"] > 0.0)
    np.testing.assert_allclose(cached["amortized"], cached["total"] / 6)
    assert cached["_benchmark"]["samples"] == 6
    assert not list((tmp_path / "sample").glob("seed_*"))


def test_data_generation_benchmark_honors_cache_write_policy(tmp_path: Path, monkeypatch) -> None:
    graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
    benchmark = GenDataBenchmark(
        generator={"kind": "source", "graph": graph, "samples": 1, "seed": 0},
        samples=2,
        repeat=1,
        warmup=0,
        hardware="test",
    )
    monkeypatch.setattr(benchmark, "_benchmark", lambda path, **kwargs: np.asarray([4.0]))
    benchmark.generate(tmp_path / "sample", format="h5", write_policy="overwrite")

    monkeypatch.setattr(benchmark, "_benchmark", lambda path, **kwargs: pytest.fail("reuse must not benchmark"))
    benchmark.generate(tmp_path / "sample", format="h5", write_policy="reuse")
    with pytest.raises(RoutineError, match="already exist"):
        benchmark.generate(tmp_path / "sample", format="h5", write_policy="error")

    incompatible = benchmark.model_copy(update={"repeat": 2}, deep=True)
    with pytest.raises(RoutineError, match="incompatible"):
        incompatible.generate(tmp_path / "sample", format="h5", write_policy="reuse")


def test_data_generation_benchmark_times_full_pass_and_reports_progress(tmp_path: Path, monkeypatch) -> None:
    import romjax.execution as execution

    graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
    benchmark = GenDataBenchmark(
        generator={"kind": "source", "graph": graph, "samples": 1, "seed": 0},
        samples=2, repeat=2, warmup=1, show_progress=True, name_depth=1, hardware="test",
    )
    workload_calls = []
    monkeypatch.setattr(benchmark, "_workload", lambda path: (
        lambda value: workload_calls.append(value) or np.asarray(value), ((1,), (2,))))
    ticks = iter([1., 5., 10., 22.])
    clock_state = {"timing": False}

    def clock():
        clock_state["timing"] = not clock_state["timing"]
        return next(ticks)

    monkeypatch.setattr(execution.time, "perf_counter", clock)
    progress = {"updates": 0, "messages": []}

    class FakeProgress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert not clock_state["timing"]
            return False

        def __setattr__(self, name, value):
            assert not clock_state["timing"]
            progress["messages"].append(value)

        def __call__(self, units):
            assert not clock_state["timing"]
            progress["updates"] += units

        def pause(self):
            assert not clock_state['timing']
            return nullcontext()

    def progress_factory(total, **kwargs):
        progress["total"] = total
        progress["title"] = kwargs["title"]
        return FakeProgress()

    monkeypatch.setattr(execution, "alive_bar", progress_factory)
    totals = benchmark._benchmark(tmp_path / "sample")
    np.testing.assert_allclose(totals, [4., 12.])
    assert workload_calls == [1, 2] * 3
    assert progress["total"] == 6 and progress["updates"] == 6
    assert progress["title"] == "sample"
    assert any("Repeat 2/2" in message for message in progress["messages"])


def test_data_generation_benchmark_excludes_cache_io_from_timing(tmp_path, monkeypatch):
    import romjax.data_gen as data_gen
    import romjax.execution as execution

    graph = FunctionGraph(edges={'sample': _ToySourceEdge(source='noise', target='sample')})
    benchmark = GenDataBenchmark(generator={'kind': 'source', 'graph': graph, 'samples': 1, 'seed': 0},
                                 samples=2, repeat=1, warmup=1, hardware='test', show_progress=False)
    path = tmp_path / 'sample'
    path.mkdir()
    save_h5({'previous': {'total': np.asarray([8.])}}, path / 'timings.h5', mode='w')
    state = {'timing': False}
    ticks = iter([10., 17.])

    def clock():
        state['timing'] = not state['timing']
        return next(ticks)

    def load(*args, **kwargs):
        assert not state['timing']
        return load_h5(*args, **kwargs)

    def save(*args, **kwargs):
        assert not state['timing']
        return save_h5(*args, **kwargs)

    monkeypatch.setattr(benchmark, '_workload', lambda path: (lambda value: np.asarray(value), ((1,), (2,))))
    monkeypatch.setattr(execution.time, 'perf_counter', clock)
    monkeypatch.setattr(data_gen, 'load_h5', load)
    monkeypatch.setattr(data_gen, 'save_h5', save)
    benchmark.generate(path, format='h5', write_policy='overwrite')
    entries = load_h5({}, path / 'timings.h5', jax=False)
    np.testing.assert_array_equal(entries.pop('previous')['total'], [8.])
    np.testing.assert_array_equal(next(iter(entries.values()))['total'], [7.])


@pytest.mark.parametrize("kind", ["implicit", "source"])
@pytest.mark.parametrize("executor", ["serial", "thread", "process"])
def test_data_generation_benchmark_uses_executor(tmp_path, kind, executor):
    edge = (_ToySourceEdge(source="noise", target="sample") if kind == "source"
            else _ToySampleableEdge(source="inputs", target="sample"))
    graph = FunctionGraph(edges={"sample": edge})
    options = ({"samples": 1, "seed": 3} if kind == "source" else
               {"input_samples": 1, "outputs_per_input": 3, "input_seed": 3, "output_seed": 5})
    benchmark = GenDataBenchmark(
        generator={"kind": kind, "batch_size": 2, **options}, samples=5,
        repeat=2, warmup=1, hardware="test", show_progress=False,
    )
    config = {"kind": executor, "show_progress": False}
    if executor != "serial":
        config["max_workers"] = 2
    generation = DataGeneration(root=tmp_path, graph=graph, executor=config, datasets={"sample": benchmark})
    assert generation.run() == 0
    entries = load_h5({}, tmp_path / "sample" / "timings.h5", jax=False)
    assert next(iter(entries)).startswith(f"test.{executor}.")
    entry = next(iter(entries.values()))
    assert entry["total"].shape == (2,) and np.all(entry["total"] > 0)
    np.testing.assert_allclose(entry["amortized"], entry["total"] / 5)
    assert not list(tmp_path.rglob("seed_*"))


def test_data_generation_benchmark_cache_identity_and_leaf_override(tmp_path):
    graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
    benchmark = GenDataBenchmark(generator={"kind": "source", "samples": 1, "seed": 0, "executor": "serial"},
                                 samples=2, repeat=1, warmup=1, hardware="test", show_progress=False)
    generation = DataGeneration(root=tmp_path, graph=graph, executor="mpi", datasets={"sample": benchmark})
    generation.run()
    benchmark.generator.executor = ExecutorConfig(kind="thread", max_workers=1, show_progress=False)
    generation.run()
    benchmark.generator.executor = ExecutorConfig(kind="thread", max_workers=2, show_progress=False)
    generation.run()
    entries = load_h5({}, tmp_path / "sample" / "timings.h5", jax=False)
    assert len(entries) == 3
    labels = set(entries)
    benchmark.generator.executor.show_progress = True
    benchmark.generator.name_depth = 1
    generation.run()
    assert set(load_h5({}, tmp_path / "sample" / "timings.h5", jax=False)) == labels


@pytest.mark.parametrize('batch_size', [1, 2])
@pytest.mark.parametrize('edge_type', [_ConditionedSampleableEdge, _RandomOutputEdge, _NoneInputEdge])
@pytest.mark.parametrize('skip', [False, True])
def test_data_generation_benchmark_matches_generation_numerics(tmp_path, batch_size, edge_type, skip):
    """A benchmark must execute the same numerical workload, including output tails."""
    graph = FunctionGraph(edges={'sample': edge_type(source='inputs', target='sample')})
    options = dict(input_samples=3, outputs_per_input=3, input_seed=3, output_seed=5,
                   batch_size=batch_size, skip_solve=skip, skip_evaluate=skip)
    generation = DataGeneration(root=tmp_path / 'reference', graph=graph, executor={'show_progress': False},
                                datasets={'sample': options})
    generation.run()
    benchmark = GenDataBenchmark(generator={'kind': 'implicit', 'graph': graph, **options}, samples=3,
                                 repeat=1, warmup=0, show_progress=False)
    workload, batches = benchmark._workload(tmp_path / 'sample')
    numerical = [np.asarray(leaf).reshape(-1) for args in batches
                 for leaf in jax.tree.leaves(jax.block_until_ready(workload(*args)))]
    persisted = [np.asarray(leaf).reshape(-1) for path in (tmp_path / 'reference').rglob('*.h5')
                 for leaf in jax.tree.leaves(load_h5({}, path, jax=False))]
    np.testing.assert_allclose(np.sort(np.concatenate(numerical)), np.sort(np.concatenate(persisted)))


def test_data_generation_benchmark_rejects_old_schema_at_executor_label(tmp_path):
    graph = FunctionGraph(edges={'sample': _ToySourceEdge(source='noise', target='sample')})
    benchmark = GenDataBenchmark(generator={'kind': 'source', 'graph': graph, 'samples': 1, 'seed': 0},
                                 samples=2, repeat=1, warmup=1, hardware='test', show_progress=False)
    benchmark.generate(tmp_path / 'sample', format='h5', write_policy='overwrite')
    path = tmp_path / 'sample' / 'timings.h5'
    cached = load_h5({}, path, jax=False)
    next(iter(cached.values()))['_benchmark']['schema_version'] = np.asarray(2)
    save_h5(cached, path, mode='w')
    with pytest.raises(RoutineError, match='incompatible'):
        benchmark.generate(tmp_path / 'sample', format='h5', write_policy='reuse')


def test_data_generation_expands_benchmark_cases(tmp_path: Path, monkeypatch) -> None:
    base_path = tmp_path / "benchmark.yml"
    base_path.write_text(
        "\n".join(
            [
                "sample:",
                "  kind: benchmark",
                "  repeat: 1",
                "  warmup: 0",
                "  hardware: test",
                "  generator:",
                "    kind: source",
                "    samples: 1",
                "    seed: 0",
                "    graph: !romx:FunctionGraph",
                "      edges:",
                    f"        - !pd:{__name__}._ToySourceEdge",
                "          source: noise",
                "          target: sample",
            ]
        ),
        encoding="utf-8",
    )
    generation = DataGeneration(
        root=tmp_path,
        base=base_path,
        overrides=[
            {
                "name": "samples",
                "cases": [
                    {"name": "2", "value": {"sample": {"samples": 2}}},
                    {"name": "4", "value": {"sample": {"samples": 4}}},
                ],
            }
        ],
    )
    monkeypatch.setattr(GenDataBenchmark, "_benchmark", lambda self, path, **kwargs: np.asarray([self.samples]))

    assert generation.run() == 0
    for samples in (2, 4):
        cached = next(iter(load_h5({}, tmp_path / f"samples={samples}" / "sample" / "timings.h5", jax=False).values()))
        np.testing.assert_allclose(cached["total"], [samples])
        np.testing.assert_allclose(cached["amortized"], [1.0])


def test_data_generation_expands_yaml_base_path_and_empty_base(tmp_path: Path) -> None:
    graph = FunctionGraph(edges={"source": _ToySourceEdge(source="noise", target="source")})
    base_path = tmp_path / "base.yml"
    base_path.write_text("source: {samples: 1, seed: 0}\n", encoding="utf-8")

    from_path = DataGeneration(
        root=tmp_path / "from_path",
        base=base_path.as_posix(),
        overrides=[{"name": "case", "cases": [{"name": "one", "value": None}]}],
        graph=graph,
    )
    empty_base = DataGeneration(
        root=tmp_path / "empty_base",
        overrides=[
            {"name": "case", "cases": [{"name": "one", "value": {"source": {"samples": 1, "seed": 0}}}]}
        ],
        graph=graph,
    )

    assert from_path.run() == 0
    assert empty_base.run() == 0
    assert (tmp_path / "from_path" / "case=one" / "source" / "seed_0" / "sample_0" / "source.h5").exists()
    assert (tmp_path / "empty_base" / "case=one" / "source" / "seed_0" / "sample_0" / "source.h5").exists()


def test_data_generation_rejects_mixed_dataset_modes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="both 'datasets' and base/overrides"):
        DataGeneration(
            root=tmp_path,
            datasets={"source": {"samples": 1, "seed": 0}},
            base={"source": {"samples": 1, "seed": 0}},
            overrides=[{"name": "source", "cases": [{"name": "one", "value": None}]}],
        )


def test_data_loader_source_lru_cache_reuses_samples(tmp_path, monkeypatch):
    _write_source_dataset(tmp_path, sample_count=1)
    counts: dict[Path, int] = {}
    real_load_h5 = load_h5

    def counting_load_h5(data, filename, mode="r", jax=False):
        path = Path(filename)
        counts[path] = counts.get(path, 0) + 1
        return real_load_h5(data, filename, mode=mode, jax=jax)

    monkeypatch.setattr("romjax.data_gen.load_h5", counting_load_h5)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "source": {
                "kind": "source",
                "batch_size": 1,
                "max_epochs": 2,
                "cache_policy": "lru",
                "cache_max_items": 1,
            }
        },
    )

    batches = list(loader)

    assert len(batches) == 2
    assert sorted(counts.values()) == [1]


def test_data_loader_source_cache_storage_device_returns_device_arrays(tmp_path, monkeypatch):
    _write_source_dataset(tmp_path, sample_count=1)
    device_put_calls: list[object] = []
    real_device_put = jax.device_put

    def counting_device_put(value, device=None):
        device_put_calls.append(value)
        return real_device_put(value, device=device)

    monkeypatch.setattr("romjax.data_gen.jax.device_put", counting_device_put)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "source": {
                "kind": "source",
                "batch_size": 1,
                "max_epochs": 2,
                "cache_policy": "lru",
                "cache_storage": "device",
                "cache_max_items": 1,
            }
        },
    )

    batch = next(loader)

    assert len(device_put_calls) == 1
    assert len(batch["source"]) == 1
    assert isinstance(batch["source"][0]["state"]["x"], jax.Array)


def test_data_loader_source_cache_max_bytes_skips_oversized_items(tmp_path, monkeypatch):
    root = tmp_path / "source"
    small_dir = root / "train" / "source" / "seed_0" / "sample_0"
    small_dir.mkdir(parents=True, exist_ok=True)
    save_h5({"state": {"x": np.asarray([0.0], dtype=np.float32)}}, small_dir / "source.h5", mode="w")
    large_dir = root / "train" / "source" / "seed_0" / "sample_1"
    large_dir.mkdir(parents=True, exist_ok=True)
    save_h5({"state": {"x": np.arange(1024, dtype=np.float32)}}, large_dir / "source.h5", mode="w")

    counts: dict[Path, int] = {}
    real_load_h5 = load_h5

    def counting_load_h5(data, filename, mode="r", jax=False):
        path = Path(filename)
        counts[path] = counts.get(path, 0) + 1
        return real_load_h5(data, filename, mode=mode, jax=jax)

    monkeypatch.setattr("romjax.data_gen.load_h5", counting_load_h5)
    loader = DataLoader(
        root=root,
        datasets={
            "train": {
                "source": {
                    "kind": "source",
                    "batch_size": 1,
                    "max_epochs": 2,
                    "cache_policy": "lru",
                    "cache_max_bytes": 64,
                }
            }
        },
    )

    list(loader)

    assert counts[small_dir / "source.h5"] == 1
    assert counts[large_dir / "source.h5"] == 2


def test_data_loader_implicit_epoch_cache_reuses_full_epoch(tmp_path, monkeypatch):
    _write_implicit_dataset(tmp_path, sample_count=2)
    counts: dict[Path, int] = {}
    real_load_h5 = load_h5

    def counting_load_h5(data, filename, mode="r", jax=False):
        path = Path(filename)
        counts[path] = counts.get(path, 0) + 1
        return real_load_h5(data, filename, mode=mode, jax=jax)

    monkeypatch.setattr("romjax.data_gen.load_h5", counting_load_h5)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "toy": {
                "kind": "implicit",
                "batch_size": 1,
                "max_epochs": 2,
                "load_solution": False,
                "cache_policy": "epoch",
            }
        },
    )

    batches = list(loader)

    assert len(batches) == 4
    assert counts[tmp_path / "toy" / "seed_0" / "sample_0" / "input.h5"] == 1
    assert counts[tmp_path / "toy" / "seed_0" / "sample_0" / "seed_0" / "sample_0" / "output.h5"] == 1
    assert counts[tmp_path / "toy" / "seed_0" / "sample_1" / "input.h5"] == 1
    assert counts[tmp_path / "toy" / "seed_0" / "sample_1" / "seed_0" / "sample_0" / "output.h5"] == 1


def test_generate_galerkin_compression_from_transport_data(tmp_path: Path) -> None:
    _write_transport_dataset(tmp_path)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "train": {
                "transport": {
                    "kind": "implicit",
                    "batch_size": 1,
                    "load_solution": False,
                }
            }
        },
    )
    artifact_path = tmp_path / "train" / "compression" / "galerkin_compression.h5"
    generator = GenLatent(
        loader=loader,
        gather_paths=[("outputs", "phi")],
        compression=_DummyCompression(scale=2.0, rank=1),
        filename="galerkin_compression.h5",
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
    _write_transport_dataset(tmp_path, sample_count=5)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "train": {
                "transport": {
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
    assert sum(len(batch["transport"]) for batch in batches) == 3
    inputs = [
        float(np.asarray(sample["inputs"]["x"]).reshape(-1)[0])
        for batch in batches
        for sample in batch["transport"]
    ]
    assert len(np.unique(np.asarray(inputs))) == 3


def test_load_implicit_model_respects_global_input_and_output_caps(tmp_path: Path) -> None:
    _write_transport_dataset(tmp_path, sample_count=4, outputs_per_input=3)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "train": {
                "transport": {
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
    inputs = [
        float(np.asarray(sample["inputs"]["x"]).reshape(-1)[0])
        for batch in batches
        for sample in batch["transport"]
    ]

    assert sum(len(batch["transport"]) for batch in batches) == 4
    assert len(np.unique(np.asarray(inputs))) == 2
    assert all(count <= 2 for count in np.unique(np.asarray(inputs), return_counts=True)[1])


def test_loader_selects_epoch_pool_once_per_dataset(tmp_path: Path) -> None:
    _write_source_dataset(tmp_path, dataset_name="train/source", sample_count=6)
    source_cfg = _TrackingLoadSource(
        batch_size=2,
        max_samples=3,
        max_epochs=2,
        shuffle_seed=0,
    )
    loader = DataLoader(root=tmp_path, datasets={"train": {"source": source_cfg}})

    batches = list(loader)

    assert source_cfg.select_calls == 1
    assert sum(len(batch["source"]) for batch in batches) == 6


def test_generate_svd_galerkin_compression_with_template_cache(tmp_path: Path) -> None:
    _write_transport_dataset(tmp_path)
    for sample_idx in range(2):
        output_dir = (
            tmp_path
            / "train"
            / "transport"
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
                "transport": {
                    "kind": "implicit",
                    "batch_size": 1,
                    "load_solution": False,
                }
            }
        },
    )
    artifact_path = tmp_path / "train" / "compression" / "galerkin_compression.h5"
    generator = GenLatent(
        loader=loader,
        gather_paths=[("outputs", "phi"), ("outputs", "psi")],
        compression=SVD(rank=1, center=False),
        filename="galerkin_compression.h5",
    )

    generator.generate(tmp_path / "train" / "compression", format="h5", write_policy="overwrite")
    compression = Compression.load(artifact_path)
    assert isinstance(compression, SVD)
    assert compression.rank == 1
    assert compression.latent_size() == 1
    assert compression.template is not None
    reconstructed = compression.reconstruct(compression.compress(compression.template))
    assert reconstructed["outputs"]["phi"].shape == (2,)
    assert reconstructed["outputs"]["psi"].shape == (2,)


def test_generate_latent_applies_norm_before_compression_fit(tmp_path: Path) -> None:
    _write_implicit_dataset(tmp_path, dataset_name="train/toy", sample_count=2, output_width=1)
    loader = DataLoader(
        root=tmp_path,
        datasets={"train": {"toy": {"kind": "implicit", "batch_size": 1, "load_solution": False, "max_epochs": 1}}},
    )
    artifact_path = tmp_path / "train" / "compression" / "normalized_compression.h5"
    generator = GenLatent(
        loader=loader,
        gather_paths=[("outputs", "y")],
        norm=NormTree(root={"outputs": {"y": {"callable": "zscore", "mean": 1.0, "std": 2.0}}}),
        compression=_DummyCompression(),
        filename="normalized_compression.h5",
    )

    generator.generate(tmp_path / "train" / "compression", format="h5", write_policy="overwrite")
    compression = Compression.load(artifact_path)

    assert jnp.allclose(compression.template["outputs"]["y"], jnp.asarray([-0.5]))


def test_generate_latent_uses_dataset_specific_norms(tmp_path: Path) -> None:
    _write_implicit_dataset(tmp_path, dataset_name="train/toy", sample_count=1, output_width=1)
    _write_implicit_dataset(tmp_path, dataset_name="train/alt", sample_count=1, output_width=1)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "train": {
                "toy": {"kind": "implicit", "batch_size": 1, "load_solution": False, "max_epochs": 1},
                "alt": {"kind": "implicit", "batch_size": 1, "load_solution": False, "max_epochs": 1},
            }
        },
    )
    generator = GenLatent(
        loader=loader,
        dataset="toy",
        norm={
            "toy": NormTree(root={"outputs": {"y": {"callable": "zscore", "mean": 1.0, "std": 1.0}}}),
            "alt": NormTree(root={"outputs": {"y": {"callable": "zscore", "mean": 0.0, "std": 1.0}}}),
        },
        compression=_DummyCompression(),
    )

    samples = list(generator._iter_samples())

    assert len(samples) == 1
    assert jnp.allclose(samples[0]["outputs"]["y"], jnp.asarray([-1.0]))


def test_generate_latent_reuses_single_norm_for_all_datasets(tmp_path: Path) -> None:
    _write_implicit_dataset(tmp_path, dataset_name="train/toy", sample_count=1, output_width=1)
    _write_implicit_dataset(tmp_path, dataset_name="train/alt", sample_count=1, output_width=1)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "train": {
                "toy": {"kind": "implicit", "batch_size": 1, "load_solution": False, "max_epochs": 1},
                "alt": {"kind": "implicit", "batch_size": 1, "load_solution": False, "max_epochs": 1},
            }
        },
    )
    generator = GenLatent(
        loader=loader,
        dataset="toy",
        norm=NormTree(root={"outputs": {"y": {"callable": "zscore", "mean": 1.0, "std": 1.0}}}),
        compression=_DummyCompression(),
    )

    samples = list(generator._iter_samples())

    assert len(samples) == 1
    assert jnp.allclose(samples[0]["outputs"]["y"], jnp.asarray([-1.0]))


def test_data_generation_config_keeps_normalized_genlatent_as_latent(tmp_path: Path) -> None:
    _write_implicit_dataset(tmp_path, dataset_name="data/toy", sample_count=2, output_width=1)
    generation = DataGeneration(
        root=tmp_path,
        datasets={
            "artifacts": {
                "compression": {
                    "loader": {
                        "root": tmp_path,
                        "datasets": {
                            "data": {
                                "toy": {
                                    "kind": "implicit",
                                    "batch_size": 1,
                                    "load_solution": False,
                                    "max_epochs": 1,
                                }
                            }
                        },
                    },
                    "gather_paths": [("toy", "outputs", "y")],
                    "norm": {"toy": {"outputs": {"y": {"callable": "zscore", "mean": 1.0, "std": 2.0}}}},
                    "compression": _DummyCompression(),
                    "filename": "normalized_compression.h5",
                }
            }
        },
        write_policy="overwrite",
    )

    assert isinstance(generation.datasets["artifacts"]["compression"], GenLatent)
    generation.run()

    expected_artifact = tmp_path / "artifacts" / "compression" / "normalized_compression.h5"
    unexpected_artifact = tmp_path / "artifacts" / "compression" / "toy_normalized_compression.h5"
    assert expected_artifact.exists()
    assert not unexpected_artifact.exists()


def test_generate_zscore_norm_artifact_from_loaded_data(tmp_path: Path) -> None:
    _write_implicit_dataset(tmp_path, dataset_name="train/toy", sample_count=2, output_width=2)
    loader = DataLoader(
        root=tmp_path,
        datasets={"train": {"toy": {"kind": "implicit", "batch_size": 1, "load_solution": False, "max_epochs": 1}}},
    )
    generator = GenNorm(
        loader=loader,
        norm={"toy": {"outputs": {"y": {"callable": "zscore"}}}},
        filename="toy_norm.h5",
    )

    generator.generate(tmp_path / "train" / "norm", format="h5", write_policy="overwrite")

    artifact = tmp_path / "train" / "norm" / "toy_toy_norm.h5"
    norm = NormTree(root=str(artifact))
    out = norm({"outputs": {"y": jnp.asarray([0.0, 1.0], dtype=jnp.float32)}})

    assert artifact.exists()
    assert jnp.allclose(out["outputs"]["y"], jnp.asarray([-1.4142135, 0.0]))


def test_generate_minmax_norm_artifact_with_leaf_axes_and_opts(tmp_path: Path) -> None:
    for sample_idx in range(2):
        sample_dir = tmp_path / "source" / "seed_0" / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        values = np.asarray([[sample_idx, sample_idx + 10], [sample_idx + 1, sample_idx + 11]], dtype=np.float32)
        save_h5({"state": {"x": values}}, sample_dir / "source.h5", mode="w")

    loader = DataLoader(root=tmp_path, datasets={"source": {"kind": "source", "batch_size": 1, "max_epochs": 1}})
    generator = GenNorm(
        loader=loader,
        norm={"source": {"state": {"x": {"callable": "minmax", "axes": -1, "ymin": -1.0, "ymax": 1.0}}}},
        filename="source_norm.h5",
    )

    generator.generate(tmp_path / "norm", format="h5", write_policy="overwrite")

    constants = load_h5({}, tmp_path / "norm" / "source_source_norm.h5", jax=False)
    x_constants = constants["tree"]["state"]["x"]
    norm = NormTree(root=str(tmp_path / "norm" / "source_source_norm.h5"))
    out = norm({"state": {"x": jnp.asarray([[0.0, 10.0], [2.0, 12.0]])}})

    assert x_constants["xmin"].shape == (2, 1)
    assert x_constants["xmax"].shape == (2, 1)
    assert jnp.allclose(out["state"]["x"], jnp.asarray([[-1.0, 0.8181819], [-0.8181818, 1.0]]))


def test_generate_norm_outlier_filter_limits_minmax_range(tmp_path: Path) -> None:
    values = [0.0, 1.0, 2.0, 1000.0]
    for sample_idx, value in enumerate(values):
        sample_dir = tmp_path / "source" / "seed_0" / f"sample_{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_h5({"state": {"x": np.asarray([value], dtype=np.float32)}}, sample_dir / "source.h5", mode="w")

    loader = DataLoader(root=tmp_path, datasets={"source": {"kind": "source", "batch_size": 1, "max_epochs": 1}})
    generator = GenNorm(
        loader=loader,
        norm={"source": {"state": {"x": {"callable": "minmax"}}}},
        outlier_filter={"threshold": 2.0, "warmup_samples": 3},
    )

    generator.generate(tmp_path / "norm", format="h5", write_policy="overwrite")

    constants = load_h5({}, tmp_path / "norm" / "source_norm.h5", jax=False)
    assert np.allclose(constants["tree"]["state"]["x"]["xmax"], np.asarray([2.0]))


def test_generate_norm_write_policy_reuse_and_error(tmp_path: Path) -> None:
    _write_source_dataset(tmp_path, dataset_name="source", sample_count=2)
    loader = DataLoader(root=tmp_path, datasets={"source": {"kind": "source", "batch_size": 1, "max_epochs": 1}})
    generator = GenNorm(loader=loader, norm={"source": {"state": {"x": "zscore"}}})

    generator.generate(tmp_path / "norm", format="h5", write_policy="overwrite")
    artifact = tmp_path / "norm" / "source_norm.h5"
    original_mtime = artifact.stat().st_mtime_ns
    generator.generate(tmp_path / "norm", format="h5", write_policy="reuse")
    assert artifact.stat().st_mtime_ns == original_mtime

    with pytest.raises(Exception, match="policy='error'"):
        generator.generate(tmp_path / "norm", format="h5", write_policy="error")


def test_generate_norm_writes_one_artifact_per_dataset(tmp_path: Path) -> None:
    _write_source_dataset(tmp_path, dataset_name="source", sample_count=2)
    _write_implicit_dataset(tmp_path, dataset_name="toy", sample_count=2, output_width=1)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "source": {"kind": "source", "batch_size": 1, "max_epochs": 1},
            "toy": {"kind": "implicit", "batch_size": 1, "load_solution": False, "max_epochs": 1},
        },
    )
    generator = GenNorm(
        loader=loader,
        norm={
            "source": {"state": {"x": "zscore"}},
            "toy": {"outputs": {"y": "minmax"}},
        },
    )

    generator.generate(tmp_path / "norm", format="h5", write_policy="overwrite")

    source_artifact = tmp_path / "norm" / "source_norm.h5"
    toy_artifact = tmp_path / "norm" / "toy_norm.h5"
    source_norm = NormTree(root=str(source_artifact))
    toy_norm = NormTree(root=str(toy_artifact))
    source_tree = load_h5({}, source_artifact, jax=False)["tree"]
    toy_tree = load_h5({}, toy_artifact, jax=False)["tree"]

    assert source_artifact.exists()
    assert toy_artifact.exists()
    assert "source" not in source_tree
    assert "toy" not in toy_tree
    assert "state" in source_tree
    assert "outputs" in toy_tree
    assert "state" in source_norm.resolve_root()
    assert "outputs" in toy_norm.resolve_root()


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
    assert len(batch["alpha"]) == 1
    assert len(batch["beta"]) == 1
    assert batch["alpha"][0]["inputs"]["x"].shape == (1,)
    assert batch["alpha"][0]["outputs"]["y"].shape == (1,)
    assert batch["beta"][0]["inputs"]["x"].shape == (1,)
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
    assert len(batch["toy"]) == 3
    assert len(batch["source"]) == 2
    assert batch["toy"][0]["inputs"]["x"].shape == (1,)
    assert batch["toy"][0]["outputs"]["y"].shape == (1,)
    assert batch["toy"][0]["residuals"]["residual"].shape == (1,)
    assert jax.device_get(batch["source"][0]["state"]["x"]).shape == ()
    assert batch["source"][0]["state"]["meta"].shape == (1,)


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
    assert len(batch) == 2
    assert [sample["outputs"]["y"][0] for sample in batch] == [2.0, 3.0]
    assert [sample["residuals"]["r"][0] for sample in batch] == [-1.0, 0.5]


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


def test_load_implicit_model_solution_only_skips_output_samples(tmp_path):
    input_dir = tmp_path / "toy" / "seed_0" / "sample_0"
    input_dir.mkdir(parents=True, exist_ok=True)
    save_h5({"x": jnp.asarray([1.0], dtype=jnp.float32)}, input_dir / "input.h5", mode="w")
    save_h5({"y": jnp.asarray([2.0], dtype=jnp.float32)}, input_dir / "solution.h5", mode="w")
    save_h5({"r": jnp.asarray([-1.0], dtype=jnp.float32)}, input_dir / "solution_residual.h5", mode="w")

    output_dir = input_dir / "seed_0" / "sample_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_h5({"y": jnp.asarray([3.0], dtype=jnp.float32)}, output_dir / "output.h5", mode="w")
    save_h5({"r": jnp.asarray([0.5], dtype=jnp.float32)}, output_dir / "residual.h5", mode="w")

    refs = LoadImplicitModel(batch_size=4, solution_only=True).discover_sample_refs(tmp_path / "toy")

    assert [sample_kind for _, _, sample_kind in refs] == ["solution"]
    batch = LoadImplicitModel(batch_size=4, solution_only=True).load_batch(refs)
    assert len(batch) == 1
    assert [sample["outputs"]["y"][0] for sample in batch] == [2.0]
    assert [sample["residuals"]["r"][0] for sample in batch] == [-1.0]


def test_load_implicit_model_solution_only_requires_solution_loading() -> None:
    with pytest.raises(ValueError, match="solution_only=True requires load_solution=True"):
        LoadImplicitModel(solution_only=True, load_solution=False)


@pytest.mark.parametrize('kind', ['serial', 'thread', 'process'])
@pytest.mark.parametrize('source', [False, True])
def test_generation_executors_match_serial(tmp_path, monkeypatch, kind, source):
    from romjax.data_gen import GenImplicitModel

    if kind == 'process':
        # Spawned workers otherwise compete with the parent for GPU preallocation.
        monkeypatch.setenv('JAX_PLATFORMS', 'cpu')

    edge = _ToySourceEdge(source='noise', target='sample') if source else _ToySampleableEdge(
        source='inputs', target='sample', shift=3.5)
    graph = FunctionGraph(edges={'sample': edge})
    options = {'samples': 5, 'seed': 7} if source else {
        'input_samples': 5, 'outputs_per_input': 3, 'input_seed': 7, 'output_seed': 8}
    generator_type = GenSource if source else GenImplicitModel
    roots = [tmp_path / 'reference', tmp_path / 'actual']
    for root, backend in zip(roots, ['serial', kind]):
        config = {'kind': backend, 'show_progress': False}
        if backend in {'thread', 'process'}:
            config['max_workers'] = 2
        generation = DataGeneration(root=root, graph=graph, executor=config, datasets={
            'sample': generator_type(**options, batch_size=2)})
        assert generation.run() == 0
    paths = sorted(path.relative_to(roots[0]) for path in roots[0].rglob('*.h5'))
    assert paths == sorted(path.relative_to(roots[1]) for path in roots[1].rglob('*.h5'))
    for path in paths:
        before = load_h5({}, roots[0] / path)
        after = load_h5({}, roots[1] / path)
        for left, right in zip(jax.tree.leaves(before), jax.tree.leaves(after)):
            np.testing.assert_allclose(left, right)


def test_generation_leaf_executor_overrides_parent(tmp_path):
    graph = FunctionGraph(edges={'sample': _ToySourceEdge(source='noise', target='sample')})
    generation = DataGeneration(root=tmp_path, graph=graph, executor='mpi', datasets={
        'sample': GenSource(samples=2, seed=1, executor='serial')})
    assert generation.run() == 0
    assert len(list(tmp_path.rglob('source.h5'))) == 2


class _FileCompressionEdge(_CompressionSampleableEdge):
    """Load a separately declared sampler using the normal YAML validator."""
    outputs_sampler: Annotated[CompressionSampler, BeforeValidator(from_yaml)]


def test_process_generation_round_trips_external_graph_and_compression(tmp_path, monkeypatch):
    import romjax

    # Keep GPU execution in the parent; only spawned workers inherit this setting.
    monkeypatch.setenv('JAX_PLATFORMS', 'cpu')

    config_dir = tmp_path / 'configuration'
    config_dir.mkdir()
    compression = SVD(rank=1).fit([
        {'outputs': {'y': jnp.asarray([value], dtype=jnp.float32)}} for value in [1., 2., 3.]])
    compression.dump(config_dir / 'compression.h5')
    (config_dir / 'sampler.yml').write_text(
        '!romx:CompressionSampler\ncompression: __parent__/compression.h5\n'
        'distribution: uniform\nreconstruct: true\n', encoding='utf-8')
    (config_dir / 'graph.yml').write_text(
        '!romx:FunctionGraph\nedges:\n  sample: !pd:tests.test_data_gen._FileCompressionEdge\n'
        '    source: inputs\n    target: outputs\n    outputs_sampler: __parent__/sampler.yml\n', encoding='utf-8')
    config = config_dir / 'generation.yml'
    config.write_text(
        f'!romx:DataGeneration\nroot: {tmp_path / "data"}\n'
        'executor: {kind: process, max_workers: 2, show_progress: false}\n'
        'graph: __parent__/graph.yml\ndatasets:\n  sample:\n'
        '    input_samples: 3\n    outputs_per_input: 2\n    input_seed: 1\n    output_seed: 2\n'
        '    batch_size: 2\n    skip_solve: true\n    skip_evaluate: true\n',
        encoding='utf-8')
    generation = romjax.load(config)
    edge = generation.graph.edges['sample']
    assert isinstance(edge.outputs_sampler, CompressionSampler)
    text = romjax.YamlLoader.dump(generation.datasets['sample'], _preserve_model_types=True)
    restored = romjax.load(text)
    restored_edge = restored.graph.edges['sample']
    assert type(restored_edge) is type(edge)
    assert Path(restored_edge.outputs_sampler.compression) == config_dir / 'compression.h5'
    key = jax.random.key(10)
    np.testing.assert_allclose(edge.sample_outputs(key)['y'], restored_edge.sample_outputs(key)['y'])
    # The cached compression contains arrays; they must never enter the worker spec.
    monkeypatch.chdir(tmp_path)
    assert generation.run() == 0
    assert len(list((tmp_path / 'data').rglob('output.h5'))) == 6


def test_remote_generation_rejects_nonserializable_model_before_writing(tmp_path):
    graph = FunctionGraph(edges={'sample': _ToySourceEdge(source='noise', target='sample')})
    # A local runtime callable cannot be imported by a fresh worker.
    from pydantic import Field

    from romjax.typing import CallableModel

    class LocalEdge(_ToySourceEdge):
        sampler: Any = Field(default_factory=lambda: CallableModel(callable=lambda: 1))

    graph.edges['sample'] = LocalEdge(source='noise', target='sample')
    generator = GenSource(graph=graph, samples=2, seed=1, executor='process')
    with pytest.raises(ExecutionError, match='Cannot reconstruct'):
        generator.generate(tmp_path / 'sample', format='h5', write_policy='error')
    assert not (tmp_path / 'sample').exists()


def test_generation_error_policy_checked_before_batch_submission(tmp_path):
    graph = FunctionGraph(edges={'sample': _ToySourceEdge(source='noise', target='sample')})
    destination = tmp_path / 'sample'
    destination.mkdir()
    generator = GenSource(graph=graph, samples=4, seed=1, executor='thread', batch_size=2)
    with pytest.raises(RoutineError, match="policy='error'"):
        generator.generate(destination, format='h5', write_policy='error')
    assert not list(destination.iterdir())


@pytest.mark.parametrize('batched', [False, True])
def test_direct_generation_cache_distinguishes_graph_edges(tmp_path, batched):
    graph = FunctionGraph(edges={
        name: _ToySourceEdge(source='noise', target=name, bias=bias)
        for name, bias in [('first', 1.), ('second', 3.)]})
    generator = GenSource(graph=graph, samples=1, seed=4)
    generate = generator._generate_batch if batched else generator._generate_serial
    for name in ['first', 'second']:
        generate(tmp_path / name, (0,), 'h5', 'reuse')
    first = load_h5({}, next((tmp_path / 'first').rglob('source.h5')))
    second = load_h5({}, next((tmp_path / 'second').rglob('source.h5')))
    assert float(second['state']['x']) - float(first['state']['x']) == pytest.approx(2.)


def test_thread_generation_throw_false_preserves_other_samples(tmp_path, monkeypatch):
    monkeypatch.setattr('romjax.data_gen.eqx.filter_jit', lambda fn: fn)
    graph = FunctionGraph(edges={'source': _FailingSourceEdge(source='noise', target='source')})
    generation = DataGeneration(root=tmp_path, graph=graph,
                                executor={'kind': 'thread', 'max_workers': 2, 'show_progress': False},
                                datasets={'source': {'samples': 6, 'seed': 0, 'batch_size': 2,
                                                     'throw': False, }})
    assert generation.run() == 0
    failed = _failed_sample_dirs(tmp_path / 'source')
    successful = list((tmp_path / 'source').rglob('source.h5'))
    assert failed and successful
    assert len(failed) + len(successful) == 6
    assert all((path / 'failure.log').exists() for path in failed)
