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
    GenNorm,
    GenSource,
    LoadImplicitModel,
    LoadSource,
)
from romjax.graph import FunctionGraph
from romjax.model import Edge, ImplicitSampleable, SourceSampleable
from romjax.norm import NormTree
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


def _get_graph():
    graph = FunctionGraph(
        edges={
            "low": _ToySampleableEdge(source="inputs", target="low", shift=1.0),
            "high": _ToySampleableEdge(source="inputs", target="high", shift=2.0),
        }
    )
    return graph


@pytest.mark.parametrize("batch_size", [1, 2])
def test_generate_implicit_persists_and_loads_conditions(tmp_path, batch_size):
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
        sample = LoadImplicitModel().load_sample((input_path, output_path, "output"))
        assert np.allclose(sample["conditions"]["x"], 4.0)
        assert np.allclose(sample["outputs"]["y"], 5.0)


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
    artifact_path = tmp_path / "train" / "compression" / "galerkin_compression.npz"
    generator = GenLatent(
        loader=loader,
        gather_paths=[("transport", "outputs", "phi")],
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
    artifact_path = tmp_path / "train" / "compression" / "galerkin_compression.npz"
    generator = GenLatent(
        loader=loader,
        gather_paths=[("transport", "outputs", "phi"), ("transport", "outputs", "psi")],
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
    assert reconstructed["transport"]["outputs"]["phi"].shape == (2,)
    assert reconstructed["transport"]["outputs"]["psi"].shape == (2,)


def test_generate_latent_applies_norm_before_compression_fit(tmp_path: Path) -> None:
    _write_implicit_dataset(tmp_path, dataset_name="train/toy", sample_count=2, output_width=1)
    loader = DataLoader(
        root=tmp_path,
        datasets={"train": {"toy": {"kind": "implicit", "batch_size": 1, "load_solution": False, "max_epochs": 1}}},
    )
    artifact_path = tmp_path / "train" / "compression" / "normalized_compression.npz"
    generator = GenLatent(
        loader=loader,
        gather_paths=[("toy", "outputs", "y")],
        norm=NormTree(root={"outputs": {"y": {"callable": "zscore", "mean": 1.0, "std": 2.0}}}),
        compression=_DummyCompression(),
        filename="normalized_compression.npz",
    )

    generator.generate(tmp_path / "train" / "compression", format="h5", write_policy="overwrite")
    compression = Compression.load(artifact_path)

    assert jnp.allclose(compression.template["toy"]["outputs"]["y"], jnp.asarray([-0.5]))


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
        norm={
            "toy": NormTree(root={"outputs": {"y": {"callable": "zscore", "mean": 1.0, "std": 1.0}}}),
            "alt": NormTree(root={"outputs": {"y": {"callable": "zscore", "mean": 0.0, "std": 1.0}}}),
        },
        compression=_DummyCompression(),
    )

    samples = list(generator._iter_samples(progress=False))

    assert jnp.allclose(samples[0]["toy"]["outputs"]["y"], jnp.asarray([-1.0]))
    assert jnp.allclose(samples[1]["alt"]["outputs"]["y"], jnp.asarray([0.0]))


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
        norm=NormTree(root={"outputs": {"y": {"callable": "zscore", "mean": 1.0, "std": 1.0}}}),
        compression=_DummyCompression(),
    )

    samples = list(generator._iter_samples(progress=False))

    assert jnp.allclose(samples[0]["toy"]["outputs"]["y"], jnp.asarray([-1.0]))
    assert jnp.allclose(samples[1]["alt"]["outputs"]["y"], jnp.asarray([-1.0]))


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
                    "filename": "normalized_compression.npz",
                }
            }
        },
        write_policy="overwrite",
    )

    assert isinstance(generation.datasets["artifacts"]["compression"], GenLatent)
    generation.run()

    expected_artifact = tmp_path / "artifacts" / "compression" / "normalized_compression.npz"
    unexpected_artifact = tmp_path / "artifacts" / "compression" / "toy_normalized_compression.npz"
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
