import json
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any

import equinox as eqx
import h5py
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
    GenImplicitModel,
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
from romjax.storage import ComponentStatus, H5Store, Record
from romjax.tree import pytree_stack
from romjax.typing import from_yaml
from romjax.utils import load_h5
from tests.storage_helpers import write_component as save_h5


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


class _FamilySampleableEdge(_ToySampleableEdge):
    """Make every family, output key, and condition visible in the result."""

    def sample_conditions(self, key):
        return {"offset": jax.random.normal(key, (1,))}

    def sample_outputs(self, key, inputs=None, solution=None, conditions=None):
        base = inputs["x"] if solution is None else solution["y"]
        return {"y": base + conditions["offset"] + jax.random.uniform(key, (1,))}


@pytest.mark.parametrize("width", [0, 1, 2])
@pytest.mark.parametrize("mix", [False, True])
@pytest.mark.parametrize(
    "edge_type,solve,count",
    [
        (_FamilySampleableEdge, True, 3),
        (_FamilySampleableEdge, False, 3),
        (_ToySampleableEdge, True, 3),
        (_NoneInputEdge, False, 3),
        (_FamilySampleableEdge, True, 0),
    ],
)
def test_family_kernel_matches_scalar_calls(tmp_path, width, mix, edge_type, solve, count):
    model = edge_type(source="inputs", target="sample")
    generator = GenImplicitModel(
        graph=FunctionGraph(edges={"sample": model}),
        input_samples=3,
        outputs_per_input=count,
        input_seed=3,
        output_seed=5,
        mix_output_seed=mix,
        map_batch_size=width,
    )
    conditioned = isinstance(model, _FamilySampleableEdge)
    evaluate = hasattr(model, "evaluate")
    keys = pytree_stack([jax.random.fold_in(jax.random.key(3), i) for i in (2, 4, 7)])
    output_keys = GenDataBenchmark._output_keys(generator, keys)
    expected_inputs, expected_outputs = [], []
    for i, key in enumerate(keys):
        inputs = model.sample_inputs(key)
        solution = model.solve(inputs) if solve else None
        residual = model.evaluate(inputs, solution) if solve and evaluate else None
        expected_inputs.append((inputs, solution, residual))
        family = []
        for j in range(count):
            out_key = output_keys[i, j]
            conditions = model.sample_conditions(out_key) if conditioned else None
            kwargs = {"conditions": conditions} if conditioned else {}
            outputs = model.sample_outputs(out_key, inputs=inputs, solution=solution, **kwargs)
            family.append((conditions, outputs, model.evaluate(inputs, outputs) if evaluate else None))
        if count:
            expected_outputs.append(pytree_stack(family))
    expected = (pytree_stack(expected_inputs), pytree_stack(expected_outputs) if count else None)
    kernel = generator._family_kernel(
        tmp_path / "sample",
        (True, solve, solve and evaluate),
        (conditioned, True, evaluate) if count else None,
    )
    actual = jax.block_until_ready(kernel(keys, (None, None, None), output_keys, (None, None, None)))
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for lhs, rhs in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("width", [0, 1, 2])
def test_generation_submits_complete_family_batches(tmp_path, monkeypatch, width):
    generator = GenImplicitModel(
        graph=FunctionGraph(edges={"sample": _FamilySampleableEdge(source="inputs", target="sample")}),
        input_samples=5,
        outputs_per_input=3,
        input_seed=3,
        output_seed=5,
        batch_size=3,
        map_batch_size=width,
        executor={"show_progress": False},
    )
    reference = generator.model_copy(update={"throw": False}, deep=True)
    reference.generate(tmp_path / "scalar" / "sample", "h5", "overwrite")
    calls = []
    original = generator._family_kernel

    def counted(*args):
        kernel = original(*args)

        def run(*values):
            calls.append(len(values[0]))
            return kernel(*values)

        return run

    monkeypatch.setattr(generator, "_family_kernel", counted)
    generator.generate(tmp_path / "mapped" / "sample", "h5", "overwrite")
    assert calls == [3, 2]
    left, right = [H5Store(tmp_path / kind / "sample") for kind in ("scalar", "mapped")]
    refs = [store.discover(include_incomplete=True) for store in (left, right)]
    assert [ref.identity for ref in refs[0]] == [ref.identity for ref in refs[1]]
    for a, b in zip(left.read_components(refs[0]), right.read_components(refs[1]), strict=True):
        assert jax.tree.structure(a) == jax.tree.structure(b)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6)
    for ref in refs[0]:
        assert left.status(ref.seed, ref.input_index, ref.output_seed, ref.output_index) == right.status(
            ref.seed, ref.input_index, ref.output_seed, ref.output_index
        )
    calls.clear()
    generator.generate(tmp_path / "mapped" / "sample", "h5", "reuse")
    assert calls == []


def test_family_generation_partial_reuse(tmp_path):
    generator = GenImplicitModel(
        graph=FunctionGraph(edges={"sample": _FamilySampleableEdge(source="inputs", target="sample")}),
        input_samples=4,
        outputs_per_input=3,
        input_seed=3,
        output_seed=5,
        batch_size=4,
        executor={"show_progress": False},
    )
    path = tmp_path / "sample"
    generator.generate(path, "h5", "overwrite")
    store = H5Store(path)
    refs = [ref for ref in store.discover(include_incomplete=True) if ref.kind != "solution"]
    before = {ref.identity: data for ref, data in zip(refs, store.read_components(refs), strict=True)}
    # One ragged family, one residual-only family, one missing solve, and one complete family.
    store.write(
        [
            Record(3, 0, "output", {}, 5, 0, {"values": ComponentStatus.STALE}),
            Record(3, 0, "output", {}, 5, 2, {"residuals": ComponentStatus.STALE}),
            Record(3, 1, "output", {}, 5, 2, {"residuals": ComponentStatus.STALE}),
            Record(3, 2, "input", {}, states={"solutions": ComponentStatus.STALE}),
            # A changed parent invalidates its dependent conditions in the storage contract.
            *[Record(3, 2, "output", {}, 5, j, {"conditions": ComponentStatus.STALE}) for j in range(3)],
        ]
    )
    plans = generator._plans(store, range(4), "reuse")
    assert [plan["index"] for plan in plans] == [0, 1, 2]
    assert [output["index"] for output in plans[0]["outputs"]] == [0, 2]
    expected = generator._compute(path, plans, mapped=False)
    actual = generator._compute(path, plans)
    assert actual.fatal is None
    assert [(r.input_index, r.output_index) for r in actual.records] == [
        (r.input_index, r.output_index) for r in expected.records
    ]
    for a, b in zip(actual.records, expected.records, strict=True):
        assert a.states == b.states
        assert jax.tree.structure(a.components) == jax.tree.structure(b.components)
        for x, y in zip(jax.tree.leaves(a.components), jax.tree.leaves(b.components), strict=True):
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6)
    store.write(actual.records)
    for ref, data in zip(refs, store.read_components(refs), strict=True):
        for x, y in zip(jax.tree.leaves(data), jax.tree.leaves(before[ref.identity]), strict=True):
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6)
    assert generator._plans(store, range(4), "reuse") == []


class _FailingFamilyEdge(_FamilySampleableEdge):
    fail_input: float

    def sample_outputs(self, key, inputs=None, solution=None, conditions=None):
        outputs = super().sample_outputs(key, inputs, solution, conditions)
        return eqx.error_if(outputs, jnp.isclose(inputs["x"][0], self.fail_input), "family output failure")


@pytest.mark.parametrize("throw", [False, True])
def test_family_failure_recovers_original_identity(tmp_path, throw):
    fail_input = float(jax.random.uniform(jax.random.fold_in(jax.random.key(3), 1)))
    generator = GenImplicitModel(
        graph=FunctionGraph(
            edges={
                "sample": _FailingFamilyEdge(
                    source="inputs",
                    target="sample",
                    fail_input=fail_input,
                )
            }
        ),
        input_samples=3,
        outputs_per_input=2,
        input_seed=3,
        output_seed=5,
        batch_size=3,
        throw=throw,
        map_batch_size=2,
    )
    path = tmp_path / "sample"
    store = H5Store(path, "implicit")
    plans = generator._plans(store, range(3), "overwrite")
    result = generator._compute(path, plans)
    failed = [record for record in result.records if record.failure]
    assert [(r.input_index, r.output_index, r.kind) for r in failed] == (
        [(1, 0, "output")] if throw else [(1, 0, "output"), (1, 1, "output")]
    )
    assert all("family output failure" in record.failure for record in failed)
    assert bool(result.fatal) is throw
    successful_outputs = [
        (r.input_index, r.output_index) for r in result.records if r.kind == "output" and not r.failure
    ]
    assert successful_outputs == ([(0, 0), (0, 1)] if throw else [(0, 0), (0, 1), (2, 0), (2, 1)])


def test_family_transform_failure_is_not_silently_ignored(tmp_path, monkeypatch):
    generator = GenImplicitModel(
        graph=FunctionGraph(edges={"sample": _ToySampleableEdge(source="inputs", target="sample")}),
        input_samples=2,
        outputs_per_input=1,
        input_seed=3,
        output_seed=5,
    )
    path = tmp_path / "sample"
    plans = generator._plans(H5Store(path, "implicit"), range(2), "overwrite")

    def fail(*args):
        raise ValueError("mapped-only failure")

    monkeypatch.setattr(generator, "_family_kernel", fail)
    with pytest.raises(RuntimeError, match="individual samples succeeded: mapped-only failure"):
        generator._compute(path, plans)


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
def test_generate_implicit_persists_and_loads_conditions(tmp_path, batch_size):
    graph = FunctionGraph(edges={"conditioned": _ConditionedSampleableEdge(source="inputs", target="conditioned")})
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
    output_refs = [ref for ref in refs if ref.kind == "output"]
    assert len(output_refs) == 4
    for conditions, expected_inputs, has_conditions in (
        ("ignore", 1.0, False),
        ("include", 1.0, True),
        ("merge", 4.0, False),
    ):
        loader = LoadImplicitModel(conditions=conditions)
        for ref in output_refs:
            sample = loader.load_sample(ref)
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
    store = H5Store(tmp_path / "conditioned")
    samples = store.read_batch(store.discover())
    assert len(samples) == 4
    for sample in samples:
        assert sample["outputs"]["y"].shape == (1,)
        assert jnp.isfinite(sample["outputs"]["y"]).all()


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

    store = H5Store(tmp_path / "mixed" / "random")
    mixed_values = [sample["outputs"]["y"] for sample in store.read_batch(store.discover())]
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

    store = H5Store(tmp_path / "legacy" / "random")
    legacy_values = [sample["outputs"]["y"] for sample in store.read_batch(store.discover())]
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
        save_h5({"x": np.asarray(sample_idx, dtype=np.float32)}, input_dir / "input.h5", mode="w")
        save_h5({"phi": phi}, input_dir / "solution.h5", mode="w")
        save_h5({"phi_residual": residual}, input_dir / "solution_residual.h5", mode="w")
        for output_idx in range(outputs_per_input):
            output_dir = input_dir / "seed_0" / f"sample_{output_idx}"
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
        save_h5({"state": {"x": np.asarray([sample_idx], dtype=np.float32)}}, sample_dir / "source.h5", mode="w")


def _write_implicit_dataset(
    root: Path,
    dataset_name: str = "toy",
    sample_count: int = 2,
    output_width: int = 1,
) -> None:
    for sample_idx in range(sample_count):
        input_dir = root / dataset_name / "seed_0" / f"sample_{sample_idx}"
        save_h5({"x": np.asarray([sample_idx], dtype=np.float32)}, input_dir / "input.h5", mode="w")
        save_h5({"y": np.asarray([sample_idx + 1], dtype=np.float32)}, input_dir / "solution.h5", mode="w")
        save_h5({"r": np.asarray([0.0], dtype=np.float32)}, input_dir / "solution_residual.h5", mode="w")

        output_dir = input_dir / "seed_0" / "sample_0"
        save_h5(
            {"y": np.arange(output_width, dtype=np.float32) + sample_idx},
            output_dir / "output.h5",
            mode="w",
        )
        save_h5({"r": np.asarray([0.0], dtype=np.float32)}, output_dir / "residual.h5", mode="w")


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
            store = H5Store(edge_dir)
            refs = store.discover()
            assert len(refs) == input_samples * 2
            assert {ref.seed for ref in refs} == {seed}
            samples = store.read_batch(refs)
            assert all(set(sample) == {"inputs", "outputs", "residuals"} for sample in samples)


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
    files = list(edge_dir.glob("*.h5"))
    failures = []
    for file in files:
        with h5py.File(file) as handle:
            failures.extend(json.loads(value) for value in handle["index/failures"].asstr()[:])
    assert failures
    assert all("failure" in item["message"] and "Traceback" in item["message"] for item in failures)
    store = H5Store(edge_dir)
    refs = store.discover()
    assert refs
    assert len({ref.identity for ref in refs}) == len(refs)


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
    files = list((edge_dir).glob("*.h5"))
    failures = []
    for file in files:
        with h5py.File(file) as handle:
            failures.extend(json.loads(value) for value in handle["index/failures"].asstr()[:])
    assert failures
    assert all("failure" in item["message"] and "Traceback" in item["message"] for item in failures)
    store = H5Store(edge_dir)
    refs = store.discover()
    assert refs
    assert len({ref.identity for ref in refs}) == len(refs)


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

    edge_dir = tmp_path / "source"
    files = list(edge_dir.glob("*.h5"))
    failures = []
    for file in files:
        with h5py.File(file) as handle:
            failures.extend(json.loads(value) for value in handle["index/failures"].asstr()[:])
    assert failures
    assert all("failure" in item["message"] and "Traceback" in item["message"] for item in failures)
    store = H5Store(edge_dir)
    refs = store.discover()
    assert refs
    assert len({ref.identity for ref in refs}) == len(refs)


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
            assert {ref.seed for ref in H5Store(source_root).discover()} == {seed}


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
        samples=2,
        repeat=2,
        warmup=1,
        show_progress=True,
        name_depth=1,
        hardware="test",
    )
    workload_calls = []
    monkeypatch.setattr(
        benchmark,
        "_workload",
        lambda path: (lambda value: workload_calls.append(value) or np.asarray(value), ((1,), (2,))),
    )
    ticks = iter([1.0, 5.0, 10.0, 22.0])
    clock_state = {"timing": False}

    def clock():
        clock_state["timing"] = not clock_state["timing"]
        return next(ticks)

    monkeypatch.setattr(execution.time, "perf_counter", clock)
    progress = {"updates": [], "messages": [], "pauses": 0}

    class FakeProgress:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __setattr__(self, name, value):
            progress["messages"].append(value)

        def __call__(self, units):
            progress["updates"].append(units)

        def pause(self):
            progress["pauses"] += 1
            return nullcontext()

    def progress_factory(total, **kwargs):
        progress["total"] = total
        progress["title"] = kwargs["title"]
        return FakeProgress()

    monkeypatch.setattr(execution, "alive_bar", progress_factory)
    totals = benchmark._benchmark(tmp_path / "sample")
    np.testing.assert_allclose(totals, [4.0, 12.0])
    assert workload_calls == [1, 2] * 3
    assert progress["total"] == 4 and progress["updates"] == [1, 1, 1, 1]
    assert progress["title"] == "sample"
    assert "Warming up..." in progress["messages"]
    assert progress["pauses"] == 0


def test_data_generation_benchmark_excludes_cache_io_from_timing(tmp_path, monkeypatch):
    import romjax.data_gen as data_gen
    import romjax.execution as execution

    graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
    benchmark = GenDataBenchmark(
        generator={"kind": "source", "graph": graph, "samples": 1, "seed": 0},
        samples=2,
        repeat=1,
        warmup=1,
        hardware="test",
        show_progress=False,
    )
    path = tmp_path / "sample"
    path.mkdir()
    save_h5({"previous": {"total": np.asarray([8.0])}}, path / "timings.h5", mode="w")
    state = {"timing": False}
    ticks = iter([10.0, 17.0])

    def clock():
        state["timing"] = not state["timing"]
        return next(ticks)

    def load(*args, **kwargs):
        assert not state["timing"]
        return load_h5(*args, **kwargs)

    def save(*args, **kwargs):
        assert not state["timing"]
        return save_h5(*args, **kwargs)

    monkeypatch.setattr(benchmark, "_workload", lambda path: (lambda value: np.asarray(value), ((1,), (2,))))
    monkeypatch.setattr(execution.time, "perf_counter", clock)
    monkeypatch.setattr(data_gen, "load_h5", load)
    monkeypatch.setattr(data_gen, "save_h5", save)
    benchmark.generate(path, format="h5", write_policy="overwrite")
    entries = load_h5({}, path / "timings.h5", jax=False)
    np.testing.assert_array_equal(entries.pop("previous")["total"], [8.0])
    np.testing.assert_array_equal(next(iter(entries.values()))["total"], [7.0])


@pytest.mark.parametrize("kind", ["implicit", "source"])
@pytest.mark.parametrize("executor", ["serial", "thread", "process"])
def test_data_generation_benchmark_uses_executor(tmp_path, kind, executor):
    if executor in {"thread", "process"} and jax.default_backend() == "gpu":
        pytest.skip("thread and process benchmark executors are unsupported on GPU")

    edge = (
        _ToySourceEdge(source="noise", target="sample")
        if kind == "source"
        else _ToySampleableEdge(source="inputs", target="sample")
    )
    graph = FunctionGraph(edges={"sample": edge})
    options = (
        {"samples": 1, "seed": 3}
        if kind == "source"
        else {"input_samples": 1, "outputs_per_input": 3, "input_seed": 3, "output_seed": 5}
    )
    repeat, warmup = (2, 1) if executor == "process" else (1, 0)
    benchmark = GenDataBenchmark(
        generator={"kind": kind, "batch_size": 2, **options},
        samples=5,
        repeat=repeat,
        warmup=warmup,
        hardware="test",
        show_progress=False,
    )
    config = {"kind": executor, "show_progress": False}
    if executor != "serial":
        config["max_workers"] = 2
    generation = DataGeneration(root=tmp_path, graph=graph, executor=config, datasets={"sample": benchmark})
    assert generation.run() == 0
    entries = load_h5({}, tmp_path / "sample" / "timings.h5", jax=False)
    assert next(iter(entries)).startswith(f"test.{executor}.")
    entry = next(iter(entries.values()))
    assert entry["total"].shape == (repeat,) and np.all(entry["total"] > 0)
    np.testing.assert_allclose(entry["amortized"], entry["total"] / 5)
    assert not list(tmp_path.rglob("seed_*"))


def test_data_generation_benchmark_cache_identity_and_leaf_override(tmp_path):
    graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
    benchmark = GenDataBenchmark(
        generator={"kind": "source", "samples": 1, "seed": 0, "executor": "serial"},
        samples=2,
        repeat=1,
        warmup=1,
        hardware="test",
        show_progress=False,
    )
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


@pytest.mark.parametrize(
    "width,mix,count,skip_solve,skip_evaluate",
    [
        (0, True, 3, False, False),
        (1, False, 3, False, True),
        (2, True, 3, True, False),
        (2, False, 0, False, False),
        (0, False, 3, True, True),
    ],
)
@pytest.mark.parametrize("edge_type", [_FamilySampleableEdge, _RandomOutputEdge, _NoneInputEdge])
def test_data_generation_benchmark_matches_generation_numerics(
    tmp_path, monkeypatch, width, mix, count, skip_solve, skip_evaluate, edge_type
):
    """Compare each component at its family/output identity, including task tails."""
    graph = FunctionGraph(edges={"sample": edge_type(source="inputs", target="sample")})
    generator = GenImplicitModel(
        graph=graph,
        input_samples=5,
        outputs_per_input=count,
        input_seed=3,
        output_seed=5,
        batch_size=3,
        map_batch_size=width,
        mix_output_seed=mix,
        skip_solve=skip_solve,
        skip_evaluate=skip_evaluate,
        executor={"show_progress": False},
    )
    generator.generate(tmp_path / "reference" / "sample", "h5", "overwrite")
    benchmark = GenDataBenchmark(generator=generator, samples=5, repeat=1, warmup=0, show_progress=False)
    kernels = []
    original = generator._family_kernel

    def capture(*args):
        kernel = original(*args)
        kernels.append(kernel)
        return kernel

    monkeypatch.setattr(generator, "_family_kernel", capture)
    workload, batches = benchmark._workload(tmp_path / "sample")
    assert kernels == [workload]
    assert not (tmp_path / "sample").exists()
    numerical = {}
    for batch_index, args in enumerate(batches):
        inputs, outputs = jax.block_until_ready(workload(*args))
        for i in range(len(args[0])):
            index = batch_index * generator.batch_size + i
            numerical["input", 3, index, None, None] = dict(
                zip(
                    ("inputs", "solutions", "solution_residuals"),
                    jax.tree.map(lambda x: x[i], inputs),
                    strict=True,
                )
            )
            for j in range(count):
                numerical["output", 3, index, 5, j] = dict(
                    zip(
                        ("conditions", "values", "residuals"),
                        jax.tree.map(lambda x: x[i, j], outputs),
                        strict=True,
                    )
                )
    store = H5Store(tmp_path / "reference" / "sample")
    refs = [ref for ref in store.discover(include_incomplete=True) if ref.kind != "solution"]
    assert set(numerical) == {ref.identity for ref in refs}
    for ref, persisted in zip(refs, store.read_components(refs), strict=True):
        actual = numerical[ref.identity]
        assert jax.tree.structure(actual) == jax.tree.structure(persisted)
        for x, y in zip(jax.tree.leaves(actual), jax.tree.leaves(persisted), strict=True):
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-6)


def test_data_generation_benchmark_rejects_old_schema_at_executor_label(tmp_path):
    graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
    benchmark = GenDataBenchmark(
        generator={"kind": "source", "graph": graph, "samples": 1, "seed": 0},
        samples=2,
        repeat=1,
        warmup=1,
        hardware="test",
        show_progress=False,
    )
    benchmark.generate(tmp_path / "sample", format="h5", write_policy="overwrite")
    path = tmp_path / "sample" / "timings.h5"
    cached = load_h5({}, path, jax=False)
    metadata = next(iter(cached.values()))["_benchmark"]
    assert metadata["schema_version"] == 5
    metadata["schema_version"] = np.asarray(4)
    metadata["compute_policy"] = np.bytes_("lax_map_staged")
    save_h5(cached, path, mode="w")
    with pytest.raises(RoutineError, match="incompatible"):
        benchmark.generate(tmp_path / "sample", format="h5", write_policy="reuse")


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
        overrides=[{"name": "case", "cases": [{"name": "one", "value": {"source": {"samples": 1, "seed": 0}}}]}],
        graph=graph,
    )

    assert from_path.run() == 0
    assert empty_base.run() == 0
    assert H5Store(tmp_path / "from_path" / "case=one" / "source").discover()
    assert H5Store(tmp_path / "empty_base" / "case=one" / "source").discover()


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
    loader = DataLoader(
        root=tmp_path,
        datasets={"source": {"kind": "source", "batch_size": 1, "max_epochs": 2, "cache_policy": "epoch"}},
    )
    assert len(list(loader)) == 2
    assert len(loader.datasets["source"]._cache_entries) == 1


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
                "cache_policy": "epoch",
                "cache_storage": "device",
            }
        },
    )

    batch = next(loader)

    assert len(device_put_calls) == 1
    assert len(batch["source"]) == 1
    assert isinstance(batch["source"][0]["state"]["x"], jax.Array)


def test_data_loader_source_cache_max_bytes_skips_oversized_items(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Cache limits"):
        LoadSource(cache_max_bytes=64)
    with pytest.raises(ValueError, match="Cache limits"):
        LoadSource(cache_max_items=1)
    with pytest.raises(ValueError):
        LoadSource(cache_policy="lru")


def test_data_loader_implicit_epoch_cache_reuses_full_epoch(tmp_path, monkeypatch):
    _write_implicit_dataset(tmp_path, sample_count=2)
    loader = DataLoader(
        root=tmp_path,
        datasets={
            "toy": {
                "kind": "implicit",
                "batch_size": 2,
                "load_solution": False,
                "cache_policy": "epoch",
                "max_epochs": 2,
            }
        },
    )
    first = next(loader)

    def fail_read(*args, **kwargs):
        raise AssertionError("epoch cache reread numerical data")

    monkeypatch.setattr(H5Store, "read_batch", fail_read)
    second = next(loader)
    assert len(first["toy"]) == len(second["toy"]) == 2


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
        float(np.asarray(sample["inputs"]["x"]).reshape(-1)[0]) for batch in batches for sample in batch["transport"]
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
        float(np.asarray(sample["inputs"]["x"]).reshape(-1)[0]) for batch in batches for sample in batch["transport"]
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
        output_dir = tmp_path / "train" / "transport" / "seed_0" / f"sample_{sample_idx}" / "seed_0" / "sample_0"
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

        save_h5({"x": jnp.asarray([value], dtype=jnp.float32)}, input_dir / "input.h5", mode="w")
        save_h5({"y": jnp.asarray([value + 1.0], dtype=jnp.float32)}, input_dir / "solution.h5", mode="w")
        save_h5({"r": jnp.asarray([1.0], dtype=jnp.float32)}, input_dir / "solution_residual.h5", mode="w")
        output_dir = input_dir / "seed_1" / "sample_0"
        save_h5({"y": jnp.asarray([value + 2.0], dtype=jnp.float32)}, output_dir / "output.h5", mode="w")
        save_h5({"r": jnp.asarray([0.0], dtype=jnp.float32)}, output_dir / "residual.h5", mode="w")

    loader = DataLoader(
        root=nested_root,
        datasets={
            "train": {"alpha": {"kind": "implicit", "batch_size": 1, "max_epochs": 1, "load_solution": False}},
            "validation": {"beta": {"kind": "implicit", "batch_size": 1, "max_epochs": 1, "load_solution": False}},
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


def test_load_implicit_model_solution_loading_options(tmp_path):
    input_dir = tmp_path / "toy" / "seed_0" / "sample_0"
    save_h5({"x": jnp.asarray([1.0], dtype=jnp.float32)}, input_dir / "input.h5", mode="w")
    save_h5({"y": jnp.asarray([2.0], dtype=jnp.float32)}, input_dir / "solution.h5", mode="w")
    save_h5({"r": jnp.asarray([-1.0], dtype=jnp.float32)}, input_dir / "solution_residual.h5", mode="w")

    output_dir = input_dir / "seed_0" / "sample_0"
    save_h5({"y": jnp.asarray([3.0], dtype=jnp.float32)}, output_dir / "output.h5", mode="w")
    save_h5({"r": jnp.asarray([0.5], dtype=jnp.float32)}, output_dir / "residual.h5", mode="w")

    refs = LoadImplicitModel(batch_size=4).discover_sample_refs(tmp_path / "toy")
    assert [ref.kind for ref in refs] == ["solution", "output"]

    batch = LoadImplicitModel(batch_size=4).load_batch(refs)
    assert len(batch) == 2
    assert [sample["outputs"]["y"][0] for sample in batch] == [2.0, 3.0]
    assert [sample["residuals"]["r"][0] for sample in batch] == [-1.0, 0.5]
    refs = LoadImplicitModel(batch_size=4, load_solution=False).discover_sample_refs(tmp_path / "toy")
    assert [ref.kind for ref in refs] == ["output"]
    solution_only = LoadImplicitModel(batch_size=4, solution_only=True)
    refs = solution_only.discover_sample_refs(tmp_path / "toy")

    assert [ref.kind for ref in refs] == ["solution"]
    batch = solution_only.load_batch(refs)
    assert len(batch) == 1
    assert [sample["outputs"]["y"][0] for sample in batch] == [2.0]
    assert [sample["residuals"]["r"][0] for sample in batch] == [-1.0]
    with pytest.raises(ValueError, match="solution_only=True requires load_solution=True"):
        LoadImplicitModel(solution_only=True, load_solution=False)


@pytest.mark.parametrize("source", [False, True])
def test_generation_executors_match_serial(tmp_path, monkeypatch, source):
    from romjax.data_gen import GenImplicitModel

    # Spawned workers otherwise compete with the parent for GPU preallocation.
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    edge = (
        _ToySourceEdge(source="noise", target="sample")
        if source
        else _ToySampleableEdge(source="inputs", target="sample", shift=3.5)
    )
    graph = FunctionGraph(edges={"sample": edge})
    options = (
        {"samples": 5, "seed": 7}
        if source
        else {"input_samples": 5, "outputs_per_input": 3, "input_seed": 7, "output_seed": 8}
    )
    generator_type = GenSource if source else GenImplicitModel
    reference_root = tmp_path / "reference"
    reference = DataGeneration(
        root=reference_root,
        graph=graph,
        executor={"kind": "serial", "show_progress": False},
        datasets={"sample": generator_type(**options, batch_size=2)},
    )
    assert reference.run() == 0
    before = H5Store(reference_root / "sample")
    left = before.discover()

    for backend in ("thread", "process"):
        actual_root = tmp_path / backend
        config = {"kind": backend, "show_progress": False, "max_workers": 2}
        generation = DataGeneration(
            root=actual_root,
            graph=graph,
            executor=config,
            datasets={"sample": generator_type(**options, batch_size=2)},
        )
        assert generation.run() == 0
        after = H5Store(actual_root / "sample")
        right = after.discover()
        assert [ref.identity for ref in left] == [ref.identity for ref in right]
        for lhs, rhs in zip(before.read_batch(left), after.read_batch(right), strict=True):
            for a, b in zip(jax.tree.leaves(lhs), jax.tree.leaves(rhs), strict=True):
                np.testing.assert_allclose(a, b)


def test_generation_leaf_executor_overrides_parent(tmp_path):
    graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
    generation = DataGeneration(
        root=tmp_path, graph=graph, executor="mpi", datasets={"sample": GenSource(samples=2, seed=1, executor="serial")}
    )
    assert generation.run() == 0
    assert len(H5Store(tmp_path / "sample").discover()) == 2


class _FileCompressionEdge(_CompressionSampleableEdge):
    """Load a separately declared sampler using the normal YAML validator."""

    outputs_sampler: Annotated[CompressionSampler, BeforeValidator(from_yaml)]


def test_process_generation_round_trips_external_graph_and_compression(tmp_path, monkeypatch):
    import romjax

    # Keep GPU execution in the parent; only spawned workers inherit this setting.
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")

    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    compression = SVD(rank=1).fit(
        [{"outputs": {"y": jnp.asarray([value], dtype=jnp.float32)}} for value in [1.0, 2.0, 3.0]]
    )
    compression.dump(config_dir / "compression.h5")
    (config_dir / "sampler.yml").write_text(
        "!romx:CompressionSampler\ncompression: __parent__/compression.h5\ndistribution: uniform\nreconstruct: true\n",
        encoding="utf-8",
    )
    (config_dir / "graph.yml").write_text(
        "!romx:FunctionGraph\nedges:\n  sample: !pd:tests.test_data_gen._FileCompressionEdge\n"
        "    source: inputs\n    target: outputs\n    outputs_sampler: __parent__/sampler.yml\n",
        encoding="utf-8",
    )
    config = config_dir / "generation.yml"
    config.write_text(
        f"!romx:DataGeneration\nroot: {tmp_path / 'data'}\n"
        "executor: {kind: process, max_workers: 2, show_progress: false}\n"
        "graph: __parent__/graph.yml\ndatasets:\n  sample:\n"
        "    input_samples: 3\n    outputs_per_input: 2\n    input_seed: 1\n    output_seed: 2\n"
        "    batch_size: 2\n    skip_solve: true\n    skip_evaluate: true\n",
        encoding="utf-8",
    )
    generation = romjax.load(config)
    edge = generation.graph.edges["sample"]
    assert isinstance(edge.outputs_sampler, CompressionSampler)
    text = romjax.YamlLoader.dump(generation.datasets["sample"], _preserve_model_types=True)
    restored = romjax.load(text)
    restored_edge = restored.graph.edges["sample"]
    assert type(restored_edge) is type(edge)
    assert Path(restored_edge.outputs_sampler.compression) == config_dir / "compression.h5"
    key = jax.random.key(10)
    np.testing.assert_allclose(edge.sample_outputs(key)["y"], restored_edge.sample_outputs(key)["y"])
    # The cached compression contains arrays; they must never enter the worker spec.
    monkeypatch.chdir(tmp_path)
    assert generation.run() == 0
    assert len(H5Store(tmp_path / "data" / "sample").discover()) == 6


def test_remote_generation_rejects_nonserializable_model_before_writing(tmp_path):
    graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
    # A local runtime callable cannot be imported by a fresh worker.
    from pydantic import Field

    from romjax.typing import CallableModel

    class LocalEdge(_ToySourceEdge):
        sampler: Any = Field(default_factory=lambda: CallableModel(callable=lambda: 1))

    graph.edges["sample"] = LocalEdge(source="noise", target="sample")
    generator = GenSource(graph=graph, samples=2, seed=1, executor="process")
    with pytest.raises(ExecutionError, match="Cannot reconstruct"):
        generator.generate(tmp_path / "sample", format="h5", write_policy="error")
    assert not (tmp_path / "sample").exists()


def test_generation_error_policy_checked_before_batch_submission(tmp_path):
    graph = FunctionGraph(edges={"sample": _ToySourceEdge(source="noise", target="sample")})
    destination = tmp_path / "sample"
    destination.mkdir()
    generator = GenSource(graph=graph, samples=4, seed=1, executor="thread", batch_size=2)
    with pytest.raises(RoutineError, match="policy='error'"):
        generator.generate(destination, format="h5", write_policy="error")
    assert not list(destination.iterdir())


@pytest.mark.parametrize("batched", [False, True])
def test_direct_generation_cache_distinguishes_graph_edges(tmp_path, batched):
    graph = FunctionGraph(
        edges={
            name: _ToySourceEdge(source="noise", target=name, bias=bias)
            for name, bias in [("first", 1.0), ("second", 3.0)]
        }
    )
    generator = GenSource(graph=graph, samples=1, seed=4, map_batch_size=0 if batched else 1)
    for name in ["first", "second"]:
        generator.generate(tmp_path / name, "h5", "reuse")
    stores = [H5Store(tmp_path / name) for name in ["first", "second"]]
    first, second = [store.read_batch(store.discover())[0] for store in stores]
    assert float(second["state"]["x"]) - float(first["state"]["x"]) == pytest.approx(2.0)


def test_thread_generation_throw_false_preserves_other_samples(tmp_path, monkeypatch):
    monkeypatch.setattr("romjax.data_gen.eqx.filter_jit", lambda fn: fn)
    graph = FunctionGraph(edges={"source": _FailingSourceEdge(source="noise", target="source")})
    generation = DataGeneration(
        root=tmp_path,
        graph=graph,
        executor={"kind": "thread", "max_workers": 2, "show_progress": False},
        datasets={"source": {"samples": 6, "seed": 0, "batch_size": 2, "throw": False}},
    )
    assert generation.run() == 0
    store = H5Store(tmp_path / "source")
    refs = store.discover()
    assert 0 < len(refs) < 6
    failures = []
    for path in store.root.glob("*.h5"):
        with h5py.File(path) as handle:
            failures.extend(handle["index/failures"][:])
    assert len(refs) + len(failures) == 6
