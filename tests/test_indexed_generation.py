"""Generation/loading integration for the indexed backend."""

import jax
import numpy as np
import pytest

from romjax.data_gen import DataGeneration, DataLoader, GenImplicitModel
from romjax.graph import FunctionGraph
from romjax.storage import H5Store
from tests.test_data_gen import _RandomOutputEdge, _ToySampleableEdge


@pytest.mark.parametrize("map_size", [0, 1, 2, 8])
@pytest.mark.parametrize("layout", ["packed", "grouped"])
def test_indexed_generation_keys_reuse_and_loading(tmp_path, map_size, layout):
    graph = FunctionGraph(edges={"toy": _RandomOutputEdge(source="inputs", target="outputs")})
    cfg = GenImplicitModel(
        graph=graph,
        input_samples=5,
        outputs_per_input=3,
        input_seed=2,
        output_seed=3,
        batch_size=2,
        map_batch_size=map_size,
        storage={"layout": layout, "samples_per_file": 2},
    )
    cfg.generate(tmp_path / "toy", "h5", "reuse")
    store = H5Store(tmp_path / "toy")
    refs = store.discover()
    assert len(refs) == 15
    for ref, sample in zip(refs, store.read_batch(refs), strict=True):
        ikey = jax.random.fold_in(jax.random.key(2), ref.input_index)
        okey = jax.random.fold_in(jax.random.fold_in(ikey, 3), ref.output_index)
        np.testing.assert_array_equal(sample["inputs"]["x"], jax.random.uniform(ikey))
        np.testing.assert_array_equal(sample["outputs"]["y"], jax.random.uniform(okey))
    stamps = {p: p.stat().st_mtime_ns for p in store.root.glob("*.h5")}
    cfg.generate(store.root, "h5", "reuse")
    assert stamps == {p: p.stat().st_mtime_ns for p in stamps}
    loader = DataLoader(
        root=tmp_path, datasets={"toy": {"kind": "implicit", "batch_size": 4, "cache_policy": "epoch", "max_epochs": 2}}
    )
    batches = list(loader)
    assert sum(len(b["toy"]) for b in batches) == 30


@pytest.mark.parametrize("executor", ["serial", "thread", "process"])
def test_indexed_executor_single_file(tmp_path, executor, monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    graph = FunctionGraph(edges={"toy": _ToySampleableEdge(source="inputs", target="outputs")})
    config = {"kind": executor, "show_progress": False}
    if executor != "serial":
        config["max_workers"] = 2
    generation = DataGeneration(
        root=tmp_path,
        graph=graph,
        executor=config,
        storage={"samples_per_file": None},
        datasets={
            "toy": {"input_samples": 5, "outputs_per_input": 2, "input_seed": 1, "output_seed": 2, "batch_size": 2}
        },
    )
    generation.run()
    store = H5Store(tmp_path / "toy")
    refs = store.discover()
    assert len(refs) == 15
    assert len(list(store.root.glob("*.h5"))) == 1
    for sample in store.read_batch(refs):
        np.testing.assert_allclose(sample["residuals"]["residual"], 0, atol=1e-6)


def test_append_new_output_seed_and_skip_backfill(tmp_path):
    graph = FunctionGraph(edges={"toy": _ToySampleableEdge(source="inputs", target="outputs")})
    cfg = GenImplicitModel(
        graph=graph,
        input_samples=2,
        outputs_per_input=1,
        input_seed=1,
        output_seed=2,
        skip_solve=True,
        skip_evaluate=True,
    )
    cfg.generate(tmp_path / "toy", "h5", "reuse")
    cfg.skip_solve = cfg.skip_evaluate = False
    cfg.output_seed = 3
    cfg.generate(tmp_path / "toy", "h5", "reuse")
    refs = H5Store(tmp_path / "toy").discover()
    assert len(refs) == 4
    assert {ref.output_seed for ref in refs if ref.kind == "output"} == {3}
    # Adding the previously absent base solution invalidates old output dependents.
    from romjax.storage import ComponentStatus

    assert H5Store(tmp_path / "toy").status(1, 0, 2, 0)["values"] == ComponentStatus.STALE


def test_generation_throw_false_isolates_output_failures(tmp_path, monkeypatch):
    from tests.test_data_gen import _FailingImplicitEdge

    monkeypatch.setattr("romjax.data_gen.eqx.filter_jit", lambda fn: fn)
    graph = FunctionGraph(
        edges={"toy": _FailingImplicitEdge(source="inputs", target="outputs", fail_stage="output_evaluate")}
    )
    cfg = GenImplicitModel(
        graph=graph, input_samples=3, outputs_per_input=2, input_seed=0, output_seed=0, throw=False, batch_size=2
    )
    cfg.generate(tmp_path / "toy", "h5", "reuse")
    refs = H5Store(tmp_path / "toy").discover()
    assert any(ref.kind == "solution" for ref in refs)
    from romjax.storage import ComponentStatus

    store = H5Store(tmp_path / "toy")
    failed = sum(store.status(0, i, 0, j)["values"] == ComponentStatus.FAILED for i in range(3) for j in range(2))
    outputs = sum(ref.kind == "output" for ref in refs)
    assert failed > 0
    assert failed + outputs == 6


def test_reuse_rejects_changed_key_scheme(tmp_path):
    graph = FunctionGraph(edges={"toy": _RandomOutputEdge(source="inputs", target="outputs")})
    cfg = GenImplicitModel(graph=graph, input_samples=1, outputs_per_input=1, input_seed=1, output_seed=2)
    cfg.generate(tmp_path / "toy", "h5", "reuse")
    cfg.mix_output_seed = False
    with pytest.raises(ValueError, match="key|seed"):
        cfg.generate(tmp_path / "toy", "h5", "reuse")


def test_resume_and_selection_independent_of_storage(tmp_path):
    from romjax.storage import RepackData

    graph = FunctionGraph(edges={"toy": _ToySampleableEdge(source="inputs", target="outputs")})
    cfg = GenImplicitModel(
        graph=graph,
        input_samples=5,
        outputs_per_input=2,
        input_seed=1,
        output_seed=2,
        batch_size=2,
        storage={"samples_per_file": 2},
    )
    cfg.generate(tmp_path / "original" / "toy", "h5", "reuse")
    RepackData(
        source=tmp_path / "original" / "toy",
        destination=tmp_path / "copy" / "toy",
        storage={"layout": "grouped", "samples_per_file": None},
    ).run()
    options = {
        "kind": "implicit",
        "batch_size": 2,
        "max_samples": 7,
        "max_input_samples": 3,
        "max_outputs_per_input": 3,
        "max_epochs": 2,
        "shuffle_seed": 7,
    }
    loaders = [DataLoader(root=tmp_path / root, datasets={"toy": dict(options)}) for root in ("original", "copy")]
    orders = [
        [tuple(ref.identity for ref in batch["toy"]) for batch in loader._iter_datasets(refs_only=True)]
        for loader in loaders
    ]
    assert orders[0] == orders[1]
    resumed = list(loaders[0]._iter_datasets(train_step=3, refs_only=True))
    assert [tuple(ref.identity for ref in batch["toy"]) for batch in resumed] == orders[0][3:]


def test_overwrite_skip_solve_does_not_use_old_solution(tmp_path):
    graph = FunctionGraph(edges={"toy": _ToySampleableEdge(source="inputs", target="outputs", shift=1.0)})
    cfg = GenImplicitModel(graph=graph, input_samples=1, outputs_per_input=1, input_seed=1, output_seed=2)
    cfg.generate(tmp_path / "toy", "h5", "reuse")
    graph.edges["toy"].shift = 100.0
    cfg.skip_solve = True
    cfg._compiled.clear()
    cfg.generate(tmp_path / "toy", "h5", "overwrite")
    store = H5Store(tmp_path / "toy")
    sample = store.read_batch(store.discover())[0]
    np.testing.assert_allclose(sample["outputs"]["y"], sample["inputs"]["x"] + 100.0)
    np.testing.assert_allclose(sample["residuals"]["residual"], 0.0)


class _FailSecondOutput(_RandomOutputEdge):
    def sample_outputs(self, key, inputs=None, solution=None):
        import equinox as eqx

        result = super().sample_outputs(key, inputs, solution)
        input_key = jax.random.fold_in(jax.random.key(1), 0)
        failed_key = jax.random.fold_in(jax.random.fold_in(input_key, 2), 1)
        result["y"] = eqx.error_if(
            result["y"],
            jax.numpy.all(jax.random.key_data(key) == jax.random.key_data(failed_key)),
            "second output failed",
        )
        return result


def test_fatal_output_failure_preserves_preceding_outputs(tmp_path):
    from romjax.storage import ComponentStatus

    graph = FunctionGraph(edges={"toy": _FailSecondOutput(source="inputs", target="outputs")})
    cfg = GenImplicitModel(graph=graph, input_samples=1, outputs_per_input=3, input_seed=1, output_seed=2)
    with pytest.raises(RuntimeError, match="second output failed"):
        cfg.generate(tmp_path / "toy", "h5", "reuse")
    store = H5Store(tmp_path / "toy")
    assert store.status(1, 0, 2, 0)["values"] == ComponentStatus.COMPLETE
    assert store.status(1, 0, 2, 1)["values"] == ComponentStatus.FAILED
    assert [ref.output_index for ref in store.discover()] == [0]


def test_complete_reuse_does_not_read_numerical_fields(tmp_path, monkeypatch):
    import h5py

    graph = FunctionGraph(edges={"toy": _ToySampleableEdge(source="inputs", target="outputs")})
    cfg = GenImplicitModel(graph=graph, input_samples=3, outputs_per_input=2, input_seed=1, output_seed=2)
    cfg.generate(tmp_path / "toy", "h5", "reuse")
    original = h5py.Dataset.__getitem__

    def checked(dataset, key, *args, **kwargs):
        assert dataset.name.startswith("/index/"), dataset.name
        return original(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", checked)
    cfg.generate(tmp_path / "toy", "h5", "reuse")


def test_skip_input_receives_parent_reference(tmp_path):
    from romjax.data_gen import LoadImplicitModel

    graph = FunctionGraph(edges={"toy": _ToySampleableEdge(source="inputs", target="outputs")})
    cfg = GenImplicitModel(graph=graph, input_samples=2, outputs_per_input=2, input_seed=1, output_seed=2)
    cfg.generate(tmp_path / "toy", "h5", "reuse")
    seen = []

    def skip(ref):
        seen.append(ref.kind)
        return ref.input_index == 0

    refs = LoadImplicitModel(skip_input=skip).discover_sample_refs(tmp_path / "toy")
    assert refs and all(ref.input_index == 1 for ref in refs)
    assert seen == ["input", "input"]


def test_modified_model_configuration_overwrite_recompiles(tmp_path):
    graph = FunctionGraph(edges={"toy": _ToySampleableEdge(source="inputs", target="outputs", shift=1.0)})
    cfg = GenImplicitModel(graph=graph, input_samples=1, outputs_per_input=0, input_seed=1, output_seed=2)
    cfg.generate(tmp_path / "toy", "h5", "reuse")
    graph.edges["toy"].shift = 5.0
    cfg.generate(tmp_path / "toy", "h5", "overwrite")
    store = H5Store(tmp_path / "toy")
    sample = store.read_batch(store.discover())[0]
    np.testing.assert_allclose(sample["outputs"]["y"], sample["inputs"]["x"] + 5.0)
