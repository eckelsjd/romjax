"""Indexed HDF5 storage contracts."""

import h5py
import numpy as np
import pytest

from romjax.storage import H5StorageConfig, H5Store, Record, RepackData


@pytest.mark.parametrize("layout", ["packed", "grouped"])
def test_storage_roundtrip_append_and_repack(tmp_path, layout):
    store = H5Store(tmp_path / "data", "implicit", H5StorageConfig(layout=layout, samples_per_file=2))
    for i in range(3):
        store.write(
            [
                Record(
                    2,
                    i,
                    "input",
                    {"inputs": {"x": np.array([i], np.float32)}, "solutions": {"y": np.array([i + 1], np.float32)}},
                )
            ]
        )
        store.write(
            [
                Record(
                    2,
                    i,
                    "output",
                    {"values": {"y": np.array([i + 2], np.float32)}, "residuals": {"r": np.array([1], np.float32)}},
                    3,
                    0,
                )
            ]
        )
    refs = store.discover()
    assert len(refs) == 6
    assert len(list(store.root.glob("*.h5"))) == 2
    samples = store.read_batch([refs[-1], refs[0], refs[-1]])
    np.testing.assert_array_equal(samples[0]["outputs"]["y"], [4])
    np.testing.assert_array_equal(samples[1]["inputs"]["x"], [0])
    np.testing.assert_array_equal(samples[2]["outputs"]["y"], [4])
    RepackData(source=store.root, destination=tmp_path / "copy", storage={"samples_per_file": None}).run()
    copied = H5Store(tmp_path / "copy").discover()
    assert [r.identity for r in refs] == [r.identity for r in copied]
    assert len(list((tmp_path / "copy").glob("*.h5"))) == 1
    with h5py.File(copied[0].file, "r") as f:
        assert "index/inputs" in f


def test_storage_replaces_rows_and_changes_capacity_for_new_files(tmp_path):
    store = H5Store(tmp_path, "source", H5StorageConfig(samples_per_file=2))
    store.write([Record(0, i, "source", {"values": {"x": np.array(i)}}) for i in range(3)])
    original = {r.identity: r.file for r in store.discover()}
    store = H5Store(tmp_path, "source", H5StorageConfig(samples_per_file=3))
    store.write([Record(0, 3, "source", {"values": {"x": np.array(3)}})])
    store.write([Record(0, 1, "source", {"values": {"x": np.array(10)}})])
    refs = store.discover()
    assert len(refs) == 4
    assert refs[1].file == original[refs[1].identity]
    assert len(list(tmp_path.glob("*.h5"))) == 3
    assert store.read_batch([refs[1]])[0]["x"] == 10


def test_storage_rejects_legacy_and_incompatible_packed_shape(tmp_path):
    (tmp_path / "seed_0" / "sample_0").mkdir(parents=True)
    with pytest.raises(ValueError, match="legacy|Legacy"):
        H5Store(tmp_path).discover()
    other = H5Store(tmp_path / "new", "source")
    other.write([Record(0, 0, "source", {"values": {"x": np.zeros(2)}})])
    with pytest.raises(ValueError, match="schema|shape"):
        other.write([Record(0, 1, "source", {"values": {"x": np.zeros(3)}})])


def test_storage_invalidates_only_changed_dependencies(tmp_path):
    store = H5Store(tmp_path, "implicit")
    store.write(
        [
            Record(0, 0, "input", {"inputs": {"x": np.array(1.0)}, "solutions": {"y": np.array(2.0)}}),
            Record(0, 0, "output", {"values": {"y": np.array(3.0)}}, 1, 0),
        ]
    )
    store.write([Record(0, 0, "input", {"inputs": {"x": np.array(1.0)}})])
    assert len(store.discover()) == 2
    store.write([Record(0, 0, "input", {"inputs": {"x": np.array(4.0)}})])
    assert store.discover() == []


def test_lock_and_schema_failure_preserve_existing_values(tmp_path):
    store = H5Store(tmp_path, "source")
    store.write([Record(0, 0, "source", {"values": {"x": np.zeros(2)}})])
    with store.locked():
        with pytest.raises(RuntimeError, match="already"):
            with H5Store(tmp_path).locked():
                pass
    with pytest.raises(ValueError, match="schema"):
        store.write([Record(0, 0, "source", {"values": {"x": np.zeros(3)}})])
    np.testing.assert_array_equal(store.read_batch(store.discover())[0]["x"], np.zeros(2))


def test_repack_preserves_failure_information(tmp_path):
    from romjax.storage import ComponentStatus

    store = H5Store(tmp_path / "original", "source")
    store.write([Record(0, 0, "source", {}, states={"values": ComponentStatus.FAILED}, failure="example failure")])
    RepackData(source=store.root, destination=tmp_path / "copy").run()
    path = next((tmp_path / "copy").glob("*.h5"))
    with h5py.File(path) as file:
        assert "example failure" in file["index/failures"].asstr()[0]


def test_mode_change_requires_repack_even_for_existing_ids(tmp_path):
    store = H5Store(tmp_path, "source", H5StorageConfig(samples_per_file=None))
    store.write([Record(0, 0, "source", {"values": {"x": np.array(1)}})])
    with pytest.raises(ValueError, match="RepackData"):
        H5Store(tmp_path, "source", H5StorageConfig(samples_per_file=2)).write(
            [Record(0, 0, "source", {"values": {"x": np.array(2)}})]
        )


def test_discovery_reads_only_indices_and_packed_reads_are_bulk(tmp_path, monkeypatch):
    store = H5Store(tmp_path, "implicit", H5StorageConfig(samples_per_file=None))
    records = []
    for i in range(8):
        records.append(
            Record(
                0,
                i,
                "input",
                {
                    "inputs": {"x": np.ones(16) * i},
                    "solutions": {"y": np.ones(16)},
                    "solution_residuals": {"r": np.zeros(16)},
                },
            )
        )
        for j in range(4):
            records.append(Record(0, i, "output", {"values": {"y": np.ones(16) * j}}, 1, j))
    store.write(records)
    reads = []
    original = h5py.Dataset.__getitem__

    def counted(dataset, key, *args, **kwargs):
        reads.append(dataset.name)
        return original(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", counted)
    fresh = H5Store(tmp_path)
    refs = fresh.discover()
    refs = [ref for ref in refs if ref.kind == "output"]
    assert len(refs) == 32
    assert reads == ["/index/inputs", "/index/outputs"]
    reads.clear()
    samples = fresh.read_batch(refs[::-1])
    assert len(samples) == 32
    assert sorted(reads) == ["/seeds/seed_0/inputs/x", "/seeds/seed_0/outputs/seed_1/values/y"]


def test_none_parent_replacement_invalidates_outputs(tmp_path):
    store = H5Store(tmp_path, "implicit")
    store.write(
        [Record(0, 0, "input", {"inputs": None}), Record(0, 0, "output", {"values": {"y": np.array(1.0)}}, 1, 0)]
    )
    store.write([Record(0, 0, "input", {"inputs": {"x": np.array(99.0)}})])
    assert store.discover() == []


def test_pending_residual_is_not_discoverable(tmp_path):
    from romjax.storage import ComponentStatus

    store = H5Store(tmp_path, "implicit")
    store.write(
        [
            Record(0, 0, "input", {"inputs": {"x": np.array(1.0)}}),
            Record(0, 0, "output", {"values": {"y": np.array(1.0)}}, 1, 0),
        ]
    )
    store.write([Record(0, 0, "output", {}, 1, 0, states={"residuals": ComponentStatus.PENDING})])
    assert store.discover() == []


def test_loader_rediscovers_changed_data_and_invalidates_epoch_cache(tmp_path):
    from romjax.data_gen import LoadSource

    store = H5Store(tmp_path, "source")
    store.write([Record(0, 0, "source", {"values": {"x": np.array(1.0)}})])
    loader = LoadSource(cache_policy="epoch")
    refs = loader.discover_sample_refs(tmp_path)
    assert loader.load_batch(refs)[0]["x"] == 1
    store.write(
        [
            Record(0, 0, "source", {"values": {"x": np.array(2.0)}}),
            Record(0, 1, "source", {"values": {"x": np.array(3.0)}}),
        ]
    )
    refs = loader.discover_sample_refs(tmp_path)
    assert len(refs) == 2
    assert [sample["x"] for sample in loader.load_batch(refs)] == [2, 3]
