"""Indexed, inspectable HDF5 storage for generated source and implicit data."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from pydantic import ConfigDict, Field, PositiveInt

from romjax.routine import Routine
from romjax.typing import DictModel

_COMPONENTS = ("inputs", "solutions", "solution_residuals", "conditions", "values", "residuals")
_INDEX_DTYPE = np.dtype(
    [(name, "<i8") for name in ("seed", "input_index", "output_seed", "output_index", "row")]
    + [(name, "u1") for name in _COMPONENTS]
)


class ComponentStatus(IntEnum):
    """Completion state recorded in the flat index."""

    PENDING = 0
    COMPLETE = 1
    SKIPPED = 2
    NONE = 3
    FAILED = 4
    STALE = 5


class H5StorageConfig(DictModel):
    """HDF5 layout and capacity, independently of numerical batching.

    :param layout: packed field arrays or individual grouped records
    :param samples_per_file: source samples/input families per shard; None uses one file
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    layout: Literal["packed", "grouped"] = "packed"
    samples_per_file: PositiveInt | None = 256


@dataclass(frozen=True)
class SampleRef:
    """Logical sample identity and its physical HDF5 location.

    :param root: dataset root
    :param file: shard path
    :param kind: source, input, solution, or output sample
    :param seed: source/input seed
    :param input_index: source/input sample index
    :param row: input row or output row
    :param input_row: parent input row
    :param output_seed: optional output seed
    :param output_index: optional output index
    """

    root: Path
    file: Path
    kind: Literal["source", "input", "solution", "output"]
    seed: int
    input_index: int
    row: int
    input_row: int
    output_seed: int | None = None
    output_index: int | None = None

    @property
    def input_id(self) -> tuple[int, int]:
        """Return the parent/source seed and sample index."""
        return self.seed, self.input_index

    @property
    def identity(self) -> tuple:
        """Return identity independently of file placement."""
        return self.kind, self.seed, self.input_index, self.output_seed, self.output_index


@dataclass
class Record:
    """Controller-side component update for one input/source/output identity."""

    seed: int
    input_index: int
    kind: Literal["source", "input", "output"]
    components: dict[str, Any]
    output_seed: int | None = None
    output_index: int | None = None
    states: dict[str, ComponentStatus] = field(default_factory=dict)
    failure: str | None = None


class H5Store:
    """Read and update a dataset from its owning controller.

    No open HDF5 handles are retained or transported to workers. Metadata is
    indexed once and refreshed after this instance writes.

    :param root: dataset directory
    :param kind: source or implicit; inferred when opening existing storage
    :param storage: settings for newly allocated files
    """

    def __init__(self, root: Path, kind: str | None = None, storage: H5StorageConfig | None = None) -> None:
        self.root = Path(root).resolve()
        self.kind = kind
        self.storage = storage or H5StorageConfig()
        self._tables: dict[Path, dict[str, np.ndarray]] = {}
        self._settings: dict[Path, tuple[str, int | None]] = {}
        self._inputs: dict[tuple[int, int], tuple[Path, int]] = {}
        self._outputs: dict[tuple[int, int, int, int], tuple[Path, int]] = {}
        self._loaded = False
        self._key_schemes: dict[str, bool] = {}

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold exclusive controller ownership during generation/repacking."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / ".romjax-generation.lock"
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(f"Dataset is already being generated: {self.root}; lock: {path}") from exc
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(f"pid={os.getpid()}\n")
            yield
        finally:
            path.unlink(missing_ok=True)

    def _load(self) -> None:
        if self._loaded:
            return
        if self.root.exists() and any(self.root.glob("seed_*/sample_*")):
            raise ValueError(f"Legacy dataset at {self.root}; regenerate it in the indexed HDF5 format.")
        for path in sorted(self.root.glob("*.h5")):
            with h5py.File(path, "r") as f:
                if "romjax_storage_version" not in f.attrs:
                    raise ValueError(f"Not an indexed romjax dataset: {path}")
                if int(f.attrs["romjax_storage_version"]) != 1:
                    raise ValueError(f"Unsupported storage version in {path}")
                kind = str(f.attrs["kind"])
                if self.kind is not None and self.kind != kind:
                    raise ValueError(f"Mixed dataset kinds in {self.root}")
                self.kind = kind
                for key, value in json.loads(f.attrs.get("key_schemes", "{}")).items():
                    if key in self._key_schemes and self._key_schemes[key] != value:
                        raise ValueError(f"Conflicting output key scheme: {key}")
                    self._key_schemes[key] = value
                self._settings[path] = (str(f.attrs["layout"]), int(f.attrs["capacity"]) or None)
                self._tables[path] = {name: f[f"index/{name}"][()] for name in ("inputs", "outputs")}
        self._reindex()
        self._loaded = True

    def _reindex(self) -> None:
        self._inputs.clear()
        self._outputs.clear()
        for path, tables in self._tables.items():
            for row in tables["inputs"]:
                key = int(row["seed"]), int(row["input_index"])
                if key in self._inputs:
                    raise ValueError(f"Duplicate input identity {key}")
                self._inputs[key] = path, int(row["row"])
            for row in tables["outputs"]:
                key = tuple(int(row[name]) for name in ("seed", "input_index", "output_seed", "output_index"))
                if key in self._outputs:
                    raise ValueError(f"Duplicate output identity {key}")
                self._outputs[key] = path, int(row["row"])

    def validate_storage(self) -> None:
        """Check file-mode compatibility without rewriting existing storage."""
        self._load()
        capacity = self.storage.samples_per_file
        if any((old is None) != (capacity is None) for _, old in self._settings.values()):
            raise ValueError("Switching between single-file and sharded storage requires RepackData.")
        if capacity is None and any(layout != self.storage.layout for layout, _ in self._settings.values()):
            raise ValueError("Changing a single-file layout requires RepackData.")

    def configure_key_scheme(self, seed: int, output_seed: int, mixed: bool) -> None:
        """Reject reinterpretation of existing output identities with different keys."""
        self._load()
        key = f"{seed}/{output_seed}"
        if key in self._key_schemes and self._key_schemes[key] != mixed:
            raise ValueError(f"Output key scheme changed for seeds {key}; choose a new output seed.")
        self._key_schemes[key] = mixed

    def _allocate_file(self, seed: int) -> Path:
        capacity = self.storage.samples_per_file
        if self._settings and any((cap is None) != (capacity is None) for _, cap in self._settings.values()):
            raise ValueError("Switching between single-file and sharded storage requires RepackData.")
        if capacity is None:
            path = self.root / "dataset.h5"
            if path in self._settings and self._settings[path][0] != self.storage.layout:
                raise ValueError("Changing a single-file layout requires RepackData.")
            return path
        files = sorted(p for p in self._tables if p.name.startswith(f"seed_{seed}_part_"))
        if files:
            last = files[-1]
            if self._settings[last] == (self.storage.layout, capacity) and len(self._tables[last]["inputs"]) < capacity:
                return last
        part = max((int(p.stem.rsplit("_", 1)[1]) for p in files), default=-1) + 1
        return self.root / f"seed_{seed}_part_{part:06d}.h5"

    def _reserve(self, record: Record) -> tuple[Path, int, str]:
        key = record.seed, record.input_index
        if key not in self._inputs:
            path = self._allocate_file(record.seed)
            if path not in self._tables:
                self._tables[path] = {name: np.zeros(0, _INDEX_DTYPE) for name in ("inputs", "outputs")}
                self._settings[path] = self.storage.layout, self.storage.samples_per_file
            rows = self._tables[path]["inputs"]
            row = np.zeros(1, _INDEX_DTYPE)
            for name in _COMPONENTS:
                row[name] = ComponentStatus.SKIPPED
            row["values" if record.kind == "source" else "inputs"] = ComponentStatus.PENDING
            row["seed"], row["input_index"], row["row"] = *key, len(rows)
            row["output_seed"], row["output_index"] = -1, -1
            self._tables[path]["inputs"] = np.concatenate((rows, row))
            self._inputs[key] = path, len(rows)
        path, input_row = self._inputs[key]
        if record.kind != "output":
            return path, input_row, "inputs"
        out_key = *key, record.output_seed, record.output_index
        if out_key not in self._outputs:
            rows = self._tables[path]["outputs"]
            row = np.zeros(1, _INDEX_DTYPE)
            for name in _COMPONENTS:
                row[name] = ComponentStatus.SKIPPED
            row["values"] = ComponentStatus.PENDING
            for name, value in zip(("seed", "input_index", "output_seed", "output_index"), out_key, strict=True):
                row[name] = value
            row["row"] = len(rows)
            self._tables[path]["outputs"] = np.concatenate((rows, row))
            self._outputs[out_key] = path, len(rows)
        return path, self._outputs[out_key][1], "outputs"

    @staticmethod
    def _schema(value: Any) -> tuple[Any, dict[str, np.ndarray]]:
        arrays = {}

        def visit(node, path):
            if node is None:
                return ["none"]
            if isinstance(node, Mapping):
                return ["dict", [[str(key), visit(val, (*path, str(key)))] for key, val in node.items()]]
            array = np.asarray(node)
            if array.dtype.hasobject:
                raise ValueError("Object arrays are not supported in generated HDF5 data.")
            arrays["/".join(path) or "_value"] = array
            return ["array", list(array.shape), array.dtype.str]

        return visit(value, ()), arrays

    @staticmethod
    def _restore(schema: Any, arrays: Mapping[str, Any], path: tuple[str, ...] = ()) -> Any:
        if schema[0] == "none":
            return None
        if schema[0] == "dict":
            return {key: H5Store._restore(child, arrays, (*path, key)) for key, child in schema[1]}
        return arrays["/".join(path) or "_value"]

    @staticmethod
    def _component_path(seed: int, component: str, output_seed: int | None = None) -> str:
        base = f"seeds/seed_{seed}"
        if output_seed is not None:
            base += f"/outputs/seed_{output_seed}"
        return f"{base}/{component}"

    @staticmethod
    def _chunks(array: np.ndarray, capacity: int | None) -> tuple[int, ...]:
        shape = [max(1, n) for n in array.shape]
        size = max(1, array.nbytes)
        rows = min(capacity or 65536, max(1, 65536 // size)) if size < 10240 else 1
        while rows * int(np.prod(shape)) * array.dtype.itemsize > 1048576 and max(shape, default=1) > 1:
            axis = int(np.argmax(shape))
            shape[axis] = max(1, (shape[axis] + 1) // 2)
        return rows, *shape

    def write(self, records: Sequence[Record]) -> None:
        """Commit component updates; the caller must own generation access."""
        self.validate_storage()
        if not records:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        grouped = defaultdict(list)
        for record in records:
            path, row, table = self._reserve(record)
            grouped[path].append((record, row, table))
        try:
            for path, updates in grouped.items():
                self._write_file(path, updates)
        except BaseException:
            # Discard reservations/status changes which did not reach disk.
            self._tables.clear()
            self._settings.clear()
            self._loaded = False
            raise

    def _write_file(self, path: Path, updates: Sequence[tuple[Record, int, str]]) -> None:
        layout, capacity = self._settings[path]
        with h5py.File(path, "a") as f:
            f.attrs.update(romjax_storage_version=1, kind=self.kind, layout=layout, capacity=capacity or 0)
            f.attrs["status_legend"] = json.dumps({status.name: int(status) for status in ComponentStatus})
            f.attrs["key_schemes"] = json.dumps(self._key_schemes)
            index = f.require_group("index")
            for name in ("inputs", "outputs"):
                if name not in index:
                    index.create_dataset(name, shape=(0,), maxshape=(None,), dtype=_INDEX_DTYPE)
            if "failures" not in index:
                index.create_dataset("failures", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype())
            writes = defaultdict(list)
            for record, row, table in updates:
                for component, value in record.components.items():
                    if component not in _COMPONENTS:
                        raise ValueError(f"Unknown component: {component}")
                    group_path = self._component_path(record.seed, component, record.output_seed)
                    schema, arrays = self._schema(value)
                    if value is not None:
                        writes[group_path].append((row, schema, arrays, table))
                if record.failure:
                    failures = index["failures"]
                    failures.resize((len(failures) + 1,))
                    failures[-1] = json.dumps(
                        {
                            "seed": record.seed,
                            "input_index": record.input_index,
                            "output_seed": record.output_seed,
                            "output_index": record.output_index,
                            "message": record.failure,
                        }
                    )
            # Validate every packed component before replacing any numerical data.
            if layout == "packed":
                for group_path, items in writes.items():
                    expected = json.dumps(items[0][1])
                    if any(json.dumps(item[1]) != expected for item in items):
                        raise ValueError(f"Incompatible packed schema/shape: {group_path}")
                    if group_path in f and f[group_path].attrs.get("schema") != expected:
                        raise ValueError(f"Incompatible packed schema/shape: {group_path}; use grouped storage.")
            # Invalidate dependencies only when an existing parent value changes.
            if self.kind == "implicit":
                for record, row, table in updates:
                    if table != "inputs":
                        continue
                    for component in ("inputs", "solutions"):
                        group_path = self._component_path(record.seed, component)
                        if component not in record.components:
                            continue
                        state = self._tables[path][table][row][component]
                        old = (
                            self._read_component(f, group_path, [row])[0]
                            if state == ComponentStatus.COMPLETE and group_path in f
                            else None
                        )
                        old_schema, old_arrays = self._schema(old)
                        new_schema, new_arrays = self._schema(record.components[component])
                        equal = old_schema == new_schema and all(
                            np.array_equal(
                                old_arrays[key], new_arrays[key], equal_nan=old_arrays[key].dtype.kind in "fc"
                            )
                            for key in old_arrays
                        )
                        if equal:
                            continue
                        names = (
                            ("solutions", "solution_residuals") if component == "inputs" else ("solution_residuals",)
                        )
                        for name in names:
                            if self._tables[path][table][row][name] == ComponentStatus.COMPLETE:
                                self._tables[path][table][row][name] = ComponentStatus.STALE
                        outputs = self._tables[path]["outputs"]
                        mask = (outputs["seed"] == record.seed) & (outputs["input_index"] == record.input_index)
                        for name in ("conditions", "values", "residuals"):
                            affected = mask & (outputs[name] == ComponentStatus.COMPLETE)
                            outputs[name][affected] = ComponentStatus.STALE
            for record, row, table in updates:
                for component in record.components.keys() | record.states.keys():
                    self._tables[path][table][row][component] = ComponentStatus.PENDING
            self._flush_indices(f)
            f.flush()
            for group_path, items in writes.items():
                group = f.require_group(group_path)
                if layout == "grouped":
                    for row, schema, arrays, _ in items:
                        name = f"row_{row}"
                        if name in group:
                            del group[name]
                        record_group = group.create_group(name)
                        record_group.attrs["schema"] = json.dumps(schema)
                        for key, array in arrays.items():
                            record_group.create_dataset(key, data=array)
                else:
                    group.attrs["schema"] = json.dumps(items[0][1])
                    rows = np.asarray([item[0] for item in items])
                    order = np.argsort(rows)
                    rows = rows[order]
                    total = len(self._tables[path][items[0][3]])
                    for key in items[0][2]:
                        arrays = [items[i][2][key] for i in order]
                        sample = arrays[0]
                        if key not in group:
                            group.create_dataset(
                                key,
                                shape=(total, *sample.shape),
                                maxshape=(None, *(None if n == 0 else n for n in sample.shape)),
                                dtype=sample.dtype,
                                chunks=self._chunks(sample, capacity),
                            )
                        dataset = group[key]
                        if len(dataset) < total:
                            dataset.resize(total, axis=0)
                        dataset[rows] = np.stack(arrays)
            f.flush()
            for record, row, table in updates:
                for component, value in record.components.items():
                    self._tables[path][table][row][component] = record.states.get(
                        component, ComponentStatus.NONE if value is None else ComponentStatus.COMPLETE
                    )
                for component, state in record.states.items():
                    self._tables[path][table][row][component] = state
            self._flush_indices(f)
            f.flush()

    def _flush_indices(self, f: h5py.File) -> None:
        for name, rows in self._tables[Path(f.filename)].items():
            dataset = f[f"index/{name}"]
            dataset.resize((len(rows),))
            dataset[:] = rows

    def reserve(self, records: Sequence[Record]) -> None:
        """Reserve logical locations in controller memory before task submission."""
        self._load()
        for record in records:
            self._reserve(record)

    def reference(
        self, seed: int, index: int, kind: str, output_seed: int | None = None, output_index: int | None = None
    ) -> SampleRef | None:
        """Resolve one identity without scanning the discovery tables."""
        self._load()
        parent = self._inputs.get((seed, index))
        if parent is None:
            return None
        path, input_row = parent
        row = input_row
        if kind == "output":
            location = self._outputs.get((seed, index, output_seed, output_index))
            if location is None:
                return None
            path, row = location
        return SampleRef(self.root, path, kind, seed, index, row, input_row, output_seed, output_index)

    def status(
        self, seed: int, index: int, output_seed: int | None = None, output_index: int | None = None
    ) -> dict[str, ComponentStatus]:
        """Return recorded component states, or pending states for a new identity."""
        self._load()
        location = (
            self._inputs.get((seed, index))
            if output_seed is None
            else self._outputs.get((seed, index, output_seed, output_index))
        )
        if location is None:
            return {name: ComponentStatus.PENDING for name in _COMPONENTS}
        path, row = location
        table = "inputs" if output_seed is None else "outputs"
        return {name: ComponentStatus(int(self._tables[path][table][row][name])) for name in _COMPONENTS}

    def discover(self, *, include_incomplete: bool = False) -> list[SampleRef]:
        """Discover logical references by bulk-reading indices, not numerical fields."""
        self._load()
        refs = []
        valid = (ComponentStatus.COMPLETE, ComponentStatus.NONE)
        for (seed, index), (path, row) in self._inputs.items():
            state = self._tables[path]["inputs"][row]
            if self.kind == "source":
                if include_incomplete or state["values"] == ComponentStatus.COMPLETE:
                    refs.append(SampleRef(self.root, path, "source", seed, index, row, row))
                continue
            if state["inputs"] not in valid and not include_incomplete:
                continue
            if include_incomplete:
                refs.append(SampleRef(self.root, path, "input", seed, index, row, row))
            if state["solutions"] == ComponentStatus.COMPLETE and state["solution_residuals"] in (
                *valid,
                ComponentStatus.SKIPPED,
            ):
                refs.append(SampleRef(self.root, path, "solution", seed, index, row, row))
        for (seed, index, out_seed, out_index), (path, row) in self._outputs.items():
            input_path, input_row = self._inputs[seed, index]
            parent = self._tables[input_path]["inputs"][input_row]
            state = self._tables[path]["outputs"][row]
            if include_incomplete or (
                parent["inputs"] in valid
                and state["values"] == ComponentStatus.COMPLETE
                and all(state[name] in (*valid, ComponentStatus.SKIPPED) for name in ("conditions", "residuals"))
            ):
                refs.append(SampleRef(self.root, path, "output", seed, index, row, input_row, out_seed, out_index))
        return sorted(
            refs,
            key=lambda ref: (
                f"seed_{ref.seed}",
                f"sample_{ref.input_index}",
                ref.kind == "output",
                f"seed_{ref.output_seed}",
                f"sample_{ref.output_index}",
            ),
        )

    @staticmethod
    def _read_component(f: h5py.File, path: str, rows: Sequence[int]) -> list[Any]:
        group = f[path]
        if f.attrs["layout"] == "grouped":
            result = []
            for row in rows:
                item = group[f"row_{row}"]
                schema = json.loads(item.attrs["schema"])
                _, paths = H5Store._schema_paths(schema)
                result.append(H5Store._restore(schema, {key: item[key][()] for key in paths}))
            return result
        schema = json.loads(group.attrs["schema"])
        _, paths = H5Store._schema_paths(schema)
        unique, inverse = np.unique(rows, return_inverse=True)
        arrays = {}
        for key in paths:
            dataset = group[key]
            if len(unique) and unique[-1] - unique[0] + 1 == len(unique):
                arrays[key] = dataset[int(unique[0]) : int(unique[-1]) + 1]
            else:
                arrays[key] = np.concatenate(
                    [dataset[part] for part in (unique[i : i + 1000] for i in range(0, len(unique), 1000))]
                )
        return [H5Store._restore(schema, {key: value[i] for key, value in arrays.items()}) for i in inverse]

    @staticmethod
    def _schema_paths(schema: Any) -> tuple[Any, list[str]]:
        paths = []

        def visit(node, path):
            if node[0] == "dict":
                for key, child in node[1]:
                    visit(child, (*path, key))
            elif node[0] == "array":
                paths.append("/".join(path) or "_value")

        visit(schema, ())
        return schema, paths

    def read_components(
        self, refs: Sequence[SampleRef], *, components: Mapping[str, Sequence[str]] | None = None
    ) -> list[dict[str, Any]]:
        """Bulk-read raw components for numerical generation or repacking."""
        self._load()
        result = [{} for _ in refs]
        requests = defaultdict(list)
        for i, ref in enumerate(refs):
            names = (
                ("values",)
                if ref.kind == "source"
                else ("conditions", "values", "residuals")
                if ref.kind == "output"
                else ("inputs", "solutions", "solution_residuals")
            )
            if components is not None:
                names = components.get(ref.kind, names)
            states = self.status(ref.seed, ref.input_index, ref.output_seed, ref.output_index)
            for component in names:
                if states[component] in (ComponentStatus.COMPLETE, ComponentStatus.STALE):
                    requests[ref.file, self._component_path(ref.seed, component, ref.output_seed)].append(
                        (i, ref.row, component)
                    )
                elif states[component] in (ComponentStatus.NONE, ComponentStatus.SKIPPED):
                    result[i][component] = None
        by_file = defaultdict(list)
        for (file, path), items in requests.items():
            by_file[file].append((path, items))
        for file, groups in by_file.items():
            with h5py.File(file, "r") as f:
                for path, items in groups:
                    values = self._read_component(f, path, [row for _, row, _ in items])
                    for (i, _, name), value in zip(items, values, strict=True):
                        result[i][name] = value
        return result

    def read_batch(self, refs: Sequence[SampleRef], *, include_conditions: bool = True) -> list[Any]:
        """Read training samples, reusing parent inputs within the batch."""
        if not refs:
            return []
        parents = list(
            dict.fromkeys(
                SampleRef(ref.root, ref.file, "input", ref.seed, ref.input_index, ref.input_row, ref.input_row)
                for ref in refs
                if ref.kind == "output"
            )
        )
        projection = {"input": ("inputs",)}
        if not include_conditions:
            projection["output"] = ("values", "residuals")
        raw = self.read_components([*refs, *parents], components=projection)
        inputs = {ref.input_id: value.get("inputs") for ref, value in zip(parents, raw[len(refs) :], strict=True)}
        result = []
        for ref, value in zip(refs, raw[: len(refs)], strict=True):
            if ref.kind == "source":
                result.append(value.get("values"))
                continue
            sample = {}
            input_value = inputs.get(ref.input_id) if ref.kind == "output" else value.get("inputs")
            if input_value is not None:
                sample["inputs"] = input_value
            for source, target in (
                ("solutions" if ref.kind == "solution" else "values", "outputs"),
                ("solution_residuals" if ref.kind == "solution" else "residuals", "residuals"),
                ("conditions", "conditions"),
            ):
                if value.get(source) is not None:
                    sample[target] = value[source]
            result.append(sample)
        return result


class RepackData(Routine):
    """Copy new-format data to a different layout/capacity without regeneration.

    :param source: existing indexed dataset
    :param destination: nonexistent destination directory
    :param storage: destination storage settings
    """

    source: Path
    destination: Path
    storage: H5StorageConfig = Field(default_factory=H5StorageConfig)

    def run(self) -> int:
        """Copy records, preserving identities and component statuses."""
        source, destination = self.source.resolve(), self.destination.resolve()
        if (
            destination.exists()
            or source == destination
            or source in destination.parents
            or destination in source.parents
        ):
            raise ValueError("RepackData requires a nonexistent, non-overlapping destination.")
        old = H5Store(source)
        refs = old.discover(include_incomplete=True)
        if old.kind is None:
            raise ValueError(f"No indexed data found in {source}")
        new = H5Store(destination, old.kind, self.storage)
        new._key_schemes = dict(old._key_schemes)
        with new.locked():
            for ref in refs:
                if ref.kind == "solution":
                    continue
                raw = old.read_components([ref])[0]
                states = old.status(ref.seed, ref.input_index, ref.output_seed, ref.output_index)
                names = (
                    ("values",)
                    if ref.kind == "source"
                    else ("conditions", "values", "residuals")
                    if ref.kind == "output"
                    else ("inputs", "solutions", "solution_residuals")
                )
                new.write(
                    [
                        Record(
                            ref.seed,
                            ref.input_index,
                            ref.kind,
                            raw,
                            ref.output_seed,
                            ref.output_index,
                            {name: states[name] for name in names},
                        )
                    ]
                )
            for file in old._tables:
                with h5py.File(file) as handle:
                    failures = list(handle["index/failures"].asstr()[:])
                for encoded in failures:
                    item = json.loads(encoded)
                    kind = "source" if old.kind == "source" else "input" if item["output_seed"] is None else "output"
                    new.write(
                        [
                            Record(
                                item["seed"],
                                item["input_index"],
                                kind,
                                {},
                                item["output_seed"],
                                item["output_index"],
                                failure=item["message"],
                            )
                        ]
                    )
        return 0
