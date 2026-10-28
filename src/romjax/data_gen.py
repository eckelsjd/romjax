"""Data generation routine and data loading."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import traceback
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterator, Literal, Mapping, Sequence, get_args

import equinox as eqx
import h5py
import jax
import numpy as np
import yaml
from alive_progress import alive_bar
from jaxtyping import Key, PyTree
from loguru import logger
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    PrivateAttr,
    model_validator,
)

from romjax.compression import SVD, Compression
from romjax.execution import (
    ExecutionAction,
    ExecutionError,
    ExecutorConfig,
    ImmediateResult,
    TaskContext,
    TaskSpec,
    TaskWorker,
)
from romjax.graph import Edge, FunctionGraph
from romjax.model import ImplicitSampleable
from romjax.norm import NormOperator, NormTree
from romjax.rng import gen_keys
from romjax.routine import CompositeOverride, Routine, RoutineError
from romjax.storage import ComponentStatus, H5StorageConfig, H5Store, Record, SampleRef
from romjax.tree import (
    TreePath,
    coerce_tree_paths,
    get_subtree,
    pytree_iter,
    pytree_merge,
    pytree_path_iter,
    pytree_stack,
    set_subtree,
)
from romjax.typing import CallableModel, from_yaml
from romjax.utils import _NullProgress, load_h5, required_fields, save_h5

__all__ = [
    "DataGeneration",
    "DataLoader",
    "GenDataBenchmark",
    "GenDataConfig",
    "GenSymlink",
    "LoadDataConfig",
]


def _get_kwargs(fn: Callable) -> set[str]:
    signature = inspect.signature(fn)
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _accepts_kwarg(fn: Callable, name: str) -> bool:
    """Return whether a callable accepts a named keyword argument."""
    parameters = inspect.signature(fn).parameters.values()
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == name for parameter in parameters)


def _resolve_model_samplers(model: Any, roles: Sequence[str]) -> None:
    """Resolve model and sampler-owned runtime state before entering JIT regions."""
    for role in roles:
        model_hook = getattr(model, f"resolve_{role}_sampler", None)
        if callable(model_hook):
            model_hook()
        sampler = getattr(model, f"{role}_sampler", None)
        sampler_hook = getattr(sampler, "resolve_sampler", None)
        if callable(sampler_hook):
            sampler_hook()


def _has_custom_sample_conditions(model: Any) -> bool:
    """Return whether a model overrides the optional condition sampler."""
    if hasattr(model, "conditions_sampler") and getattr(model, "conditions_sampler") is None:
        return False
    return type(model).sample_conditions is not ImplicitSampleable.sample_conditions


def _call_sample_outputs(
    model: Any,
    key: Key,
    one_input: PyTree | None,
    one_solution: PyTree | None,
    one_conditions: PyTree | None,
) -> PyTree:
    """Call a model output sampler while preserving legacy signatures."""
    kwargs: dict[str, PyTree] = {}
    if _accepts_kwarg(model.sample_outputs, "inputs"):
        kwargs["inputs"] = one_input
    if _accepts_kwarg(model.sample_outputs, "solution"):
        kwargs["solution"] = one_solution
    if one_conditions is not None:
        if not _accepts_kwarg(model.sample_outputs, "conditions"):
            raise TypeError("sample_outputs must accept conditions when sample_conditions returns a value.")
        kwargs["conditions"] = one_conditions
    return model.sample_outputs(key, **kwargs)


type SUPPORTED_FORMATS = Literal["h5"]
type SUPPORTED_POLICIES = Literal["reuse", "overwrite", "error"]
type SUPPORTED_CONDITIONS = Literal["ignore", "include", "merge"]


class GenDataConfig(BaseModel, ABC):
    """Abstract class for generating datasets."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    format: SUPPORTED_FORMATS | None = None
    write_policy: SUPPORTED_POLICIES | None = None

    def _validate_format_and_policy(self, format, write_policy):
        """Make sure we have format and policy at runtime."""
        format = self.format or format
        write_policy = self.write_policy or write_policy

        if format is None:
            raise ValueError(f"Must specify a write format. Supported: {get_args(SUPPORTED_FORMATS)}")
        if write_policy is None:
            raise ValueError(f"Must specify a write policy. Supported: {get_args(SUPPORTED_POLICIES)}")

        return format, write_policy

    @abstractmethod
    def generate(
        self, path: Path, format: SUPPORTED_FORMATS | None = None, write_policy: SUPPORTED_POLICIES | None = None
    ) -> None:
        """Generate data at the provided path using the specified format and write_policy."""
        raise NotImplementedError


class GenSymlink(GenDataConfig):
    """Create a symbolic link to an existing dataset or artifact.

    Relative source paths are interpreted from the working directory when
    :meth:`generate` is called.

    :param symlink: existing file or directory to use as the link target
    """

    symlink: Path

    def generate(
        self,
        path: Path,
        format: SUPPORTED_FORMATS | None = None,
        write_policy: SUPPORTED_POLICIES | None = None,
    ) -> None:
        """Create the configured symbolic link at ``path``.

        :param path: destination of the symbolic link
        :param format: inherited data format, unused except for configuration validation
        :param write_policy: behavior when the destination already exists
        :raises RoutineError: if the source is missing or the destination conflicts with the write policy
        """
        _, write_policy = self._validate_format_and_policy(format, write_policy)

        configured_source = self.symlink
        source = configured_source if configured_source.is_absolute() else Path.cwd() / configured_source
        try:
            source = source.resolve(strict=True)
        except FileNotFoundError as exc:
            raise RoutineError(f"Symlink source does not exist: {source.absolute()}") from exc

        destination = Path(path).absolute()
        destination_exists = destination.exists() or destination.is_symlink()
        if destination_exists:
            if write_policy == "reuse":
                return
            if write_policy == "error":
                raise RoutineError(f"Dataset already exists at {destination} and policy='error'")
            if not destination.is_symlink():
                raise RoutineError(f"Refusing to overwrite non-symlink destination at {destination}")
            destination.unlink()

        destination.parent.mkdir(parents=True, exist_ok=True)
        link_target = source
        if not configured_source.is_absolute():
            link_target = Path(os.path.relpath(source, start=destination.parent.resolve()))
        destination.symlink_to(link_target, target_is_directory=source.is_dir())


class GenGraph(GenDataConfig, ABC):
    """Graph sampling with independent task, compute, and storage batch sizes.

    :param batch_size: source samples or complete input families per executor task
    :param map_batch_size: source/family vectorization width; zero maps the full axis.
        Implicit outputs are vectorized within each family. Positive widths bound
        concurrent families, not the memory occupied by the returned batch. Irregular
        cached output plans retain staged mapping with this width for each stage.
    :param storage: settings for newly created files; inherits the routine default
    :param executor: task executor; inherits the routine executor
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None
    name_depth: int = 3
    batch_size: PositiveInt = 1
    map_batch_size: NonNegativeInt = 0
    storage: H5StorageConfig | None = None
    executor: Annotated[
        ExecutorConfig | None, BeforeValidator(lambda value: None if value is None else ExecutorConfig.from_dict(value))
    ] = None
    _required_methods: list[str] = PrivateAttr(default_factory=list)
    _compiled: dict[Any, Callable] = PrivateAttr(default_factory=dict)

    def bar_text(self, path: Path) -> str:
        return "/".join(path.parts[-self.name_depth :])

    def _edge_from_path(self, path: Path) -> Edge:
        if self.graph is None or path.name not in self.graph.edges:
            raise ValueError(f"Last folder name must be an edge name: {path.name}")
        return self.graph.edges[path.name]

    def _validate_required_methods(self, path: Path) -> None:
        model = self._edge_from_path(path)
        for name in self._required_methods:
            if not hasattr(model, name):
                raise ValueError(f"Graph edge {path.name} must implement {name}.")

    def _kernel(self, path: Path, stage: str, flags: tuple[bool, ...], mapped: bool) -> Callable:
        model = self._edge_from_path(path)
        cache_key = id(model), stage, flags, mapped, self.map_batch_size
        if cache_key in self._compiled:
            return self._compiled[cache_key]
        if stage == "source":
            single = model.sample_source
        elif stage == "input":

            def single(key, inputs, solution, residual):
                if flags[0]:
                    inputs = model.sample_inputs(key)
                if flags[1]:
                    solution = model.solve(inputs) if inputs is not None else None
                if flags[2]:
                    residual = model.evaluate(inputs, solution) if solution is not None else None
                return inputs, solution, residual
        else:

            def single(key, conditions, outputs, residual, inputs, solution):
                if flags[0]:
                    conditions = model.sample_conditions(key)
                if flags[1]:
                    outputs = _call_sample_outputs(model, key, inputs, solution, conditions)
                if flags[2]:
                    residual = model.evaluate(inputs, outputs)
                return conditions, outputs, residual

        if not mapped:
            result = eqx.filter_jit(single)
        elif stage == "source":
            result = eqx.filter_jit(lambda keys: jax.lax.map(single, keys, batch_size=self.map_batch_size))
        elif stage == "input":
            result = eqx.filter_jit(
                lambda keys, inputs, solutions, residuals: jax.lax.map(
                    lambda item: single(*item), (keys, inputs, solutions, residuals), batch_size=self.map_batch_size
                )
            )
        else:
            result = eqx.filter_jit(
                lambda keys, conditions, outputs, residuals, inputs, solution: jax.lax.map(
                    lambda item: single(*item, inputs, solution),
                    (keys, conditions, outputs, residuals),
                    batch_size=self.map_batch_size,
                )
            )
        self._compiled[cache_key] = result
        return result

    def _family_kernel(
        self, path: Path, input_flags: tuple[bool, ...], output_flags: tuple[bool, ...] | None
    ) -> Callable[..., PyTree]:
        """Compile complete families, keeping input-to-output context on the device."""
        cache_key = id(self._edge_from_path(path)), "family", input_flags, output_flags, self.map_batch_size
        if cache_key in self._compiled:
            return self._compiled[cache_key]
        input_kernel = self._kernel(path, "input", input_flags, False)
        output_kernel = None if output_flags is None else self._kernel(path, "output", output_flags, False)
        width = self.map_batch_size

        def family(item):
            key, input_values, output_keys, output_values = item
            inputs, solution, residual = input_kernel(key, *input_values)
            outputs = None
            if output_kernel is not None:
                outputs = jax.vmap(output_kernel, in_axes=(0, 0, 0, 0, None, None))(
                    output_keys, *output_values, inputs, solution
                )
            return (inputs, solution, residual), outputs

        @eqx.filter_jit
        def workload(input_keys, input_values, output_keys, output_values):
            return jax.lax.map(family, (input_keys, input_values, output_keys, output_values), batch_size=width)

        self._compiled[cache_key] = workload
        return workload

    @staticmethod
    def _unstack(tree: PyTree, count: int) -> list[PyTree]:
        return [jax.tree.map(lambda leaf: leaf[i], tree) for i in range(count)]

    @staticmethod
    def _group_key(flags: tuple, values: Any) -> tuple:
        leaves, structure = jax.tree.flatten(values)
        return flags, str(structure), tuple((np.shape(leaf), str(np.asarray(leaf).dtype)) for leaf in leaves)

    def _apply(
        self,
        path: Path,
        stage: str,
        flags: tuple,
        keys: Sequence[Key],
        values: Sequence[PyTree],
        *context: Any,
        mapped: bool = True,
    ) -> list[PyTree]:
        kernel = self._kernel(path, stage, flags, mapped)
        if mapped:
            args = () if stage == "source" else tuple(pytree_stack(values))
            result = kernel(pytree_stack(keys), *args, *context)
            return self._unstack(jax.device_get(result), len(keys))
        result = []
        for key, value in zip(keys, values, strict=True):
            args = () if stage == "source" else tuple(value)
            result.append(jax.device_get(kernel(key, *args, *context)))
        return result

    def _compute(self, path: Path, plans: Sequence[dict[str, Any]], *, mapped: bool = True) -> _GeneratedBatch:
        """Compute mapped implicit families, replaying scalar work to identify failures."""
        if not mapped or not self.throw or isinstance(self, GenSource):
            return self._compute_staged(path, plans, mapped=mapped)
        try:
            return self._compute_families(path, plans)
        except Exception as exc:
            return self._recover_families(path, plans, exc)

    def _recover_families(self, path: Path, plans: Sequence[dict[str, Any]], failure: Exception) -> _GeneratedBatch:
        """Replay pure kernels to identify the failing identity and salvage records."""
        recovered = []
        for plan in plans:
            result = self._compute(path, [plan], mapped=False)
            recovered.extend(result.records)
            if result.fatal:
                return _GeneratedBatch(recovered, result.fatal)
        raise RuntimeError(f"Mapped generation failed although individual samples succeeded: {failure}") from failure

    def _input_record(self, model: Edge, plan: dict[str, Any], values: PyTree) -> Record:
        """Build only the input component updates requested by a plan."""
        components = {
            name: value
            for name, value, needed in zip(
                ("inputs", "solutions", "solution_residuals"), values, plan["flags"], strict=True
            )
            if needed
        }
        states = {}
        for name, skip in (
            ("solutions", self.skip_solve or not hasattr(model, "solve")),
            ("solution_residuals", self.skip_evaluate or not hasattr(model, "evaluate")),
        ):
            if skip and (plan["overwrite"] or name not in plan["old"]):
                components[name], states[name] = None, ComponentStatus.SKIPPED
        return Record(self.input_seed, plan["index"], "input", components, states=states)

    def _output_record(self, model: Edge, index: int, output: dict[str, Any], values: PyTree) -> Record:
        """Build an output update with the existing skipped/absent component states."""
        fields = {
            name: value
            for name, value, needed in zip(("conditions", "values", "residuals"), values, output["flags"], strict=True)
            if needed
        }
        states = {}
        if self.skip_evaluate or not hasattr(model, "evaluate"):
            fields["residuals"], states["residuals"] = None, ComponentStatus.SKIPPED
        if not _has_custom_sample_conditions(model):
            fields["conditions"], states["conditions"] = None, ComponentStatus.NONE
        return Record(self.input_seed, index, "output", fields, self.output_seed, output["index"], states)

    def _compute_families(self, path: Path, plans: Sequence[dict[str, Any]]) -> _GeneratedBatch:
        """Batch compatible plans; retain staged execution for irregular cached outputs."""
        model = self._edge_from_path(path)
        groups, by_index = {}, {}
        for plan in plans:
            outputs = plan["outputs"]
            signatures = [self._group_key(item["flags"], item["values"]) for item in outputs]
            if signatures and any(signature != signatures[0] for signature in signatures[1:]):
                result = self._compute_staged(path, [plan])
                if result.fatal:
                    # Let the caller replay in original plan order, including earlier families.
                    raise RuntimeError(result.fatal)
                by_index[plan["index"]] = result.records
                continue
            group_key = self._group_key(plan["flags"], plan["values"]), len(outputs), tuple(signatures[:1])
            groups.setdefault(group_key, []).append(plan)
        for group in groups.values():
            first = group[0]
            has_outputs = bool(first["outputs"])
            keys = [jax.random.fold_in(jax.random.key(self.input_seed), plan["index"]) for plan in group]
            output_keys = None
            output_values = (None, None, None)
            if has_outputs:
                output_keys = pytree_stack(
                    [
                        pytree_stack([self._output_key(key, item["index"]) for item in plan["outputs"]])
                        for key, plan in zip(keys, group, strict=True)
                    ]
                )
                output_values = pytree_stack(
                    [pytree_stack([item["values"] for item in plan["outputs"]]) for plan in group]
                )
            kernel = self._family_kernel(path, first["flags"], first["outputs"][0]["flags"] if has_outputs else None)
            result = kernel(
                pytree_stack(keys), pytree_stack([plan["values"] for plan in group]), output_keys, output_values
            )
            # One transfer for the whole group, after input-to-output dependencies finish on device.
            for plan, (inputs, outputs) in zip(group, self._unstack(jax.device_get(result), len(group)), strict=True):
                records = [self._input_record(model, plan, inputs)]
                if has_outputs:
                    records.extend(
                        self._output_record(model, plan["index"], output, values)
                        for output, values in zip(
                            plan["outputs"], self._unstack(outputs, len(plan["outputs"])), strict=True
                        )
                    )
                by_index[plan["index"]] = records
        return _GeneratedBatch([record for plan in plans for record in by_index[plan["index"]]])

    def _compute_staged(self, path: Path, plans: Sequence[dict[str, Any]], *, mapped: bool = True) -> _GeneratedBatch:
        """Compute independent families; this method has no target-file access."""
        model = self._edge_from_path(path)
        records: list[Record] = []
        source = isinstance(self, GenSource)
        seed = self.seed if source else self.input_seed
        if not self.throw and mapped:
            # Host orchestration isolates Python/model exceptions; numerical kernels are shared.
            for plan in plans:
                try:
                    records.extend(self._compute(path, [plan], mapped=False).records)
                except Exception as exc:
                    records.append(
                        Record(
                            seed,
                            plan["index"],
                            "source" if source else "input",
                            {},
                            states={"values" if source else "inputs": ComponentStatus.FAILED},
                            failure=f"{type(exc).__name__}: {exc}",
                        )
                    )
            return _GeneratedBatch(records)
        groups = {}
        for plan in plans:
            flags, values = plan["flags"], plan["values"]
            groups.setdefault(self._group_key(flags, values), []).append(plan)
        try:
            for group in groups.values():
                flags = group[0]["flags"]
                keys = [jax.random.fold_in(jax.random.key(seed), item["index"]) for item in group]
                values = [item["values"] for item in group]
                computed = self._apply(path, "source" if source else "input", flags, keys, values, mapped=mapped)
                for plan, key, value in zip(group, keys, computed, strict=True):
                    index = plan["index"]
                    if source:
                        records.append(Record(seed, index, "source", {"values": value}))
                        continue
                    inputs, solution, residual = value
                    records.append(self._input_record(model, plan, value))
                    out_groups = {}
                    for output in plan["outputs"]:
                        group_key = self._group_key(output["flags"], output["values"])
                        if not mapped or not self.throw:
                            group_key = (*group_key, output["index"])
                        out_groups.setdefault(group_key, []).append(output)
                    for outputs in out_groups.values():
                        out_keys = [self._output_key(key, item["index"]) for item in outputs]
                        try:
                            results = self._apply(
                                path,
                                "output",
                                outputs[0]["flags"],
                                out_keys,
                                [item["values"] for item in outputs],
                                inputs,
                                solution,
                                mapped=mapped,
                            )
                        except Exception as failure:
                            if mapped:
                                raise
                            message = f"{type(failure).__name__}: {failure}\n{traceback.format_exc()}"
                            records.append(
                                Record(
                                    seed,
                                    index,
                                    "output",
                                    {},
                                    self.output_seed,
                                    outputs[0]["index"],
                                    {"values": ComponentStatus.FAILED},
                                    message,
                                )
                            )
                            if self.throw:
                                return _GeneratedBatch(records, message)
                            continue
                        for output, result in zip(outputs, results, strict=True):
                            records.append(self._output_record(model, index, output, result))
            return _GeneratedBatch(records)
        except Exception as exc:
            if not mapped:
                index = plans[0]["index"]
                message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                records.append(
                    Record(
                        seed,
                        index,
                        "source" if source else "input",
                        {},
                        states={"values" if source else "inputs": ComponentStatus.FAILED},
                        failure=message,
                    )
                )
                return _GeneratedBatch(records, message if self.throw else None)
            return self._recover_families(path, plans, exc)

    def _plans(self, store: H5Store, indices: Sequence[int], policy: str) -> list[dict[str, Any]]:
        source = isinstance(self, GenSource)
        model = self._edge_from_path(store.root)
        seed = self.seed if source else self.input_seed
        requested = []
        for index in indices:
            ref = store.reference(seed, index, "source" if source else "input")
            if ref is not None:
                requested.append(ref)
            if not source:
                for j in range(self.outputs_per_input):
                    ref = store.reference(seed, index, "output", self.output_seed, j)
                    if ref is not None:
                        requested.append(ref)
        raw = {}
        plans = []
        valid = (ComponentStatus.COMPLETE, ComponentStatus.NONE)
        for index in indices:
            old = raw.get(("source" if source else "input", seed, index, None, None), {})
            status = store.status(seed, index)
            overwrite = policy == "overwrite"
            needed = overwrite or status["values" if source else "inputs"] not in valid
            if source:
                if needed:
                    plans.append({"index": index, "flags": (), "values": (), "old": old, "overwrite": overwrite})
                continue
            solve = (
                not self.skip_solve
                and hasattr(model, "solve")
                and (needed or overwrite or status["solutions"] not in valid)
            )
            evaluate = (
                not self.skip_evaluate
                and hasattr(model, "evaluate")
                and (solve or overwrite or status["solution_residuals"] not in valid)
            )
            outputs = []
            for j in range(self.outputs_per_input):
                state = store.status(seed, index, self.output_seed, j)
                previous = raw.get(("output", seed, index, self.output_seed, j), {})
                conditions = _has_custom_sample_conditions(model) and (overwrite or state["conditions"] not in valid)
                values = overwrite or needed or solve or conditions or state["values"] not in valid
                residual = (
                    not self.skip_evaluate
                    and hasattr(model, "evaluate")
                    and (values or state["residuals"] not in valid)
                )
                if conditions or values or residual:
                    outputs.append(
                        {
                            "index": j,
                            "flags": (conditions, values, residual),
                            "values": tuple(previous.get(name) for name in ("conditions", "values", "residuals")),
                        }
                    )
            if needed or solve or evaluate or outputs or overwrite:
                plans.append(
                    {
                        "index": index,
                        "flags": (needed, solve, evaluate),
                        "old": old,
                        "values": tuple(old.get(name) for name in ("inputs", "solutions", "solution_residuals")),
                        "outputs": outputs,
                        "overwrite": overwrite,
                    }
                )
        # Decide work from metadata first: a complete reuse pass reads no numerical fields.
        identities = set()
        for plan in plans:
            identities.add(("source" if source else "input", seed, plan["index"], None, None))
            for output in plan.get("outputs", []):
                identities.add(("output", seed, plan["index"], self.output_seed, output["index"]))
        requested = [ref for ref in requested if ref.identity in identities]
        raw = {ref.identity: data for ref, data in zip(requested, store.read_components(requested), strict=True)}
        for plan in plans:
            old = raw.get(("source" if source else "input", seed, plan["index"], None, None), {})
            plan["old"] = old
            if source:
                continue
            # Overwrite cannot use a solution which this request explicitly discards.
            plan["values"] = (
                old.get("inputs"),
                None if plan["overwrite"] and self.skip_solve else old.get("solutions"),
                None if plan["overwrite"] and self.skip_evaluate else old.get("solution_residuals"),
            )
            for output in plan["outputs"]:
                previous = raw.get(("output", seed, plan["index"], self.output_seed, output["index"]), {})
                output["values"] = tuple(previous.get(name) for name in ("conditions", "values", "residuals"))
        reservations = []
        for plan in plans:
            reservations.append(Record(seed, plan["index"], "source" if source else "input", {}))
            for output in plan.get("outputs", []):
                reservations.append(Record(seed, plan["index"], "output", {}, self.output_seed, output["index"]))
        store.reserve(reservations)
        return plans

    def _generate(
        self,
        path: Path,
        format: str,
        write_policy: str,
        executor: ExecutorConfig,
        jax_env: Mapping[str, str] | None = None,
        storage: H5StorageConfig | None = None,
    ) -> None:
        path = Path(path).resolve()
        self._validate_required_methods(path)
        if format != "h5":
            raise RoutineError(f"Unsupported sample format: {format}")
        if write_policy == "error" and path.exists():
            raise RoutineError(f"Dataset already exists at {path} and policy='error'")
        source = isinstance(self, GenSource)
        store = H5Store(path, "source" if source else "implicit", self.storage or storage)
        worker = _GenerationWorker(self, path)
        # Reconstruction must succeed before creating target directories or locks.
        worker.prepare(remote=executor.kind in {"process", "mpi"})
        if executor.kind == "hybrid":
            raise ExecutionError("Hybrid generation is not supported.")
        count = self.samples if source else self.input_samples
        fatal = []

        def tasks():
            for start in range(0, count, self.batch_size):
                stop = min(count, start + self.batch_size)
                plans = self._plans(store, range(start, stop), write_policy)
                yield TaskSpec(
                    str(start),
                    f"inputs {start}:{stop}",
                    plans,
                    units=stop - start,
                    immediate=None if plans else ImmediateResult(_GeneratedBatch([])),
                )

        def commit(outcome):
            store.write(outcome.value.records)
            if outcome.value.fatal:
                fatal.append(outcome.value.fatal)
            return ExecutionAction.STOP if fatal else ExecutionAction.CONTINUE

        with store.locked():
            store.validate_storage()
            if not source:
                store.configure_key_scheme(self.input_seed, self.output_seed, self.mix_output_seed)
            executor.execute(
                tasks(),
                worker,
                on_result=commit,
                total_units=count,
                collect_results=False,
                progress_bar={"title": self.bar_text(path)},
                jax_env=jax_env,
            )
        if fatal:
            raise RoutineError(fatal[0])

    def generate(self, path: Path, format: str | None = None, write_policy: str | None = None) -> None:
        """Generate records through the selected executor and controller-owned writer."""
        format, write_policy = self._validate_format_and_policy(format, write_policy)
        self._generate(Path(path), format, write_policy, self.executor or ExecutorConfig())


class GenImplicitModel(GenGraph):
    """Generate reproducible implicit input families with optional solves/conditions.

    :param input_samples: number of input identities to request
    :param outputs_per_input: sampled output identities requested for each input
    :param input_seed: input key seed
    :param output_seed: output key seed
    :param mix_output_seed: fold output seed into each input key
    :param skip_solve: omit base solutions
    :param skip_evaluate: omit residual evaluation
    :param throw: stop after a model failure; false isolates failures
    """

    input_samples: NonNegativeInt
    outputs_per_input: NonNegativeInt
    input_seed: int
    output_seed: int
    mix_output_seed: bool = True
    throw: bool = True
    skip_solve: bool = False
    skip_evaluate: bool = False
    _required_methods: list[str] = PrivateAttr(default=["sample_inputs"])

    def _output_key(self, input_key: Key, index: int) -> Key:
        base = (
            jax.random.fold_in(input_key, self.output_seed)
            if self.mix_output_seed
            else jax.random.key(self.output_seed)
        )
        return jax.random.fold_in(base, index)


class GenSource(GenGraph):
    """Generate indexed source samples with optional per-sample failure isolation."""

    samples: NonNegativeInt
    seed: int
    throw: bool = True
    _required_methods: list[str] = PrivateAttr(default=["sample_source"])


@dataclass
class _GeneratedBatch:
    records: list[Record]
    fatal: str | None = None


@lru_cache(maxsize=4)
def _generation_worker(identity: str, configuration: str) -> GenGraph:
    import romjax

    generator = romjax.load(configuration)
    roles = ("source",) if isinstance(generator, GenSource) else ("inputs", "outputs", "conditions")
    for model in generator.graph.edges.values():
        _resolve_model_samplers(model, roles)
    return generator


@dataclass
class _GenerationWorker(TaskWorker):
    generator: GenGraph | None
    path: Path
    configuration: str = ""

    def prepare(self, *, remote: bool) -> _GenerationWorker:
        import romjax

        if remote:
            try:
                text = romjax.YamlLoader.dump(self.generator, _preserve_yaml_sources=True, _preserve_model_types=True)
                restored = romjax.load(text)
                restored._validate_required_methods(self.path)
            except Exception as exc:
                raise ExecutionError(f"Cannot reconstruct {type(self.generator).__name__}: {exc}") from exc
            return _GenerationWorker(None, self.path, text)
        self.generator._compiled.clear()
        roles = ("source",) if isinstance(self.generator, GenSource) else ("inputs", "outputs", "conditions")
        _resolve_model_samplers(self.generator._edge_from_path(self.path), roles)
        return self

    def run(self, plans: Sequence[dict[str, Any]], context: TaskContext) -> _GeneratedBatch:
        generator = self.generator or _generation_worker(context.run_id, self.configuration)
        return generator._compute(self.path, plans)


class LoadDataConfig[T: Any](BaseModel, ABC):
    """Batch loading with all-or-none epoch caching.

    :param batch_size: samples returned per batch
    :param cache_policy: none loads one batch; epoch retains every encountered sample
    :param cache_storage: host or device placement for cached samples
    :param stack_batch: explicitly request a stacked PyTree instead of sample lists
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    batch_size: PositiveInt = 16
    shuffle_seed: int = 0
    max_epochs: PositiveInt | None = None
    cache_policy: Literal["none", "epoch"] = "none"
    cache_storage: Literal["host", "device"] = "host"
    stack_batch: bool = False
    _cache_entries: dict[Any, PyTree] = PrivateAttr(default_factory=dict)
    _stores: dict[Path, H5Store] = PrivateAttr(default_factory=dict)
    _store_versions: dict[Path, tuple] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _reject_cache_limits(cls, value):
        if isinstance(value, Mapping) and {"cache_max_items", "cache_max_bytes"} & value.keys():
            raise ValueError("Cache limits were removed; use cache_policy=none or epoch.")
        return value

    @abstractmethod
    def discover_sample_refs(self, root: Path) -> list[T]:
        """Discover references from index metadata."""
        raise NotImplementedError

    def _store(self, root: Path, *, refresh: bool = False) -> H5Store:
        root = Path(root).resolve()
        if refresh:
            version = tuple(
                (path.name, stat.st_mtime_ns, stat.st_size)
                for path in sorted(root.glob("*.h5"))
                for stat in [path.stat()]
            )
            if root in self._stores and self._store_versions.get(root) != version:
                del self._stores[root]
                self._cache_entries = {ref: value for ref, value in self._cache_entries.items() if ref.root != root}
            self._store_versions[root] = version
        if root not in self._stores:
            self._stores[root] = H5Store(root)
        return self._stores[root]

    def select_epoch_refs(self, refs: Sequence[T], indices: np.ndarray) -> list[T]:
        """Select the fixed pool before epoch reshuffling."""
        return [refs[int(index)] for index in indices]

    def count_epoch_refs(self, refs: Sequence[T], indices: np.ndarray) -> int:
        """Count selected references."""
        return len(self.select_epoch_refs(refs, indices))

    def _read_samples(self, refs: Sequence[SampleRef]) -> list[PyTree]:
        result = [None] * len(refs)
        by_root = {}
        for i, ref in enumerate(refs):
            by_root.setdefault(ref.root, []).append((i, ref))
        for root, items in by_root.items():
            values = self._store(root).read_batch(
                [ref for _, ref in items], include_conditions=getattr(self, "conditions", None) != "ignore"
            )
            for (i, _), sample in zip(items, values, strict=True):
                result[i] = sample
        return result

    def load_sample(self, ref: SampleRef) -> PyTree:
        """Load an individual sample through the same bulk reader."""
        return self._read_samples([ref])[0]

    def load_batch(self, refs: Sequence[SampleRef]) -> PyTree | list[PyTree]:
        """Bulk-read cache misses and restore requested sample order."""
        if self.cache_policy == "none":
            samples = self._read_samples(refs)
        else:
            missing = list(dict.fromkeys(ref for ref in refs if ref not in self._cache_entries))
            loaded = self._read_samples(missing)
            for ref, sample in zip(missing, loaded, strict=True):
                self._cache_entries[ref] = jax.device_put(sample) if self.cache_storage == "device" else sample
            samples = [self._cache_entries[ref] for ref in refs]
        if self.stack_batch and samples:
            try:
                return pytree_stack(samples)
            except (ValueError, TypeError) as exc:
                raise ValueError("Cannot stack incompatible sample schemas/shapes; use stack_batch=False.") from exc
        return samples


type ImplicitSampleRef = SampleRef


class LoadImplicitModel(LoadDataConfig[SampleRef]):
    """Select/load implicit solutions and sampled outputs from flat HDF5 indices."""

    max_samples: PositiveInt | None = None
    max_input_samples: PositiveInt | None = None
    max_outputs_per_input: PositiveInt | None = None
    skip_input: Callable[[SampleRef], bool] | None = None
    skip_output: Callable[[SampleRef], bool] | None = None
    load_solution: bool = True
    conditions: SUPPORTED_CONDITIONS = "include"
    solution_only: bool = False

    @model_validator(mode="after")
    def _validate_solution_flags(self) -> LoadImplicitModel:
        if self.solution_only and not self.load_solution:
            raise ValueError("solution_only=True requires load_solution=True.")
        return self

    def discover_sample_refs(self, root: Path) -> list[SampleRef]:
        """Read indices and apply each input predicate once to its parent reference."""
        store = self._store(root, refresh=True)
        refs = store.discover()
        skipped = set()
        if self.skip_input is not None:
            for seed, index in dict.fromkeys(ref.input_id for ref in refs):
                if self.skip_input(store.reference(seed, index, "input")):
                    skipped.add((seed, index))
        return [
            ref
            for ref in refs
            if ref.input_id not in skipped
            and (
                (ref.kind == "solution" and self.load_solution)
                or (
                    ref.kind == "output"
                    and not self.solution_only
                    and (not self.skip_output or not self.skip_output(ref))
                )
            )
        ]

    def select_epoch_refs(self, refs: Sequence[SampleRef], indices: np.ndarray) -> list[SampleRef]:
        """Select a reproducible pool with global input/output/sample caps."""
        selected, inputs, outputs = [], set(), {}
        for index in indices:
            ref = refs[int(index)]
            if self.max_samples is not None and len(selected) >= self.max_samples:
                break
            if (
                self.max_input_samples is not None
                and ref.input_id not in inputs
                and len(inputs) >= self.max_input_samples
            ):
                continue
            if self.max_outputs_per_input is not None and outputs.get(ref.input_id, 0) >= self.max_outputs_per_input:
                continue
            selected.append(ref)
            inputs.add(ref.input_id)
            outputs[ref.input_id] = outputs.get(ref.input_id, 0) + 1
        return selected

    def _read_samples(self, refs: Sequence[SampleRef]) -> list[PyTree]:
        samples = super()._read_samples(refs)
        for sample in samples:
            if self.conditions == "ignore":
                sample.pop("conditions", None)
            elif self.conditions == "merge" and "conditions" in sample:
                sample["inputs"] = pytree_merge(sample.get("inputs"), sample.pop("conditions"))
        return samples


class LoadSource(LoadDataConfig[SampleRef]):
    """Load source samples from flat HDF5 indices."""

    max_samples: PositiveInt | None = None
    skip_sample: Callable[[SampleRef], bool] | None = None

    def discover_sample_refs(self, root: Path) -> list[SampleRef]:
        """Discover complete source references."""
        return [
            ref
            for ref in self._store(root, refresh=True).discover()
            if ref.kind == "source" and (self.skip_sample is None or not self.skip_sample(ref))
        ]

    def count_epoch_refs(self, refs: Sequence[SampleRef], indices: np.ndarray) -> int:
        """Count without invoking selection hooks during list preallocation."""
        return len(indices) if self.max_samples is None else min(len(indices), self.max_samples)

    def select_epoch_refs(self, refs: Sequence[SampleRef], indices: np.ndarray) -> list[SampleRef]:
        """Apply the global source-sample cap."""
        selected = super().select_epoch_refs(refs, indices)
        return selected if self.max_samples is None else selected[: self.max_samples]


def _validate_benchmark_generator(value: Any) -> GenImplicitModel | GenSource:
    """Validate a benchmark's nested generator from a YAML-friendly mapping."""
    if isinstance(value, GenImplicitModel | GenSource):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("Benchmark generator must be an implicit or source generator configuration.")
    options = dict(value)
    kind = options.pop("kind", None)
    if kind == "implicit":
        return GenImplicitModel(**options)
    if kind == "source":
        return GenSource(**options)
    raise ValueError("Benchmark generator requires kind='implicit' or kind='source'.")


class GenDataBenchmark(GenDataConfig):
    """Benchmark a data generator without persisting generated samples.

    The configured generator supplies the graph, random seeds, batching, and numerical
    operations. It stores repeat-wise wall times in an HDF5 cache. Each hardware group
    contains ``total`` in seconds per repeat and ``amortized`` in seconds per input
    sample. Each repeat times all input batches through the selected executor,
    including dispatch and synchronization. Preparation, warmup, progress, and
    file I/O are excluded. With ``warmup=0``, first-use compilation is measured.

    Note that each worker (for parallel executors) will iterate the entire dataset `warmup` times, to account
    for changing batch sizes and compilation.

    :param generator: implicit or source data generator to benchmark
    :param samples: number of input samples included in each complete timed pass
    :param repeat: number of complete timed passes over all batches
    :param warmup: number of untimed full passes on each participating worker
    :param filename: HDF5 timing-cache filename written below the dataset path
    :param hardware: hardware-label prefix; inferred from JAX when omitted, then suffixed with executor identity
    """

    show_progress: bool = True

    generator: Annotated[GenImplicitModel | GenSource, BeforeValidator(_validate_benchmark_generator)]
    samples: PositiveInt
    repeat: PositiveInt = 5
    warmup: NonNegativeInt = 1
    filename: str = "timings.h5"
    hardware: str | None = None
    name_depth: int = 3

    @model_validator(mode="after")
    def _validate_benchmark(self) -> "GenDataBenchmark":
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError("Benchmark filename must be a non-empty basename.")
        if not self.generator.throw:
            raise ValueError("Data-generation benchmarks require generator.throw=True.")
        self.hardware = self._hardware_label(self.hardware)
        return self

    @staticmethod
    def _hardware_label(configured: str | None) -> str:
        """Return a stable explicit or environment-derived cache label."""
        label = configured or os.environ.get("DATA_GEN_HARDWARE")
        if label is None:
            if jax.default_backend() == "gpu":
                label = "gpu"
            else:
                threads = os.environ.get("CPU_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
                label = f"cpu_{threads}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError("Hardware labels may contain only letters, numbers, '_', '-', and '.'.")
        return label

    def bar_text(self, path: Path):
        return "/".join(path.parts[-self.name_depth :])

    def _metadata(self) -> dict[str, Any]:
        """Return cache metadata sufficient to reject incompatible reuse."""
        import romjax

        generator = self.generator
        # Executor settings are tracked separately; presentation changes must
        # not invalidate an otherwise identical numerical benchmark.
        numerical = generator.model_copy(update={"executor": None, "name_depth": 3, "storage": None})
        serialized = romjax.YamlLoader.dump(numerical, sort_keys=True, _preserve_yaml_sources=True)
        return {
            "schema_version": np.asarray(5),
            "timing": np.bytes_("full_pass_wall"),
            "samples": np.asarray(int(self.samples)),
            "repeat": np.asarray(int(self.repeat)),
            "warmup": np.asarray(int(self.warmup)),
            "batch_size": np.asarray(int(generator.batch_size)),
            "map_batch_size": np.asarray(int(generator.map_batch_size)),
            "compute_policy": np.bytes_("lax_map_families_vmap_outputs"),
            "generator": np.bytes_(type(generator).__name__),
            "generator_hash": np.bytes_(hashlib.sha256(serialized.encode()).hexdigest()),
        }

    @staticmethod
    def _metadata_matches(cached: Mapping[str, Any], requested: Mapping[str, Any]) -> bool:
        """Return whether cached scalar metadata matches the requested benchmark."""
        if set(cached) != set(requested):
            return False
        return all(np.array_equal(np.asarray(cached[key]), np.asarray(value)) for key, value in requested.items())

    def _input_batches(self, seed: int, batch_size: int) -> tuple[jax.Array, ...]:
        """Construct deterministic key batches without creating sample directories."""
        keys = tuple(gen_keys(self.samples, seed))
        return tuple(pytree_stack(keys[start : start + batch_size]) for start in range(0, len(keys), batch_size))

    @staticmethod
    def _output_keys(generator: GenImplicitModel, input_keys: jax.Array) -> jax.Array | None:
        """Construct one output-key row per input key using normal generation semantics."""
        if generator.outputs_per_input == 0:
            return None
        output_base = jax.random.key(generator.output_seed)

        def keys_for_input(input_key: Key) -> jax.Array:
            base = jax.random.fold_in(input_key, generator.output_seed) if generator.mix_output_seed else output_base
            return jax.vmap(lambda index: jax.random.fold_in(base, index))(np.arange(generator.outputs_per_input))

        return jax.vmap(keys_for_input)(input_keys)

    def _workload(self, path: Path) -> tuple[Callable[..., PyTree], tuple[tuple[Any, ...], ...]]:
        """Prepare the same mapped numerical kernels without persistence or array transport."""
        generator = self.generator
        generator._validate_required_methods(path)
        model = generator._edge_from_path(path)
        roles = ("source",) if isinstance(generator, GenSource) else ("inputs", "outputs", "conditions")
        _resolve_model_samplers(model, roles)
        seed = generator.seed if isinstance(generator, GenSource) else generator.input_seed
        batches = self._input_batches(seed, generator.batch_size)
        if isinstance(generator, GenSource):
            return generator._kernel(path, "source", (), True), tuple((keys,) for keys in batches)
        solve = hasattr(model, "solve") and not generator.skip_solve
        evaluate = hasattr(model, "evaluate") and not generator.skip_evaluate
        output_flags = (_has_custom_sample_conditions(model), True, evaluate) if generator.outputs_per_input else None
        workload = generator._family_kernel(path, (True, solve, solve and evaluate), output_flags)
        args = tuple(
            (keys, (None, None, None), self._output_keys(generator, keys), (None, None, None)) for keys in batches
        )
        jax.block_until_ready(args)
        return workload, args

    def _benchmark(
        self,
        path: Path,
        executor: ExecutorConfig | None = None,
        jax_env: Mapping[str, str] | None = None,
    ) -> np.ndarray:
        """Measure each complete input pass through the selected executor."""
        executor = executor or self.generator.executor or ExecutorConfig()
        executor = executor.model_copy(update={"show_progress": self.show_progress and executor.show_progress})
        count = (self.samples + self.generator.batch_size - 1) // self.generator.batch_size
        tasks = tuple(
            TaskSpec(
                str(index),
                f"batch {index + 1}/{count}",
                index,
                units=min(self.generator.batch_size, self.samples - index * self.generator.batch_size),
            )
            for index in range(count)
        )
        durations = executor.execute_passes(
            tasks,
            _BenchmarkWorker(self, Path(path).absolute()),
            warmup=self.warmup,
            repeat=self.repeat,
            progress_bar={"title": self.bar_text(path)},
            jax_env=jax_env,
        )
        return np.asarray(durations)

    def _executor_metadata(self, executor: ExecutorConfig, jax_env: Mapping[str, str] | None) -> str:
        """Describe timing-relevant settings without presentation options."""
        count = (self.samples + self.generator.batch_size - 1) // self.generator.batch_size
        settings = executor.model_dump(mode="json", exclude={"show_progress", "progress_bar"})
        settings["workers"] = executor._pass_worker_count(count)
        settings["jax_env"] = {
            "JAX_ENABLE_X64": str(jax.config.jax_enable_x64).lower(),
            "JAX_PLATFORMS": jax.config.jax_platforms,
            **(jax_env or {}),
        }
        return json.dumps(settings, sort_keys=True)

    def generate(
        self,
        path: Path,
        format: SUPPORTED_FORMATS | None = None,
        write_policy: SUPPORTED_POLICIES | None = None,
    ) -> None:
        """Reuse or collect timings and persist the current hardware entry."""
        format, write_policy = self._validate_format_and_policy(format, write_policy)
        self._generate(Path(path), format, write_policy, self.generator.executor or ExecutorConfig())

    def _generate(
        self,
        path: Path,
        format: str,
        write_policy: str,
        executor: ExecutorConfig,
        jax_env: Mapping[str, str] | None = None,
    ) -> None:
        """Collect or reuse one executor-specific hardware entry."""
        if format != "h5":
            raise RoutineError(f"Save format {format!r} not recognized.")
        path = Path(path)
        cache_path = path / self.filename
        cached = load_h5({}, cache_path, jax=False) if cache_path.exists() else {}
        assert self.hardware is not None
        execution = self._executor_metadata(executor, jax_env)
        fingerprint = hashlib.sha256(execution.encode()).hexdigest()[:12]
        label = f"{self.hardware}.{executor.kind}.{fingerprint}"
        exists = label in cached
        metadata = self._metadata()
        metadata["executor"] = np.bytes_(execution)
        if exists and write_policy == "error":
            raise RoutineError(f"Benchmark timings already exist for {label!r} at {cache_path}.")
        if exists and write_policy == "reuse":
            entry = cached[label]
            if not isinstance(entry, Mapping) or not self._metadata_matches(entry.get("_benchmark", {}), metadata):
                raise RoutineError(f"Cached benchmark at {cache_path} is incompatible; use write_policy='overwrite'.")
            return

        total = self._benchmark(path, executor=executor, jax_env=jax_env)
        cached[label] = {
            "_benchmark": metadata,
            "total": total,
            "amortized": total / self.samples,
        }
        path.mkdir(parents=True, exist_ok=True)
        save_h5(cached, cache_path, mode="w")


@dataclass
class _BenchmarkWorker(TaskWorker[int, None]):
    benchmark: GenDataBenchmark | None
    path: Path
    configuration: str = ""
    workload: tuple[Callable, tuple] | None = None

    def prepare(self, *, remote: bool) -> _BenchmarkWorker:
        import romjax

        if not remote:
            return _BenchmarkWorker(self.benchmark, self.path, workload=self.benchmark._workload(self.path))
        try:
            configuration = romjax.YamlLoader.dump(
                self.benchmark, _preserve_yaml_sources=True, _preserve_model_types=True
            )
            restored = romjax.load(configuration)
            restored.generator._validate_required_methods(self.path)
        except Exception as exc:
            raise ExecutionError(f"Cannot reconstruct benchmark for remote execution: {exc}") from exc
        return _BenchmarkWorker(None, self.path, configuration)

    def initialize(self, payloads: Sequence[int], context: TaskContext) -> None:
        import romjax

        if self.workload is None:
            self.workload = romjax.load(self.configuration)._workload(self.path)

    def run(self, payload: int, context: TaskContext) -> None:
        workload, batches = self.workload
        jax.block_until_ready(workload(*batches[payload]))


type NormAxes = int | Sequence[int] | None


class NormOutlierFilter(BaseModel):
    """
    Streaming outlier filter for normalization-stat accumulation.

    :param method: filtering method
    :param threshold: number of standard deviations used by sigma clipping
    :param warmup_samples: number of initial samples used before filtering begins
    """

    method: Literal["sigma"] = "sigma"
    threshold: float = 5.0
    warmup_samples: NonNegativeInt = 32


class GenNormLeaf(BaseModel):
    """
    Generation-time normalization config for one pytree leaf.

    Extra fields are preserved as runtime normalization options in the saved artifact.

    :param callable: runtime normalization callable or registered name
    :param axes: axes to reduce over within each single-sample leaf (use empty axes to keep all)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    callable: str | Callable[..., Any] = "zscore"
    axes: NormAxes = None

    @model_validator(mode="before")
    @classmethod
    def _from_spec(cls, value: Any) -> Any:
        if isinstance(value, GenNormLeaf):
            return value
        if isinstance(value, str) or callable(value):
            return {"callable": value}
        return value

    @property
    def opts(self) -> dict[str, Any]:
        """Return static runtime normalization options."""
        return dict(self.model_extra or {})

    @property
    def callable_name(self) -> str:
        """Return an importable or registered callable name for artifact persistence."""
        if isinstance(self.callable, str):
            return self.callable
        return f"{self.callable.__module__}.{self.callable.__qualname__}"


def _is_gennorm_leaf_spec(value: Any) -> bool:
    return (
        isinstance(value, GenNormLeaf)
        or isinstance(value, str)
        or callable(value)
        or (isinstance(value, Mapping) and "callable" in value)
    )


def _validate_gennorm_spec(value: Any) -> Any:
    if isinstance(value, GenNormTree):
        return value.root
    if _is_gennorm_leaf_spec(value):
        return GenNormLeaf.model_validate(value)
    if isinstance(value, Mapping):
        return {key: _validate_gennorm_spec(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_validate_gennorm_spec(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_validate_gennorm_spec(item) for item in value)
    return value


class GenNormTree(BaseModel):
    """
    Pytree of generation-time normalization leaf configs.

    :param root: one broadcast leaf config or a pytree of per-leaf configs
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Any

    @model_validator(mode="before")
    @classmethod
    def _from_spec(cls, value: Any) -> Any:
        if isinstance(value, GenNormTree):
            return {"root": value.root}
        if isinstance(value, Mapping) and set(value.keys()) == {"root"}:
            return value
        return {"root": value}

    @model_validator(mode="after")
    def _validate_root(self) -> "GenNormTree":
        self.root = _validate_gennorm_spec(self.root)
        return self


class NormStatsCallable(CallableModel):
    """
    Custom streaming normalization-stat callable.

    The configured callable should provide ``init(sample_leaf, axes, **opts)``,
    ``update(state, sample_leaf)``, and ``finalize(state)`` methods.
    """

    callable: Callable[..., Any]

    def init(self, sample_leaf: Any, axes: tuple[int, ...], **opts: Any) -> Any:
        return self.callable.init(sample_leaf, axes=axes, **opts)

    def update(self, state: Any, sample_leaf: Any) -> Any:
        return self.callable.update(state, sample_leaf)

    def finalize(self, state: Any) -> dict[str, Any]:
        return self.callable.finalize(state)


def _normalize_norm_axes(axes: NormAxes, ndim: int) -> tuple[int, ...]:
    """Normalize reduction axes against a leaf rank."""
    if axes is None:
        return tuple(range(ndim))
    axes_tuple = (axes,) if isinstance(axes, int) else tuple(axes)
    normalized: list[int] = []
    for axis in axes_tuple:
        axis_int = int(axis)
        if axis_int < 0:
            axis_int += ndim
        if axis_int < 0 or axis_int >= ndim:
            raise ValueError(f"Normalization axis {axis!r} is out of range for leaf rank {ndim}.")
        if axis_int in normalized:
            raise ValueError(f"Duplicate normalization axis {axis!r}.")
        normalized.append(axis_int)
    return tuple(normalized)


def _finite_group_stats(arr: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite count, mean, and M2 over axes with keepdims."""
    arr = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(arr)
    count = np.sum(finite, axis=axes, keepdims=True, dtype=np.float64) if axes else finite.astype(np.float64)
    total = np.sum(np.where(finite, arr, 0.0), axis=axes, keepdims=True) if axes else np.where(finite, arr, 0.0)
    mean = np.divide(total, count, out=np.zeros_like(total, dtype=np.float64), where=count > 0)
    delta = np.where(finite, arr - mean, 0.0)
    m2 = np.sum(np.square(delta), axis=axes, keepdims=True) if axes else np.square(delta)
    return count, mean, m2


def _combine_moments(
    count_a: np.ndarray,
    mean_a: np.ndarray,
    m2_a: np.ndarray,
    count_b: np.ndarray,
    mean_b: np.ndarray,
    m2_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine two sets of streaming moments (Welford/Chan algorithm)."""
    total = count_a + count_b
    delta = mean_b - mean_a
    mean = np.divide(
        mean_a * count_a + mean_b * count_b,
        total,
        out=np.zeros_like(mean_a, dtype=np.float64),
        where=total > 0,
    )
    correction = np.divide(
        np.square(delta) * count_a * count_b,
        total,
        out=np.zeros_like(m2_a, dtype=np.float64),
        where=total > 0,
    )
    return total, mean, m2_a + m2_b + correction


class _BuiltinNormAccumulator:
    """Streaming per-leaf accumulator for built-in norm constants."""

    def __init__(self, leaf: GenNormLeaf, sample_leaf: Any, outlier_filter: NormOutlierFilter | None):
        self.leaf = leaf
        self.callable_name = leaf.callable_name
        self.input_shape = tuple(np.asarray(sample_leaf).shape)
        self.axes = _normalize_norm_axes(leaf.axes, len(self.input_shape))
        self.outlier_filter = outlier_filter
        self.samples_seen = 0
        self.count: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None
        self.xmin: np.ndarray | None = None
        self.xmax: np.ndarray | None = None
        self.update(sample_leaf)

    def _check_shape(self, sample_leaf: Any) -> np.ndarray:
        arr = np.asarray(sample_leaf, dtype=np.float64)
        if tuple(arr.shape) != self.input_shape:
            raise ValueError(f"Normalization leaf shape changed from {self.input_shape} to {tuple(arr.shape)}.")
        return arr

    def _filter_array(self, arr: np.ndarray) -> np.ndarray:
        if (
            self.outlier_filter is None
            or self.samples_seen < self.outlier_filter.warmup_samples
            or self.mean is None
            or self.m2 is None
            or self.count is None
        ):
            return arr

        std = np.sqrt(np.divide(self.m2, self.count, out=np.zeros_like(self.m2), where=self.count > 0))
        lower = self.mean - self.outlier_filter.threshold * std
        upper = self.mean + self.outlier_filter.threshold * std
        return np.where((arr >= lower) & (arr <= upper), arr, np.nan)

    def _update_moments(self, arr: np.ndarray) -> None:
        count_b, mean_b, m2_b = _finite_group_stats(arr, self.axes)
        if self.count is None or self.mean is None or self.m2 is None:
            self.count, self.mean, self.m2 = count_b, mean_b, m2_b
            return
        self.count, self.mean, self.m2 = _combine_moments(self.count, self.mean, self.m2, count_b, mean_b, m2_b)

    def _update_minmax(self, arr: np.ndarray) -> None:
        if len(self.axes) == 0:
            xmin_b = arr
            xmax_b = arr
        else:
            if not np.isfinite(arr).any():
                return
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                xmin_b = np.nanmin(arr, axis=self.axes, keepdims=True)
                xmax_b = np.nanmax(arr, axis=self.axes, keepdims=True)
        if self.xmin is None or self.xmax is None:
            self.xmin, self.xmax = xmin_b, xmax_b
            return
        self.xmin = np.fmin(self.xmin, xmin_b)
        self.xmax = np.fmax(self.xmax, xmax_b)

    def update(self, sample_leaf: Any) -> None:
        """Update the accumulator with one sample leaf."""
        raw = self._check_shape(sample_leaf)
        filtered = self._filter_array(raw)
        if self.callable_name == "zscore":
            self._update_moments(filtered)
        elif self.callable_name == "minmax":
            self._update_minmax(filtered)
            self._update_moments(raw)
        else:
            raise ValueError(f"Built-in GenNorm does not know how to compute constants for {self.callable_name!r}.")
        self.samples_seen += 1

    def finalize(self) -> dict[str, Any]:
        """Return runtime norm config for the accumulated leaf."""
        payload: dict[str, Any] = {
            "callable": self.callable_name,
            "axes": self.axes,
            "input_shape": self.input_shape,
            **self.leaf.opts,
        }
        if self.callable_name == "zscore":
            assert self.count is not None and self.mean is not None and self.m2 is not None
            variance = np.divide(self.m2, self.count, out=np.zeros_like(self.m2), where=self.count > 0)
            payload.update({"mean": self.mean, "std": np.sqrt(variance)})
        elif self.callable_name == "minmax":
            assert self.xmin is not None and self.xmax is not None
            payload.update({"xmin": self.xmin, "xmax": self.xmax})
        return payload


class _CustomNormAccumulator:
    """Adapter for custom streaming normalization-stat callables."""

    def __init__(self, leaf: GenNormLeaf, sample_leaf: Any):
        if not callable(leaf.callable):
            raise TypeError("Custom GenNorm stats require a callable object.")
        stats = NormStatsCallable(callable=leaf.callable)
        input_shape = tuple(np.asarray(sample_leaf).shape)
        axes = _normalize_norm_axes(leaf.axes, len(input_shape))
        self.leaf = leaf
        self.stats = stats
        self.input_shape = input_shape
        self.axes = axes
        self.state = stats.init(sample_leaf, axes=axes, **leaf.opts)

    def update(self, sample_leaf: Any) -> None:
        arr = np.asarray(sample_leaf)
        if tuple(arr.shape) != self.input_shape:
            raise ValueError(f"Normalization leaf shape changed from {self.input_shape} to {tuple(arr.shape)}.")
        self.state = self.stats.update(self.state, sample_leaf)

    def finalize(self) -> dict[str, Any]:
        payload = {
            "callable": self.leaf.callable_name,
            "axes": self.axes,
            "input_shape": self.input_shape,
            **self.leaf.opts,
        }
        payload.update(self.stats.finalize(self.state))
        return payload


def _make_norm_accumulator(
    leaf: GenNormLeaf,
    sample_leaf: Any,
    outlier_filter: NormOutlierFilter | None,
) -> _BuiltinNormAccumulator | _CustomNormAccumulator:
    if isinstance(leaf.callable, str):
        if leaf.callable not in {"zscore", "minmax"}:
            NormOperator(callable=leaf.callable, **leaf.opts)
            raise ValueError(f"GenNorm cannot infer constants for registered norm {leaf.callable!r}.")
        return _BuiltinNormAccumulator(leaf, sample_leaf, outlier_filter)
    return _CustomNormAccumulator(leaf, sample_leaf)


def _iter_gennorm_configured_leaves(
    spec: Any,
    sample: PyTree,
    path: TreePath = (),
) -> Generator[tuple[TreePath, GenNormLeaf, Any], None, None]:
    """Yield configured sample leaves from a GenNorm tree."""
    if isinstance(spec, GenNormLeaf):
        for leaf_path, sample_leaf in pytree_path_iter(sample, is_leaf=lambda leaf: eqx.is_array_like(leaf)):
            if eqx.is_array_like(sample_leaf):
                yield path + leaf_path, spec, sample_leaf
        return
    if isinstance(spec, Mapping):
        if not isinstance(sample, Mapping):
            raise TypeError(f"GenNorm spec at {path!r} expects a mapping sample.")
        for key, child_spec in spec.items():
            if key not in sample:
                raise KeyError(f"GenNorm sample is missing configured path {path + (key,)!r}.")
            yield from _iter_gennorm_configured_leaves(child_spec, sample[key], path + (key,))
        return
    if isinstance(spec, list | tuple):
        if not isinstance(sample, Sequence) or isinstance(sample, str | bytes):
            raise TypeError(f"GenNorm spec at {path!r} expects a sequence sample.")
        for index, child_spec in enumerate(spec):
            if index >= len(sample):
                raise IndexError(f"GenNorm sample is missing configured index {path + (index,)!r}.")
            yield from _iter_gennorm_configured_leaves(child_spec, sample[index], path + (index,))


def _set_nested_mapping(tree: dict[str, Any], path: TreePath, value: Any) -> None:
    """Set a nested dictionary path."""
    cursor = tree
    for token in path[:-1]:
        cursor = cursor.setdefault(str(token), {})
    cursor[str(path[-1])] = value


def _write_norm_value(group: h5py.Group, key: str, value: Any) -> None:
    """Persist one normalization option as an HDF5 attribute or dataset."""
    if isinstance(value, tuple | list):
        group.attrs[key] = json.dumps(list(value), separators=(",", ":"))
        return
    if isinstance(value, str | int | float | bool) or value is None:
        group.attrs[key] = "null" if value is None else value
        return
    group.create_dataset(key, data=np.asarray(value))


def _write_norm_tree_group(group: h5py.Group, tree: Any) -> None:
    """Write a finalized norm tree to an HDF5 group."""
    if isinstance(tree, Mapping) and "callable" in tree:
        group.attrs["callable"] = tree["callable"]
        for key, value in tree.items():
            if key == "callable":
                continue
            _write_norm_value(group, key, value)
        return
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            _write_norm_tree_group(group.create_group(str(key), track_order=True), value)
        return
    raise TypeError("Finalized GenNorm tree must contain mapping containers and leaf norm configs.")


class GenNorm(GenDataConfig):
    """Compute and save normalization constants from loaded datasets."""

    show_progress: bool = True

    loader: DataLoader
    filename: str | Path = "norm.h5"
    norm: GenNormTree
    outlier_filter: NormOutlierFilter | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_loader(cls, value: Any) -> Any:
        if isinstance(value, str | Path | DataLoader):
            return {"loader": value}
        return value

    @model_validator(mode="after")
    def _set_max_epochs(self) -> "GenNorm":
        """Only load data once."""
        for _, ds_cfg in pytree_path_iter(self.loader.datasets, is_leaf=lambda leaf: isinstance(leaf, LoadDataConfig)):
            ds_cfg.max_epochs = 1
        return self

    @staticmethod
    def _iter_loaded_samples(loaded: PyTree) -> Generator[PyTree, None, None]:
        """Yield individual samples from either a stacked pytree or a plain sample list."""
        if isinstance(loaded, (list, tuple)):
            yield from loaded
            return
        yield from pytree_iter(loaded)

    def _iter_samples(self) -> Generator[tuple[str, PyTree], None, None]:
        ctxt = alive_bar(len(self.loader), title=self.filename) if self.show_progress else _NullProgress()

        with ctxt as bar:
            bar.text("Loading normalization samples...")
            for batch in self.loader:
                for dataset_name, loaded in batch.items():
                    for sample in self._iter_loaded_samples(loaded):
                        yield dataset_name, sample
                bar()

    def _finalize_tree(self, accumulators: Mapping[TreePath, Any]) -> dict[str, Any]:
        tree: dict[str, Any] = {}
        for path, accumulator in accumulators.items():
            _set_nested_mapping(tree, path, accumulator.finalize())
        return tree

    def _dataset_norm_spec(self, dataset_name: str) -> Any:
        """Return the sample-relative norm spec for one loaded dataset."""
        root = self.norm.root
        if isinstance(root, Mapping) and dataset_name in root:
            return root[dataset_name]
        return root

    def _artifact_path(self, root: Path, dataset_name: str) -> Path:
        """Return the dataset-specific norm artifact path."""
        filename = Path(self.filename)
        suffix = filename.suffix or ".h5"
        return root / f"{dataset_name}_{filename.stem}{suffix}"

    def generate(self, path, format=None, write_policy=None):
        """Compute normalization constants from loaded data and emit one HDF5 norm artifact per dataset."""
        format, write_policy = self._validate_format_and_policy(format, write_policy)
        if format != "h5":
            raise RoutineError(f"Save format '{format}' not recognized.")

        root_path = Path(path)
        dataset_names = [
            dataset_path[-1]
            for dataset_path, _ in pytree_path_iter(
                self.loader.datasets,
                is_leaf=lambda leaf: isinstance(leaf, LoadDataConfig),
            )
        ]
        artifact_paths = {
            dataset_name: self._artifact_path(root_path, str(dataset_name)) for dataset_name in dataset_names
        }
        existing_paths = [artifact_path for artifact_path in artifact_paths.values() if artifact_path.exists()]
        if write_policy == "error" and existing_paths:
            raise RoutineError(f"Normalization artifact already exists at {existing_paths[0]} and policy='error'")
        if write_policy == "reuse" and existing_paths and len(existing_paths) == len(artifact_paths):
            return

        accumulators: dict[str, dict[TreePath, _BuiltinNormAccumulator | _CustomNormAccumulator]] = {}
        sample_counts: dict[str, int] = {}
        for dataset_name, sample in self._iter_samples():
            artifact_path = self._artifact_path(root_path, dataset_name)
            if write_policy == "reuse" and artifact_path.exists():
                continue

            sample_counts[dataset_name] = sample_counts.get(dataset_name, 0) + 1
            dataset_accumulators = accumulators.setdefault(dataset_name, {})
            norm_spec = self._dataset_norm_spec(dataset_name)
            for leaf_path, leaf_spec, sample_leaf in _iter_gennorm_configured_leaves(norm_spec, sample):
                if leaf_path not in dataset_accumulators:
                    dataset_accumulators[leaf_path] = _make_norm_accumulator(
                        leaf_spec,
                        sample_leaf,
                        self.outlier_filter,
                    )
                else:
                    dataset_accumulators[leaf_path].update(sample_leaf)

        if not accumulators:
            raise ValueError("GenNorm did not find any configured array leaves to normalize.")

        root_path.mkdir(parents=True, exist_ok=True)
        for dataset_name, dataset_accumulators in accumulators.items():
            artifact_path = self._artifact_path(root_path, dataset_name)
            finalized = self._finalize_tree(dataset_accumulators)
            with h5py.File(artifact_path, "w", track_order=True) as h5:
                h5.attrs["romjax_type"] = "norm_tree"
                h5.attrs["version"] = 1
                h5.attrs["dataset"] = dataset_name
                _write_norm_tree_group(h5.create_group("tree", track_order=True), finalized)

            manifest = {
                "dataset": dataset_name,
                "sample_count": sample_counts.get(dataset_name, 0),
                "leaf_count": len(dataset_accumulators),
                "leaf_paths": [list(path) for path in dataset_accumulators],
                "outlier_filter": None if self.outlier_filter is None else self.outlier_filter.model_dump(),
            }
            artifact_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))


class GenLatent(GenDataConfig):
    """Fit a latent-space compressor and emit the compression artifact."""

    show_progress: bool = True

    loader: DataLoader
    dataset: str | None = None
    filename: str | Path = "compression.h5"
    gather_paths: Annotated[Sequence[TreePath], BeforeValidator(coerce_tree_paths)] = Field(default_factory=list)
    gather_template: Any | None = None
    norm: Any | None = None
    compression: Compression = Field(default_factory=lambda: SVD(energy_tol=0.999))
    name_depth: int = 2

    @staticmethod
    def _iter_loaded_samples(loaded: PyTree) -> Generator[PyTree, None, None]:
        """Yield individual samples from either a stacked pytree or a plain sample list."""
        if isinstance(loaded, (list, tuple)):
            yield from loaded
            return

        yield from pytree_iter(loaded)

    @model_validator(mode="before")
    @classmethod
    def _from_loader(cls, value):
        if isinstance(value, str | Path | DataLoader):
            return {"loader": value}
        return value

    @model_validator(mode="after")
    def _set_max_epochs(self):
        """Only load data once"""
        for _, ds_cfg in pytree_path_iter(self.loader.datasets, is_leaf=lambda leaf: isinstance(leaf, LoadDataConfig)):
            ds_cfg.max_epochs = 1
        return self

    @model_validator(mode="after")
    def _normalize_norm_config(self) -> "GenLatent":
        """Canonicalize the latent normalization config."""
        self.norm = self._coerce_norm_config(self.norm)
        return self

    def _selected_paths(self) -> list[TreePath]:
        paths = list(self.gather_paths)
        if self.gather_template is not None:

            def _collect(template: PyTree, prefix: TreePath = ()) -> list[TreePath]:
                if isinstance(template, Mapping):
                    paths: list[TreePath] = []
                    for key, value in template.items():
                        paths.extend(_collect(value, prefix + (key,)))
                    return paths
                if isinstance(template, tuple):
                    paths: list[TreePath] = []
                    for index, value in enumerate(template):
                        paths.extend(_collect(value, prefix + (index,)))
                    return paths
                if isinstance(template, list):
                    paths: list[TreePath] = []
                    for index, value in enumerate(template):
                        paths.extend(_collect(value, prefix + (index,)))
                    return paths
                if template is None or template is False:
                    return []
                return [prefix]

            paths.extend(_collect(self.gather_template))
        return paths

    def _merge_selected_sample(self, sample: PyTree) -> PyTree:
        selected_paths = self._selected_paths()
        if not selected_paths:
            return sample

        merged: PyTree | None = None
        for path in selected_paths:
            merged = set_subtree(merged, path, get_subtree(sample, path))
        return merged if merged is not None else sample

    def _coerce_norm_config(self, value: Any) -> NormTree | Mapping[str, NormTree] | None:
        """Return a canonical latent norm config.

        A single :class:`NormTree` is used for every dataset. A mapping is treated as dataset-specific only when
        every value is already a :class:`NormTree`, keeping bare tree specs compatible with the single-tree form.
        """
        if value is None or isinstance(value, NormTree):
            return value
        if isinstance(value, Mapping) and value and all(isinstance(item, NormTree) for item in value.values()):
            return dict(value)
        return NormTree.model_validate(value)

    def _dataset_norm(self, dataset_name: str) -> NormTree | None:
        """Return the normalization tree for one dataset, if configured."""
        if self.norm is None:
            return None
        if isinstance(self.norm, NormTree):
            return self.norm
        if dataset_name not in self.norm:
            raise ValueError(f"No latent normalization configured for dataset {dataset_name!r}.")
        return self.norm[dataset_name]

    def _apply_dataset_norm(self, sample: PyTree, dataset_name: str, norm: NormTree) -> PyTree:
        """Apply a selected dataset's normalization to one unwrapped sample."""
        del dataset_name
        norm.resolve_root()
        return norm(sample)

    def bar_text(self, path: Path):
        return "/".join(path.parts[-self.name_depth :])

    def _iter_samples(self, path: Path | None = None) -> Generator[PyTree, None, None]:
        if path is None:
            path = Path(self.filename)

        ctxt = alive_bar(len(self.loader), title=self.bar_text(path)) if self.show_progress else _NullProgress()

        with ctxt as bar:
            bar.text("Loading compression samples...")
            selected_name = self.dataset
            for batch in self.loader:
                names = tuple(batch)
                if selected_name is None:
                    if len(names) != 1:
                        raise ValueError("GenLatent.dataset is required when the loader yields multiple datasets.")
                    selected_name = names[0]
                if selected_name not in batch:
                    raise ValueError(f"GenLatent dataset {selected_name!r} is absent from a loader batch.")
                dataset_norm = self._dataset_norm(selected_name)
                if dataset_norm is not None:
                    dataset_norm.resolve_root()
                for sample in self._iter_loaded_samples(batch[selected_name]):
                    selected = self._merge_selected_sample(sample)
                    if dataset_norm is not None:
                        selected = self._apply_dataset_norm(selected, selected_name, dataset_norm)
                    yield selected
                bar()

    def generate(self, path, format=None, write_policy=None):
        """Fit latent coordinates from loaded data and emit the compression artifact."""
        _, write_policy = self._validate_format_and_policy(format, write_policy)
        artifact_path = Path(path) / self.filename

        if write_policy == "error" and artifact_path.exists():
            raise RoutineError(f"Latent compression artifact already exists at {artifact_path} and policy='error'")
        if write_policy == "reuse" and artifact_path.exists():
            return

        samples = list(self._iter_samples(artifact_path))

        if hasattr(self.compression, "show_progress"):
            object.__setattr__(self.compression, "show_progress", self.show_progress)
        compression = self.compression.fit(samples)
        logger.debug(f"Compression finished. Latent space: {compression.latent_size()}")

        compression.dump(artifact_path)

        if hasattr(compression, "save_orbax"):
            compression.save_orbax(path)

        latent_bounds = compression.latent_bounds()
        latent_normal = compression.latent_normal()
        manifest_path = artifact_path.with_suffix(".manifest.json")
        manifest = {
            "latent_size": compression.latent_size(),
            "latent_bounds": None
            if latent_bounds is None
            else [np.asarray(bounds).tolist() for bounds in latent_bounds],
            "latent_normal": None if latent_normal is None else [np.asarray(stats).tolist() for stats in latent_normal],
        }
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))


def _validate_gendata_pytree(template: PyTree) -> PyTree[GenDataConfig]:
    """Validate every leaf in a pytree-like template as a :class:`GenDataConfig`. Leave anything else untouched."""
    if isinstance(template, Mapping):
        if template.get("kind") == "benchmark":
            return GenDataBenchmark(**{key: value for key, value in template.items() if key != "kind"})
        if all(field in template for field in required_fields(GenSymlink)):
            return GenSymlink(**template)
        latent_fields = {"compression", "gather_paths", "gather_template"}
        filename = template.get("filename")
        latent_filename = isinstance(filename, str | Path) and Path(filename).suffix == ".h5"
        if (any(field in template for field in latent_fields) or latent_filename) and all(
            field in template for field in required_fields(GenLatent)
        ):
            return GenLatent(**template)
        if all(field in template for field in required_fields(GenNorm)):
            return GenNorm(**template)
        if all(field in template for field in required_fields(GenLatent)):
            return GenLatent(**template)
        if all(field in template for field in required_fields(GenImplicitModel)):
            return GenImplicitModel(**template)
        if all(field in template for field in required_fields(GenSource)):
            return GenSource(**template)
        return {key: _validate_gendata_pytree(value) for key, value in template.items()}
    if isinstance(template, tuple):
        return tuple(_validate_gendata_pytree(value) for value in template)
    if isinstance(template, list):
        return [_validate_gendata_pytree(value) for value in template]

    return template


def _validate_loaddata_pytree(template: PyTree) -> PyTree[LoadDataConfig]:
    """Validate every leaf in a pytree-like template as a :class:`LoadDataConfig`. Leave anything else untouched."""
    if isinstance(template, Mapping):
        if len(template) == 0:
            return {}

        implicit_fields = set(LoadImplicitModel.model_fields)
        source_fields = set(LoadSource.model_fields)
        template_fields = set(template)

        if (kind := template.pop("kind", None)) is not None:
            if kind == "implicit":
                return LoadImplicitModel(**template)
            elif kind == "source":
                return LoadSource(**template)
            else:
                raise ValueError(f"Load config '{kind}' not recognized. Supported: ['implicit', 'source']")

        if template_fields <= implicit_fields:
            return LoadImplicitModel(**template)
        if template_fields <= source_fields:
            return LoadSource(**template)
        return {key: _validate_loaddata_pytree(value) for key, value in template.items()}
    if isinstance(template, tuple):
        return tuple(_validate_loaddata_pytree(value) for value in template)
    if isinstance(template, list):
        return [_validate_loaddata_pytree(value) for value in template]

    return template


type GenDataPyTree = Annotated[PyTree, BeforeValidator(_validate_gendata_pytree)]
type LoadDataPyTree = Annotated[PyTree, BeforeValidator(_validate_loaddata_pytree)]


class DataGeneration(Routine):
    """
    File-based data generation routine.

    :ivar root: root directory for saving data
    :ivar executor: batch execution backend for implicit and source generators (default: serial)
    :ivar datasets: pytree template for datasets to generate under root
    :ivar base: optional YAML configuration template or YAML path to expand
    :ivar overrides: named Cartesian-product configuration overrides
    :ivar format: the data format to save samples. Only `h5` supported.
    :ivar write_policy: reuse existing data, overwrite existing data, or throw an error if existing data found
    :ivar graph: graph object or YAML path (optional, for graph-related datasets)
    """

    root: Path
    storage: H5StorageConfig = Field(default_factory=H5StorageConfig)
    executor: Annotated[ExecutorConfig, BeforeValidator(ExecutorConfig.from_dict)] = Field(
        default_factory=ExecutorConfig
    )
    datasets: GenDataPyTree | None = None
    base: Any | None = None
    overrides: tuple[CompositeOverride, ...] | None = None

    format: SUPPORTED_FORMATS = "h5"
    write_policy: SUPPORTED_POLICIES = "reuse"

    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None
    _source_path: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _bind_graph(self):
        import romjax

        object.__setattr__(self, "_source_path", romjax.YamlLoader.current_source_path())
        # Pass graph object to graph datasets
        if self.graph is not None and self.datasets is not None:
            self._bind_graph_to_datasets(self.datasets)

        has_expansion = self.base is not None or self.overrides is not None
        if self.datasets is not None and has_expansion:
            raise ValueError("DataGeneration cannot specify both 'datasets' and base/overrides expansion.")
        if self.datasets is None and not self.overrides:
            raise ValueError("DataGeneration requires either 'datasets' or a non-empty 'overrides' list.")
        if self.overrides is not None and len({item.name for item in self.overrides}) != len(self.overrides):
            raise ValueError("DataGeneration override names must be unique.")

        return self

    def _bind_graph_to_datasets(self, datasets: PyTree) -> None:
        """Pass the configured graph to graph-based dataset leaves."""
        leaves, _ = jax.tree.flatten(datasets, is_leaf=lambda leaf: isinstance(leaf, GenDataConfig))
        for ds in leaves:
            if isinstance(ds, GenDataBenchmark) and ds.generator.graph is None:
                ds.generator.graph = self.graph
            if hasattr(ds, "graph") and ds.graph is None:
                ds.graph = self.graph

    def _base_node(self) -> tuple[Any, Path | None]:
        """Return the raw YAML node and source path for expansion.

        :return: base node, or ``None`` for an empty base, and its source path
        """
        import romjax

        if self.base is None:
            return None, self._source_path
        if isinstance(self.base, romjax.YamlSource):
            return self.base.node, self.base.source_path
        if isinstance(self.base, str | Path):
            path = romjax.YamlLoader._resolve_override_path(str(self.base), self._source_path)
            return romjax.YamlLoader._compose_resolved_node(path)
        node = yaml.compose(
            romjax.YamlLoader.dump(self.base, sort_keys=False, _preserve_yaml_sources=True),
            Loader=yaml.SafeLoader,
        )
        return node, self._source_path

    def _expanded_datasets(self) -> Iterator[tuple[Path, PyTree]]:
        """Yield each merged configuration and its Cartesian-product output root.

        :return: output root and validated dataset PyTree for each override combination
        """
        import romjax

        base_node, base_path = self._base_node()
        assert self.overrides is not None
        for selected in product(*(item.cases for item in self.overrides)):
            parts = tuple(f"{override.name}={case.name}" for override, case in zip(self.overrides, selected))
            merged = romjax.YamlLoader._resolve_parent_refs(base_node, base_path)
            for case in selected:
                if case.value is None:
                    continue
                override_node = yaml.compose(
                    romjax.YamlLoader.dump(case.value, sort_keys=False, _preserve_yaml_sources=True),
                    Loader=yaml.SafeLoader,
                )
                if override_node is not None:
                    override_node = romjax.YamlLoader._resolve_parent_refs(override_node, self._source_path)
                    merged = romjax.YamlLoader._merge_nodes(merged, override_node)

            datasets = _validate_gendata_pytree(romjax.load(romjax.YamlSource(merged, base_path)))
            self._bind_graph_to_datasets(datasets)
            if not any(
                isinstance(leaf, GenDataConfig)
                for leaf in jax.tree.leaves(datasets, is_leaf=lambda leaf: isinstance(leaf, GenDataConfig))
            ):
                raise ValueError("Expanded data-generation configuration does not contain a supported data generator.")
            yield self.root.joinpath(*parts), datasets

    def _generate_dataset(self, dataset: GenDataConfig, path: Path) -> None:
        if isinstance(dataset, GenImplicitModel | GenSource | GenDataBenchmark):
            format, policy = dataset._validate_format_and_policy(self.format, self.write_policy)
            generator = dataset.generator if isinstance(dataset, GenDataBenchmark) else dataset
            dataset._generate(
                path,
                format,
                policy,
                generator.executor or self.executor,
                self._jax_child_env(),
                **({} if isinstance(dataset, GenDataBenchmark) else {"storage": self.storage}),
            )
        else:
            dataset.generate(path, format=self.format, write_policy=self.write_policy)

    def _run_expanded_cases(self) -> None:
        """Generate every Cartesian-product data-generation case."""
        for case_root, datasets in self._expanded_datasets():
            for path, dataset in pytree_path_iter(datasets, is_leaf=lambda leaf: isinstance(leaf, GenDataConfig)):
                self._generate_dataset(dataset, case_root / "/".join(path))

    def run(self) -> int:
        """Generate all datasets."""
        if self.overrides:
            self._run_expanded_cases()
            return 0

        assert self.datasets is not None
        for path, dataset in pytree_path_iter(self.datasets, is_leaf=lambda leaf: isinstance(leaf, GenDataConfig)):
            self._generate_dataset(dataset, self.root / "/".join(path))

        return 0


class DataLoader(BaseModel):
    """
    File-backed mini-batch loader for datasets created by :class:`romjax.data_gen.DataGeneration`.

    The yielded batch payload is a mapping keyed by sampled dataset names, e.g. for implicit models:
    ``{dataset_name: {"inputs": batch, "outputs": batch, "residuals": batch}}``.

    If `datasets` is a PyTree, then data will be loaded from the corresponding path under `root`. The name of the last
    directory in the path will be used as the `dataset_name` in the returned batch.

    :param root: root directory for loading datasets
    :param datasets: configs for loading independent datasets under root, defaults to all top-level directories
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    root: Path
    datasets: LoadDataPyTree = Field(default_factory=dict)

    _iterator: Iterator[dict[str, PyTree]] | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _from_root(cls, value):
        """Validate a loader with all default options from just a plain root directory."""
        if isinstance(value, str | Path):
            return {"root": value}
        return value

    @model_validator(mode="after")
    def _infer_datasets(self):
        """Try to infer datasets from top-level directories. Can only infer ImplicitModel and Source."""
        if len(self.datasets) == 0:
            dataset_dirs = sorted(d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith("."))

            def _infer_dataset_dir(ds_dir: Path) -> LoadDataConfig | None:
                store = H5Store(ds_dir)
                store.discover()
                if store.kind == "implicit":
                    return LoadImplicitModel()
                if store.kind == "source":
                    return LoadSource()
                return None

            datasets = {}
            for ds_dir in dataset_dirs:
                if (load_cfg := _infer_dataset_dir(ds_dir)) is not None:
                    datasets[ds_dir.name] = load_cfg

            if datasets:
                self.datasets = datasets

        return self

    def _iter_datasets(
        self,
        train_step: int = 0,
        refs_only: bool = False,
        max_epochs: int | None = None,
    ) -> Generator[dict[str, PyTree], None, None]:
        """
        Walk over all available datasets at once, using available configurations to limit sample sizes.
        Yields a mapping of dataset names to PyTree batches of data.

        :param train_step: initialize from this training step (default: 0)
        :param refs_only: only return the sample paths (don't load from disk)
        :param max_epochs: override local dataset settings for max_epochs
        """
        ds_config = {}
        ds_paths = {}
        ds_selected_refs = {}
        ds_totals = {}

        for path, ds_cfg in pytree_path_iter(self.datasets, is_leaf=lambda leaf: isinstance(leaf, LoadDataConfig)):
            ds_root = self.root / "/".join(path)
            ds_name = ds_root.name

            ds_paths[ds_name] = ds_cfg.discover_sample_refs(ds_root)
            if len(ds_paths[ds_name]) == 0:
                raise ValueError(f"No samples found for dataset '{ds_name}' in {ds_root}")

            initial_indices = np.random.default_rng(np.random.SeedSequence([ds_cfg.shuffle_seed, 0])).permutation(
                len(ds_paths[ds_name])
            )
            ds_selected_refs[ds_name] = ds_cfg.select_epoch_refs(ds_paths[ds_name], initial_indices)
            if len(ds_selected_refs[ds_name]) == 0:
                raise ValueError(f"No samples selected for dataset '{ds_name}' in {ds_root}")

            ds_totals[ds_name] = len(ds_selected_refs[ds_name])
            ds_config[ds_name] = ds_cfg

        # Advance dataset epoch and cursor indices based on current training step
        # Epoch - number of iterations through an entire dataset
        # Cursor - starting index within a dataset for the next mini-batch

        ds_epochs = {name: 0 for name in ds_config}
        ds_cursors = {name: 0 for name in ds_config}
        ds_active = {name: True for name in ds_config}

        def _shuffle_indices(name: str, epoch: int) -> np.ndarray:
            """Build a deterministic per-epoch permutation without invoking JAX random ops."""
            seed = np.random.SeedSequence([ds_config[name].shuffle_seed, epoch])
            return np.random.default_rng(seed).permutation(ds_totals[name])

        ds_indices = {name: _shuffle_indices(name, ds_epochs[name]) for name in ds_config}

        def _refresh_epoch(name: str) -> None:
            """Advance a dataset to the next epoch and reshuffle the selected sample pool."""
            ds_epochs[name] += 1
            _max_epochs = max_epochs or ds_config[name].max_epochs
            if _max_epochs is not None and ds_epochs[name] >= _max_epochs:
                ds_active[name] = False
                ds_cursors[name] = len(ds_selected_refs[name])
                return

            ds_indices[name] = _shuffle_indices(name, ds_epochs[name])
            ds_cursors[name] = 0

        def _advance_dataset(name: str) -> None:
            """Advance one dataset by a single batch and update its termination state."""
            if not ds_active[name]:
                return

            ds_cursors[name] += ds_config[name].batch_size
            if ds_cursors[name] >= len(ds_selected_refs[name]):
                _refresh_epoch(name)

        for _ in range(train_step):
            for name in ds_config:
                _advance_dataset(name)

            if not any(ds_active.values()):
                return

        while True:
            active_names = [name for name in ds_paths if ds_active[name]]
            if len(active_names) == 0:
                return

            ds_batch = {}

            for name in active_names:
                epoch_refs = ds_selected_refs[name]
                batch_size = ds_config[name].batch_size
                batch_paths = [
                    epoch_refs[int(index)]
                    for index in ds_indices[name][ds_cursors[name] : ds_cursors[name] + batch_size]
                ]
                ds_batch[name] = batch_paths if refs_only else ds_config[name].load_batch(batch_paths)
                ds_cursors[name] += len(batch_paths)

                if ds_cursors[name] >= len(epoch_refs):
                    _refresh_epoch(name)

            yield ds_batch

    def set_iterator(self, train_step: int = 0):
        """Set the iterator based on the current train step."""
        self._iterator = self._iter_datasets(train_step)

    def __next__(self):
        """Continue existing iterator if possible, otherwise start from beginning."""
        if self._iterator is None:
            self.set_iterator()
        return next(self._iterator)

    def __iter__(self):
        """Restart iteration from 0."""
        self.set_iterator()
        return self

    def __len__(self):
        """Return the number of mini-batches yielded in one epoch.

        This avoids materializing the iterator so callers like ``list(loader)`` do not trigger an
        extra dataset-selection pass via ``__len__`` preallocation.
        """
        batch_counts: list[int] = []

        for path, ds_cfg in pytree_path_iter(self.datasets, is_leaf=lambda leaf: isinstance(leaf, LoadDataConfig)):
            ds_root = self.root / "/".join(path)
            refs = ds_cfg.discover_sample_refs(ds_root)
            if len(refs) == 0:
                batch_counts.append(0)
                continue

            initial_indices = np.random.default_rng(np.random.SeedSequence([ds_cfg.shuffle_seed, 0])).permutation(
                len(refs)
            )
            selected_count = ds_cfg.count_epoch_refs(refs, initial_indices)
            if selected_count == 0:
                batch_counts.append(0)
                continue

            batch_counts.append((selected_count + ds_cfg.batch_size - 1) // ds_cfg.batch_size)

        return max(batch_counts, default=0)
