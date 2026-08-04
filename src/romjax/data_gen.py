"""Data generation routine and data loading."""
from __future__ import annotations

import inspect
import json
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterator, Literal, Mapping, Sequence, get_args

import equinox as eqx
import h5py
import jax
import numpy as np
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
from romjax.graph import Edge, FunctionGraph
from romjax.model import ImplicitSampleable
from romjax.norm import NormOperator, NormTree
from romjax.rng import gen_keys
from romjax.routine import Routine, RoutineError
from romjax.tree import (
    TreePath,
    coerce_tree_path,
    coerce_tree_paths,
    get_subtree,
    pytree_iter,
    pytree_merge,
    pytree_path_iter,
    pytree_stack,
    set_subtree,
)
from romjax.typing import CallableModel, DictModel, from_yaml
from romjax.utils import _NullProgress, load_h5, required_fields, save_h5

__all__ = [
    "DataGeneration",
    "DataLoader",
    "GenDataConfig",
    "LoadDataConfig",
]

_BAR_TITLE_LEN = 20
_FAILURE_MARKER = ".romjax_failed"


def _get_kwargs(fn: Callable) -> set[str]:
    signature = inspect.signature(fn)
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _accepts_kwarg(fn: Callable, name: str) -> bool:
    """Return whether a callable accepts a named keyword argument."""
    parameters = inspect.signature(fn).parameters.values()
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


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


def _build_batched_sample_outputs(model):
    """Build one batched ``sample_outputs`` function for a model."""
    has_conditions = _has_custom_sample_conditions(model)

    def _sample_outputs(keys, one_input, one_solution, one_conditions=None):
        if has_conditions:
            return jax.vmap(
                lambda key, conditions: _call_sample_outputs(
                    model, key, one_input, one_solution, conditions
                )
            )(keys, one_conditions)
        return jax.vmap(
            lambda key: _call_sample_outputs(model, key, one_input, one_solution, None)
        )(keys)

    return eqx.filter_jit(_sample_outputs)


def _is_failed_sample_dir(sample_dir: Path) -> bool:
    """Return whether a sample directory has been marked as failed."""
    return (sample_dir / _FAILURE_MARKER).exists()


def _log_sample_failure(sample_dir: Path, message: str, exc: BaseException) -> None:
    """Persist a detailed failure record for one sample directory."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / _FAILURE_MARKER).touch(exist_ok=True)

    failure_log = sample_dir / "failure.log"
    sink_id = logger.add(
        failure_log,
        level="DEBUG",
        mode="a",
        backtrace=True,
        diagnose=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}\n{exception}",
    )
    try:
        logger.opt(exception=exc).error(message)
    finally:
        logger.remove(sink_id)


type SUPPORTED_FORMATS = Literal["h5"]
type SUPPORTED_POLICIES = Literal["reuse", "overwrite", "error"]


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
        self, 
        path: Path, 
        format: SUPPORTED_FORMATS | None = None,
        write_policy: SUPPORTED_POLICIES | None = None
    ) -> None:
        """Generate data at the provided path using the specified format and write_policy."""
        raise NotImplementedError


class GenGraph(GenDataConfig, ABC):
    """
    Datasets that use a FunctionGraph for sample generation.

    Concrete classes must specify how to generate both in serial and in batch.
    
    :ivar graph: the FunctionGraph or Yaml path
    :ivar name_depth: depth in the path for displaying current dataset name (default: 2)
    :ivar batch_size: batch size
    :ivar _required_methods: names of the sampling methods that corresponding graph edges must implement.
    """

    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None
    name_depth: int = 2
    batch_size: PositiveInt = 1
    _required_methods: list[str] = PrivateAttr(default_factory=list)

    @abstractmethod
    def generate_serial(self, path, format, write_policy):
        """Generate data in serial (batch=1)."""
        raise NotImplementedError
    
    @abstractmethod
    def generate_batch(self, path, format, write_policy):
        """Genereate data in batches (batch>1)."""
        raise NotImplementedError

    def bar_text(self, path: Path):
        return "/".join(path.parts[-self.name_depth:])
    
    def _edge_from_path(self, path: Path) -> Edge:
        edge_name = path.name

        if edge_name not in self.graph.edges:
            raise ValueError(f"Last folder name must be an edge name in the graph. '{edge_name}' not recognized.")
        
        return self.graph.edges[edge_name]

    def _validate_required_methods(self, path: Path):
        """Make sure all required methods are implemented."""
        edge = self._edge_from_path(Path(path))

        for meth in self._required_methods:
            if not hasattr(edge, meth):
                raise ValueError(f"Graph edge '{edge}' must implement '{meth}' method.")

    def generate(self, path, format=None, write_policy=None):
        """
        Generate data for a FunctionGraph. Assumes graph edge name is the last name in the `path`.
        """
        format, write_policy = self._validate_format_and_policy(format, write_policy)

        if self.graph is None:
            raise ValueError("Must specify a graph to generate data.")
        
        self._validate_required_methods(path)
        
        if self.batch_size > 1:
            self.generate_batch(path, format, write_policy)
        else:
            self.generate_serial(path, format, write_policy)


class LoadDataConfig[T: Any](BaseModel, ABC):
    """
    Abstract class for batch-loading datasets.

    :param batch_size: number of output samples per yielded mini-batch
    :param shuffle_seed: seed for shuffling mini-batch data
    :param max_epochs: maximum times to iterate through all available data (defaults to infinite loop)
    :param cache_policy: file-cache policy for loaded samples
    :param cache_storage: storage location for cached samples
    :param cache_max_items: maximum number of cached samples, if any
    :param cache_max_bytes: maximum total cached sample size, if any
    :param stack_batch: stack samples into a batched pytree before returning them
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_size: PositiveInt = 16
    shuffle_seed: int = 0
    max_epochs: PositiveInt | None = None
    cache_policy: Literal["none", "lru", "epoch"] = "none"
    cache_storage: Literal["host", "device"] = "host"
    cache_max_items: NonNegativeInt | None = None
    cache_max_bytes: NonNegativeInt | None = None
    stack_batch: bool = False

    _cache_entries: OrderedDict[Any, tuple[PyTree, int]] = PrivateAttr(default_factory=OrderedDict)
    _cache_bytes: int = PrivateAttr(default=0)
    _epoch_cache_items: int | None = PrivateAttr(default=None)

    @abstractmethod
    def discover_sample_refs(self, root: Path) -> list[T]:
        """Discover sample references below one dataset root."""
        raise NotImplementedError

    @abstractmethod
    def load_sample(self, ref: T) -> PyTree:
        """Load one sample referenced from disk."""
        raise NotImplementedError

    def configure_epoch_cache(self, epoch_size: int | None) -> None:
        """Set the selected epoch size used by ``cache_policy='epoch'``."""
        self._epoch_cache_items = epoch_size
        if self.cache_policy == "epoch":
            self._enforce_cache_limits()

    def select_epoch_refs(self, refs: Sequence[T], indices: jax.Array) -> list[T]:
        """Select the full epoch reference order from a shuffled dataset index order."""
        return [refs[int(index)] for index in indices]

    def count_epoch_refs(self, refs: Sequence[T], indices: jax.Array) -> int:
        """Count the number of references selected for one epoch without materializing them."""
        return len(indices)

    def load_batch(self, refs: Sequence[T]) -> PyTree | list[PyTree]:
        """Load a selected batch of dataset references."""
        samples = [self._load_cached_sample(ref) for ref in refs]
        if len(samples) == 0:
            return {}
        if self.stack_batch:
            return pytree_stack(samples)
        return samples

    def _cache_key(self, ref: T) -> Any:
        """Return a hashable cache key for one sample reference."""
        def _normalize(value: Any) -> Any:
            # if isinstance(value, Path):
            #     return value.resolve()
            if isinstance(value, tuple):
                return tuple(_normalize(item) for item in value)
            if isinstance(value, list):
                return tuple(_normalize(item) for item in value)
            return value

        return _normalize(ref)

    def _cache_limit_items(self) -> int | None:
        """Return the effective item cap for the current cache policy."""
        if self.cache_policy != "epoch":
            return self.cache_max_items
        if self._epoch_cache_items is None:
            return self.cache_max_items
        if self.cache_max_items is None:
            return self._epoch_cache_items
        return max(self.cache_max_items, self._epoch_cache_items)

    def _cache_limit_bytes(self) -> int | None:
        """Return the effective byte cap for the current cache policy."""
        return self.cache_max_bytes

    @staticmethod
    def _sample_nbytes(sample: PyTree) -> int:
        """Estimate the in-memory size of one cached sample in bytes."""
        total = 0
        for leaf in jax.tree.leaves(sample):
            if leaf is None:
                continue
            if hasattr(leaf, "nbytes"):
                total += int(leaf.nbytes)
                continue
            if hasattr(leaf, "size") and hasattr(leaf, "dtype"):
                total += int(leaf.size * leaf.dtype.itemsize)
                continue
            total += int(np.asarray(leaf).nbytes)
        return total

    def _materialize_for_cache(self, sample: PyTree) -> PyTree:
        """Convert one loaded sample to the configured cache storage."""
        if self.cache_storage == "device":
            return jax.device_put(sample)
        return sample

    def _load_cached_sample(self, ref: T) -> PyTree:
        """Load a sample through the in-memory cache when enabled."""
        if self.cache_policy == "none":
            return self.load_sample(ref)

        # key = self._cache_key(ref)
        key = ref
        if key in self._cache_entries:
            cached_sample, cached_size = self._cache_entries.pop(key)
            self._cache_entries[key] = (cached_sample, cached_size)
            return cached_sample

        sample = self._materialize_for_cache(self.load_sample(ref))
        sample_size = self._sample_nbytes(sample)
        byte_limit = self._cache_limit_bytes()
        item_limit = self._cache_limit_items()
        if item_limit == 0:
            return sample
        if byte_limit is not None and sample_size > byte_limit:
            return sample

        self._evict_for_insert(sample_size)
        self._cache_entries[key] = (sample, sample_size)
        self._cache_bytes += sample_size
        return sample

    def _evict_for_insert(self, sample_size: int) -> None:
        """Evict least-recently-used samples until the next insert fits."""
        item_limit = self._cache_limit_items()
        byte_limit = self._cache_limit_bytes()

        while self._cache_entries:
            over_items = item_limit is not None and len(self._cache_entries) >= item_limit
            over_bytes = byte_limit is not None and self._cache_bytes + sample_size > byte_limit
            if not over_items and not over_bytes:
                break
            _, (_, evicted_size) = self._cache_entries.popitem(last=False)
            self._cache_bytes -= evicted_size

    def _enforce_cache_limits(self) -> None:
        """Prune the cache to the current effective policy limits."""
        item_limit = self._cache_limit_items()
        byte_limit = self._cache_limit_bytes()

        while self._cache_entries and item_limit is not None and len(self._cache_entries) > item_limit:
            _, (_, evicted_size) = self._cache_entries.popitem(last=False)
            self._cache_bytes -= evicted_size

        if byte_limit is None:
            return

        while self._cache_entries and self._cache_bytes > byte_limit:
            _, (_, evicted_size) = self._cache_entries.popitem(last=False)
            self._cache_bytes -= evicted_size


class GenImplicitModel(GenGraph):
    """
    Sampling configuration for a nested input/output strategy for implicit models in a FunctionGraph.

    A nested folder structure will be generated with the outer seed_i/sample_j set corresponding to the
    result of `sample_inputs`, and the inner seed/sample set corresponding to the result of `sample_outputs`.
    Generally, output samples are conditioned on input samples, and so several output samples may be requested
    for a single input sample. When implemented, `sample_conditions` is called for each inner sample and its
    result is saved as `conditions.h5` alongside the output.

    If an Edge implements `solve` and `evaluate`, a solution will be generated and saved alongside each input, 
    and a residual will be computed and saved alongside each output.

    :ivar input_samples: number of input samples
    :ivar outputs_per_input: number of output samples for each input sample
    :ivar input_seed: random seed for inputs
    :ivar output_seed: random seed for outputs
    :ivar throw: whether solve/evaluate failures should propagate immediately
    """

    input_samples: int
    outputs_per_input: int
    input_seed: int
    output_seed: int
    throw: bool = True
    _required_methods: list[str] = PrivateAttr(default=["sample_inputs"])

    def generate_serial(self, path, format, write_policy):
        """Generate data in serial for an ImplicitModel."""
        path = Path(path)
        model = self._edge_from_path(path)
        failed_dirs: set[Path] = set()
        failed_cases = 0

        def _record_failure(sample_dir: Path, message: str, exc: BaseException) -> None:
            nonlocal failed_cases
            if sample_dir not in failed_dirs:
                failed_dirs.add(sample_dir)
                failed_cases += 1
            _log_sample_failure(sample_dir, message, exc)

        with alive_bar(self.input_samples, title=self.bar_text(path), title_length=_BAR_TITLE_LEN) as bar:
            if write_policy == "error" and path.exists():
                raise RoutineError(f"Dataset already exists at {path} and policy='error'")

            path.mkdir(parents=True, exist_ok=True)

            sample_inputs = eqx.filter_jit(model.sample_inputs)
            solve = eqx.filter_jit(model.solve) if hasattr(model, "solve") else None
            evaluate = eqx.filter_jit(model.evaluate) if hasattr(model, "evaluate") else None
            sample_conditions = (
                eqx.filter_jit(model.sample_conditions)
                if _has_custom_sample_conditions(model)
                else None
            )
            sample_outputs = eqx.filter_jit(
                lambda key, one_input, one_solution, one_conditions=None: _call_sample_outputs(
                    model, key, one_input, one_solution, one_conditions
                )
            )

            for input_key, input_dir in gen_keys(self.input_samples, self.input_seed, path=path):
                input_path = input_dir / "input.h5"
                solution_path = input_dir / "solution.h5"
                solution_residual_path = input_dir / "solution_residual.h5"

                if write_policy == "reuse" and input_path.exists():
                    if format != "h5":
                        raise RoutineError(f"Save format '{format}' not recognized")
                    one_input = load_h5({}, input_path, jax=True)
                    one_solution = load_h5({}, solution_path, jax=True) if solution_path.exists() else None
                else:
                    one_input = sample_inputs(input_key)
                    one_solution = None
                    if solve is not None:
                        if self.throw:
                            one_solution = solve(one_input)
                        else:
                            try:
                                one_solution = solve(one_input)
                            except Exception as exc:  # noqa: BLE001
                                _record_failure(input_dir, f"Failed to solve sample at {input_dir}", exc)
                    one_solution_residual = None
                    if evaluate is not None and one_solution is not None:
                        if self.throw:
                            one_solution_residual = evaluate(one_input, one_solution)
                        else:
                            try:
                                one_solution_residual = evaluate(one_input, one_solution)
                            except Exception as exc:  # noqa: BLE001
                                _record_failure(
                                    input_dir,
                                    f"Failed to evaluate solution sample at {input_dir}",
                                    exc,
                                )

                    if format == "h5":
                        save_h5(one_input, input_path, mode="w")
                        if one_solution is not None:
                            save_h5(one_solution, solution_path, mode="w")
                        if one_solution_residual is not None:
                            save_h5(one_solution_residual, solution_residual_path, mode="w")
                    else:
                        raise RoutineError(f"Save format '{format}' not recognized")

                skip = "existing" if write_policy == "reuse" else None
                for output_key, output_dir in gen_keys(
                    self.outputs_per_input, self.output_seed, path=input_dir, skip=skip
                ):
                    one_conditions = sample_conditions(output_key) if sample_conditions is not None else None
                    if one_conditions is not None:
                        save_h5(one_conditions, output_dir / "conditions.h5", mode="w")

                    one_output = sample_outputs(output_key, one_input, one_solution, one_conditions)
                    one_residual = None
                    if evaluate is not None:
                        if self.throw:
                            one_residual = evaluate(one_input, one_output)
                        else:
                            try:
                                one_residual = evaluate(one_input, one_output)
                            except Exception as exc:  # noqa: BLE001
                                _record_failure(
                                    output_dir,
                                    f"Failed to evaluate output sample at {output_dir}",
                                    exc,
                                )

                    if format == "h5":
                        save_h5(one_output, output_dir / "output.h5", mode="w")
                        if one_residual is not None:
                            save_h5(one_residual, output_dir / "residual.h5", mode="w")
                    else:
                        raise RoutineError(f"Save format '{format}' not recognized")

                bar()

        if failed_cases:
            logger.debug(f"Implicit generation completed with {failed_cases} failed cases at {path}")

    def generate_batch(self, path, format, write_policy):
        """Generate data for an ImplicitModel using batches and vmap."""
        if not self.throw:
            logger.warning(
                "GenImplicitModel.throw=False requires per-sample failure handling, "
                "so batch generation falls back to serial execution."
            )
            return self.generate_serial(path, format, write_policy)

        def _save_outputs(
            sample_outputs: Callable,
            sample_conditions: Callable | None,
            one_input: PyTree,
            one_solution: PyTree,
            output_keys: tuple[Key, ...],
            output_paths: tuple[Path, ...],
            evaluate: Callable | None,
        ) -> None:
            conditions = sample_conditions(pytree_stack(output_keys)) if sample_conditions is not None else None
            conditions = jax.device_get(conditions) if conditions is not None else None
            if conditions is not None:
                condition_iter = pytree_iter(conditions)
                for output_path in output_paths:
                    if format != "h5":
                        raise RoutineError(f"Save format '{format}' not recognized")
                    save_h5(next(condition_iter), output_path / "conditions.h5", mode="w")

            outputs = (
                sample_outputs(pytree_stack(output_keys), one_input, one_solution, conditions)
                if conditions is not None
                else sample_outputs(pytree_stack(output_keys), one_input, one_solution)
            )
            residuals = evaluate(one_input, outputs) if evaluate is not None else None
            outputs = jax.device_get(outputs)
            residuals = jax.device_get(residuals) if residuals is not None else None

            residual_iter = pytree_iter(residuals) if residuals is not None else None
            for one_output, output_path in zip(pytree_iter(outputs), output_paths):
                one_residual = next(residual_iter) if residual_iter is not None else None

                if format == "h5":
                    save_h5(one_output, output_path / "output.h5", mode="w")
                    if one_residual is not None:
                        save_h5(one_residual, output_path / "residual.h5", mode="w")
                else:
                    raise RoutineError(f"Save format '{format}' not recognized")

        def _process_input_batch(
            input_batch: list[tuple[Key, Path]],
            sample_inputs: Callable,
            solve: Callable | None,
            evaluate_solution: Callable | None,
            sample_outputs: Callable,
            sample_conditions: Callable | None,
            evaluate: Callable | None,
            bar: Callable,
        ) -> None:
            if not input_batch:
                return

            # Only sample/solve missing inputs for policy=reuse
            inputs_by_index: dict[int, object] = {}
            solutions_by_index: dict[int, object] = {}
            solution_residuals_by_index: dict[int, object] = {}

            missing_indices = range(len(input_batch))
            if write_policy == "reuse":
                missing_indices = [
                    i for i, (_, input_path) in enumerate(input_batch) if not (input_path / "input.h5").exists()
                ]

            if missing_indices:
                missing_keys = tuple(input_batch[i][0] for i in missing_indices)
                generated_inputs = sample_inputs(pytree_stack(missing_keys))
                generated_solutions = solve(generated_inputs) if solve is not None else None
                generated_solution_residuals = (
                    evaluate_solution(generated_inputs, generated_solutions)
                    if evaluate_solution is not None and generated_solutions is not None
                    else None
                )
                generated_inputs = jax.device_get(generated_inputs)
                generated_solutions = jax.device_get(generated_solutions) if generated_solutions is not None else None
                generated_solution_residuals = (
                    jax.device_get(generated_solution_residuals)
                    if generated_solution_residuals is not None
                    else None
                )
                input_samples = list(pytree_iter(generated_inputs))
                solution_samples = list(pytree_iter(generated_solutions)) if generated_solutions is not None else None
                solution_residual_samples = (
                    list(pytree_iter(generated_solution_residuals))
                    if generated_solution_residuals is not None
                    else None
                )

                for batch_index, input_index in enumerate(missing_indices):
                    inputs_by_index[input_index] = input_samples[batch_index]
                    if solution_samples is not None:
                        solutions_by_index[input_index] = solution_samples[batch_index]
                    if solution_residual_samples is not None:
                        solution_residuals_by_index[input_index] = solution_residual_samples[batch_index]

            for i, (_, input_path) in enumerate(input_batch):
                if i in inputs_by_index:
                    one_input = inputs_by_index[i]
                    one_solution = solutions_by_index.get(i)
                    one_solution_residual = solution_residuals_by_index.get(i)

                    if format == "h5":
                        save_h5(one_input, input_path / "input.h5", mode="w")
                        if one_solution is not None:
                            save_h5(one_solution, input_path / "solution.h5", mode="w")
                        if one_solution_residual is not None:
                            save_h5(one_solution_residual, input_path / "solution_residual.h5", mode="w")
                    else:
                        raise RoutineError(f"Save format '{format}' not recognized")

                else:
                    if format != "h5":
                        raise RoutineError(f"Save format '{format}' not recognized")
                    one_input = load_h5({}, input_path / "input.h5", jax=True)
                    solution_path = input_path / "solution.h5"
                    one_solution = load_h5({}, solution_path, jax=True) if solution_path.exists() else None

                output_batch: list[tuple[Key, Path]] = []
                skip = 'existing' if write_policy == 'reuse' else None

                for output_key, output_dir in gen_keys(
                    self.outputs_per_input, self.output_seed, path=input_path, skip=skip
                ):
                    if len(output_batch) < self.batch_size:
                        output_batch.append((output_key, output_dir))
                        continue

                    output_keys, output_paths = zip(*output_batch)
                    _save_outputs(
                        sample_outputs,
                        sample_conditions,
                        one_input,
                        one_solution,
                        output_keys,
                        output_paths,
                        evaluate,
                    )
                    output_batch.clear()
                    output_batch.append((output_key, output_dir))

                if output_batch:
                    output_keys, output_paths = zip(*output_batch)
                    _save_outputs(
                        sample_outputs,
                        sample_conditions,
                        one_input,
                        one_solution,
                        output_keys,
                        output_paths,
                        evaluate,
                    )

                bar()

            input_batch.clear()
        
        path = Path(path)
        model = self._edge_from_path(path)

        with alive_bar(self.input_samples, title=self.bar_text(path), title_length=_BAR_TITLE_LEN) as bar:

            if write_policy == 'error' and path.exists():
                raise RoutineError(f"Dataset already exists at {path} and policy='error'")

            path.mkdir(parents=True, exist_ok=True)

            sample_inputs = eqx.filter_jit(eqx.filter_vmap(model.sample_inputs))
            solve = eqx.filter_jit(eqx.filter_vmap(model.solve)) if hasattr(model, "solve") else None
            sample_outputs = _build_batched_sample_outputs(model)
            sample_conditions = (
                eqx.filter_jit(eqx.filter_vmap(model.sample_conditions))
                if _has_custom_sample_conditions(model)
                else None
            )
            evaluate_solution = eqx.filter_jit(eqx.filter_vmap(model.evaluate)) if hasattr(model, "evaluate") else None
            evaluate = (eqx.filter_jit(eqx.filter_vmap(model.evaluate, in_axes=(None, 0))) 
                        if hasattr(model, "evaluate") else None)

            input_batch: list[tuple[Key, Path]] = []

            for input_key, input_dir in gen_keys(self.input_samples, self.input_seed, path=path):
                if len(input_batch) < self.batch_size:
                    input_batch.append((input_key, input_dir))
                    continue

                _process_input_batch(
                    input_batch,
                    sample_inputs,
                    solve,
                    evaluate_solution,
                    sample_outputs,
                    sample_conditions,
                    evaluate,
                    bar,
                )
                input_batch.append((input_key, input_dir))

            _process_input_batch(
                input_batch,
                sample_inputs,
                solve,
                evaluate_solution,
                sample_outputs,
                sample_conditions,
                evaluate,
                bar,
            )


type ImplicitSampleRef = tuple[Path, Path, Literal["output", "solution"]]


class LoadImplicitModel(LoadDataConfig[ImplicitSampleRef]):
    """
    File-based load configuration for implicit-model datasets corresponding to `GenImplicitModel`.

    :param max_samples: maximum loaded samples per epoch (defaults to all)
    :param max_input_samples: maximum loaded input samples  (defaults to all)
    :param max_outputs_per_input: maximum loaded output samples below each input sample  (defaults to all)
    :param skip_input: decide whether to skip a particular input sample when loading
    :param skip_output: decide whether to skip a particular output sample when loading
    :param load_solution: whether to include ``solution.h5`` and ``solution_residual.h5`` as an extra
        sample for each input (defaults to ``True``)
    Output samples include ``conditions.h5`` as ``conditions`` when present.
    :param solution_only: whether to load only ``solution.h5`` and skip all output samples (defaults to ``False``)
    """

    max_samples: PositiveInt | None = None
    max_input_samples: PositiveInt | None = None
    max_outputs_per_input: PositiveInt | None = None
    skip_input: Callable[[Path], bool] | None = None
    skip_output: Callable[[Path], bool] | None = None
    load_solution: bool = True
    solution_only: bool = False

    @model_validator(mode="after")
    def _validate_solution_flags(self) -> LoadImplicitModel:
        """Validate solution-loading mode flags."""
        if self.solution_only and not self.load_solution:
            raise ValueError("solution_only=True requires load_solution=True.")
        return self

    @staticmethod
    def walk_sample_directories(
        root: Path,
        skip_input: Callable[[Path], bool] | None = None,
        skip_output: Callable[[Path], bool] | None = None,
        load_solution: bool = True,
        solution_only: bool = False,
    ) -> Generator[ImplicitSampleRef, None, None]:
        """Walk a nested input/output directory structure generated by `gen_keys` via `DataGeneration`."""
        if skip_input is None:
            skip_input = lambda p: False
        if skip_output is None:
            skip_output = lambda p: False

        for input_seed_dir in sorted(root.iterdir()):
            if not input_seed_dir.is_dir() or not input_seed_dir.name.startswith("seed_"):
                continue

            for input_sample_dir in sorted(input_seed_dir.iterdir()):
                if (not input_sample_dir.is_dir() or
                    not input_sample_dir.name.startswith("sample_") or
                    skip_input(input_sample_dir) or
                    _is_failed_sample_dir(input_sample_dir)):
                    continue

                if load_solution and (input_sample_dir / "solution.h5").exists():
                    yield input_sample_dir, input_sample_dir, "solution"

                if solution_only:
                    continue

                for output_seed_dir in sorted(input_sample_dir.iterdir()):
                    if not output_seed_dir.is_dir() or not output_seed_dir.name.startswith("seed_"):
                        continue

                    for output_sample_dir in sorted(output_seed_dir.iterdir()):
                        if (not output_sample_dir.is_dir() or
                            not output_sample_dir.name.startswith("sample_") or
                            skip_output(output_sample_dir) or
                            _is_failed_sample_dir(output_sample_dir)):
                            continue

                        yield input_sample_dir, output_sample_dir, "output"

    def discover_sample_refs(self, root: Path) -> list[ImplicitSampleRef]:
        """Discover nested implicit-model input/output sample paths."""
        return list(
            self.walk_sample_directories(
                root,
                skip_input=self.skip_input,
                skip_output=self.skip_output,
                load_solution=self.load_solution,
                solution_only=self.solution_only,
            )
        )

    def select_epoch_refs(self, refs: Sequence[ImplicitSampleRef], indices: jax.Array) -> list[ImplicitSampleRef]:
        """Select a global shuffled subset of implicit samples subject to epoch-level caps."""
        selected_refs: list[ImplicitSampleRef] = []
        unique_inputs: set[Path] = set()
        outputs_per_input: dict[Path, int] = {}

        for index in indices:
            sample_ref = refs[int(index)]
            input_path, _, _ = sample_ref

            next_total = len(selected_refs) + 1
            next_input_total = len(unique_inputs) + int(input_path not in unique_inputs)
            next_outputs_per_input = outputs_per_input.get(input_path, 0) + 1

            if self.max_samples is not None and next_total > self.max_samples:
                break
            if self.max_input_samples is not None and next_input_total > self.max_input_samples:
                continue
            if self.max_outputs_per_input is not None and next_outputs_per_input > self.max_outputs_per_input:
                continue

            selected_refs.append(sample_ref)
            unique_inputs.add(input_path)
            outputs_per_input[input_path] = next_outputs_per_input

        return selected_refs

    def count_epoch_refs(self, refs: Sequence[ImplicitSampleRef], indices: jax.Array) -> int:
        """Count a global shuffled subset of implicit samples subject to epoch-level caps."""
        selected_count = 0
        unique_inputs: set[Path] = set()
        outputs_per_input: dict[Path, int] = {}

        for index in indices:
            sample_ref = refs[int(index)]
            input_path, _, _ = sample_ref

            next_total = selected_count + 1
            next_input_total = len(unique_inputs) + int(input_path not in unique_inputs)
            next_outputs_per_input = outputs_per_input.get(input_path, 0) + 1

            if self.max_samples is not None and next_total > self.max_samples:
                break
            if self.max_input_samples is not None and next_input_total > self.max_input_samples:
                continue
            if self.max_outputs_per_input is not None and next_outputs_per_input > self.max_outputs_per_input:
                continue

            selected_count += 1
            unique_inputs.add(input_path)
            outputs_per_input[input_path] = next_outputs_per_input

        return selected_count

    def load_sample(self, ref: ImplicitSampleRef) -> PyTree:
        """Load one implicit-model sample from disk."""
        input_path, output_path, sample_kind = ref
        sample: dict[str, PyTree] = {"inputs": load_h5({}, input_path / "input.h5", jax=False)}

        output_file = output_path / ("solution.h5" if sample_kind == "solution" else "output.h5")
        if output_file.exists():
            sample["outputs"] = load_h5({}, output_file, jax=False)

        conditions_file = output_path / "conditions.h5"
        if sample_kind == "output" and conditions_file.exists():
            sample["conditions"] = load_h5({}, conditions_file, jax=False)

        residual_name = "solution_residual.h5" if sample_kind == "solution" else "residual.h5"
        residual_file = output_path / residual_name
        if residual_file.exists():
            sample["residuals"] = load_h5({}, residual_file, jax=False)

        return sample


class GenSource(GenGraph):
    """
    Sampling configuration for a plain source node in a FunctionGraph.

    :ivar throw: whether sample_source failures should propagate immediately
    """

    samples: int
    seed: int
    throw: bool = True
    _required_methods: list[str] = PrivateAttr(default=["sample_source"])

    def generate_serial(self, path, format, write_policy):
        """Generate source node data in serial."""
        path = Path(path)
        model = self._edge_from_path(path)
        failed_dirs: set[Path] = set()
        failed_cases = 0

        def _record_failure(sample_dir: Path, message: str, exc: BaseException) -> None:
            nonlocal failed_cases
            if sample_dir not in failed_dirs:
                failed_dirs.add(sample_dir)
                failed_cases += 1
            _log_sample_failure(sample_dir, message, exc)

        with alive_bar(self.samples, title=self.bar_text(path), title_length=_BAR_TITLE_LEN) as bar:
            if write_policy == "error" and path.exists():
                raise RoutineError(f"Dataset already exists at {path} and policy='error'")

            path.mkdir(parents=True, exist_ok=True)

            if hasattr(model, "resolve_source_sampler"):
                model.resolve_source_sampler()  # may need to load from a compression

            sample_source = eqx.filter_jit(model.sample_source)
            skip = (
                lambda p: (Path(p) / "source.h5").exists()
                if write_policy == "reuse"
                else False
            )

            for input_key, input_dir in gen_keys(self.samples, self.seed, path=path):
                if skip(input_dir):
                    bar()
                    continue

                if format != "h5":
                    raise RoutineError(f"Save format '{format}' not recognized.")

                if self.throw:
                    save_h5(sample_source(input_key), input_dir / "source.h5", mode="w")
                else:
                    try:
                        save_h5(sample_source(input_key), input_dir / "source.h5", mode="w")
                    except Exception as exc:  # noqa: BLE001
                        _record_failure(input_dir, f"Failed to sample source at {input_dir}", exc)

                bar()

        if failed_cases:
            logger.debug(f"Source generation completed with {failed_cases} failed cases at {path}")

    def generate_batch(self, path, format, write_policy):
        """Generate source node data in batches using vmap."""
        if not self.throw:
            logger.warning(
                "GenSource.throw=False requires per-sample failure handling, "
                "so batch generation falls back to serial execution."
            )
            return self.generate_serial(path, format, write_policy)

        def _process_batch(batch, sample_source, bar):
            if not batch:
                return
            
            input_keys, input_paths = zip(*batch)
            inputs = jax.device_get(sample_source(pytree_stack(input_keys)))

            for one_input, one_path in zip(pytree_iter(inputs), input_paths):
                if format == "h5":
                    save_h5(one_input, one_path / "source.h5", mode="w")
                else:
                    raise RoutineError(f"Save format '{format}' not recognized.")
                bar()
        
        path = Path(path)
        model = self._edge_from_path(path)

        with alive_bar(self.samples, title=self.bar_text(path), title_length=_BAR_TITLE_LEN) as bar:

            if write_policy == "error" and path.exists():
                raise RoutineError(f"Dataset already exists at {path} and policy='error'")

            path.mkdir(parents=True, exist_ok=True)

            if hasattr(model, "resolve_source_sampler"):
                model.resolve_source_sampler()  # may need to load from a compression

            sample_source = eqx.filter_jit(eqx.filter_vmap(model.sample_source))
            skip = (
                lambda p: (Path(p) / "source.h5").exists()
                if write_policy == "reuse"
                else False
            )

            batch: list[tuple[Key, Path]] = []

            for input_key, input_dir in gen_keys(self.samples, self.seed, path=path):
                if skip(input_dir):
                    bar()
                    continue

                if len(batch) < self.batch_size:
                    batch.append((input_key, input_dir))
                    continue

                _process_batch(batch, sample_source, bar)
                batch.clear()
                batch.append((input_key, input_dir))
            
            _process_batch(batch, sample_source, bar)


class LoadSource(LoadDataConfig[Path]):
    """
    File-based load configuration for source-node datasets corresponding to `GenSource`.

    :param max_samples: maximum loaded samples per epoch (defaults to all)
    :param skip_sample: decide whether to skip a particular source sample when loading
    """

    max_samples: PositiveInt | None = None
    skip_sample: Callable[[Path], bool] | None = None

    @staticmethod
    def walk_sample_directories(
        root: Path,
        skip_sample: Callable[[Path], bool] | None = None,
    ) -> Generator[Path, None, None]:
        """
        Walk a seed/sample directory structure generated by `gen_keys` via `GenSource`.

        :param root: Root directory containing seed/sample subdirectories.
        :param skip_sample: Optional predicate for skipping sample directories.
        :return: Generator yielding discovered sample directories.
        """
        if skip_sample is None:
            skip_sample = lambda p: False

        for seed_dir in sorted(root.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            for sample_dir in sorted(seed_dir.iterdir()):
                if (not sample_dir.is_dir() or
                    not sample_dir.name.startswith("sample_") or
                    skip_sample(sample_dir) or
                    _is_failed_sample_dir(sample_dir)):
                    continue

                yield sample_dir

    def discover_sample_refs(self, root: Path) -> list[Path]:
        """
        Discover source sample paths.

        :param root: Dataset root directory.
        :return: List of sample directories containing source data.
        """
        return list(self.walk_sample_directories(root, skip_sample=self.skip_sample))

    def select_epoch_refs(self, refs: Sequence[Path], indices: jax.Array) -> list[Path]:
        """Select a global shuffled subset of source samples subject to epoch-level caps."""
        selected_refs = [refs[int(index)] for index in indices]
        if self.max_samples is not None:
            return selected_refs[: self.max_samples]
        return selected_refs

    def count_epoch_refs(self, refs: Sequence[Path], indices: jax.Array) -> int:
        """Count a global shuffled subset of source samples subject to epoch-level caps."""
        if self.max_samples is not None:
            return min(len(indices), self.max_samples)
        return len(indices)

    def load_sample(self, ref: Path) -> PyTree:
        """Load one source sample from disk."""
        return load_h5({}, ref / "source.h5", jax=False)


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
    :param axes: axes to reduce over within each single-sample leaf
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
    return isinstance(value, GenNormLeaf) or isinstance(value, str) or callable(value) or (
        isinstance(value, Mapping) and "callable" in value
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
            raise ValueError(
                f"Normalization leaf shape changed from {self.input_shape} to {tuple(arr.shape)}."
            )
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
            raise ValueError(
                f"Normalization leaf shape changed from {self.input_shape} to {tuple(arr.shape)}."
            )
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

    def _iter_samples(self, progress: bool = True) -> Generator[tuple[str, PyTree], None, None]:
        ctxt = (
            alive_bar(len(self.loader), title=self.filename, title_length=_BAR_TITLE_LEN)
            if progress
            else _NullProgress()
        )

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
            dataset_name: self._artifact_path(root_path, str(dataset_name))
            for dataset_name in dataset_names
        }
        existing_paths = [artifact_path for artifact_path in artifact_paths.values() if artifact_path.exists()]
        if write_policy == "error" and existing_paths:
            raise RoutineError(
                f"Normalization artifact already exists at {existing_paths[0]} and policy='error'"
            )
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

    loader: DataLoader
    filename: str | Path = "compression.npz"
    gather_paths: Annotated[Sequence[TreePath], BeforeValidator(coerce_tree_paths)] = Field(default_factory=list)
    gather_template: Any | None = None
    norm: Any | None = None
    compression: Compression = Field(default_factory=lambda: SVD(energy_tol=0.999))

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
        """Apply one dataset's norm to either the wrapped payload or the nested dataset sample."""
        resolved_root = norm.resolve_root()
        if isinstance(sample, Mapping) and dataset_name in sample and isinstance(resolved_root, Mapping):
            if dataset_name in resolved_root:
                return norm(sample)

            normalized = dict(sample)
            normalized[dataset_name] = norm(sample[dataset_name])
            return normalized
        return norm(sample)

    def _iter_samples(self, progress: bool = True) -> Generator[PyTree, None, None]:
        ctxt = (
            alive_bar(len(self.loader), title=self.filename, title_length=_BAR_TITLE_LEN)
            if progress
            else _NullProgress()
        )

        with ctxt as bar:
            bar.text("Loading compression samples...")
            for batch in self.loader:
                for dataset_name, loaded in batch.items():
                    dataset_norm = self._dataset_norm(dataset_name)
                    if dataset_norm is not None:
                        dataset_norm.resolve_root()
                    for sample in self._iter_loaded_samples(loaded):
                        selected = self._merge_selected_sample({dataset_name: sample})
                        if dataset_norm is not None:
                            selected = self._apply_dataset_norm(selected, dataset_name, dataset_norm)
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
        
        samples = list(self._iter_samples())
        
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
            "latent_normal": None
            if latent_normal is None
            else [np.asarray(stats).tolist() for stats in latent_normal],
        }
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))


def _validate_gendata_pytree(template: PyTree) -> PyTree[GenDataConfig]:
    """Validate every leaf in a pytree-like template as a :class:`GenDataConfig`. Leave anything else untouched."""
    if isinstance(template, Mapping):
        latent_fields = {"compression", "gather_paths", "gather_template"}
        filename = template.get("filename")
        latent_filename = isinstance(filename, str | Path) and Path(filename).suffix == ".npz"
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


class DataGenerationBase(DictModel):
    """Named, arbitrary configuration used as one data-generation base case.

    All fields other than ``name`` are intentionally unrestricted at this stage. They are validated as a
    :class:`GenDataConfig` or dataset PyTree when the generation routine runs.

    :param name: directory name for this base case
    """

    name: str

    @model_validator(mode="after")
    def _validate_name(self) -> "DataGenerationBase":
        if not self.name or self.name in {".", ".."} or any(separator in self.name for separator in ("/", "\\")):
            raise ValueError("Data-generation base names must be non-empty single path components.")
        return self


class DataGenerationOverride(BaseModel):
    """One path and its candidate values for base configuration expansion.

    :param path: path in a base configuration to override
    :param cases: candidate values for the override path
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: TreePath
    cases: tuple[Any, ...]

    @classmethod
    def _coerce_path(cls, value: Any) -> TreePath:
        path = coerce_tree_path(value)
        if not isinstance(path, tuple) or not all(isinstance(token, str | int) for token in path):
            raise ValueError("Data-generation override paths must be sequences of string or integer tokens.")
        if not path:
            raise ValueError("Data-generation override paths cannot be empty.")
        if any(isinstance(token, int) and token < 0 for token in path):
            raise ValueError("Data-generation override paths do not support negative list indices.")
        return path

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("Data-generation overrides must be mappings with 'path' and 'cases'.")
        normalized = dict(value)
        normalized["path"] = cls._coerce_path(normalized.get("path"))
        cases = normalized.get("cases")
        if isinstance(cases, str | bytes) or not isinstance(cases, Sequence):
            cases = [cases]
        if not cases:
            raise ValueError("Data-generation override cases must contain at least one value.")
        normalized["cases"] = tuple(cases)
        return normalized


def _override_directory_name(override: DataGenerationOverride, value: Any) -> str:
    """Return a readable, safe directory name for one override value."""
    if isinstance(value, str | int | float | bool) or value is None:
        rendered = str(value)
    elif isinstance(value, Path):
        rendered = value.as_posix()
    else:
        rendered = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    rendered = rendered.replace("/", "_").replace("\\", "_")
    return f"{override.path[-1]}={rendered}"


class DataGeneration(Routine):
    """
    File-based data generation routine.

    :ivar root: root directory for saving data
    :ivar datasets: pytree template for datasets to generate under root
    :ivar bases: named arbitrary base configurations to expand under root
    :ivar overrides: Cartesian-product overrides applied to each base configuration
    :ivar format: the data format to save samples. Only `h5` supported.
    :ivar write_policy: reuse existing data, overwrite existing data, or throw an error if existing data found
    :ivar graph: graph object or YAML path (optional, for graph-related datasets)
    """

    root: Path
    datasets: GenDataPyTree | None = None
    bases: list[DataGenerationBase] = Field(default_factory=list)
    overrides: list[DataGenerationOverride] = Field(default_factory=list)

    format: SUPPORTED_FORMATS = "h5"
    write_policy: SUPPORTED_POLICIES = "reuse"

    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None

    @model_validator(mode="after")
    def _bind_graph(self):
        # Pass graph object to graph datasets
        if self.graph is not None and self.datasets is not None:
            self._bind_graph_to_datasets(self.datasets)

        if self.bases and self.datasets is not None:
            raise ValueError("DataGeneration cannot specify both 'datasets' and 'bases'.")
        if self.overrides and not self.bases:
            raise ValueError("DataGeneration overrides require at least one base configuration.")
        if not self.datasets and not self.bases:
            raise ValueError("DataGeneration requires either 'datasets' or at least one base configuration.")

        names = [base.name for base in self.bases]
        if len(names) != len(set(names)):
            raise ValueError("Data-generation base names must be unique.")

        return self

    def _bind_graph_to_datasets(self, datasets: PyTree) -> None:
        """Pass the configured graph to graph-based dataset leaves."""
        leaves, _ = jax.tree.flatten(datasets, is_leaf=lambda leaf: isinstance(leaf, GenDataConfig))
        for ds in leaves:
            if hasattr(ds, "graph") and ds.graph is None:
                ds.graph = self.graph

    def _base_datasets(self, base: DataGenerationBase, values: Sequence[Any]) -> PyTree:
        """Validate one overridden base as a generic dataset PyTree."""
        config = base.model_dump(mode="python")
        for override, value in zip(self.overrides, values):
            if override.path[0] == "name":
                raise ValueError("Data-generation overrides cannot modify the reserved base 'name'.")
            config = pytree_merge(config, set_subtree(None, override.path, value))
        config.pop("name", None)
        datasets = _validate_gendata_pytree(config)
        self._bind_graph_to_datasets(datasets)
        if not any(isinstance(leaf, GenDataConfig) for leaf in jax.tree.leaves(
            datasets, is_leaf=lambda leaf: isinstance(leaf, GenDataConfig)
        )):
            raise ValueError(f"Base configuration {base.name!r} does not contain a supported data generator.")
        return datasets

    def _run_base_cases(self) -> None:
        """Generate every Cartesian-product base case."""
        case_values = product(*(override.cases for override in self.overrides))
        for values in case_values:
            override_root = self.root
            for override, value in zip(self.overrides, values):
                override_root /= _override_directory_name(override, value)

            for base in self.bases:
                datasets = self._base_datasets(base, values)
                base_root = override_root / base.name
                for path, dataset in pytree_path_iter(
                    datasets, is_leaf=lambda leaf: isinstance(leaf, GenDataConfig)
                ):
                    dataset.generate(base_root / "/".join(path), format=self.format, write_policy=self.write_policy)

    def run(self) -> int:
        """Generate all datasets."""
        if self.bases:
            self._run_base_cases()
            return 0

        assert self.datasets is not None
        for path, dataset in pytree_path_iter(self.datasets, is_leaf=lambda leaf: isinstance(leaf, GenDataConfig)):
            dataset.generate(self.root / "/".join(path), format=self.format, write_policy=self.write_policy)
        
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
                """If nested seed/sample, then ImplicitModel. If flat seed/sample, then Source. Can't tell otherwise."""
                seed_dirs = [d for d in ds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

                if len(seed_dirs) > 0 and all("seed_" in d.name for d in seed_dirs):
                    # Just check the first seed directory
                    sample_dirs = [
                        d
                        for d in seed_dirs[0].iterdir()
                        if d.is_dir() and not d.name.startswith(".") and not _is_failed_sample_dir(d)
                    ]

                    if len(sample_dirs) > 0 and all("sample_" in d.name for d in sample_dirs):
                        if all((s / "input.h5").exists() and (s / "solution.h5").exists() for s in sample_dirs):
                            return LoadImplicitModel()
                        if all((s / "source.h5").exists() for s in sample_dirs):
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
            ds_cfg.configure_epoch_cache(ds_totals[ds_name])
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
                    epoch_refs[int(index)] for index in ds_indices[name][ds_cursors[name]:ds_cursors[name] + batch_size]
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
