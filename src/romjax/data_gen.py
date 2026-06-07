"""Data generation routine and data loading."""
from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterator, Literal, Mapping, Sequence, get_args

import equinox as eqx
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
    PositiveInt,
    PrivateAttr,
    model_validator,
)

from romjax.compression import SVD, Compression
from romjax.graph import Edge, FunctionGraph
from romjax.rng import gen_keys
from romjax.routine import Routine, RoutineError
from romjax.tree import (
    TreePath,
    coerce_tree_paths,
    get_subtree,
    pytree_iter,
    pytree_path_iter,
    pytree_stack,
    set_subtree,
)
from romjax.typing import from_yaml
from romjax.utils import _NullProgress, load_h5, required_fields, save_h5

__all__ = [
    "DataGeneration",
    "DataLoader",
    "GenDataConfig",
    "LoadDataConfig",
]

_BAR_TITLE_LEN = 20


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


def _build_batched_sample_outputs(model):
    """Build one batched ``sample_outputs`` function for a model."""
    sample_output_kwargs = _get_kwargs(model.sample_outputs)

    if all(arg in sample_output_kwargs for arg in ["inputs", "solution"]):
        def _sample_outputs(keys, one_input, one_solution):
            return jax.vmap(
                lambda key: model.sample_outputs(key, inputs=one_input, solution=one_solution)
            )(keys)
    elif "solution" in sample_output_kwargs:
        def _sample_outputs(keys, _, one_solution):
            return jax.vmap(
                lambda key: model.sample_outputs(key, solution=one_solution)
            )(keys)
    else:
        def _sample_outputs(keys, *_):
            return jax.vmap(model.sample_outputs)(keys)

    return eqx.filter_jit(_sample_outputs)


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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_size: PositiveInt = 16
    shuffle_seed: int = 0
    max_epochs: PositiveInt | None = None

    @abstractmethod
    def discover_sample_refs(self, root: Path) -> list[T]:
        """Discover sample references below one dataset root."""
        raise NotImplementedError

    def select_epoch_refs(self, refs: Sequence[T], indices: jax.Array) -> list[T]:
        """Select the full epoch reference order from a shuffled dataset index order."""
        return [refs[int(index)] for index in indices]

    def count_epoch_refs(self, refs: Sequence[T], indices: jax.Array) -> int:
        """Count the number of references selected for one epoch without materializing them."""
        return len(indices)

    @abstractmethod
    def load_batch(self, refs: Sequence[T]) -> dict[str, PyTree]:
        """Load and stack a selected batch of dataset references."""
        raise NotImplementedError


class GenImplicitModel(GenGraph):
    """
    Sampling configuration for a nested input/output strategy for implicit models in a FunctionGraph.

    A nested folder structure will be generated with the outer seed_i/sample_j set corresponding to the 
    result of `sample_inputs`, and the inner seed/sample set corresponding to the result of `sample_outputs`.
    Generally, output samples are conditioned on input samples, and so several output samples may be requested 
    for a single input sample.

    If an Edge implements `solve` and `evaluate`, a solution will be generated and saved alongside each input, 
    and a residual will be computed and saved alongside each output.

    :ivar input_samples: number of input samples
    :ivar outputs_per_input: number of output samples for each input sample
    :ivar input_seed: random seed for inputs
    :ivar output_seed: random seed for outputs
    """

    input_samples: int
    outputs_per_input: int
    input_seed: int
    output_seed: int
    _required_methods: list[str] = PrivateAttr(default=["sample_inputs"])

    def generate_serial(self, path, format, write_policy):
        """Generate data in serial for an ImplicitModel."""
        path = Path(path)
        model = self._edge_from_path(path)

        with alive_bar(self.input_samples, title=self.bar_text(path), title_length=_BAR_TITLE_LEN) as bar:

            if write_policy == "error" and path.exists():
                raise RoutineError(f"Dataset already exists at {path} and policy='error'")

            path.mkdir(parents=True, exist_ok=True)

            sample_inputs = eqx.filter_jit(model.sample_inputs)
            solve = eqx.filter_jit(model.solve) if hasattr(model, "solve") else None
            evaluate = eqx.filter_jit(model.evaluate) if hasattr(model, "evaluate") else None

            sample_output_kwargs = _get_kwargs(model.sample_outputs)
            if all(arg in sample_output_kwargs for arg in ["inputs", "solution"]):
                sample_outputs = eqx.filter_jit(
                    lambda key, one_input, one_solution: model.sample_outputs(
                        key, inputs=one_input, solution=one_solution
                    )
                )
            elif "solution" in sample_output_kwargs:
                sample_outputs = eqx.filter_jit(
                    lambda key, _, one_solution: model.sample_outputs(key, solution=one_solution)
                )
            else:
                sample_outputs = eqx.filter_jit(lambda key, *_: model.sample_outputs(key))

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
                    one_solution = solve(one_input) if solve is not None else None
                    one_solution_residual = (
                        evaluate(one_input, one_solution)
                        if evaluate is not None and one_solution is not None
                        else None
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
                    one_output = sample_outputs(output_key, one_input, one_solution)
                    one_residual = evaluate(one_input, one_output) if evaluate is not None else None

                    if format == "h5":
                        save_h5(one_output, output_dir / "output.h5", mode="w")
                        if one_residual is not None:
                            save_h5(one_residual, output_dir / "residual.h5", mode="w")
                    else:
                        raise RoutineError(f"Save format '{format}' not recognized")

                bar()
    
    def generate_batch(self, path, format, write_policy):
        """Generate data for an ImplicitModel using batches and vmap."""

        def _save_outputs(
            sample_outputs: Callable,
            one_input: PyTree,
            one_solution: PyTree,
            output_keys: tuple[Key, ...],
            output_paths: tuple[Path, ...],
            evaluate: Callable | None,
        ) -> None:
            outputs = sample_outputs(pytree_stack(output_keys), one_input, one_solution)
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
                    _save_outputs(sample_outputs, one_input, one_solution, output_keys, output_paths, evaluate)
                    output_batch.clear()
                    output_batch.append((output_key, output_dir))

                if output_batch:
                    output_keys, output_paths = zip(*output_batch)
                    _save_outputs(sample_outputs, one_input, one_solution, output_keys, output_paths, evaluate)

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
                    evaluate,
                    bar,
                )
                input_batch.append((input_key, input_dir))

            _process_input_batch(input_batch, sample_inputs, solve, evaluate_solution, sample_outputs, evaluate, bar)


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
                    skip_input(input_sample_dir)):
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
                            skip_output(output_sample_dir)):
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

    def load_batch(self, refs: Sequence[ImplicitSampleRef]) -> dict[str, PyTree]:
        """Load stacked implicit-model mini-batches."""
        loaded: dict[str, list[PyTree]] = {"inputs": [], "outputs": [], "residuals": []}

        for input_path, output_path, sample_kind in refs:
            loaded["inputs"].append(load_h5({}, input_path / "input.h5", jax=True))

            output_file = output_path / ("solution.h5" if sample_kind == "solution" else "output.h5")
            if output_file.exists():
                loaded["outputs"].append(load_h5({}, output_file, jax=True))

            residual_name = "solution_residual.h5" if sample_kind == "solution" else "residual.h5"
            residual_file = output_path / residual_name
            if residual_file.exists():
                loaded["residuals"].append(load_h5({}, residual_file, jax=True))

        return {
            key: pytree_stack(value)
            for key, value in loaded.items()
            if len(value) > 0
        }


class GenSource(GenGraph):
    """
    Sampling configuration for a plain source node in a FunctionGraph.
    """

    samples: int
    seed: int
    _required_methods: list[str] = PrivateAttr(default=["sample_source"])

    def generate_serial(self, path, format, write_policy):
        """Generate source node data in serial."""
        path = Path(path)
        model = self._edge_from_path(path)

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

                if format == "h5":
                    save_h5(sample_source(input_key), input_dir / "source.h5", mode="w")
                else:
                    raise RoutineError(f"Save format '{format}' not recognized.")
                
                bar()
    
    def generate_batch(self, path, format, write_policy):
        """Generate source node data in batches using vmap."""
        
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
                    skip_sample(sample_dir)):
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

    def load_batch(self, refs: Sequence[Path]) -> dict[str, PyTree]:
        """
        Load stacked source mini-batches.

        :param refs: Sample directories to load.
        :return: Mapping containing the stacked ``source`` batch.
        """
        return pytree_stack([load_h5({}, ref / "source.h5", jax=True) for ref in refs])


class GenLatent(GenDataConfig):
    """Fit a latent-space compressor and emit the compression artifact."""

    loader: DataLoader
    filename: str | Path = "compression.npz"
    gather_paths: Annotated[Sequence[TreePath], BeforeValidator(coerce_tree_paths)] = Field(default_factory=list)
    gather_template: Any | None = None
    compression: Compression = Field(default_factory=lambda: SVD(energy_tol=0.999))

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

    def _iter_samples(self, progress: bool = True) -> Generator[PyTree, None, None]:
        
        ctxt = (alive_bar(len(self.loader), title=self.filename, title_length=_BAR_TITLE_LEN) 
                if progress else _NullProgress())

        with ctxt as bar:
            bar.text("Loading compression samples...")
            for batch in self.loader:
                for dataset_name, loaded in batch.items():
                    for sample in pytree_iter(loaded):
                        yield self._merge_selected_sample({dataset_name: sample})
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
        
        logger.info("Fitting compression...")
        compression = self.compression.fit(samples)
        logger.info(f"Compression finished. Latent space: {compression.latent_size()}")

        compression.dump(artifact_path)
        
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
    :ivar datasets: pytree template for datasets to generate under root
    :ivar format: the data format to save samples. Only `h5` supported.
    :ivar write_policy: reuse existing data, overwrite existing data, or throw an error if existing data found
    :ivar graph: graph object or YAML path (optional, for graph-related datasets)
    """

    root: Path
    datasets: GenDataPyTree

    format: SUPPORTED_FORMATS = "h5"
    write_policy: SUPPORTED_POLICIES = "reuse"

    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None

    @model_validator(mode="after")
    def _bind_graph(self):
        # Pass graph object to graph datasets
        if self.graph is not None:
            leaves, _ = jax.tree.flatten(self.datasets, is_leaf=lambda leaf: isinstance(leaf, GenDataConfig))
            for ds in leaves:
                if hasattr(ds, "graph"):
                    if ds.graph is None:
                        ds.graph = self.graph
        
        return self
    
    def run(self) -> int:
        """Generate all datasets."""
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
                    sample_dirs = [d for d in seed_dirs[0].iterdir() if d.is_dir() and not d.name.startswith(".")]

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
