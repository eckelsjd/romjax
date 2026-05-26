"""Data generation routine."""
import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Callable, Literal, Mapping, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
from alive_progress import alive_bar
from jaxtyping import Key, PyTree
from pydantic import BaseModel, BeforeValidator, ConfigDict, PositiveInt, model_validator

from romjax.graph import FunctionGraph
from romjax.rng import gen_keys
from romjax.routine import Routine, RoutineError
from romjax.tree import pytree_iter, pytree_path_iter
from romjax.typing import from_yaml
from romjax.utils import load_h5, save_h5

__all__ = ["DataGeneration", "DatasetConfig"]


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


def _stack_batch(items: tuple[object, ...]) -> object:
    """Stack matching pytree items into a single batched pytree."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *items)


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


def _required_fields(model_cls: type[BaseModel]) -> set[str]:
    inherited = set()

    # for base in model_cls.__mro__[1:]:
    #     if issubclass(base, BaseModel):
    #         inherited.update(getattr(base, "model_fields", {}))

    return {
        name
        for name, field in model_cls.model_fields.items()
        if name not in inherited and field.is_required()
    }


type SUPPORTED_FORMATS = Literal["h5"]
type SUPPORTED_POLICIES = Literal["reuse", "overwrite", "error"]


class DatasetConfig(BaseModel, ABC):
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


class GraphDataset(DatasetConfig):
    """Datasets that use a FunctionGraph for sample generation."""

    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None


class ImplicitModelDataset(GraphDataset):
    """
    Sampling configuration for a nested input/output strategy for implicit models in a FunctionGraph.

    :ivar input_samples: number of input samples
    :ivar outputs_per_input: number of output samples for each input sample
    :ivar input_seed: random seed for inputs
    :ivar output_seed: random seed for outputs
    :ivar batch_size: number of samples to generate at a time
    :ivar name_depth: depth in the path for displaying current dataset name (default: 2)
    """

    input_samples: int
    outputs_per_input: int
    input_seed: int
    output_seed: int
    batch_size: PositiveInt = 1
    name_depth: int = 2

    def _generate_serial(self, path, format, write_policy):
        """Generate data in serial for an ImplicitModel."""
        path = Path(path)
        bar_text = "/".join(path.parts[-self.name_depth:])
        edge_name = path.name

        if edge_name not in self.graph.edges:
            raise ValueError(f"Last folder name must be an edge name in the graph. '{edge_name}' not recognized.")
        
        model = self.graph.edges[edge_name]

        with alive_bar(self.input_samples) as bar:
            bar.text(bar_text)

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

                if write_policy == "reuse" and input_path.exists():
                    if format != "h5":
                        raise RoutineError(f"Save format '{format}' not recognized")
                    one_input = load_h5({}, input_path, jax=True)
                    one_solution = load_h5({}, solution_path, jax=True) if solution_path.exists() else None
                else:
                    one_input = sample_inputs(input_key)
                    one_solution = solve(one_input) if solve is not None else None

                    if format == "h5":
                        save_h5(one_input, input_path, mode="w")
                        if one_solution is not None:
                            save_h5(one_solution, solution_path, mode="w")
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
    
    def _generate_batch(self, path, format, write_policy):
        """Generate data for an ImplicitModel using batches and vmap."""

        def _save_outputs(
            sample_outputs: Callable,
            one_input: PyTree,
            one_solution: PyTree,
            output_keys: tuple[Key, ...],
            output_paths: tuple[Path, ...],
            evaluate: Callable | None,
        ) -> None:
            outputs = sample_outputs(_stack_batch(output_keys), one_input, one_solution)
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
            sample_outputs: Callable,
            evaluate: Callable | None,
            bar: Callable,
        ) -> None:
            if not input_batch:
                return

            # Only sample/solve missing inputs for policy=reuse
            inputs_by_index: dict[int, object] = {}
            solutions_by_index: dict[int, object] = {}

            missing_indices = range(len(input_batch))
            if write_policy == "reuse":
                missing_indices = [
                    i for i, (_, input_path) in enumerate(input_batch) if not (input_path / "input.h5").exists()
                ]

            if missing_indices:
                missing_keys = tuple(input_batch[i][0] for i in missing_indices)
                generated_inputs = sample_inputs(_stack_batch(missing_keys))
                generated_solutions = solve(generated_inputs) if solve is not None else None
                generated_inputs = jax.device_get(generated_inputs)
                generated_solutions = jax.device_get(generated_solutions) if generated_solutions is not None else None
                input_samples = list(pytree_iter(generated_inputs))
                solution_samples = list(pytree_iter(generated_solutions)) if generated_solutions is not None else None

                for batch_index, input_index in enumerate(missing_indices):
                    inputs_by_index[input_index] = input_samples[batch_index]
                    if solution_samples is not None:
                        solutions_by_index[input_index] = solution_samples[batch_index]

            for i, (_, input_path) in enumerate(input_batch):
                if i in inputs_by_index:
                    one_input = inputs_by_index[i]
                    one_solution = solutions_by_index.get(i)

                    if format == "h5":
                        save_h5(one_input, input_path / "input.h5", mode="w")
                        if one_solution is not None:
                            save_h5(one_solution, input_path / "solution.h5", mode="w")
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
        bar_text = "/".join(path.parts[-self.name_depth:])
        edge_name = path.name

        if edge_name not in self.graph.edges:
            raise ValueError(f"Last folder name must be an edge name in the graph. '{edge_name}' not recognized.")
        
        model = self.graph.edges[edge_name]

        with alive_bar(self.input_samples) as bar:
            bar.text(bar_text)

            if write_policy == 'error' and path.exists():
                raise RoutineError(f"Dataset already exists at {path} and policy='error'")

            path.mkdir(parents=True, exist_ok=True)

            sample_inputs = eqx.filter_jit(eqx.filter_vmap(model.sample_inputs))
            solve = eqx.filter_jit(eqx.filter_vmap(model.solve)) if hasattr(model, "solve") else None
            sample_outputs = _build_batched_sample_outputs(model)
            evaluate = (eqx.filter_jit(eqx.filter_vmap(model.evaluate, in_axes=(None, 0))) 
                        if hasattr(model, "evaluate") else None)

            input_batch: list[tuple[Key, Path]] = []

            for input_key, input_dir in gen_keys(self.input_samples, self.input_seed, path=path):
                if len(input_batch) < self.batch_size:
                    input_batch.append((input_key, input_dir))
                    continue

                _process_input_batch(input_batch, sample_inputs, solve, sample_outputs, evaluate, bar)
                input_batch.append((input_key, input_dir))

            _process_input_batch(input_batch, sample_inputs, solve, sample_outputs, evaluate, bar)

    def generate(self, path, format=None, write_policy=None):
        """
        Generate data for an ImplicitModel. Assumes graph edge name is the last name in the `path`.
        
        A nested folder structure will be generated with the outer seed_i/sample_j set corresponding to the 
        result of `sample_inputs`, and the inner seed/sample set corresponding to the result of `sample_outputs`.
        Generally, output samples are conditioned on input samples, and so several output samples may be requested 
        for a single input sample.

        If an Edge implements `solve` and `evaluate`, a solution will be generated and saved alongside each input, 
        and a residual will be computed and saved alongside each output.
        """
        format, write_policy = self._validate_format_and_policy(format, write_policy)

        if self.graph is None:
            raise ValueError("Must specify a graph to generate data.")
        
        if self.batch_size > 1:
            self._generate_batch(path, format, write_policy)
        else:
            self._generate_serial(path, format, write_policy)


class SourceDataset(GraphDataset):
    """
    Sampling configuration for a plain source node in a FunctionGraph.
    """

    samples: int
    seed: int

    def generate(self, path, format=None, write_policy=None):
        format, write_policy = self._validate_format_and_policy(format, write_policy)

        if self.graph is None:
            raise ValueError("Must specify a graph to generate data.")
        
        pass


def validate_dataset_pytree(template: PyTree) -> PyTree:
    """Validate every leaf in a pytree-like template as a :class:`DatasetConfig`. Leave anything else untouched."""
    if isinstance(template, Mapping):
        if all(field in template for field in _required_fields(ImplicitModelDataset)):
            return ImplicitModelDataset(**template)
        if all(field in template for field in _required_fields(SourceDataset)):
            return SourceDataset(**template)
        return {key: validate_dataset_pytree(value) for key, value in template.items()}
    if isinstance(template, tuple):
        return tuple(validate_dataset_pytree(value) for value in template)
    if isinstance(template, list):
        return [validate_dataset_pytree(value) for value in template]
    
    return template


type DatasetPyTree = Annotated[PyTree, BeforeValidator(validate_dataset_pytree)]


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
    datasets: DatasetPyTree

    format: SUPPORTED_FORMATS = "h5"
    write_policy: SUPPORTED_POLICIES = "reuse"

    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None

    @model_validator(mode="after")
    def _bind_graph(self):
        # Pass graph object to graph datasets
        if self.graph is not None:
            leaves, _ = jax.tree.flatten(self.datasets, is_leaf=lambda leaf: isinstance(leaf, DatasetConfig))
            for ds in leaves:
                if hasattr(ds, "graph"):
                    if ds.graph is None:
                        ds.graph = self.graph
        
        return self
    
    def run(self) -> int:
        """Generate all datasets."""
        for path, dataset in pytree_path_iter(self.datasets, is_leaf=lambda leaf: isinstance(leaf, DatasetConfig)):
            dataset.generate(self.root / "/".join(path), format=self.format, write_policy=self.write_policy)
        
        return 0
