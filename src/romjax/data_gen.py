"""Data generation routine."""
import inspect
import os
from pathlib import Path
from typing import Annotated, Callable, Literal, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from alive_progress import alive_bar
from jaxtyping import Key, PyTree
from pydantic import (
    BaseModel,
    BeforeValidator,
    PositiveInt,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from romjax.graph import FunctionGraph
from romjax.model import ImplicitSampleable
from romjax.rng import gen_keys
from romjax.routine import Routine, RoutineError
from romjax.tree import pytree_iter
from romjax.typing import from_yaml
from romjax.utils import load_h5, save_h5

__all__ = ["DataGeneration"]


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


class SampleConfig(BaseModel):
    """
    Sampling configuration for a nested input/output sampling strategy.

    :ivar input_samples: number of input samples
    :ivar outputs_per_input: number of output samples for each input sample
    :ivar input_seed: random seed for inputs
    :ivar output_seed: random seed for outputs
    """

    input_samples: int
    outputs_per_input: int
    input_seed: int
    output_seed: int


class DataGeneration(Routine):
    """
    Data generation routine for a FunctionGraph.
    
    :ivar root: root directory for saving data
    :ivar graph: the FunctionGraph object specifying all models and connections. 
                 May point to a yaml file with the FunctionGraph spec implemented at the top-level
    :ivar train: sampling configuration for each model (see `SampleConfig`) for training dataset
    :ivar validation: sampling configurations for validation dataset
    :ivar to_sample: the names of the models to sample. Each must implement the `ImplicitSampleable` protocol
    :ivar batch_size: number of samples to generate at a time
    :ivar format: the data format to save samples. Only `h5` supported.
    :ivar write_policy: reuse existing data, overwrite existing data, or throw an error if existing data found
    """

    root: Path 
    graph: Annotated[FunctionGraph, BeforeValidator(from_yaml)]
    train: Sequence[SampleConfig]
    validation: Sequence[SampleConfig]
    to_sample: list[str] | None = None
    batch_size: PositiveInt = 1
    format: Literal["h5"] = "h5"
    write_policy: Literal["reuse", "overwrite", "error"] = "reuse"

    def run(self) -> int:
        if self.batch_size > 1:
            return self._generate_data_batch()
        else:
            return self._generate_data_serial()

    @field_validator("root", mode="after")
    @classmethod
    def _make_root(cls, value: Path):
        if not value.exists():
            os.makedirs(value, exist_ok=True)
        os.makedirs(value / "train", exist_ok=True)
        os.makedirs(value / "validation", exist_ok=True)
        return value
    
    @field_validator("train", "validation", mode="before")
    @classmethod
    def _allow_single_config(cls, value: SampleConfig | Sequence[SampleConfig]) -> Sequence[SampleConfig]:
        if not isinstance(value, Sequence):
            return [value]
        return value
    
    @field_validator("to_sample", mode="wrap")
    @classmethod
    def _validate_sampleable(
        cls, 
        value: str | list[str] | None, 
        handler: ValidatorFunctionWrapHandler, 
        info: ValidationInfo
    ) -> list[str]:
        """If none, default to all sampleables in the graph."""
        if value is None:
            value = []
            for edge_name, edge in info.data['graph'].edges.items():
                if isinstance(edge, ImplicitSampleable):
                    value.append(edge_name)
        
        if not isinstance(value, list):
            value = [value]
        
        value = handler(value)

        # Check for the Sampleable required methods
        for name in value:
            if name not in info.data['graph'].edges:
                raise ValueError(f"Model name '{name}' not an edge in the graph, so it cannot be sampled.")
            edge = info.data['graph'].edges[name]
            if not hasattr(edge, 'sample_inputs') or not hasattr(edge, 'sample_outputs'):
                raise ValueError(f"Graph edge object {edge} does not have the required 'sample_inputs' and "
                                 f"'sample_outputs' methods, so it cannot be sampled.")
        return value
    
    @model_validator(mode="after")
    def _validate_sequences(self):
        """Make sure train, validation configs match the length of sampleables (will broadcast len=1)."""
        num_sample = len(self.to_sample)

        if num_sample == 0:
            raise ValueError("Must specify at least one model to sample.")

        if num_sample > 1:
            if len(self.train) == 1:
                for i in range(1, num_sample):
                    new_config = self.train[0].copy()
                    new_config.seed += i
                    self.train.append(new_config)
            
            if len(self.validation) == 1:
                for i in range(1, num_sample):
                    new_config = self.validation[0].copy()
                    new_config.seed += i
                    self.validation.append(new_config)
        
        if len(self.train) != num_sample:
            raise ValueError(f"Number of training configs: {len(self.train)}. Expected {num_sample}")
        if len(self.validation) != num_sample:
            raise ValueError(f"Number of validation configs: {len(self.validation)}. Expected {num_sample}")
        
        return self

    def _get_dataset_totals(self):
        """Sum over just input samples for all sampleable models."""
        total_train_samples = 0
        total_validation_samples = 0
        for edge_idx, edge_name in enumerate(self.to_sample):
            total_train_samples += self.train[edge_idx].input_samples
            total_validation_samples += self.validation[edge_idx].input_samples
        
        return {"train": total_train_samples, "validation": total_validation_samples}

    def _generate_data_batch(self):
        """
        Generate training and validation data for a FunctionGraph using batches and vmap.
        
        Compatible with Edges stored in a FunctionGraph that implement the Sampleable protocol.
        A nested folder structure will be generated with the outer seed_i/sample_j set corresponding to the 
        result of `sample_inputs`, and the inner seed/sample set corresponding to the result of `sample_outputs`.
        Generally, output samples are conditioned on input samples, and so several output samples may be requested 
        for a single input sample.

        Of particular interest are ImplictModels, which implement the `solve` and `evaluate` methods. If an Edge
        implements these methods, a solution will be generated and saved alongside each input, and a residual will be
        computed and saved alongside each output.

        Only the `h5` storage format is supported for all samples, so make sure your samples adhere to a simple nested
        dict structure with array leaves.
        """

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

                if self.format == "h5":
                    save_h5(one_output, output_path / "output.h5", mode="w")
                    if one_residual is not None:
                        save_h5(one_residual, output_path / "residual.h5", mode="w")
                else:
                    raise RoutineError(f"Save format '{self.format}' not recognized")

        def _process_input_batch(
            input_batch: list[tuple[Key, Path]],
            sample_inputs: Callable,
            solve: Callable | None,
            sample_outputs: Callable,
            evaluate: Callable | None,
            sample_config: object,
            bar: Callable,
        ) -> None:
            if not input_batch:
                return

            # Only sample/solve missing inputs for policy=reuse
            inputs_by_index: dict[int, object] = {}
            solutions_by_index: dict[int, object] = {}

            missing_indices = range(len(input_batch))
            if self.write_policy == "reuse":
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

                    if self.format == "h5":
                        save_h5(one_input, input_path / "input.h5", mode="w")
                        if one_solution is not None:
                            save_h5(one_solution, input_path / "solution.h5", mode="w")
                    else:
                        raise RoutineError(f"Save format '{self.format}' not recognized")

                else:
                    if self.format != "h5":
                        raise RoutineError(f"Save format '{self.format}' not recognized")
                    one_input = load_h5({}, input_path / "input.h5", jax=True)
                    solution_path = input_path / "solution.h5"
                    one_solution = load_h5({}, solution_path, jax=True) if solution_path.exists() else None

                output_batch: list[tuple[Key, Path]] = []
                skip = 'existing' if self.write_policy == 'reuse' else None

                for output_key, output_dir in gen_keys(
                    sample_config.outputs_per_input, sample_config.output_seed, path=input_path, skip=skip
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

        dataset_totals = self._get_dataset_totals()

        for dataset_name in ["train", "validation"]:
            with alive_bar(dataset_totals[dataset_name], title=f"{dataset_name} data", title_length=15) as bar:
                for edge_idx, edge_name in enumerate(self.to_sample):
                    sample_config = getattr(self, dataset_name)[edge_idx]
                    bar.text(f"current model={edge_name}")

                    working_dir = self.root / dataset_name / edge_name

                    if self.write_policy == 'error' and working_dir.exists():
                        raise RoutineError(f"Dataset already exists at {working_dir} and policy='error'")

                    working_dir.mkdir(parents=True, exist_ok=True)

                    model = self.graph.edges[edge_name]
                    sample_inputs = eqx.filter_jit(eqx.filter_vmap(model.sample_inputs))
                    solve = eqx.filter_jit(eqx.filter_vmap(model.solve)) if hasattr(model, "solve") else None
                    sample_outputs = _build_batched_sample_outputs(model)
                    evaluate = (eqx.filter_jit(eqx.filter_vmap(model.evaluate, in_axes=(None, 0))) 
                                if hasattr(model, "evaluate") else None)

                    input_batch: list[tuple[Key, Path]] = []

                    for input_key, input_dir in gen_keys(
                        sample_config.input_samples, sample_config.input_seed, path=working_dir
                    ):
                        if len(input_batch) < self.batch_size:
                            input_batch.append((input_key, input_dir))
                            continue

                        _process_input_batch(
                            input_batch, sample_inputs, solve, sample_outputs, evaluate, sample_config, bar
                        )
                        input_batch.append((input_key, input_dir))

                    _process_input_batch(
                        input_batch, sample_inputs, solve, sample_outputs, evaluate, sample_config, bar
                    )
        return 0
    
    def _generate_data_serial(self):
        """
        Generate training and validation data for a FunctionGraph, in serial. See `generate_data_batch`.
        """

        dataset_totals = self._get_dataset_totals()

        for dataset_name in ["train", "validation"]:
            with alive_bar(dataset_totals[dataset_name], title=f"{dataset_name} data", title_length=15) as bar:
                for edge_idx, edge_name in enumerate(self.to_sample):
                    sample_config = getattr(self, dataset_name)[edge_idx]
                    bar.text(f"current model={edge_name}")

                    working_dir = self.root / dataset_name / edge_name

                    if self.write_policy == "error" and working_dir.exists():
                        raise RoutineError(f"Dataset already exists at {working_dir} and policy='error'")

                    working_dir.mkdir(parents=True, exist_ok=True)

                    model = self.graph.edges[edge_name]
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

                    for input_key, input_dir in gen_keys(
                        sample_config.input_samples, sample_config.input_seed, path=working_dir
                    ):
                        input_path = input_dir / "input.h5"
                        solution_path = input_dir / "solution.h5"

                        if self.write_policy == "reuse" and input_path.exists():
                            if self.format != "h5":
                                raise RoutineError(f"Save format '{self.format}' not recognized")
                            one_input = load_h5({}, input_path, jax=True)
                            one_solution = load_h5({}, solution_path, jax=True) if solution_path.exists() else None
                        else:
                            one_input = sample_inputs(input_key)
                            one_solution = solve(one_input) if solve is not None else None

                            if self.format == "h5":
                                save_h5(one_input, input_path, mode="w")
                                if one_solution is not None:
                                    save_h5(one_solution, solution_path, mode="w")
                            else:
                                raise RoutineError(f"Save format '{self.format}' not recognized")

                        skip = "existing" if self.write_policy == "reuse" else None
                        for output_key, output_dir in gen_keys(
                            sample_config.outputs_per_input, sample_config.output_seed, path=input_dir, skip=skip
                        ):
                            one_output = sample_outputs(output_key, one_input, one_solution)
                            one_residual = evaluate(one_input, one_output) if evaluate is not None else None

                            if self.format == "h5":
                                save_h5(one_output, output_dir / "output.h5", mode="w")
                                if one_residual is not None:
                                    save_h5(one_residual, output_dir / "residual.h5", mode="w")
                            else:
                                raise RoutineError(f"Save format '{self.format}' not recognized")

                        bar()
        return 0
    