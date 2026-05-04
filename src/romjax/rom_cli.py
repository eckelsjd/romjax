"""ROM CLI for building reduced-order models with romjax."""

import argparse
import inspect
import shutil
import sys
from pathlib import Path
from typing import Callable

import jax
import equinox as eqx
import jax.numpy as jnp
from alive_progress import alive_bar
from jaxtyping import Key

from romjax import YamlLoader
from romjax.config import GenDataConfig
from romjax.rng import gen_keys
from romjax.utils import load_h5, save_h5
from romjax.tree import pytree_iter


class RomWorkflowError(RuntimeError):
    """Raised when rom workflow encounters invalid local state."""


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


def generate_data_batch(config: GenDataConfig):
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

    :param config: See `GenDataConfig`
    """

    total_train_samples = 0
    total_validation_samples = 0
    for edge_idx, edge_name in enumerate(config.to_sample):
        total_train_samples += getattr(config, "train")[edge_idx].input_samples
        total_validation_samples += getattr(config, "validation")[edge_idx].input_samples

    def _save_outputs(
        sample_outputs: Callable,
        one_input: object,
        one_solution: object,
        output_keys: tuple[object, ...],
        output_paths: tuple[Path, ...],
        evaluate: Callable | None,
        config: GenDataConfig,
    ) -> None:
        outputs = sample_outputs(_stack_batch(output_keys), one_input, one_solution)
        residuals = evaluate(one_input, outputs) if evaluate is not None else None
        outputs = jax.device_get(outputs)
        residuals = jax.device_get(residuals) if residuals is not None else None

        residual_iter = pytree_iter(residuals) if residuals is not None else None
        for one_output, output_path in zip(pytree_iter(outputs), output_paths):
            one_residual = next(residual_iter) if residual_iter is not None else None

            if config.format == "h5":
                save_h5(one_output, output_path / "output.h5", mode="w")
                if one_residual is not None:
                    save_h5(one_residual, output_path / "residual.h5", mode="w")
            else:
                raise RomWorkflowError(f"Save format '{config.format}' not recognized")

    def _process_input_batch(
        *,
        input_batch: list[tuple[object, Path]],
        model: object,
        sample_inputs: Callable,
        solve: Callable | None,
        sample_outputs: Callable,
        evaluate: Callable | None,
        sample_config: object,
        config: GenDataConfig,
        bar: Callable,
    ) -> None:
        if not input_batch:
            return

        # Only sample/solve missing inputs for policy=reuse
        inputs_by_index: dict[int, object] = {}
        solutions_by_index: dict[int, object] = {}

        missing_indices = range(len(input_batch))
        if config.dataset_policy == "reuse":
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

                if config.format == "h5":
                    save_h5(one_input, input_path / "input.h5", mode="w")
                    if one_solution is not None:
                        save_h5(one_solution, input_path / "solution.h5", mode="w")
                else:
                    raise RomWorkflowError(f"Save format '{config.format}' not recognized")

            else:
                if config.format != "h5":
                    raise RomWorkflowError(f"Save format '{config.format}' not recognized")
                one_input = load_h5({}, input_path / "input.h5", jax=True)
                solution_path = input_path / "solution.h5"
                one_solution = load_h5({}, solution_path, jax=True) if solution_path.exists() else None

            output_batch: list[tuple[Key, Path]] = []
            skip = 'existing' if config.dataset_policy == 'reuse' else None

            for output_key, output_dir in gen_keys(
                sample_config.outputs_per_input, sample_config.output_seed, path=input_path, skip=skip
            ):
                if len(output_batch) < config.batch_size:
                    output_batch.append((output_key, output_dir))
                    continue

                output_keys, output_paths = zip(*output_batch)
                _save_outputs(sample_outputs, one_input, one_solution, output_keys, output_paths, evaluate, config)
                output_batch.clear()
                output_batch.append((output_key, output_dir))

            if output_batch:
                output_keys, output_paths = zip(*output_batch)
                _save_outputs(sample_outputs, one_input, one_solution, output_keys, output_paths, evaluate, config)

            bar()

        input_batch.clear()

    dataset_totals = {
        "train": total_train_samples,
        "validation": total_validation_samples,
    }

    for dataset_name in ["train", "validation"]:
        with alive_bar(dataset_totals[dataset_name], title=f"{dataset_name} data", title_length=15) as bar:
            for edge_idx, edge_name in enumerate(config.to_sample):
                sample_config = getattr(config, dataset_name)[edge_idx]
                bar.text(f"current model={edge_name}")

                working_dir = config.root / dataset_name / edge_name

                if config.dataset_policy == 'error' and working_dir.exists():
                    raise RomWorkflowError(f"Dataset already exists at {working_dir} and policy='error'")

                working_dir.mkdir(parents=True, exist_ok=True)

                model = config.graph.edges[edge_name]
                sample_inputs = eqx.filter_jit(eqx.filter_vmap(model.sample_inputs))
                solve = eqx.filter_jit(eqx.filter_vmap(model.solve)) if hasattr(model, "solve") else None
                sample_outputs = _build_batched_sample_outputs(model)
                evaluate = (eqx.filter_jit(eqx.filter_vmap(model.evaluate, in_axes=(None, 0))) 
                            if hasattr(model, "evaluate") else None)

                input_batch: list[tuple[Key, Path]] = []

                for input_key, input_dir in gen_keys(
                    sample_config.input_samples, sample_config.input_seed, path=working_dir
                ):
                    if len(input_batch) < config.batch_size:
                        input_batch.append((input_key, input_dir))
                        continue

                    _process_input_batch(
                        input_batch=input_batch,
                        model=model,
                        sample_inputs=sample_inputs,
                        solve=solve,
                        sample_outputs=sample_outputs,
                        evaluate=evaluate,
                        sample_config=sample_config,
                        config=config,
                        bar=bar,
                    )
                    input_batch.append((input_key, input_dir))

                _process_input_batch(
                    input_batch=input_batch,
                    model=model,
                    sample_inputs=sample_inputs,
                    solve=solve,
                    sample_outputs=sample_outputs,
                    evaluate=evaluate,
                    sample_config=sample_config,
                    config=config,
                    bar=bar,
                )


def generate_data_serial(config: GenDataConfig):
    """
    Generate training and validation data for a FunctionGraph, in serial. See `generate_data_batch`.
    
    :param config: See `GenDataConfig`
    """

    total_train_samples = 0
    total_validation_samples = 0
    for edge_idx, edge_name in enumerate(config.to_sample):
        total_train_samples += getattr(config, "train")[edge_idx].input_samples
        total_validation_samples += getattr(config, "validation")[edge_idx].input_samples

    dataset_totals = {
        "train": total_train_samples,
        "validation": total_validation_samples,
    }

    for dataset_name in ["train", "validation"]:
        with alive_bar(dataset_totals[dataset_name], title=f"{dataset_name} data", title_length=15) as bar:
            for edge_idx, edge_name in enumerate(config.to_sample):
                sample_config = getattr(config, dataset_name)[edge_idx]
                bar.text(f"current model={edge_name}")

                working_dir = config.root / dataset_name / edge_name

                if config.dataset_policy == "error" and working_dir.exists():
                    raise RomWorkflowError(f"Dataset already exists at {working_dir} and policy='error'")

                working_dir.mkdir(parents=True, exist_ok=True)

                model = config.graph.edges[edge_name]
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

                    if config.dataset_policy == "reuse" and input_path.exists():
                        if config.format != "h5":
                            raise RomWorkflowError(f"Save format '{config.format}' not recognized")
                        one_input = load_h5({}, input_path, jax=True)
                        one_solution = load_h5({}, solution_path, jax=True) if solution_path.exists() else None
                    else:
                        one_input = sample_inputs(input_key)
                        one_solution = solve(one_input) if solve is not None else None

                        if config.format == "h5":
                            save_h5(one_input, input_path, mode="w")
                            if one_solution is not None:
                                save_h5(one_solution, solution_path, mode="w")
                        else:
                            raise RomWorkflowError(f"Save format '{config.format}' not recognized")

                    skip = "existing" if config.dataset_policy == "reuse" else None
                    for output_key, output_dir in gen_keys(
                        sample_config.outputs_per_input, sample_config.output_seed, path=input_dir, skip=skip
                    ):
                        one_output = sample_outputs(output_key, one_input, one_solution)
                        one_residual = evaluate(one_input, one_output) if evaluate is not None else None

                        if config.format == "h5":
                            save_h5(one_output, output_dir / "output.h5", mode="w")
                            if one_residual is not None:
                                save_h5(one_residual, output_dir / "residual.h5", mode="w")
                        else:
                            raise RomWorkflowError(f"Save format '{config.format}' not recognized")

                    bar()


def build_parser() -> argparse.ArgumentParser:
    """Build the rom CLI argument parser."""
    parser = argparse.ArgumentParser(description="romjax reduced-order model building workflow")
    subparsers = parser.add_subparsers(dest="command", required = True)
    
    gen = subparsers.add_parser("generate", help="Generate training and validation data")
    gen.add_argument("config", help="Path to config file")

    return parser


def cli(argv: list[str] | None = None) -> int:
    """Run the rom CLI.

    :param argv: CLI arguments excluding the interpreter name
    :return: process exit code
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:

        if not Path(args.config).exists():
            raise RomWorkflowError(f"Config file '{args.config}' not found")
        
        config = YamlLoader.load(args.config)

        if hasattr(config, "root"):
            shutil.copy(Path(args.config), config.root / Path(args.config).name)

        if args.command == "generate":
            if hasattr(config, "batch_size") and config.batch_size > 1:
                generate_data_batch(config)
            else:
                generate_data_serial(config)
            return 0
        
    except RomWorkflowError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    
    parser.error(f"Unhandled command: {args.command}")
    return 2


def main():
    """Console-script entrypoint for the rom CLI."""
    raise SystemExit(cli())


__all__ = [
    "RomWorkflowError",
    "cli",
    "generate_data_batch",
    "generate_data_serial",
]
