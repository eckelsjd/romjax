"""ROM CLI for building reduced-order models with romjax."""

from pathlib import Path
import inspect
import shutil
import sys
import argparse
from typing import Callable

import jax
from alive_progress import alive_bar

from romjax import YamlLoader
from romjax.config import GenDataConfig
from romjax.rng import gen_keys
from romjax.utils import pytree_at, save_h5


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


def generate_data(config: GenDataConfig):
    """Generate training and validation data."""

    total_train_samples = 0
    total_validation_samples = 0
    for edge_idx, edge_name in enumerate(config.to_sample):
        total_train_samples += getattr(config, "train")[edge_idx].input_samples
        total_validation_samples += getattr(config, "validation")[edge_idx].input_samples

    def _save_outputs(
        model: object,
        one_input: object,
        one_solution: object,
        output_keys: tuple[object, ...],
        output_paths: tuple[Path, ...],
        evaluate: Callable | None,
        config: GenDataConfig,
    ) -> None:
        _kwargs = _get_kwargs(model.sample_outputs)

        if all(arg in _kwargs for arg in ["inputs", "solution"]):
            sample_outputs = jax.jit(
                jax.vmap(lambda key: model.sample_outputs(key, inputs=one_input, solution=one_solution))
            )
        elif "solution" in _kwargs:
            sample_outputs = jax.jit(jax.vmap(lambda key: model.sample_outputs(key, solution=one_solution)))
        else:
            sample_outputs = jax.jit(jax.vmap(model.sample_outputs))

        outputs = sample_outputs(output_keys)
        residuals = evaluate(one_input, outputs) if evaluate is not None else None

        for j, output_path in enumerate(output_paths):
            one_output = pytree_at(outputs, j)
            one_residual = pytree_at(residuals, j) if residuals is not None else None

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
        evaluate: Callable | None,
        sample_config: object,
        skip: str | None,
        config: GenDataConfig,
        bar: Callable,
    ) -> None:
        if not input_batch:
            return

        input_keys, input_paths = zip(*input_batch)
        inputs = sample_inputs(input_keys)
        solutions = solve(inputs) if solve is not None else None

        for i, input_path in enumerate(input_paths):
            one_input = pytree_at(inputs, i)
            one_solution = pytree_at(solutions, i) if solutions is not None else None

            if config.format == "h5":
                save_h5(one_input, input_path / "input.h5", mode="w")
                if one_solution is not None:
                    save_h5(one_solution, input_path / "solution.h5", mode="w")
            else:
                raise RomWorkflowError(f"Save format '{config.format}' not recognized")

            output_batch: list[tuple[object, Path]] = []

            for output_key, output_dir in gen_keys(
                sample_config.outputs_per_input, sample_config.output_seed, path=input_path, skip=skip
            ):
                if len(output_batch) < config.batch_size:
                    output_batch.append((output_key, output_dir))
                    continue

                output_keys, output_paths = zip(*output_batch)
                _save_outputs(model, one_input, one_solution, output_keys, output_paths, evaluate, config)
                output_batch.clear()
                output_batch.append((output_key, output_dir))

            if output_batch:
                output_keys, output_paths = zip(*output_batch)
                _save_outputs(model, one_input, one_solution, output_keys, output_paths, evaluate, config)

            bar()

        input_batch.clear()

    dataset_totals = {
        "train": total_train_samples,
        "validation": total_validation_samples,
    }

    for dataset_name in ["train", "validation"]:
        print(f"Generating {dataset_name} data...", flush=True)
        with alive_bar(dataset_totals[dataset_name], title=f"{dataset_name} data") as bar:
            for edge_idx, edge_name in enumerate(config.to_sample):
                sample_config = getattr(config, dataset_name)[edge_idx]
                bar.text(f"model={edge_name}")

                working_dir = config.root / dataset_name / edge_name

                if config.dataset_policy == 'error' and working_dir.exists():
                    raise RomWorkflowError(f"Dataset already exists at {working_dir} and policy='error'")

                working_dir.mkdir(parents=True, exist_ok=True)

                model = config.graph.edges[edge_name]
                sample_inputs = jax.jit(jax.vmap(model.sample_inputs))
                solve = jax.jit(jax.vmap(model.solve)) if hasattr(model, "solve") else None
                evaluate = jax.jit(jax.vmap(model.evaluate, in_axes=(None, 0))) if hasattr(model, "evaluate") else None

                skip = 'existing' if config.dataset_policy == 'reuse' else None
                input_batch: list[tuple[int, object, Path]] = []

                for input_key, input_dir in gen_keys(
                    sample_config.input_samples, sample_config.input_seed, path=working_dir, skip=skip
                ):
                    if len(input_batch) < config.batch_size:
                        input_batch.append((input_key, input_dir))
                        continue

                    _process_input_batch(
                        input_batch=input_batch,
                        model=model,
                        sample_inputs=sample_inputs,
                        solve=solve,
                        evaluate=evaluate,
                        sample_config=sample_config,
                        skip=skip,
                        config=config,
                        bar=bar,
                    )
                    input_batch.append((input_key, input_dir))

                _process_input_batch(
                    input_batch=input_batch,
                    model=model,
                    sample_inputs=sample_inputs,
                    solve=solve,
                    evaluate=evaluate,
                    sample_config=sample_config,
                    skip=skip,
                    config=config,
                    bar=bar,
                )


def build_parser() -> argparse.ArgumentParser:
    """Build the rom CLI argument parser."""
    parser = argparse.ArgumentParser(description="romjax rom building workflow")
    subparsers = parser.add_subparsers(dest="command", required = True)
    
    gen = subparsers.add_parser("generate", help="Generate training and valdiation data")
    gen.add_argument("config")

    return parser


def cli(argv: list[str] | None = None, repo_root: Path | None = None) -> int:
    """Run the rom CLI.

    :param argv: CLI arguments excluding the interpreter name
    :param repo_root: optional repository root override, useful for tests
    :return: process exit code
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not Path(args.config).exists():
        raise ValueError(f"Config file '{args.config}' not found")
    
    config = YamlLoader.load(args.config)

    if hasattr(config, "root"):
        shutil.copy(Path(args.config), config.root / Path(args.config).name)

    try:
        if args.command == "generate":
            generate_data(config)
            return 0
    except RomWorkflowError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    
    parser.error(f"Unhandled command: {args.command}")
    return 2


def main():
    """Console-script entrypoint for the rom CLI."""
    raise SystemExit(cli())
