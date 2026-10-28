"""Compare staged Python dispatch, diagnostic outer JIT, and structured family batching.

Run: JAX_PLATFORMS=cpu uv run python -m demo.benchmark_implicit_batching
Use --families 17 --batch-size 8 to include a task tail. This synthetic periodic
reaction-diffusion workload needs no external artifacts and is not the Poisson
experiment from the handoff. Generation timings include HDF5 writes and host transfers.
Measurements, plots, and datasets remain in a dated run folder under
demo/implicit_batching_results (override the parent with --output-dir).
"""

import argparse
import json
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jaxtyping import Key, PyTree
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from pydantic import PositiveInt

from romjax.data_gen import GenDataBenchmark, GenImplicitModel
from romjax.graph import Edge, FunctionGraph
from romjax.model import ImplicitSampleable
from romjax.storage import H5Store
from romjax.tree import pytree_stack


class DiffusionEdge(Edge, ImplicitSampleable):
    """Small periodic reaction-diffusion stencil with a fixed Jacobi solve.

    :param size: number of grid points along each spatial axis
    :param iterations: Jacobi iterations, held fixed for reproducible timing
    """

    size: PositiveInt = 16
    iterations: PositiveInt = 20

    def forward(self, inputs: PyTree) -> PyTree:
        """Return the approximate solution for one input."""
        return self.solve(inputs)

    def backward(self, outputs: PyTree) -> PyTree:
        """Reject inversion of this demonstration edge."""
        raise NotImplementedError

    @staticmethod
    def _neighbors(field: jax.Array) -> jax.Array:
        return sum(jnp.roll(field, shift, axis) for axis in (0, 1) for shift in (-1, 1))

    def sample_inputs(self, key: Key) -> PyTree:
        """Sample a forcing field and a positive diffusion coefficient."""
        forcing_key, coefficient_key = jax.random.split(key)
        return {
            "f": jax.random.normal(forcing_key, (self.size, self.size)),
            "a": jax.random.uniform(coefficient_key, minval=0.1, maxval=0.5),
        }

    def solve(self, inputs: PyTree) -> PyTree:
        """Apply a fixed number of Jacobi steps to the periodic stencil."""
        a, f = inputs["a"], inputs["f"]
        field = jax.lax.fori_loop(
            0, self.iterations, lambda _, u: (f + a * self._neighbors(u)) / (1 + 4 * a), jnp.zeros_like(f)
        )
        return {"u": field}

    def sample_outputs(self, key: Key, inputs: PyTree | None = None, solution: PyTree | None = None) -> PyTree:
        """Sample a candidate field conditioned on the forcing or available solution."""
        base = inputs["f"] if solution is None else solution["u"]
        return {"u": base + 0.1 * jax.random.normal(key, base.shape)}

    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate the discrete reaction-diffusion residual."""
        u = outputs["u"]
        return {"r": u + inputs["a"] * (4 * u - self._neighbors(u)) - inputs["f"]}


def _legacy_workload(generator: GenImplicitModel, path: Path) -> Callable:
    solve, evaluate = not generator.skip_solve, not generator.skip_evaluate
    inputs_kernel = generator._kernel(path, "input", (True, solve, solve and evaluate), True)
    outputs_kernel = generator._kernel(path, "output", (False, True, evaluate), True)

    def workload(keys, input_values, output_keys, output_values):
        inputs, solutions, residuals = inputs_kernel(keys, *input_values)
        result = [inputs, solutions, residuals]
        if output_keys is not None:
            for i in range(len(keys)):
                one_input = jax.tree.map(lambda x: x[i], inputs)
                one_solution = jax.tree.map(lambda x: x[i], solutions)
                result.extend(outputs_kernel(output_keys[i], *output_values, one_input, one_solution))
        return result

    return workload


def _measure(function: Callable, batches: tuple, warmup: int, repeats: int) -> tuple[dict, list]:
    def run():
        return [jax.block_until_ready(function(*args)) for args in batches]

    start = perf_counter()
    result = run()
    cold = perf_counter() - start
    for _ in range(warmup):
        run()
    times = []
    for _ in range(repeats):
        start = perf_counter()
        run()
        times.append(perf_counter() - start)
    return {"cold_seconds": cold, "median_seconds": float(np.median(times)), "seconds": times}, result


def _assert_equal(left: PyTree, right: PyTree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


def _generation(generator: GenImplicitModel, root: Path, warmup: int, repeats: int, staged: bool) -> dict:
    # Keep generator/compiled caches alive, as in one prepared generation worker.
    # Each pass still plans, executes, transfers, and writes to a fresh dataset.
    if staged:
        generator._compute = generator._compute_staged
    counter = 0

    def run():
        nonlocal counter
        store = H5Store(root / str(counter) / "sample", "implicit")
        counter += 1
        with store.locked():
            store.configure_key_scheme(generator.input_seed, generator.output_seed, generator.mix_output_seed)
            for start in range(0, generator.input_samples, generator.batch_size):
                plans = generator._plans(
                    store, range(start, min(start + generator.batch_size, generator.input_samples)), "overwrite"
                )
                result = generator._compute(store.root, plans)
                if result.fatal:
                    raise RuntimeError(result.fatal)
                store.write(result.records)

    measurements, _ = _measure(lambda: run(), ((),), warmup, repeats)
    return measurements


def _plot_results(reports: Sequence[dict[str, Any]], output_dir: Path, config: dict[str, Any]) -> None:
    """Plot warmed medians/ranges and ratios to the matching staged baseline."""
    widths = list(dict.fromkeys(row["width"] for row in reports))
    positions = np.arange(len(widths))
    labels = ["0 (all)" if width == 0 else str(width) for width in widths]
    lookup = {(row["mode"], row["width"], row["path"]): row for row in reports}
    styles = {
        "python": ("#0072B2", "o"),
        "jit_python": ("#E69F00", "s"),
        "families": ("#009E73", "^"),
        "generation_staged": ("#0072B2", "o"),
        "generation_families": ("#CC79A7", "D"),
    }
    panels = [
        ("Numerical workload", [("python", "Python loop"), ("jit_python", "JIT of loop"), ("families", "Family map")]),
        ("Generation including HDF5", [("generation_staged", "Staged"), ("generation_families", "Family map")]),
    ]
    details = (
        f"{config['families']} families · {config['outputs']} sampled outputs/family in residual case · "
        f"{config['size']}×{config['size']} grid · task size {config['batch_size']}"
    )
    timing = Figure(figsize=(11, 7), layout="constrained")
    FigureCanvasAgg(timing)
    axes = timing.subplots(2, 2)
    for row, mode in enumerate(("solution", "residual")):
        for column, (title, series) in enumerate(panels):
            ax = axes[row, column]
            for i, (key, label) in enumerate(series):
                samples = np.asarray([lookup[mode, width, key]["seconds"] for width in widths]) * 1000
                medians = np.median(samples, axis=1)
                errors = np.vstack((medians - samples.min(axis=1), samples.max(axis=1) - medians))
                color, marker = styles[key]
                ax.errorbar(
                    positions + (i - (len(series) - 1) / 2) * 0.16,
                    medians,
                    yerr=errors,
                    fmt=marker,
                    color=color,
                    capsize=4,
                    markersize=7,
                    label=label,
                )
            ax.set(
                title=f"{mode.capitalize()} — {title}",
                ylabel="Full-pass time (ms, log scale)",
                yscale="log",
                xlabel="map_batch_size (families)",
                xticks=positions,
                xticklabels=labels,
            )
            ax.grid(axis="y", alpha=0.25)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.legend(fontsize=9)
    timing.suptitle(
        f"Warmed timing comparison\n{details}\n"
        f"Dots: medians; whiskers: observed min–max across {config['repeats']} passes (not confidence intervals)",
        fontsize=11,
    )
    timing.supxlabel(
        "Generation includes planning, compute, transfers, and HDF5; executor scheduling is excluded.", fontsize=9
    )

    speedup = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(speedup)
    for ax, mode in zip(speedup.subplots(1, 2), ("solution", "residual"), strict=True):
        comparisons = [
            ("jit_python", "python", "Numerical: JIT of loop"),
            ("families", "python", "Numerical: family map"),
            ("generation_families", "generation_staged", "Generation: family map"),
        ]
        for i, (key, baseline, label) in enumerate(comparisons):
            ratios = [
                lookup[mode, width, baseline]["median_seconds"] / lookup[mode, width, key]["median_seconds"]
                for width in widths
            ]
            color, marker = styles[key]
            ax.plot(
                positions + (i - 1) * 0.16,
                ratios,
                marker=marker,
                linestyle="none",
                color=color,
                markersize=8,
                label=label,
            )
        ax.axhline(1, color="0.4", linestyle="--", linewidth=1)
        ax.set(
            title=mode.capitalize(),
            ylabel="Staged median / method median (×)",
            xlabel="map_batch_size (families)",
            xticks=positions,
            xticklabels=labels,
            ylim=(0, None),
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9)
    speedup.suptitle(f"Speedup relative to each path's staged baseline\n{details}", fontsize=11)
    speedup.supxlabel(
        "Above 1× is faster; below 1× is slower. Ratios of warmed medians; uncertainty is not estimated.", fontsize=9
    )
    for name, figure in (("timings", timing), ("speedup", speedup)):
        figure.savefig(output_dir / f"{name}.png", dpi=160)
        figure.savefig(output_dir / f"{name}.svg")


def main() -> None:
    """Run synchronized numerical and persistence comparisons, emitting JSON lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("families", 8),
        ("outputs", 3),
        ("batch-size", 4),
        ("size", 16),
        ("warmup", 1),
        ("repeats", 5),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--widths", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "implicit_batching_results",
        help="parent directory for dated runs containing plots, raw timings, and generated datasets",
    )
    args = parser.parse_args()
    if (
        min(args.families, args.outputs, args.batch_size, args.size, args.repeats) < 1
        or min(args.widths) < 0
        or args.warmup < 0
    ):
        parser.error("sizes/repeats must be positive; widths/warmup must be nonnegative")
    directory = args.output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    directory.mkdir(parents=True)
    metadata = {
        "devices": [{"name": str(device), "kind": device.device_kind} for device in jax.devices()],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "x64": jax.config.jax_enable_x64,
        "environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "CPU_THREADS", "XLA_FLAGS", "JAX_PLATFORMS", "JAX_PLATFORM_NAME")
        },
        "config": vars(args) | {"output_dir": str(args.output_dir)},
        "run_directory": str(directory.resolve()),
        "peak_memory": "not measured",
        "cold_seconds": "first call through each measured path, not isolated compilation; caches may be shared",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    all_reports = []
    with (directory / "timings.jsonl").open("w") as output:

        def record(report):
            line = json.dumps(report)
            output.write(line + "\n")
            output.flush()
            all_reports.append(report)
            print(line, flush=True)

        for mode in ("solution", "residual"):
            for width in args.widths:
                generator = GenImplicitModel(
                    graph=FunctionGraph(
                        edges={"sample": DiffusionEdge(source="inputs", target="sample", size=args.size)}
                    ),
                    input_samples=args.families,
                    outputs_per_input=0 if mode == "solution" else args.outputs,
                    input_seed=2,
                    output_seed=3,
                    batch_size=args.batch_size,
                    map_batch_size=width,
                    skip_solve=mode == "residual",
                    skip_evaluate=mode == "solution",
                )
                benchmark = GenDataBenchmark(generator=generator, samples=args.families, show_progress=False)
                path = Path(directory) / "sample"
                structured, batches = benchmark._workload(path)
                legacy = _legacy_workload(generator, path)
                results, reports = {}, []
                for label, function in (
                    ("python", legacy),
                    ("jit_python", eqx.filter_jit(legacy)),
                    ("families", structured),
                ):
                    measurements, result = _measure(function, batches, args.warmup, args.repeats)
                    results[label] = result
                    reports.append(
                        {
                            "mode": mode,
                            "width": width,
                            "path": label,
                            **measurements,
                            "families_per_second": args.families / measurements["median_seconds"],
                            "sampled_outputs_per_second": args.families
                            * generator.outputs_per_input
                            / measurements["median_seconds"],
                        }
                    )
                _assert_equal(results["python"], results["jit_python"])
                for old, new in zip(results["python"], results["families"], strict=True):
                    output_rows = [tuple(old[i : i + 3]) for i in range(3, len(old), 3)]
                    _assert_equal((tuple(old[:3]), pytree_stack(output_rows) if output_rows else None), new)
                for report in reports:
                    record(report)
                for staged in (True, False):
                    root = directory / "datasets" / f"{mode}-{width}-{'staged' if staged else 'families'}"
                    measurements = _generation(generator.model_copy(), root, args.warmup, args.repeats, staged)
                    record(
                        {
                            "mode": mode,
                            "width": width,
                            "path": "generation_staged" if staged else "generation_families",
                            "includes": "planning, compute, transfer, HDF5; no executor scheduling",
                            **measurements,
                        }
                    )
    _plot_results(all_reports, directory, metadata["config"])
    print(json.dumps({"plots": [str(directory / f"{name}.png") for name in ("timings", "speedup")]}), flush=True)


if __name__ == "__main__":
    main()
