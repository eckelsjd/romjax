"""Profile JAX tracing and compilation for Poisson solution-loss training.

Run with diagnostic flags, for example:

    ROMJAX_PROFILE=1 \
    ROMJAX_PROFILE_DIR=/tmp/romjax-poisson-profile/trace \
    ROMJAX_PROFILE_LABEL=poisson-solution-loss \
    ROMJAX_PROFILE_HOST_TRACER_LEVEL=3 \
    ROMJAX_PROFILE_DEVICE_TRACER_LEVEL=1 \
    ROMJAX_PROFILE_PYTHON_TRACER_LEVEL=1 \
    JAX_LOG_COMPILES=1 \
    JAX_EXPLAIN_CACHE_MISSES=1 \
    TF_CPP_MIN_LOG_LEVEL=0 \
    JAX_DUMP_IR_TO=/tmp/romjax-poisson-profile/ir \
    JAX_DUMP_IR_MODES=eqn_count_pprof \
    uv run python demo/profile_poisson_solution_loss.py 2>&1 \
      | tee /tmp/romjax-poisson-profile/compile.log
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx

from romjax.graph import FunctionGraph
from romjax.model import FilterModel, eqx_evaluate
from romjax.nn import LinearProjection
from romjax.pde import ImplicitIterativeGalerkin, IterativeSolver
from romjax.poisson import Poisson2D
from romjax.train import BatchLoader, DiagnosticsConfig, GraphLoss, TerminationConfig, Train


def _solver(max_steps: int) -> IterativeSolver:
    """Build the Optimistix solver used by both full and Galerkin Poisson solves."""
    return IterativeSolver(
        solver=optx.Newton(rtol=1.0, atol=5.0e-4),
        max_steps=max_steps,
        adjoint=optx.ImplicitAdjoint(),
        throw=False,
    )


def _projection(latent: int, dof: int) -> LinearProjection:
    """Create a deterministic near-orthonormal row projection."""
    matrix = np.zeros((latent, dof), dtype=np.float32)
    matrix[:, :latent] = np.eye(latent, dtype=np.float32)
    matrix[0, 0] = 0.9
    return LinearProjection(matrix=matrix)


def _build_graph(grid_size: int, solver_steps: int) -> FunctionGraph:
    """Build the reduced Poisson graph used by the solution-loss path."""
    solver = _solver(solver_steps)
    grid = {"shape": [grid_size, grid_size], "bounds": [[0.0, 1.0], [0.0, 1.0]]}

    poisson = Poisson2D(
        source="hf_coord",
        target="hf_res",
        name="poisson",
        grid=grid,
        solver=solver,
        forcing={"callable": "constant", "inputs_default": {"const": -1.0}},
        conductivity={"callable": "constant", "inputs_default": {"const": 1.0}},
    )
    coordinate_transform = FilterModel(
        source="hf_coord",
        target="lf_coord",
        name="coordinate transform",
        filters=[
            {
                "forward": {
                    "callable": eqx_evaluate,
                    "input_routes": [["outputs"]],
                    "output_routes": [["outputs"]],
                    "opts": {"gather": "flat", "method": "reduce"},
                },
                "backward": {
                    "callable": eqx_evaluate,
                    "input_routes": [["outputs"]],
                    "opts": {"gather": "flat", "scatter": "flat", "method": "reconstruct"},
                },
            },
            {"forward": {"input_routes": [["inputs"]]}, "backward": {"input_routes": [["inputs"]]}},
        ],
    )
    residual_transform = FilterModel(
        source="hf_res",
        target="lf_res",
        name="residual transform",
        filters=[
            {
                "forward": {
                    "callable": eqx_evaluate,
                    "input_routes": [["residuals"]],
                    "output_routes": [["residuals"]],
                    "opts": {"gather": "flat", "method": "reduce"},
                },
                "backward": {
                    "callable": eqx_evaluate,
                    "input_routes": [["residuals"]],
                    "opts": {"gather": "flat", "scatter": "flat", "method": "reconstruct"},
                },
            },
            {"forward": {"input_routes": [["inputs"]]}, "backward": {"input_routes": [["inputs"]]}},
        ],
    )
    galerkin = ImplicitIterativeGalerkin(
        source="lf_coord",
        target="lf_res",
        name="galerkin",
        path=["coordinate transform", "poisson", "residual transform"],
        solver=solver,
    )

    return FunctionGraph(
        nodes=[
            {"name": "hf_coord", "ignore": "inputs", "error_op": "sum-square"},
            {"name": "hf_res", "ignore": "inputs", "error_op": "sum-square"},
            {"name": "lf_coord", "ignore": "inputs", "error_op": "sum-square"},
            {"name": "lf_res", "ignore": "inputs", "error_op": "sum-square"},
        ],
        edges=[poisson, coordinate_transform, residual_transform, galerkin],
    )


def _sample(poisson: Poisson2D) -> dict[str, Any]:
    """Create one fixed-shape solution-only training sample."""
    inputs = {}
    residuals = {"phi_residual": jnp.zeros(poisson.grid.shape, dtype=jnp.float32)}
    outputs = poisson.solve(inputs=inputs, residuals=residuals)
    return {"inputs": inputs, "residuals": residuals, "outputs": outputs}


def _tree_summary(tree: Any) -> Any:
    """Return a JSON-serializable shape/dtype summary for a PyTree."""
    if isinstance(tree, Mapping):
        return {str(key): _tree_summary(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [_tree_summary(value) for value in tree]
    if eqx.is_array(tree):
        return {"shape": list(tree.shape), "dtype": str(tree.dtype)}
    return type(tree).__name__


def build_train(args: argparse.Namespace) -> Train:
    """Construct the minimal Train routine."""
    graph = _build_graph(args.grid_size, args.solver_steps)
    poisson = graph.edges["poisson"]
    dof = int(np.prod(poisson.grid.shape))
    params = {
        "coordinate transform": {"call_args": _projection(args.latent_dim, dof)},
        "residual transform": {"call_args": _projection(args.latent_dim, dof)},
    }
    sample = _sample(poisson)
    batch = {"poisson": [sample for _ in range(args.batch_size)]}

    # print("Batch tree summary:")
    # print(json.dumps(_tree_summary(batch), indent=2, sort_keys=True))
    # print("Parameter tree summary:")
    # print(json.dumps(_tree_summary(params), indent=2, sort_keys=True))

    return Train(
        loss=GraphLoss(
            terms=[
                {
                    "name": "solution",
                    "term": {
                        "callable": "solution",
                        "path": ["residual transform", "galerkin", "coordinate transform"],
                        "template_paths": ["outputs"],
                        "aux_paths": [["coordinate transform", "backward", "cached_states", 0, "template"]],
                    },
                    "dataset": "poisson",
                },
                {
                    "name": "orth",
                    "term": {"callable": "orthogonal", "ref": ["coordinate transform", "call_args", "matrix"]},
                    "batch_reduce": None,
                },
            ],
            balancing={
                "kind": "ema_log",
                "update_interval": 10,
                "decay": 0.99,
                "normalize": False,
                "min_scale": 1e-13,
                "max_scale": 1e13,
                "eps": 1e-13,
            },
            graph=graph,
        ),
        init_params=params,
        optimizer=optax.adam(args.learning_rate),
        dataloader=BatchLoader(data=[batch] * (args.steps + 1), batch_size=1, max_epochs=1),
        termination=TerminationConfig(max_steps=args.steps, max_runtime=120),
        diagnostics=DiagnosticsConfig(show_progress=False, log_interval=1, plot_interval=None, test_interval=None),
        graph=graph,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=12)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--solver-steps", type=int, default=25)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    """Run the profiling demo."""
    args = parse_args()
    train = build_train(args)
    params = train()
    # print("Final parameter tree summary:")
    # print(json.dumps(_tree_summary(params), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
