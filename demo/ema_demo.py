"""Demo of EMA-balanced GraphLoss training with raw and scaled term plots.

Run from the repository root with:

    uv run python demo/ema_demo.py

The demo writes ``loss.csv``, per-term CSV diagnostics, checkpoints, and ``loss.pdf`` under the
configured output root.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from jaxtyping import PyTree
from loguru import logger

from romjax.graph import FunctionGraph
from romjax.train import DiagnosticsConfig, GraphLoss, TerminationConfig, Train


logger.disable("romjax")


class RepeatBatchLoader:
    """Repeat one in-memory batch forever."""

    def __init__(self, batch: Any):
        self.batch = batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.batch


def small_task(params: PyTree, single_data: dict[str, jax.Array], graph: FunctionGraph) -> jax.Array:
    """Small-scale task that mostly trains ``theta``."""
    del graph
    theta_error = params["theta"] - single_data["theta_target"]
    phi_error = params["phi"] - single_data["phi_target"]
    return jnp.square(theta_error) + 0.05 * jnp.square(phi_error)


def large_task(params: PyTree, single_data: dict[str, jax.Array], graph: FunctionGraph) -> jax.Array:
    """Large-scale task that mostly trains ``phi`` with a small nonlinear coupling to ``theta``."""
    del graph
    theta_error = params["theta"] - single_data["theta_target"]
    phi_error = params["phi"] - single_data["phi_target"]
    return 100.0 * (jnp.square(phi_error) + 0.05 * jnp.square(theta_error) ** 2)


def build_train(root: Path, *, steps: int, live_plot: bool) -> Train:
    """Build the EMA-balanced toy training routine.

    :param root: output directory for checkpoints, CSV histories, and plots
    :param steps: number of optimizer steps
    :param live_plot: whether to update the matplotlib figure interactively
    :return: configured training routine
    """
    batch = {"toy": [{"theta_target": jnp.asarray(2.0), "phi_target": jnp.asarray(-1.0)}]}

    loss = GraphLoss(
        terms=[
            {"name": "small_theta_task", "term": small_task, "dataset": "toy"},
            {"name": "large_phi_task_100x", "term": large_task, "dataset": "toy"},
        ],
        balancing={
            "kind": "none",
            "decay": 0.7,
            "target": 1.0,
            "min_scale": 1e-8,
            "max_scale": 1e8,
            "eps": 1e-12,
            "bootstrap": True,
            "normalize": "mean",
            "update_interval": 1000
        },
    )

    def validation_loss(params: PyTree) -> jax.Array:
        theta_error = params["theta"] - 2.0
        phi_error = params["phi"] + 1.0
        small = jnp.square(theta_error) + 0.05 * jnp.square(phi_error)
        large_normalized = jnp.square(phi_error) + 0.05 * jnp.square(theta_error) ** 2
        return small + large_normalized

    return Train(
        loss=loss,
        init_params={"theta": jnp.asarray(-3.0), "phi": jnp.asarray(3.0)},
        optimizer=optax.adam(0.1),
        test=validation_loss,
        dataloader=RepeatBatchLoader(batch),
        termination=TerminationConfig(max_steps=steps),
        diagnostics=DiagnosticsConfig(
            plot_interval=50,
            test_interval=50,
            live_plot=live_plot,
            loss_plot={"opts": {"title": "EMA-balanced total loss"}},
            test_plot={"opts": {"title": "Unscaled validation loss"}},
            raw_terms_plot={"enabled": True, "spec": {"opts": {"title": "Raw GraphLoss terms"}}},
            scaled_terms_plot={"enabled": True, "spec": {"opts": {"title": "Scaled GraphLoss terms"}}},
        ),
        root=root,
        write_policy="overwrite",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("demo/ema_balancing"),
        help="Directory where training diagnostics will be written.",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of optimizer steps.")
    parser.add_argument("--live-plot", action="store_true", help="Update the training plot interactively.")
    parser.add_argument("--show", action="store_true", help="Show the final matplotlib figure window.")
    return parser.parse_args()


def main() -> None:
    """Run the EMA balancing demo."""
    args = parse_args()
    train = build_train(args.root.resolve(), steps=args.steps, live_plot=args.live_plot)
    params = train()

    print(f"Final theta: {float(params['theta']):.6f}")
    print(f"Final phi: {float(params['phi']):.6f}")
    print(f"Final unscaled validation loss: {float(train.test(params)):.6e}")
    print(f"Diagnostics written to: {args.root.resolve()}")
    print("Inspect loss_terms_raw.csv, loss_terms_scaled.csv, loss_term_scales.csv, and loss.pdf.")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
