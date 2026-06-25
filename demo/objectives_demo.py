"""Toy comparison of hierarchical objective functions.

This demo illustrates a situation where a more restrictive objective, `J2`, decays smoothly,
`J1` follows that profile in a well-behaved way, and `J0` is more irregular while still remaining
bounded above by `J1`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt


THETA_MIN = 0.0
THETA_MAX = 2.0
N_POINTS = 4000

# Chosen constants for the scaled comparison plot.
C1 = 1.15
C2 = 1.35


def objective_curves(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Construct three nonnegative objectives with the desired hierarchy.

    Parameters
    ----------
    theta
        Free parameter values.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        The objectives `(J0, J1, J2)`.
    """
    distance = theta - 1.0
    envelope = distance**2

    # Smooth reference objective that vanishes at theta = 1 and is symmetric around that point.
    j2 = envelope * (1.0 + 0.08 * jnp.sin(1.6 * theta + 0.3) ** 2)

    # Well-behaved objective that tracks J2 closely while remaining below it.
    j1 = 0.78 * envelope * (1.0 + 0.05 * jnp.sin(2.7 * theta - 0.4) ** 2)

    # Sporadic objective with many local minima, but still bounded by J1.
    j0 = 0.52 * envelope * (
        0.08 + 0.92 * jnp.sin(5.5 * theta + 0.15) ** 2 * jnp.sin(8.9 * theta - 0.7) ** 2
    )

    return j0, j1, j2


def verify_bounds(j0: jnp.ndarray, j1: jnp.ndarray, j2: jnp.ndarray) -> None:
    """Check the intended pointwise ordering on the sampled grid."""
    eps = 1e-12

    if not bool(jnp.all(j0 >= -eps)):
        raise ValueError("J0 must remain nonnegative.")
    if not bool(jnp.all(j1 >= -eps)):
        raise ValueError("J1 must remain nonnegative.")
    if not bool(jnp.all(j2 >= -eps)):
        raise ValueError("J2 must remain nonnegative.")

    if not bool(jnp.all(j0 <= C1 * j1 + eps)):
        raise ValueError("Expected J0 <= C1 * J1 on the plotted domain.")
    if not bool(jnp.all(C1 * j1 <= C2 * j2 + eps)):
        raise ValueError("Expected C1 * J1 <= C2 * J2 on the plotted domain.")


def plot_objectives(output_path: Path) -> None:
    """Generate and save the comparison plot.

    Parameters
    ----------
    output_path
        Destination for the saved figure.
    """
    theta = jnp.linspace(THETA_MIN, THETA_MAX, N_POINTS)
    j0, j1, j2 = objective_curves(theta)
    verify_bounds(j0, j1, j2)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), layout="constrained")
    ax.plot(theta, j0, lw=2.0, label=r"$J_{{GC}}(\theta)$", color="#7f3c8d")
    ax.plot(theta, C1 * j1, lw=2.2, label=rf"$C_1 J_{{ROM}}(\theta)$", color="#11a579")
    ax.plot(theta, C2 * j2, lw=2.4, label=rf"$C_2 J_{{SR}}(\theta)$", color="#3969ac")

    ax.set_yscale("log")
    ax.set_xlim(theta[0], theta[-1])
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("objective value")
    ax.set_title("Bounded objectives converging at $\\theta=1$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax.text(
        0.02,
        0.05,
        r"$J_{{GC}} \leq C_1 J_{{ROM}} \leq C_2 J_{{SR}}$, all vanish as $\theta \to 1$",
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".png"),
        help="Path to the output image.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after saving it.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the demo."""
    args = parse_args()
    plot_objectives(args.output)
    print(f"Saved demo plot to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
