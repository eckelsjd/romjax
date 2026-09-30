"""Compare coupled and diagonal block autoencoders with corresponding POD models.

Run with, for example::

    uv run python demo/block_linear_autoencoder_demo.py --steps 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from loguru import logger

from romjax.compression import SVD, BlockLinearCompression
from romjax.nn import BlockLinearAutoencoder
from romjax.random_field import kle

logger.disable("romjax.train")


def _sample_fields(key: jax.Array, samples: int) -> tuple[jax.Array, jax.Array]:
    """Generate a KLE input field and a nonlinear output field of the same shape."""
    inputs = kle(
        key,
        shape=(32, 32),
        truncation=(8, 4),
        correlation_lengths=(0.08, 0.08),
        spectral_decay=0.5,
        nsamples=samples,
    )
    filtered = (
        4.0 * inputs
        + jnp.roll(inputs, 1, axis=-1)
        + jnp.roll(inputs, -1, axis=-1)
        + jnp.roll(inputs, 1, axis=-2)
        + jnp.roll(inputs, -1, axis=-2)
    ) / 8.0
    outputs = filtered + 0.02 * filtered**2
    return inputs, outputs


def _as_samples(inputs: jax.Array, outputs: jax.Array) -> list[dict[str, jax.Array]]:
    """Package paired fields in the canonical PyTree order used by compression."""
    return [{"input": input_field, "output": output_field} for input_field, output_field in zip(inputs, outputs)]


def _fit_autoencoder(
    key: jax.Array,
    samples: list[dict[str, jax.Array]],
    *,
    diagonal: bool,
    steps: int,
    learning_rate: float,
) -> BlockLinearCompression:
    """Fit one coupled or diagonal block autoencoder through the artifact API."""
    field_size = samples[0]["input"].size
    module = BlockLinearAutoencoder(
        input_sizes=[field_size, field_size],
        latent_sizes=[16, 16],
        key=key,
        bias=jnp.zeros(2 * field_size),
        diagonal=diagonal,
    )
    return BlockLinearCompression(
        fit_mode="pod_then_train",
        train={
            "init_params": module,
            "optimizer": optax.adam(learning_rate),
            "termination": {"max_steps": steps},
            "diagnostics": {"plot_interval": 20, "test_interval": 20, "live_plot": True},
        },
        show_progress=True,
        test=True,
    ).fit(samples)


def _reconstruct_autoencoder(
    compression: BlockLinearCompression, samples: list[dict[str, jax.Array]]
) -> tuple[jax.Array, jax.Array]:
    """Reconstruct paired fields with a fitted block compression artifact."""
    reconstructed = [compression.reconstruct(compression.compress(sample)) for sample in samples]
    return (
        jnp.stack([sample["input"] for sample in reconstructed]),
        jnp.stack([sample["output"] for sample in reconstructed]),
    )


def _relative_error(
    inputs: jax.Array,
    outputs: jax.Array,
    reconstructed_inputs: jax.Array,
    reconstructed_outputs: jax.Array,
) -> float:
    """Return relative Frobenius error over both fields and all samples."""
    reference = jnp.concatenate((inputs.reshape(inputs.shape[0], -1), outputs.reshape(outputs.shape[0], -1)), axis=1)
    approximation = jnp.concatenate(
        (reconstructed_inputs.reshape(inputs.shape[0], -1), reconstructed_outputs.reshape(outputs.shape[0], -1)),
        axis=1,
    )
    return float(jnp.linalg.norm(approximation - reference) / jnp.linalg.norm(reference))


def run(args: argparse.Namespace) -> None:
    """Fit all four compressors and compare independent test reconstruction errors."""
    if args.train_samples < 32:
        raise ValueError("At least 32 training samples are required for rank-32 POD initialization.")
    inputs, outputs = _sample_fields(jax.random.key(args.seed), args.train_samples + args.test_samples)
    train_inputs, test_inputs = inputs[: args.train_samples], inputs[args.train_samples :]
    train_outputs, test_outputs = outputs[: args.train_samples], outputs[args.train_samples :]
    train_samples = _as_samples(train_inputs, train_outputs)
    test_samples = _as_samples(test_inputs, test_outputs)
    joint_pod = SVD(rank=32, center=True).fit(train_samples)
    input_pod = SVD(rank=16, center=True).fit([{"field": field} for field in train_inputs])
    output_pod = SVD(rank=16, center=True).fit([{"field": field} for field in train_outputs])

    coupled = _fit_autoencoder(
        jax.random.key(args.seed + 1),
        train_samples,
        diagonal=False,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    diagonal = _fit_autoencoder(
        jax.random.key(args.seed + 2),
        train_samples,
        diagonal=True,
        steps=args.steps,
        learning_rate=args.learning_rate,
    )
    coupled_reconstruction = _reconstruct_autoencoder(coupled, test_samples)
    diagonal_reconstruction = _reconstruct_autoencoder(diagonal, test_samples)

    joint_reconstruction = [joint_pod.reconstruct(joint_pod.compress(sample)) for sample in test_samples]
    joint_fields = (
        jnp.stack([sample["input"] for sample in joint_reconstruction]),
        jnp.stack([sample["output"] for sample in joint_reconstruction]),
    )

    separate_fields = (
        jnp.stack([input_pod.reconstruct(input_pod.compress({"field": field}))["field"] for field in test_inputs]),
        jnp.stack(
            [output_pod.reconstruct(output_pod.compress({"field": field}))["field"] for field in test_outputs]
        ),
    )

    errors = {
        "coupled block AE": _relative_error(test_inputs, test_outputs, *coupled_reconstruction),
        "diagonal block AE": _relative_error(test_inputs, test_outputs, *diagonal_reconstruction),
        "joint POD": _relative_error(test_inputs, test_outputs, *joint_fields),
        "separate POD": _relative_error(test_inputs, test_outputs, *separate_fields),
    }
    print("Held-out relative reconstruction errors (total rank 32)")
    for name, error in errors.items():
        print(f"  {name:20s} {error:.6e}")

    figure, axis = plt.subplots(figsize=(7, 4), layout="tight")
    axis.bar(errors.keys(), errors.values())
    axis.set_ylabel("relative reconstruction error")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved comparison plot to {args.output}")


def _parse_args() -> argparse.Namespace:
    """Parse deterministic demo settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/block_linear_autoencoder.png"))
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
