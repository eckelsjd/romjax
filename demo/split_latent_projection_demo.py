"""Compare a learned split latent projection with separate POD reconstructions.

Run with, for example::

    uv run python demo/split_latent_projection_demo.py --steps 800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from romjax.model import FilterModel, eqx_evaluate
from romjax.nn import SplitLinearProjection
from romjax.random_field import kle


def _make_filter(b_shape: tuple[int, ...], u_shape: tuple[int, ...], rank_b: int, rank_u: int) -> FilterModel:
    """Build the flat-vector FilterModel route used by the split projection demo."""
    return FilterModel(
        source="full",
        target="latent",
        filters=[
            {
                "forward": {
                    "callable": eqx_evaluate,
                    "input_routes": [
                        {"outer": ["full", "b"], "inner": ["b"]},
                        {"outer": ["full", "u"], "inner": ["u"]},
                    ],
                    "output_routes": [
                        {"inner": ["lb"], "outer": ["latent", "lb"]},
                        {"inner": ["lu"], "outer": ["latent", "lu"]},
                    ],
                    "opts": {
                        "gather": "flat",
                        "method": "reduce",
                        "scatter": "flat",
                        "template": {"lb": {"shape": [rank_b]}, "lu": {"shape": [rank_u]}},
                    },
                },
                "backward": {
                    "callable": eqx_evaluate,
                    "input_routes": [
                        {"outer": ["latent", "lb"], "inner": ["lb"]},
                        {"outer": ["latent", "lu"], "inner": ["lu"]},
                    ],
                    "output_routes": [
                        {"inner": ["b"], "outer": ["full", "b"]},
                        {"inner": ["u"], "outer": ["full", "u"]},
                    ],
                    "opts": {"gather": "flat", "method": "reconstruct", "scatter": "flat"},
                },
            }
        ],
    )


def _sample_fields(key: jax.Array, nsamples: int) -> tuple[jax.Array, jax.Array]:
    """Generate a KLE field and a nonlinear, lower-resolution output field."""
    b = kle(
        key,
        shape=(12, 10),
        truncation=(8, 7),
        correlation_lengths=(0.22, 0.16),
        spectral_decay=2.5,
        nsamples=nsamples,
    )
    u = jnp.tanh(0.8 * b[:, ::2, ::2]) + 0.15 * b[:, ::2, ::2] ** 2 + 0.1 * b[:, 1::2, 1::2]
    return b, u


def _pod_reconstruct(train: jax.Array, test: jax.Array, rank: int) -> jax.Array:
    """Fit a centered POD basis and reconstruct held-out flattened samples."""
    mean = jnp.mean(train, axis=0)
    _, _, basis = jnp.linalg.svd(train - mean, full_matrices=False)
    basis = basis[:rank]
    return (test - mean) @ basis.T @ basis + mean


def _mse(reference: jax.Array, approximation: jax.Array) -> jax.Array:
    """Return mean squared reconstruction error."""
    return jnp.mean((reference - approximation) ** 2)


def _train_projection(
    key: jax.Array,
    b_train: jax.Array,
    u_train: jax.Array,
    rank_b: int,
    rank_u: int,
    steps: int,
    learning_rate: float,
) -> SplitLinearProjection:
    """Train one split projection using equally weighted field reconstruction MSE."""
    b_size = b_train.shape[1]
    u_size = u_train.shape[1]
    module = SplitLinearProjection(
        input_size=b_size + u_size,
        b_latent=rank_b,
        u_latent=rank_u,
        b_output=b_size,
        u_output=u_size,
        key=key,
        random_bias=True,
    )
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(eqx.filter(module, eqx.is_array))
    full_train = jnp.concatenate((b_train, u_train), axis=-1)

    @eqx.filter_jit
    def step(
        current: SplitLinearProjection, current_state: optax.OptState
    ) -> tuple[SplitLinearProjection, optax.OptState]:
        def loss_fn(candidate: SplitLinearProjection) -> jax.Array:
            reconstructed = candidate.reconstruct(candidate.reduce(full_train))
            bhat, uhat = jnp.split(reconstructed, (b_size,), axis=-1)
            return 0.5 * (_mse(b_train, bhat) + _mse(u_train, uhat))

        gradients = eqx.filter_grad(loss_fn)(current)
        updates, next_state = optimizer.update(gradients, current_state, eqx.filter(current, eqx.is_array))
        return eqx.apply_updates(current, updates), next_state

    for _ in range(steps):
        module, state = step(module, state)
    return module


def run(args: argparse.Namespace) -> None:
    """Train rank sweeps, compare held-out errors, and save the resulting figure."""
    data_key, init_key = jax.random.split(jax.random.key(args.seed))
    b, u = _sample_fields(data_key, args.train_samples + args.test_samples)
    b_train, b_test = b[: args.train_samples], b[args.train_samples :]
    u_train, u_test = u[: args.train_samples], u[args.train_samples :]
    b_train_flat, b_test_flat = b_train.reshape(args.train_samples, -1), b_test.reshape(args.test_samples, -1)
    u_train_flat, u_test_flat = u_train.reshape(args.train_samples, -1), u_test.reshape(args.test_samples, -1)

    rows: list[tuple[int, float, float, float, float, float, float]] = []
    for index, rank in enumerate(args.ranks):
        key = jax.random.fold_in(init_key, index)
        module = _train_projection(
            key, b_train_flat, u_train_flat, rank, rank, args.steps, args.learning_rate
        )
        reconstructed = module.reconstruct(module.reduce(jnp.concatenate((b_test_flat, u_test_flat), axis=-1)))
        split_b, split_u = jnp.split(reconstructed, (b_train_flat.shape[1],), axis=-1)
        pod_b = _pod_reconstruct(b_train_flat, b_test_flat, rank)
        pod_u = _pod_reconstruct(u_train_flat, u_test_flat, rank)
        split_b_error, split_u_error = _mse(b_test_flat, split_b), _mse(u_test_flat, split_u)
        pod_b_error, pod_u_error = _mse(b_test_flat, pod_b), _mse(u_test_flat, pod_u)
        rows.append(
            (
                rank,
                float(split_b_error),
                float(split_u_error),
                float(0.5 * (split_b_error + split_u_error)),
                float(pod_b_error),
                float(pod_u_error),
                float(0.5 * (pod_b_error + pod_u_error)),
            )
        )

    print("rank  split_b     split_u     split_total  pod_b       pod_u       pod_total")
    for row in rows:
        print(f"{row[0]:4d}  {row[1]:.3e}  {row[2]:.3e}  {row[3]:.3e}  {row[4]:.3e}  {row[5]:.3e}  {row[6]:.3e}")

    # Exercise the same FilterModel configuration used by a graph edge: it owns
    # field gathering, latent splitting, and restoration of unequal field shapes.
    filter_model = _make_filter(b.shape[1:], u.shape[1:], args.ranks[0], args.ranks[0])
    example_module = _train_projection(
        jax.random.fold_in(init_key, len(args.ranks)), b_train_flat, u_train_flat,
        args.ranks[0], args.ranks[0], args.steps, args.learning_rate,
    )
    latent, aux = filter_model.forward_aux(
        {"full": {"b": b_test[0], "u": u_test[0]}, "call_args": example_module}
    )
    restored, _ = filter_model.backward_aux({"latent": latent["latent"], "call_args": example_module}, aux=aux)
    assert restored["full"]["b"].shape == b.shape[1:]
    assert restored["full"]["u"].shape == u.shape[1:]

    ranks = [row[0] * 2 for row in rows]
    figure, axis = plt.subplots(figsize=(6, 4), layout="tight")
    axis.semilogy(ranks, [row[3] for row in rows], "o-", label="split latent")
    axis.semilogy(ranks, [row[6] for row in rows], "s--", label="separate POD")
    axis.set(xlabel="total latent size", ylabel="held-out total reconstruction MSE")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved comparison plot to {args.output}")


def _parse_args() -> argparse.Namespace:
    """Parse deterministic demo configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("split_latent/split_latent_projection.png"))
    return parser.parse_args()


if __name__ == "__main__":
    run(_parse_args())
