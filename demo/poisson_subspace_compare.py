"""Compare PCA and Adam-learned subspaces for Poisson solution reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import romjax as romx

from loguru import logger

logger.disable("romjax")


def _load_phi_matrix(data_root: Path, split: str, batch_size: int = 64) -> jax.Array:
    """Load the stacked ``outputs.phi`` fields for one dataset split.

    :param data_root: Root directory containing the ``train`` and ``validation`` folders.
    :param split: Dataset split to load, e.g. ``"train"`` or ``"validation"``.
    :param batch_size: Mini-batch size used by :class:`romjax.data_gen.DataLoader`.
    :returns: Array with shape ``(num_samples, 1024)``.
    """
    loader = romx.DataLoader(
        root=data_root,
        datasets={
            split: {
                "poisson": {
                    "kind": "implicit",
                    "batch_size": batch_size,
                    "load_solution": True,
                    "max_epochs": 1,
                    "max_samples": 300,
                    "shuffle_seed": 1,
                }
            }
        },
    )

    phi_batches: list[jax.Array] = []
    for batch in loader:
        phi_batches.append(jnp.asarray(batch["poisson"]["outputs"]["phi"]))

    if not phi_batches:
        raise ValueError(f"No Poisson samples were loaded from {data_root / split}.")

    matrix = jnp.concatenate(phi_batches, axis=0)
    return matrix.reshape(matrix.shape[0], -1)


def _rank_for_energy(matrix: jax.Array, energy_tol: float = 0.99) -> tuple[int, jax.Array, jax.Array]:
    """Compute the PCA rank needed to capture the requested energy fraction."""
    mean = jnp.mean(matrix, axis=0)
    centered = matrix - mean
    _, singular_values, vt = jnp.linalg.svd(centered, full_matrices=False)
    energy = jnp.square(singular_values)
    total_energy = jnp.sum(energy)
    cumulative = jnp.cumsum(energy) / total_energy
    rank = int(jnp.searchsorted(cumulative, energy_tol, side="left") + 1)
    rank = min(rank, int(vt.shape[0]))
    return rank, mean, singular_values


def _orthonormal_rows(matrix: jax.Array) -> jax.Array:
    """Return a row-orthonormal basis with the same span as ``matrix``."""
    q, _ = jnp.linalg.qr(matrix.T, mode="reduced")
    return q.T


def _fit_pca_basis(matrix: jax.Array, rank: int) -> tuple[jax.Array, jax.Array]:
    """Fit the centered PCA basis of the requested rank."""
    mean = jnp.mean(matrix, axis=0)
    centered = matrix - mean
    _, _, vt = jnp.linalg.svd(centered, full_matrices=False)
    basis = vt[:rank]
    return basis, mean


def _train_adam_basis(
    matrix: jax.Array,
    mean: jax.Array,
    rank: int,
    *,
    steps: int,
    learning_rate: float,
    orthonormal_penalty_weight: float,
    tikhonov_weight: float,
    seed: int,
) -> jax.Array:
    """Train a basis with Adam on reconstruction error and an orthogonality penalty."""
    centered = matrix - mean
    params = jax.random.normal(jax.random.key(seed), (rank, centered.shape[1]))
    optimizer = optax.adam(learning_rate)
    

    def loss_fn(basis_params: jax.Array, args) -> jax.Array:
        del args
        projected = centered @ basis_params.T
        recon = projected @ basis_params
        # recon_loss = jnp.mean(jnp.linalg.norm(recon - centered, axis=1))
        recon_loss = jnp.mean(jnp.sum(jnp.square(recon - centered), axis=1))
        reg = tikhonov_weight * jnp.sum(jnp.square(basis_params))
        gram = basis_params @ basis_params.T
        orthogonality_loss = jnp.mean(jnp.square(gram - jnp.eye(gram.shape[0], dtype=basis_params.dtype)))
        return recon_loss + reg + orthonormal_penalty_weight * orthogonality_loss
    
    train = romx.Train(
        loss=loss_fn,
        optimizer=optimizer,
        init_params=params,
        termination={"max_steps": steps},
        diagnostics={"live_plot": True, "plot_interval": 10}
    )
    params = train()

    # opt_state = optimizer.init(params)

    # @jax.jit
    # def train_step(basis_params: jax.Array, state: optax.OptState) -> tuple[jax.Array, optax.OptState, jax.Array]:
    #     loss, grads = jax.value_and_grad(loss_fn)(basis_params)
    #     updates, state = optimizer.update(grads, state, basis_params)
    #     basis_params = optax.apply_updates(basis_params, updates)
    #     return basis_params, state, loss

    # for step in range(steps):
    #     params, opt_state, loss = train_step(params, opt_state)
    #     if step == 0 or (step + 1) % max(steps // 10, 1) == 0:
    #         print(f"adam step {step + 1:04d}/{steps}: loss={float(loss):.6e}")

    return params


def _relative_errors(matrix: jax.Array, basis: jax.Array, mean: jax.Array) -> np.ndarray:
    """Compute relative reconstruction errors for each sample in ``matrix``."""
    centered = matrix - mean
    recon = (centered @ basis.T) @ basis + mean
    errors = jnp.linalg.norm(matrix - recon, axis=1) / jnp.linalg.norm(matrix, axis=1)
    return np.asarray(errors)


def _projector_relative_error(pca_basis: jax.Array, adam_basis: jax.Array) -> float:
    """Return the normalized Frobenius distance between the two subspace projectors."""
    pca_cols = pca_basis.T
    adam_cols = adam_basis.T
    # pca_cols, _ = jnp.linalg.qr(pca_cols, mode="reduced")
    # adam_cols, _ = jnp.linalg.qr(adam_cols, mode="reduced")
    pca_projector = pca_cols @ pca_cols.T
    adam_projector = adam_cols @ adam_cols.T
    return float(jnp.linalg.norm(adam_projector - pca_projector) / jnp.linalg.norm(pca_projector))


def _orthogonal_norm(basis):
    gram = basis @ basis.T  # (rxN) x (Nxr)
    return float(jnp.linalg.norm(gram - jnp.eye(gram.shape[0])) / jnp.linalg.norm(gram))


def main() -> None:
    """Run the Poisson subspace comparison demo."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "journal" / "poisson" / "data",
        help="Root directory containing the journal Poisson data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("poisson_subspace_compare"),
        help="Directory used to store the matrices, summary, and histogram plot.",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Adam optimization steps.")
    parser.add_argument("--lr", type=float, default=0.1, help="Adam learning rate.")
    parser.add_argument(
        "--orthonormal",
        type=float,
        default=1.0,
        help="Weight for the orthonormality penalty in the Adam objective.",
    )
    parser.add_argument("--tikhonov", type=float, default=0.1)
    args = parser.parse_args()

    train_root = args.data_root / "train"
    validation_root = args.data_root / "validation"
    if not train_root.exists():
        raise FileNotFoundError(f"Training data not found at {train_root}")
    if not validation_root.exists():
        raise FileNotFoundError(f"Validation data not found at {validation_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_matrix = _load_phi_matrix(args.data_root, "train")
    validation_matrix = _load_phi_matrix(args.data_root, "validation")

    rank, _, singular_values = _rank_for_energy(train_matrix, energy_tol=0.99)
    pca_basis, mean = _fit_pca_basis(train_matrix, rank)
    adam_basis = _train_adam_basis(
        train_matrix,
        mean,
        rank,
        steps=args.steps,
        learning_rate=args.lr,
        orthonormal_penalty_weight=args.orthonormal,
        tikhonov_weight=args.tikhonov,
        seed=0,
    )

    artifact_path = args.output_dir / "poisson_subspace_bases.npz"
    np.savez_compressed(
        artifact_path,
        pca_basis=np.asarray(pca_basis),
        adam_basis=np.asarray(adam_basis),
        mean=np.asarray(mean),
        rank=np.asarray(rank),
        train_singular_values=np.asarray(singular_values),
    )

    pca_errors = _relative_errors(validation_matrix, pca_basis, mean)
    adam_errors = _relative_errors(validation_matrix, adam_basis, mean)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    max_error = float(max(np.max(pca_errors), np.max(adam_errors)))
    bins = np.linspace(
        0.0,
        max(max_error, 1e-12),
        30,
    )
    ax.hist(pca_errors, bins=bins, alpha=0.65, label="PCA", density=True)
    ax.hist(adam_errors, bins=bins, alpha=0.65, label="Adam basis", density=True)
    ax.set_xlabel("Relative reconstruction error")
    ax.set_ylabel("Density")
    ax.set_title("Validation reconstruction error on Poisson fields")
    ax.legend()
    ax.grid(True, alpha=0.25)

    histogram_path = args.output_dir / "poisson_subspace_errors.png"
    fig.savefig(histogram_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    projector_relative_error = _projector_relative_error(pca_basis, adam_basis)
    pca_orthogonal_error = _orthogonal_norm(pca_basis)
    adam_orthogonal_error = _orthogonal_norm(adam_basis)
    summary = {
        "rank": rank,
        "train_samples": int(train_matrix.shape[0]),
        "validation_samples": int(validation_matrix.shape[0]),
        "energy_tol": 0.99,
        "projector_relative_error": projector_relative_error,
        "pca_orthogonal_norm": pca_orthogonal_error,
        "adam_orthogonal_norm": adam_orthogonal_error,
        "pca_validation_relative_error_mean": float(np.mean(pca_errors)),
        "adam_validation_relative_error_mean": float(np.mean(adam_errors)),
        "pca_validation_relative_error_median": float(np.median(pca_errors)),
        "adam_validation_relative_error_median": float(np.median(adam_errors)),
    }

    summary_path = args.output_dir / "poisson_subspace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # print(f"Saved basis archive to {artifact_path}")
    # print(f"Saved validation histogram to {histogram_path}")
    # print(f"Saved subspace summary to {summary_path}")
    print(f"Chosen rank for 99% train energy: {rank}")
    print(f"Projector relative error: {projector_relative_error:.6e}")
    print(f"PCA orthogonal norm: {pca_orthogonal_error:.6e}")
    print(f"Adam orthogonal norm: {adam_orthogonal_error:.6e}")


if __name__ == "__main__":
    main()
