"""Benchmark SVD compression on the Poisson dataset and generate summary plots."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from romjax import YamlLoader
from romjax.compression import SVD
from romjax.data_gen import GenLatent
from romjax.plotting import AxisOptions, PlotSpec, gridplot
from romjax.tree import norm as pytree_norm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
STYLE_PATH = REPO_ROOT / "src" / "romjax" / "stix.mplstyle"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config" / "gen_data.yml"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures" / "latent"


def _use_style() -> None:
    """Apply the local plotting style if it exists."""
    if STYLE_PATH.exists():
        plt.style.use(STYLE_PATH)


def _load_generator(config_path: Path) -> GenLatent:
    """Load the latent-generation config from YAML."""
    os.chdir(SCRIPT_DIR)
    config = YamlLoader.load(config_path)
    generator = config.datasets["train"]["compression"]
    if not isinstance(generator, GenLatent):
        raise TypeError("Expected the train/compression config to resolve to GenLatent.")
    return generator


def _sample_latents(compression: SVD, samples: list[dict]) -> np.ndarray:
    """Project all samples into latent space and return a dense array."""
    latent = [np.asarray(compression.compress(sample), dtype=np.float64).reshape(-1) for sample in samples]
    return np.stack(latent, axis=0)


def _relative_reconstruction_errors(compression: SVD, samples: list[dict]) -> np.ndarray:
    """Compute relative reconstruction error for each sample."""
    errors: list[float] = []
    for sample in samples:
        latent = compression.compress(sample)
        reconstructed = compression.reconstruct(latent)
        diff = jax.tree.map(lambda a, b: jnp.asarray(a) - jnp.asarray(b), sample, reconstructed)
        numerator = float(jax.device_get(pytree_norm(diff)))
        denominator = float(jax.device_get(pytree_norm(sample)))
        errors.append(numerator / max(denominator, 1e-12))
    return np.asarray(errors, dtype=np.float64)


def _field_array(sample: dict) -> np.ndarray:
    """Extract the Poisson field from one latent sample."""
    return np.asarray(sample["poisson"]["outputs"]["phi"], dtype=np.float64)


def _plot_field_comparison(samples: list[dict], compression: SVD, output_dir: Path) -> None:
    """Plot original, reconstructed, and absolute error fields for four samples."""
    n_rows = min(4, len(samples))
    if n_rows == 0:
        raise ValueError("No samples available for field comparison.")

    x = np.linspace(0.0, 1.0, _field_array(samples[0]).shape[0])
    y = np.linspace(0.0, 1.0, _field_array(samples[0]).shape[1])
    xx, yy = np.meshgrid(x, y, indexing="ij")

    rows: list[list[PlotSpec]] = []
    for row_idx, sample in enumerate(samples[:n_rows], start=1):
        original = _field_array(sample)
        reconstructed_sample = compression.reconstruct(compression.compress(sample))
        reconstructed = _field_array(reconstructed_sample)
        abs_error = np.abs(original - reconstructed)

        field_vmin = float(np.min([original.min(), reconstructed.min()]))
        field_vmax = float(np.max([original.max(), reconstructed.max()]))
        err_vmax = float(np.max(abs_error)) if np.any(abs_error) else 1.0

        rows.append(
            [
                PlotSpec(
                    kind="pcolor",
                    data=(xx, yy, original),
                    opts=AxisOptions(ax_visible=False, grid=False, title=f"Original" if row_idx==0 else ""),
                    kwargs={"cmap": "viridis", "vmin": field_vmin, "vmax": field_vmax},
                ),
                PlotSpec(
                    kind="pcolor",
                    data=(xx, yy, reconstructed),
                    opts=AxisOptions(ax_visible=False, grid=False, title=f"Reconstructed" if row_idx==0 else ""),
                    kwargs={"cmap": "viridis", "vmin": field_vmin, "vmax": field_vmax},
                ),
                PlotSpec(
                    kind="pcolor",
                    data=(xx, yy, abs_error),
                    opts=AxisOptions(ax_visible=False, grid=False, title=f"Absolute error" if row_idx==0 else ""),
                    kwargs={"cmap": "viridis", "vmin": 0.0, "vmax": err_vmax},
                ),
            ]
        )

    fig, _axs = gridplot(rows, scheme="dark", subplot_size_in=(3.2, 2.8), subplots_kwargs={"squeeze": False})
    _axs[0, 0].set_title("Original", color="white")
    _axs[0, 1].set_title("Reconstructed", color="white")
    _axs[0, 2].set_title("Absolute error", color="white")
    fig.savefig(output_dir / "field_reconstruction_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_singular_spectrum(compression: SVD, output_dir: Path) -> None:
    """Plot the singular value spectrum and cumulative energy."""
    singular_values = np.asarray(compression.singular_values, dtype=np.float64)
    if singular_values.size == 0:
        raise ValueError("Compression did not produce a singular spectrum.")

    energy = singular_values**2
    cumulative_energy = np.cumsum(energy) / np.sum(energy)
    rank = compression.latent_size()
    energy_tol = compression.energy_tol

    fig, (ax_sv, ax_energy) = plt.subplots(1, 2, figsize=(10, 4))

    ax_sv.semilogy(np.arange(1, singular_values.size + 1), singular_values, marker="o", lw=1.5)
    ax_sv.axvline(rank, color="tab:red", ls="--", lw=1.5, label=f"rank={rank}")
    ax_sv.set_xlabel("Mode")
    ax_sv.set_ylabel("Singular value")
    ax_sv.set_title("Singular Spectrum")
    ax_sv.legend(frameon=False)
    ax_sv.grid(True, alpha=0.25)

    ax_energy.plot(np.arange(1, cumulative_energy.size + 1), cumulative_energy, marker="o", lw=1.5)
    ax_energy.axhline(float(energy_tol) if energy_tol is not None else 0.0, color="tab:red", ls="--", lw=1.5)
    ax_energy.axvline(rank, color="tab:red", ls="--", lw=1.5)
    ax_energy.set_xlabel("Mode")
    ax_energy.set_ylabel("Cumulative energy")
    ax_energy.set_ylim(0.0, 1.02)
    ax_energy.set_title("Explained Energy")
    ax_energy.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "singular_value_spectrum.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_latent_histograms(latent_matrix: np.ndarray, output_dir: Path) -> None:
    """Plot histograms for the first nine latent coordinates."""
    latent_dim = latent_matrix.shape[1]
    n_plots = min(9, latent_dim)
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    axes_flat = axes.flat

    for idx, ax in enumerate(axes_flat):
        if idx >= n_plots:
            ax.axis("off")
            continue

        values = latent_matrix[:, idx]
        ax.hist(values, bins=30, density=True, color="tab:blue", alpha=0.85, edgecolor="white")
        ax.set_title(f"$z_{idx + 1}$")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Latent Coordinate Histograms")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_histograms.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_error_histogram(errors: np.ndarray, output_dir: Path) -> None:
    """Plot the relative reconstruction error histogram."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(errors, bins=40, color="tab:orange", alpha=0.9, edgecolor="white")
    ax.set_xlabel("Relative reconstruction error")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction Error Distribution")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "relative_reconstruction_error_histogram.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> dict[str, float]:
    """Run the benchmark and save plots plus summary metrics."""
    _use_style()

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = _load_generator(DEFAULT_CONFIG_PATH)
    compression = generator.compression

    t0 = time.perf_counter()
    samples = list(generator._iter_samples())
    t1 = time.perf_counter()
    compression = compression.fit(samples)
    t2 = time.perf_counter()

    latent_matrix = _sample_latents(compression, samples)
    reconstruction_errors = _relative_reconstruction_errors(compression, samples)
    minval, maxval = [b.tolist() for b in compression.latent_bounds()]

    _plot_singular_spectrum(compression, output_dir)
    _plot_latent_histograms(latent_matrix, output_dir)
    _plot_error_histogram(reconstruction_errors, output_dir)
    _plot_field_comparison(samples, compression, output_dir)

    metrics = {
        "n_samples": int(len(samples)),
        "feature_dim": int(latent_matrix.shape[1] if latent_matrix.ndim == 2 else 0),
        "latent_dim": int(compression.latent_size()),
        "iter_seconds": float(t1 - t0),
        "fit_seconds": float(t2 - t1),
        "mean_relative_error": float(np.mean(reconstruction_errors)),
        "median_relative_error": float(np.median(reconstruction_errors)),
        "p90_relative_error": float(np.quantile(reconstruction_errors, 0.90)),
        "p95_relative_error": float(np.quantile(reconstruction_errors, 0.95)),
        "max_relative_error": float(np.max(reconstruction_errors)),
        "minval": minval,
        "maxval": maxval
    }

    with (output_dir / "latent_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)

    print(
        "Samples: {n_samples:.0f}. Iter: {iter_seconds:.3f}s. Fit: {fit_seconds:.3f}s. "
        "Latent: {latent_dim:.0f}. Mean rel err: {mean_relative_error:.3e}. "
        "Median rel err: {median_relative_error:.3e}. Max rel err: {max_relative_error:.3e}.".format(
            **metrics
        )
    )
    print(f"Minval: {minval}")
    print(f"Maxval: {maxval}")

    return metrics


if __name__ == "__main__":
    main()
