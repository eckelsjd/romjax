"""Compare Python-loop and vmapped execution around a jitted JAX function."""

from __future__ import annotations

import atexit
import os
from pathlib import Path
from timeit import repeat

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from romjax import YamlLoader
from romjax.utils import pytree_at, monitor_gpu_memory

# jax.config.update('jax_platform_name', 'cpu')
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

thread, stop_event = monitor_gpu_memory(0.5)


def _cleanup_gpu_monitor() -> None:
    """Stop the background GPU monitor thread before interpreter shutdown."""
    stop_event.set()
    if thread.is_alive():
        thread.join(timeout=1.0)


atexit.register(_cleanup_gpu_monitor)


graph = YamlLoader.load("poisson_graph.yml")
poisson = graph.edges['poisson']
in_key = jax.random.key(0)

# @jax.jit
# def kernel(x: jax.Array) -> jax.Array:
#     """A small compute-heavy kernel to make batching effects visible."""
#     y = jnp.sin(x) + jnp.cos(2.0 * x)
#     for _ in range(1000):
#         y = jnp.tanh(1.5 * y) + 0.1 * y**2
#     return y.sum()

@jax.jit
def kernel(inputs):
    return poisson.solve(inputs)

vmapped_kernel = jax.jit(jax.vmap(kernel))


# def make_sample(sample_index: int, width: int) -> jax.Array:
#     """Build one deterministic sample."""
#     start = sample_index / (width + 1.0)
#     stop = start + 1.0
#     return jnp.linspace(start, stop, width, dtype=jnp.float32)


def make_batch(batch_size):
    """Build a full batch for vmapped execution."""
    in_keys = jax.random.split(in_key, batch_size)
    return jax.jit(jax.vmap(poisson.sample_inputs))(in_keys)
    # return jnp.stack([make_sample(sample_index, width) for sample_index in range(batch_size)])


def run_python_loop(batch, batch_size) -> None:
    """Call the jitted kernel one sample at a time from Python."""
    for sample_index in range(batch_size):
        res = kernel(pytree_at(batch, sample_index))
        res['phi'].block_until_ready()
        # kernel(make_sample(sample_index, width)).block_until_ready()


def run_vmapped(batch) -> None:
    """Call the vmapped+jitted kernel on the full batch."""
    res = vmapped_kernel(batch)
    res['phi'].block_until_ready()


def benchmark(batch_size: int, repetitions: int) -> tuple[np.ndarray, np.ndarray]:
    """Return repeated runtimes for looped and vmapped execution."""
    batch = make_batch(batch_size)

    run_python_loop(batch, batch_size)
    run_vmapped(batch)

    loop_times = repeat(lambda: run_python_loop(batch, batch_size), repeat=repetitions, number=1)
    vmap_times = repeat(lambda: run_vmapped(batch), repeat=repetitions, number=1)
    return np.asarray(loop_times), np.asarray(vmap_times)


def main() -> None:
    """Run the timing comparison and save a plot of the results."""
    sizes = [1, 16, 32, 64]
    repetitions = 5

    loop_means: list[float] = []
    vmap_means: list[float] = []
    loop_p05: list[float] = []
    loop_p95: list[float] = []
    vmap_p05: list[float] = []
    vmap_p95: list[float] = []

    for size in sizes:
        loop_times, vmap_times = benchmark(size, repetitions)

        loop_mean = float(loop_times.mean())
        vmap_mean = float(vmap_times.mean())
        loop_lo, loop_hi = np.percentile(loop_times, [5, 95])
        vmap_lo, vmap_hi = np.percentile(vmap_times, [5, 95])

        loop_means.append(loop_mean)
        vmap_means.append(vmap_mean)
        loop_p05.append(float(loop_lo))
        loop_p95.append(float(loop_hi))
        vmap_p05.append(float(vmap_lo))
        vmap_p95.append(float(vmap_hi))

        print(
            f"batch_size={size:>3d} | "
            f"loop={loop_mean:.6f}s [{loop_lo:.6f}, {loop_hi:.6f}] | "
            f"vmap={vmap_mean:.6f}s [{vmap_lo:.6f}, {vmap_hi:.6f}]"
        )

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.errorbar(
        sizes,
        loop_means,
        yerr=[
            np.asarray(loop_means) - np.asarray(loop_p05),
            np.asarray(loop_p95) - np.asarray(loop_means),
        ],
        marker="o",
        capsize=4,
        label="Python loop + jit",
    )
    ax.errorbar(
        sizes,
        vmap_means,
        yerr=[
            np.asarray(vmap_means) - np.asarray(vmap_p05),
            np.asarray(vmap_p95) - np.asarray(vmap_means),
        ],
        marker="o",
        capsize=4,
        label="vmap + jit",
    )
    ax.set_xlabel("Outer loop size")
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Jitted kernel timing: Python loop vs vmap")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    output_path = Path(__file__).with_suffix(".png")
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
