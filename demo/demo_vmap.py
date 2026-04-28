import time
import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')

# ----------------------------------------
# Work function (moderately heavy)
# ----------------------------------------
def f(x):
    # simulate some nontrivial compute
    for _ in range(2000):
        x = jnp.sin(x) + 0.1 * x**2
    return x


# ----------------------------------------
# Sequential Python loop (baseline)
# ----------------------------------------
def run_python_loop(x):
    out = []
    for i in range(x.shape[0]):
        out.append(f(x[i]))
    return jnp.stack(out)


# ----------------------------------------
# Vectorized + JIT version
# ----------------------------------------
vmap_f = jax.jit(jax.vmap(f))


# ----------------------------------------
# Benchmark helper
# ----------------------------------------
def benchmark(fn, x, name):
    # warmup (important for JIT)
    y = fn(x)
    if hasattr(y, "block_until_ready"):
        y.block_until_ready()

    t0 = time.time()
    y = fn(x)
    if hasattr(y, "block_until_ready"):
        y.block_until_ready()
    t1 = time.time()

    print(f"{name:20s} | N={x.shape[0]:7d} | time = {t1 - t0:.6f} s")


# ----------------------------------------
# Run experiment over increasing sizes
# ----------------------------------------
def main():
    sizes = [10, 50, 100]

    print("Device:", jax.default_backend())
    print("-" * 60)

    for N in sizes:
        x = jnp.linspace(0.0, 10.0, N)

        # Python loop (will get very slow)
        if N <= 10_000:  # avoid extreme slowdown
            benchmark(run_python_loop, x, "python loop")

        # vmap version
        benchmark(vmap_f, x, "vmap + jit")

        print()


if __name__ == "__main__":
    main()