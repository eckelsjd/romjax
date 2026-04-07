"""Based on Darcy flow by Kovachki et al. (2022):
    
    https://arxiv.org/abs/2108.08481

    Used for testing neural operators on PDEs.
"""
import argparse
import pickle
from pathlib import Path
import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimistix as optx
import optax
import lineax as lx
from jaxtyping import Key, ArrayLike

from romjax.solvers import Poisson2D
from romjax.plotting import gridplot, get_scheme
from romjax.optimization import Optimizer
from romjax.utils import get_gpu_memory, monitor_gpu_memory

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def get_darcy_solver(shape):
    """For testing Darcy flow: constant forcing, homogeneous dirichlet BCs on [0,1]^2
    with a random field conductivity kappa(x)=a(x).
    """
    darcy = Poisson2D(
        config={
            "solver": optx.Newton(rtol=1, atol=1e-3),
                                #   linear_solver=lx.GMRES(rtol=1e-3, atol=1e-20, max_steps=100, restart=30,
                                #                          stagnation_iters=30)),
            "grid": {"shape": shape, "bounds": ((0, 1), (0, 1))},
            "max_steps": 200,
            "throw": True
        },
        forcing_defaults={'const': -1.0},
        conductivity_defaults={'const': 1.0}
    )

    return darcy


def sample_darcy_random_field(coords, nsamples, *, key=None):
    """Get random field according to Sec 6.2 of Kovachki 2022."""
    x, y = coords
    nx, ny = x.shape
    if key is None:
        key = jax.random.PRNGKey(0)

    kx = jnp.arange(nx)
    ky = jnp.arange(ny)

    # Orthonormal cosine basis for Neumann Laplacian on (0, 1)
    scale_x = jnp.where(kx == 0, 1.0, jnp.sqrt(2.0))
    scale_y = jnp.where(ky == 0, 1.0, jnp.sqrt(2.0))
    phi_x = jnp.cos(jnp.pi * x[:, 0, None] * kx[None, :]) * scale_x[None, :]
    phi_y = jnp.cos(jnp.pi * y[0, :, None] * ky[None, :]) * scale_y[None, :]

    # Eigenvalues of covariance C = (-Δ + 9I)^-2
    lap_eigs = (jnp.pi ** 2) * (kx[:, None] ** 2 + ky[None, :] ** 2)
    cov_eigs = (lap_eigs + 9.0) ** -2
    sqrt_cov = jnp.sqrt(cov_eigs)

    # Sample Gaussian random field g ~ N(0, C)
    coeffs = jax.random.normal(key, (nsamples, nx, ny)) * sqrt_cov[None, :, :]
    gauss_field = jnp.einsum("ik,jl,bkl->bij", phi_x, phi_y, coeffs)

    # Nemytskii operator: piecewise-constant conductivity
    return jnp.where(gauss_field >= 0.0, 12.0, 3.0)


def plot_samples(samples, solutions, x, y, *, predictions=None):
    """Plot input fields versus solution fields (and optional predictions)."""
    input_clim = (float(jnp.min(samples)), float(jnp.max(samples)))

    if predictions is not None:
        err = jnp.abs(predictions - solutions)
    else:
        err = None

    input_specs = [
        {'kind': 'pcolor', 'data': (x, y, samples[i]), 'opts': {'clim': input_clim}}
        for i in range(len(samples))
    ]
    solution_specs = [
        {'kind': 'pcolor', 'data': (x, y, solutions[i]), 'opts': {'clim': 'auto'}}
        for i in range(len(solutions))
    ]

    if predictions is None:
        grid_spec = [[input_specs[i], solution_specs[i]] for i in range(len(samples))]
    else:
        pred_specs = []
        err_specs = []
        for i in range(len(predictions)):
            row_min = float(jnp.min(jnp.stack([predictions[i], solutions[i], err[i]])))
            row_max = float(jnp.max(jnp.stack([predictions[i], solutions[i], err[i]])))
            row_clim = (row_min, row_max)
            pred_specs.append(
                {'kind': 'pcolor', 'data': (x, y, predictions[i]), 'opts': {'clim': row_clim}}
            )
            err_specs.append(
                {'kind': 'pcolor', 'data': (x, y, err[i]), 'opts': {'clim': row_clim}}
            )
            solution_specs[i]['opts'] = {'clim': row_clim}
        grid_spec = [[input_specs[i], pred_specs[i], solution_specs[i], err_specs[i]] for i in range(len(samples))]
    scheme = 'dark'
    text_color, _ = get_scheme(scheme)

    def add_column_titles(fig, axs, *args):
        axs[0, 0].set_title("Input field", color=text_color)
        if predictions is None:
            axs[0, 1].set_title("True solution", color=text_color)
        else:
            axs[0, 1].set_title("Prediction", color=text_color)
            axs[0, 2].set_title("True solution", color=text_color)
            axs[0, 3].set_title("Absolute error", color=text_color)

    fig, axs = gridplot(grid_spec, subplot_size_in=(4, 3), scheme=scheme, adjust=add_column_titles)

    return fig, axs


def show_random_samples(shape, nsamples, key):
    """Just show some random fields and solutions."""
    darcy = get_darcy_solver(shape)
    x, y = darcy.config.grid.coords
    samples = sample_darcy_random_field((x, y), nsamples, key=key)

    @jax.jit
    def solve_one(kappa):
        return darcy.solve({"conductivity": {"const": kappa}})["phi"]

    solutions = jax.vmap(solve_one)(samples)

    fig, axs = plot_samples(samples, solutions, x, y)

    plt.show()


def sinusoid_forcing(inputs, outputs):
        """Smooth one-parameter forcing to keep Poisson solver stable."""
        xg, yg = inputs["coords"]
        return inputs["amplitude"] * jnp.sin(jnp.pi * xg) * jnp.sin(jnp.pi * yg)


def train_forcing(shape, key, save_dir):
    """Try to learn a forcing parameter over some random fields."""
    # used_mib, total_mib = get_gpu_memory()[0]
    # print(f"Initial GPU: {used_mib} / {total_mib} MiB")

    train_key, test_key = jax.random.split(key, 2)

    ntrain = 10
    ntest = 5

    darcy = get_darcy_solver(shape)
    x, y = darcy.config.grid.coords

    train_in = sample_darcy_random_field((x, y), ntrain, key=train_key)
    test_in = sample_darcy_random_field((x, y), ntest, key=test_key)

    # Use a "true" value on the training data
    darcy.forcing = sinusoid_forcing
    darcy.forcing_defaults = {"amplitude": 1.0}
    true_param = jnp.asarray(-1.0)
    param0 = jnp.asarray(-10.)

    @jax.jit
    def solve_one(kappa, param):
        return darcy.solve({"conductivity": {"const": kappa}, "forcing": {"amplitude": param}})["phi"]
    
    solve_many = jax.vmap(solve_one, in_axes=(0, None), out_axes=0)
    
    # used_mib, total_mib = get_gpu_memory()[0]
    # print(f"Before solve GPU: {used_mib} / {total_mib} MiB")

    # print("Starting GPU monitor...")

    # thread, stop_event = monitor_gpu_memory(0.3)

    train_out = solve_many(train_in, true_param)
    # time.sleep(4)
    test_out = solve_many(test_in, true_param)

    # print("Closing GPU monitor...")
    # stop_event.set()
    # thread.join()

    # used_mib, total_mib = get_gpu_memory()[0]
    # print(f"After solve GPU: {used_mib} / {total_mib} MiB")

    # Can't take grad of vmap(optx.root_find) apparently, so manually loop over train/test sets
    @jax.jit
    def loss_fn(param, *args):
        def body(i, acc):
            pred = solve_one(train_in[i], param)
            diff = pred - train_out[i]
            return acc + jnp.mean(diff**2)
        total = jax.lax.fori_loop(0, train_in.shape[0], body, 0.0)
        return total / train_in.shape[0]
    
    @jax.jit
    def test_fn(param):
        def body(i, acc):
            pred = solve_one(test_in[i], param)
            diff = pred - test_out[i]
            return acc + jnp.mean(diff**2)
        total = jax.lax.fori_loop(0, test_in.shape[0], body, 0.0)
        return total / test_in.shape[0]

    opt = Optimizer()

    res = opt.run_debug(
        loss_fn,
        params0=param0,
        optimizer=optax.adam(0.2),
        max_steps=100,
        max_runtime_s=120,
        grad_tol=1e-10,
        log_interval=5,
        plot_interval=5,
        test_fn=test_fn,
        live_plot=True,
        save=save_dir,
        prefix="darcy_"
    )

    print(f"True param: {true_param}, Result: {res}")

    # Save test data for offline validation plots
    test_data = {
        "x": jax.device_get(x),
        "y": jax.device_get(y),
        "test_in": jax.device_get(test_in),
        "test_out": jax.device_get(test_out),
    }
    with open(save_dir / "darcy_test_data.pkl", "wb") as fd:
        pickle.dump(test_data, fd)


def plot_validation(save_dir: str | Path, nshow: int = 3) -> None:
    """Load saved results and make quick validation plots."""
    save_dir = Path(save_dir)
    with open(save_dir / "darcy_opt-results.pkl", "rb") as fd:
        res = pickle.load(fd)["params"]
    with open(save_dir / "darcy_test_data.pkl", "rb") as fd:
        test_data = pickle.load(fd)

    x = jnp.asarray(test_data["x"])
    y = jnp.asarray(test_data["y"])
    test_in = jnp.asarray(test_data["test_in"])
    test_out = jnp.asarray(test_data["test_out"])

    darcy = get_darcy_solver(test_in.shape[-2:])

    darcy.forcing = sinusoid_forcing
    darcy.forcing_defaults = {"amplitude": 1.0}

    @jax.jit
    def solve_one(kappa, param):
        return darcy.solve({"conductivity": {"const": kappa}, "forcing": {"amplitude": param}})["phi"]

    solve_many = jax.vmap(solve_one, in_axes=(0, None))
    pred_test = solve_many(test_in, res)
    nshow = min(nshow, test_in.shape[0])
    plot_samples(test_in[:nshow], test_out[:nshow], x, y, predictions=pred_test[:nshow])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample Darcy random fields and solve Poisson.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the Darcy field sampler.")
    parser.add_argument("--shape", type=list[int], default=[32, 32], help="Shape of 2d grid")
    parser.add_argument("--save-dir", type=str, default="darcy-opt", help="Where to save results")
    args = parser.parse_args()
    
    key = jax.random.PRNGKey(args.seed)
    shape = tuple(args.shape)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    nshow = 3

    # show_random_samples(shape, nshow, key)
    train_forcing(shape, key, save_dir)
    plot_validation(save_dir, nshow)
