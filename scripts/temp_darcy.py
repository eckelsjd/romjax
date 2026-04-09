"""Based on Darcy flow by Kovachki et al. (2022):
    
    https://arxiv.org/abs/2108.08481

    Used for testing neural operators on PDEs.
"""
import argparse
import pickle
from pathlib import Path
import os
import shutil

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optimistix as optx
import optax

from romjax.poisson import Poisson2D, darcy_field
from romjax.plotting import gridplot, get_scheme
from romjax.optim import train, load_train_file
from romjax.utils import get_gpu_memory, monitor_gpu_memory, load_h5, save_h5, iter_pytree, pytree_at, get_logger
from romjax.rng import gen_keys

# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".50"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def get_darcy_solver(shape, bounds):
    """For testing Darcy flow: constant forcing, homogeneous dirichlet BCs on [0,1]^2
    with a random field conductivity kappa(x)=a(x).
    """
    darcy = Poisson2D(
        config={
            "solver": optx.Newton(rtol=1, atol=1e-3),
                                #   linear_solver=lx.GMRES(rtol=1e-3, atol=1e-20, max_steps=100, restart=30,
                                #                          stagnation_iters=30)),
            "grid": {"shape": shape, "bounds": bounds},
            "max_steps": 100,
            "throw": True
        },
        forcing_defaults={'const': -1.0},
        conductivity_defaults={'const': 1.0},
        conductivity_sampler='parametric',
        conductivity_sampler_opts={'const': {'distribution': darcy_field, 'shape': shape, 'bounds': bounds}}
    )

    return darcy


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


def show_random_samples(shape, bounds, nsamples, path, seed):
    """Just show some random fields and solutions."""
    darcy = get_darcy_solver(shape, bounds)
    x, y = darcy.config.grid.coords

    pairs = list(gen_keys(nsamples, path=path, seed=seed))
    keys, paths = zip(*pairs)
    keys = jnp.stack(keys)
    samples = jax.jit(jax.vmap(darcy.sample_inputs))(keys)
    solutions = jax.jit(jax.vmap(darcy.solve))(samples)
    
    for p, sample, sol in zip(paths, iter_pytree(samples), iter_pytree(solutions)):
        save_h5(sample, p / "inputs.h5")
        save_h5(sol, p / "solution.h5")

    fig, axs = plot_samples(samples['conductivity']["const"], solutions["phi"], x, y)

    plt.show()


def sinusoid_forcing(inputs, outputs):
    """Smooth one-parameter forcing to keep Poisson solver stable."""
    xg, yg = inputs["coords"]
    return inputs["amplitude"] * jnp.sin(jnp.pi * xg) * jnp.sin(jnp.pi * yg)


def train_forcing(shape, bounds, path, seed):
    """Try to learn a forcing parameter over some random fields."""
    # used_mib, total_mib = get_gpu_memory()[0]
    # print(f"Initial GPU: {used_mib} / {total_mib} MiB")
    key = jax.random.key(seed)
    train_key, test_key = jax.random.split(key, 2)

    ntrain = 10
    ntest = 5

    train_keys = jnp.stack(jax.random.split(train_key, ntrain))
    test_keys = jnp.stack(jax.random.split(test_key, ntest))

    darcy = get_darcy_solver(shape, bounds)
    x, y = darcy.config.grid.coords

    sample_many = jax.jit(jax.vmap(darcy.sample_inputs))
    train_in = sample_many(train_keys)
    test_in = sample_many(test_keys)

    # Use a "true" value on the training data
    darcy.forcing = sinusoid_forcing
    darcy.forcing_defaults = {"amplitude": 1.0}
    true_param = jnp.asarray(-1.0)
    param0 = jnp.asarray(-10.)

    @jax.jit
    def solve_one(inputs, param):
        d = {**inputs}
        d['forcing'] = {'amplitude': param}
        return darcy.solve(d)
    
    # used_mib, total_mib = get_gpu_memory()[0]
    # print(f"Before solve GPU: {used_mib} / {total_mib} MiB")
    # print("Starting GPU monitor...")
    # thread, stop_event = monitor_gpu_memory(0.3)

    solve_many = jax.vmap(solve_one, in_axes=(0, None), out_axes=0)
    train_out = solve_many(train_in, true_param)
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
            pred = solve_one(pytree_at(train_in, i), param)["phi"]
            diff = pred - pytree_at(train_out, i)["phi"]
            return acc + jnp.mean(diff**2)
        total = jax.lax.fori_loop(0, ntrain, body, 0.0)
        return total / ntrain
    
    @jax.jit
    def test_fn(param):
        def body(i, acc):
            pred = solve_one(pytree_at(test_in, i), param)["phi"]
            diff = pred - pytree_at(test_out, i)["phi"]
            return acc + jnp.mean(diff**2)
        total = jax.lax.fori_loop(0, ntest, body, 0.0)
        return total / ntest
    
    res = train(
        loss_fn,
        params0=param0,
        optimizer=optax.adam(0.2),
        max_steps=100,
        max_runtime_s=120,
        log_interval=5,
        plot_interval=5,
        test_fn=test_fn,
        test_tol=1e-6,
        live_plot=True,
        save=path,
        save_prefix="darcy_",
        logger=get_logger("darcy-opt", stdout=False, log_file=path/"train.log")
    )

    print(f"True param: {true_param}, Result: {res}")

    # Save test data for offline validation plots
    test_data = {
        "x": jax.device_get(x),
        "y": jax.device_get(y),
        "test_in": jax.device_get(test_in),
        "test_out": jax.device_get(test_out),
    }
    with open(path / "darcy_test_data.pkl", "wb") as fd:
        pickle.dump(test_data, fd)


def plot_validation(shape, bounds, path, nshow) -> None:
    """Load saved results and make quick validation plots."""
    save_dir = Path(path)
    res = load_train_file(save_dir)["params"]

    with open(save_dir / "darcy_test_data.pkl", "rb") as fd:
        test_data = pickle.load(fd)

    x = jnp.asarray(test_data["x"])
    y = jnp.asarray(test_data["y"])
    test_in = jax.tree.map(lambda x: jnp.asarray(x), test_data['test_in'])
    test_out = jax.tree.map(lambda x: jnp.asarray(x), test_data['test_out'])
    darcy = get_darcy_solver(shape, bounds)

    darcy.forcing = sinusoid_forcing
    darcy.forcing_defaults = {"amplitude": 1.0}

    @jax.jit
    def solve_one(inputs, param):
        d = {**inputs}
        d['forcing'] = {'amplitude': param}
        return darcy.solve(d)

    solve_many = jax.vmap(solve_one, in_axes=(0, None), out_axes=0)
    pred_test = solve_many(test_in, res)

    nshow = min(nshow, pred_test['phi'].shape[0])
    plot_samples(
        test_in['conductivity']['const'][:nshow], 
        test_out['phi'][:nshow], 
        x, y, 
        predictions=pred_test['phi'][:nshow]
    )
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample Darcy random fields and solve Poisson.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the Darcy field sampler.")
    parser.add_argument("--shape", type=int, default=32, help="Square shape of 2d grid")
    parser.add_argument("--path", type=str, default="darcy-opt", help="Where to save results")
    parser.add_argument("--nshow", type=int, default=3, help="How many samples to show")
    args = parser.parse_args()
    
    seed = args.seed
    bounds = ((0, 1), (0, 1))
    shape = (args.shape, args.shape)
    path = Path(args.path)
    nshow = args.nshow

    if path.exists():
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

    # thread, stop = monitor_gpu_memory(0.5)

    # show_random_samples(shape, bounds, nshow, path, seed)
    train_forcing(shape, bounds, path, seed)

    # stop.set()
    # thread.join()
    
    plot_validation(shape, bounds, path, nshow)
