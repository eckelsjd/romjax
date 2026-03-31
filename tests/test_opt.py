from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
import optax

from romtools.optimization import Optimizer
from romtools.typing import PyTree


def test_pytree_opt_debug(tmp_path):
    """Simple function approximation with optax adam and nested pytrees."""
    true_params = {
        "encoder": {
            "A_x": jnp.array([[0.7, -0.2, 0.1, 0.4],
                              [-0.3, 0.5, 0.6, -0.1]]),
            "A_u": jnp.array([[0.2, -0.4, 0.3, 0.1],
                              [0.6, 0.2, -0.5, 0.3],
                              [-0.1, 0.7, 0.2, -0.6]]),
            "b0": jnp.array([0.2, -0.1, 0.3, 0.05]),
        },
        "modulator": {
            "w": jnp.array([0.9]),
            "b": jnp.array([0.1]),
        },
        "heads": {
            "y": {
                "W": jnp.array([[0.5, -0.2],
                                [0.1, 0.6],
                                [-0.3, 0.4],
                                [0.2, -0.5]])
            },
            "z": {
                "W": jnp.array([[0.4],
                                [-0.1],
                                [0.2],
                                [0.3]]),
                "b": jnp.array([0.1]),
            },
        },
        "scales": (
            jnp.array([1.2]),
            {"z": jnp.array([0.8])},
        ),
    }
    
    def true_model(inputs: PyTree) -> PyTree:
        # Nonlinear model with multiple inputs and outputs
        x = inputs["x"]
        u = inputs["u"]
        meta = inputs["meta"]["scale"]

        h = jnp.tanh(x @ true_params["encoder"]["A_x"] + u @ true_params["encoder"]["A_u"] + true_params["encoder"]["b0"])
        mod = meta * true_params["modulator"]["w"] + true_params["modulator"]["b"]
        h2 = jnp.tanh(h * mod)

        y = h2 @ true_params["heads"]["y"]["W"] + true_params["scales"][0]
        z = jnp.tanh(h2 @ true_params["heads"]["z"]["W"] + true_params["heads"]["z"]["b"]) * true_params["scales"][1]["z"]
        energy = jnp.sum(h2**2, axis=1, keepdims=True)

        return {
            "y": y,
            "aux": {
                "z": z,
                "energy": energy,
            },
        }

    def approx_model(inputs: PyTree, params: PyTree) -> PyTree:
        # Approximate model with nested pytree parameterization
        x = inputs["x"]
        u = inputs["u"]
        meta = inputs["meta"]["scale"]

        h = jnp.tanh(x @ params["encoder"]["A_x"] + u @ params["encoder"]["A_u"] + params["encoder"]["b0"])
        mod = meta * params["modulator"]["w"] + params["modulator"]["b"]
        h2 = jnp.tanh(h * mod)

        y = h2 @ params["heads"]["y"]["W"] + params["scales"][0]
        z = jnp.tanh(h2 @ params["heads"]["z"]["W"] + params["heads"]["z"]["b"]) * params["scales"][1]["z"]
        energy = jnp.sum(h2**2, axis=1, keepdims=True)

        return {
            "y": y,
            "aux": {
                "z": z,
                "energy": energy,
            },
        }

    key = jax.random.PRNGKey(0)
    key_train, key_test = jax.random.split(key, 2)
    n_train = 128
    n_test = 64

    def sample_inputs(key: jax.Array, n_samples: int) -> PyTree:
        kx, ku, km = jax.random.split(key, 3)
        x = jax.random.normal(kx, (n_samples, 2))
        u = jax.random.normal(ku, (n_samples, 3))
        scale = 0.6 + 0.4 * jax.random.uniform(km, (n_samples, 1))
        return {
            "x": x,
            "u": u,
            "meta": {
                "scale": scale,
            },
        }

    train_inputs = sample_inputs(key_train, n_train)
    test_inputs = sample_inputs(key_test, n_test)
    train_data = {
        "inputs": train_inputs,
        "targets": true_model(train_inputs),
    }
    test_data = {
        "inputs": test_inputs,
        "targets": true_model(test_inputs),
    }

    @jax.jit
    def tree_mse(pred: PyTree, target: PyTree) -> jax.Array:
        leaves_p = jax.tree_util.tree_leaves(pred)
        leaves_t = jax.tree_util.tree_leaves(target)
        losses = [jnp.mean((p - t) ** 2) for p, t in zip(leaves_p, leaves_t)]
        return jnp.mean(jnp.stack(losses))

    def loss_fn(params: PyTree) -> PyTree:
        # MSE over training data using approx model
        preds = approx_model(train_data["inputs"], params)
        return tree_mse(preds, train_data["targets"])

    @jax.jit
    def test_score(params: PyTree) -> PyTree:
        # MSE over test data using approx model
        preds = approx_model(test_data["inputs"], params)
        return tree_mse(preds, test_data["targets"])

    opt = Optimizer()
    opt.set_logger(log_file=Path(tmp_path) / 'opt.log', stdout=False)
    options = {
        "save": tmp_path,
        "prefix": "test_",
        "live_plot": False,
        "save_interval": 100,
        "log_interval": 100,
        "plot_interval": 100,
        "max_steps": 400,
        "max_runtime_s": 10,
        "test_fn": test_score,
        "loss_tol": 1e-8
    }
    leaves, treedef = jax.tree_util.tree_flatten(true_params)
    keys = jax.random.split(jax.random.PRNGKey(123), len(leaves))
    perturbed_leaves = [leaf + 0.15 * jax.random.normal(k, leaf.shape) for leaf, k in zip(leaves, keys)]
    params0 = jax.tree_util.tree_unflatten(treedef, perturbed_leaves)
    optimizer = optax.adam(0.2)
    params_hat = opt.run_debug(loss_fn, params0, optimizer, **options)
    
    train_score = float(loss_fn(params_hat))
    test_score_val = float(test_score(params_hat))
    assert train_score < 2.5e-3
    assert test_score_val < 3.5e-3

    leaves_hat = jax.tree_util.tree_leaves(params_hat)
    leaves_true = jax.tree_util.tree_leaves(true_params)
    rel_errors = [
        float(jnp.linalg.norm(a - b) / jnp.maximum(1e-6, jnp.linalg.norm(b)))
        for a, b in zip(leaves_hat, leaves_true)
    ]
    assert max(rel_errors) < 0.1

    log_file = Path(tmp_path) / "opt.log"
    assert log_file.exists()

    results_file = Path(tmp_path) / "test_opt-results.pkl"
    history_file = Path(tmp_path) / "test_opt-history.csv"
    loss_plot = Path(tmp_path) / "test_opt-loss.pdf"
    test_plot = Path(tmp_path) / "test_opt-test.pdf"
    assert results_file.exists()
    assert history_file.exists()
    assert loss_plot.exists()
    assert test_plot.exists()

    with open(results_file, "rb") as fd:
        saved = pickle.load(fd)
    saved_params = saved["params"]
    saved_leaves = jax.tree_util.tree_leaves(saved_params)
    for a, b in zip(saved_leaves, leaves_hat):
        assert jnp.allclose(a, b, atol=1e-6, rtol=1e-6)
