"""Simple demos of the `Train` routine. No graphs. Just plain functions."""

from typing import Literal
import sys

import jax
import optax
import jax.numpy as jnp
import equinox as eqx
from loguru import logger
import matplotlib.pyplot as plt

import romjax as romx

from romjax.train import TerminationConfig, DiagnosticsConfig, CheckpointerConfig

logger.disable("romjax")


def get_method(method: Literal["linear", "mlp"]):
    """Get one of the demo cases."""
    if method == "linear":
        def approx_model(x, params):
            return params["weight"] * x + params["bias"]
        
        def true_model(x):
            true_params = {"weight": 3, "bias": -2}
            return approx_model(x, true_params)
        
        def init_params(key):
            weight_key, bias_key = jax.random.split(key)
            return {"weight": jax.random.normal(weight_key), "bias": jax.random.normal(bias_key)}

    elif method == "mlp":
        def approx_model(x, params):
            return eqx.filter_vmap(lambda x: romx.eqx_evaluate(x, params))(x)
        
        def true_model(x):
            return jnp.tanh(x)
        
        def init_params(key):
            mlp = eqx.nn.MLP(
                in_size='scalar', 
                out_size='scalar', 
                width_size=5, 
                depth=2, 
                key=key, 
                activation=jax.nn.sigmoid
            )
            return mlp

    setattr(init_params, "sample", init_params.__call__)

    return approx_model, true_model, init_params


def run(method="linear"):

    approx_model, true_model, init_params = get_method(method)

    def mse(params, data):
        x, y = data
        yhat = approx_model(x, params)
        return jnp.mean(jnp.square(y - yhat))
    
    key = jax.random.key(0)
    train_key, test_key = jax.random.split(key, 2)
    xtrain_key, ytrain_key = jax.random.split(train_key, 2)
    xtest_key, ytest_key = jax.random.split(test_key, 2)

    ntrain = 64
    ntest= 32

    xtrain = jax.random.uniform(xtrain_key, shape=ntrain) * 4 - 2
    ytrain = true_model(xtrain) + jax.random.normal(ytrain_key, shape=ntrain) * 0.1

    xtest = jax.random.uniform(xtest_key, shape=ntest) * 4 - 2
    ytest = true_model(xtest) + jax.random.normal(ytest_key, shape=ntest) * 0.1

    test_fn = lambda params: mse(params, (xtest, ytest))
    dl = romx.BatchDataLoader(data=(xtrain, ytrain), batch_size=16, shuffle_seed=0)

    train = romx.Train(
        routine_config=dict(gridplot=dict(scheme="dark", subplot_size_in=(4, 3)), mplstyle="romjax.stix"),
        loss=mse,
        init_params=init_params,
        optimizer=optax.adam(0.1),
        test=test_fn,
        dataloader=dl,
        termination=TerminationConfig(max_steps=100),
        diagnostics=DiagnosticsConfig(plot_interval=10, test_interval=10, live_plot=True),
        init_seed=0,
        # checkpointer=CheckpointerConfig(),
        # root="demo/train",
        # write_policy="reuse"
    )

    approx_params = train()

    idx_train = jnp.argsort(xtrain)
    ytrain_true = true_model(xtrain[idx_train])
    ytrain_pred = approx_model(xtrain[idx_train], approx_params)

    idx_test = jnp.argsort(xtest)
    ytest_true = true_model(xtest[idx_test])
    ytest_pred = approx_model(xtest[idx_test], approx_params)

    print(f"Train relative error: {romx.BinaryOp('relative')(ytrain_true, ytrain_pred):.5f}")
    print(f"Validation relative error: {romx.BinaryOp('relative')(ytest_true, ytest_pred)}")

    fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
    ax.plot(xtrain, ytrain, "or", label="Training data")
    ax.plot(xtest, ytest, "ob", label="Test data")
    ax.plot(xtrain[idx_train], ytrain_true, "-k", label="True")
    ax.plot(xtrain[idx_train], ytrain_pred, "--r", label="Model")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        method = sys.argv[1]
    else:
        method = "linear"
    
    supported = ["linear", "mlp"]
    if method not in supported:
        sys.exit(f"Method {method} unknown. Only {supported}")

    run(method)
