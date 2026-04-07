"""Optimization stuff."""
import logging
import time
import pickle
from pathlib import Path
from typing import Callable, Iterator, Any
from os import PathLike

import jax
import optax
from jaxtyping import PyTree
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt

from romjax.utils import tree_l2_norm


__all__ = ['train']


def _prettify_timedelta(delta: float) -> str:
    total_seconds = int(delta)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    if days > 0:
        return f"{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if minutes > 0:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{delta:.3f} s"


def train(
    loss_fn: Callable[[PyTree, Any], float],
    params0: PyTree,
    optimizer: optax.GradientTransformation,
    dataloader: Iterator[Any] | None = None,
    max_steps: int = 200,
    grad_tol: float = -1,
    param_tol: float = -1,
    loss_tol: float = -1,
    loss_window: int = 5,
    max_runtime_s: float = 300,
    log_interval: int = 1,
    plot_interval: int = 1,
    hist_interval: int = 1,
    save_interval: int = 0,
    test_fn: Callable[[PyTree], float] | None = None,
    test_tol: float = -1,
    live_plot: bool = True,
    save: str | PathLike | None = None,
    save_prefix: str = "",
    logger: logging.Logger | None = None,
) -> PyTree:
    def _infinite_loader():
        while True:
            yield ()
    if dataloader is None:
        dataloader = _infinite_loader()

    @eqx.filter_jit
    def step(params: PyTree, opt_state: optax.OptState, args: Any):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(params, args)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(params, eqx.is_array))
        params = eqx.apply_updates(params, updates)
        return params, opt_state, loss, grads

    # Initialize
    params = params0
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))
    curr_step = 1
    loss_val = jnp.inf
    test_val = jnp.inf
    loss_fig, loss_ax, test_fig, test_ax = None, None, None, None
    loss_line, test_line = None, None
    history = {"iter": [], "dt": [], "loss": []}
    if test_fn is not None:
        history["test"] = []
    
    if live_plot:
        plt.ion()
    
    if plot_interval > 0:
        loss_fig, loss_ax = plt.subplots(figsize=(6, 5), layout='tight')
        (loss_line,) = loss_ax.plot([], [], "-k")
        loss_ax.set_xlabel("Iteration")
        loss_ax.set_ylabel("Objective")
        loss_ax.set_yscale('log')
        loss_ax.grid(True)

        if test_fn is not None:
            test_fig, test_ax = plt.subplots(figsize=(6, 5), layout='tight')
            (test_line,) = test_ax.plot([], [], "-k")
            test_ax.set_xlabel("Iteration")
            test_ax.set_ylabel("Test score")
            test_ax.set_yscale('log')
            test_ax.grid(True)

    if logger is not None:
        logger.info("Initializing optimization")
    t_start = time.time()

    while True:
        # Get mini-batch 
        try:
            args = next(dataloader)
        except StopIteration:
            if logger is not None:
                logger.info(f"Args loader has stopped at k={curr_step}. Terminating...")
            break
    
        # Update step
        prev_params = params
        t0 = time.perf_counter()
        params, opt_state, loss, grads = step(params, opt_state, args)
        loss = jax.block_until_ready(loss)
        dt = time.perf_counter() - t0
        loss_val = float(loss)
        grad_norm = tree_l2_norm(grads) if grad_tol > 0 else -1

        if hist_interval > 0 and (curr_step % hist_interval == 0):
            history["iter"].append(curr_step)
            history["loss"].append(loss_val)
            history["dt"].append(dt)

        # Test set
        compute_test = test_fn is not None and hist_interval > 0 and (curr_step % hist_interval == 0)
        if compute_test:
            test_val = float(test_fn(params))
            history["test"].append(test_val)

        # Log
        if log_interval > 0 and (curr_step % log_interval == 0) and logger is not None:
            test_str = f"test={test_val:12.6e}" if compute_test else ""
            grad_str = f"grad={grad_norm:12.6e}" if grad_tol > 0 else ""
            elapsed_str = _prettify_timedelta(time.time()-t_start)
            logger.info(f"Elapsed: {elapsed_str}  |  k={curr_step:4d} dt={dt:8.3e} s loss={loss_val:12.6e} "
                        f"{test_str} {grad_str}")
        
        # Plot
        if plot_interval > 0 and (curr_step % plot_interval == 0):
            loss_line.set_data(history["iter"], history["loss"])
            loss_ax.relim()
            loss_ax.autoscale_view()
            loss_fig.canvas.draw_idle()
            loss_fig.canvas.flush_events()

            if test_fn is not None:
                test_line.set_data(history["iter"], history["test"])
                test_ax.relim()
                test_ax.autoscale_view()
                test_fig.canvas.draw_idle()
                test_fig.canvas.flush_events()

        # Check end conditions
        if curr_step >= max_steps:
            if logger is not None:
                logger.info(f"Termination criteria reached: {curr_step}/{max_steps} iterations")
            break
        if (t_diff := time.time() - t_start) >= max_runtime_s:
            if logger is not None:
                logger.info(f"Termination criteria reached: {t_diff}/{max_runtime_s} seconds")
            break
        if grad_tol > 0:
            if not jnp.isfinite(grad_norm):
                if logger is not None:
                    logger.warning("Grad norm is not finite. Terminating...")
                break
            if grad_norm < grad_tol:
                if logger is not None:
                    logger.info(f"Termination criteria reached: gradient norm {grad_norm:.2e} < {grad_tol:.2e}")
                break
        if param_tol > 0:
            param_norm = tree_l2_norm(jax.tree.map(
                lambda x, y: x - y, 
                eqx.filter(params, eqx.is_array), 
                eqx.filter(prev_params, eqx.is_array)
            ))
            if param_norm < param_tol:
                if logger is not None:
                    logger.info(f"Termination criteria reached: param step norm {param_norm:.2e} < {param_tol:.2e}")
                break
        if loss_tol > 0 and len(history["loss"]) > 2:
            loss_max = max(history["loss"][-loss_window:])
            loss_min = min(history["loss"][-loss_window:])
            if (loss_diff := abs(loss_max - loss_min) / max(1, loss_min)) < loss_tol:
                if logger is not None:
                    logger.info(f"Termination criteria reached: loss diff {loss_diff:.2e} < {loss_tol:.2e}")
                break
        if test_fn is not None and test_tol > 0:
            if len(history["test"]) > 1:
                if (test_score := max(history["test"][-loss_window:])) < test_tol:
                    if logger is not None:
                        logger.info(f"Termination criteria reached: test score {test_score:.2e} < {test_tol:.2e}")
                    break
        
        if save_interval > 0 and (curr_step % save_interval == 0):
            if save is not None:
                if logger is not None:
                    logger.info(f"Saving results at iter={curr_step}")
                with open(Path(save) / f"{save_prefix}opt-iter-{curr_step}.pkl", "wb") as fd:
                    d = {"params": params, "opt_state": opt_state, "loss": loss, "grads": grads, "history": history}
                    pickle.dump(d, fd)
                
                if plot_interval > 0:
                    loss_fig.savefig(Path(save) / f"{save_prefix}opt-loss.pdf", bbox_inches='tight')

                    if test_fn is not None:
                        test_fig.savefig(Path(save) / f"{save_prefix}opt-test.pdf", bbox_inches='tight')

        curr_step += 1

    # Save and exit
    if logger is not None:
        logger.info("Finishing Optimization")

    if live_plot:
        plt.waitforbuttonpress()
        plt.ioff()
    
    if save is not None:
        if logger is not None:
            logger.info("Saving final results...")
        with open(Path(save) / f"{save_prefix}opt-iter-{curr_step}.pkl", "wb") as fd:
            d = {"params": params, "opt_state": opt_state, "loss": loss, "grads": grads, "history": history}
            pickle.dump(d, fd)

        # History
        np.savetxt(
            Path(save) / f"{save_prefix}opt-history.csv", 
            np.array(list(history.values())).T, 
            fmt=["%d", "%8.3e", "%12.6e"] + (["%12.6e"] if test_fn is not None else []), 
            delimiter=",", 
            header=",".join(list(history.keys()))
        )

        # Plots
        if plot_interval > 0:
            loss_fig.savefig(Path(save) / f"{save_prefix}opt-loss.pdf", bbox_inches='tight')

            if test_fn is not None:
                test_fig.savefig(Path(save) / f"{save_prefix}opt-test.pdf", bbox_inches='tight')
    
    return params
