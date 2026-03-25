import logging
import time
import pickle
from pathlib import Path
from typing import Callable

import jax
import optax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ConfigDict

from romtools.utils import get_logger
from romtools.typing import PyTree


# 

class Optimizer(BaseModel):
    """Minimize scalar loss function with plotting and logging options."""
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    logger: logging.Logger = Field(default_factory=lambda: get_logger(Optimizer.__name__), exclude=True)

    def run_debug(
        self,
        loss_fn: Callable[[PyTree], float],
        params0: PyTree,
        optimizer: optax.GradientTransformation,
        max_steps: int = 200,
        grad_tol: float = 1e-6,
        param_tol: float = 1e-6,
        loss_tol: float = 1e-6,
        loss_window: int = 5,
        max_runtime_s: float = 300,
        log_interval: int = 1,
        plot_interval: int = 1,
        hist_interval: int = 1,
        save_interval: int = 0,
        test_fn: Callable[[PyTree], float] | None = None,
        test_tol: float = 0,
        live_plot: bool = True,
        save: str | Path | None = None,
        prefix: str = "",
    ) -> PyTree:
        @jax.jit
        def step(params: PyTree, opt_state: optax.OptState):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, grads
        
        @jax.jit
        def tree_l2_norm(tree: PyTree):
            return jnp.sqrt(jax.tree.reduce(lambda acc, x: acc + jnp.sum(x**2), tree, 0.0))

        # Initialize
        params = params0
        opt_state = optimizer.init(params)
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

            if test_fn is not None:
                test_fig, test_ax = plt.subplots(figsize=(6, 5), layout='tight')
                (test_line,) = test_ax.plot([], [], "-k")
                test_ax.set_xlabel("Iteration")
                test_ax.set_ylabel("Test score")
                test_ax.set_yscale('log')

        self.logger.info("Initializing optimization")
        t_start = time.time()

        while True:
            # Update step
            prev_params = params
            t0 = time.perf_counter()
            params, opt_state, loss, grads = step(params, opt_state)
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
            if log_interval > 0 and (curr_step % log_interval == 0):
                test_str = f"test={test_val:12.6e}" if compute_test else ""
                grad_str = f"grad={grad_norm:12.6e}" if grad_tol > 0 else ""
                self.logger.info(f"k={curr_step:4d} dt={dt:8.3e} s loss={loss_val:12.6e} {test_str} {grad_str}")
            
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
                self.logger.info(f"Termination criteria reached: {curr_step}/{max_steps} iterations")
                break
            if (t_diff := time.time() - t_start) >= max_runtime_s:
                self.logger.info(f"Termination criteria reached: {t_diff:d}/{max_runtime_s:d} seconds")
                break
            if grad_tol > 0:
                if not jnp.isfinite(grad_norm):
                    self.logger.warning("Grad norm is not finite. Terminating...")
                    break
                if grad_norm < grad_tol:
                    self.logger.info(f"Termination criteria reached: gradient norm {grad_norm:.2e} < {grad_tol:.2e}")
                    break
            if param_tol > 0:
                if (param_norm := tree_l2_norm(jax.tree.map(lambda x, y: x - y, params, prev_params))) < param_tol:
                    self.logger.info(f"Termination criteria reached: param step norm {param_norm:.2e} < {param_tol:.2e}")
                    break
            if len(history["loss"]) > 2:
                loss_max = max(history["loss"][-loss_window:])
                loss_min = min(history["loss"][-loss_window:])
                if (loss_diff := abs(loss_max - loss_min) / max(1, loss_min)) < loss_tol:
                    self.logger.info(f"Termination criteria reached: loss diff {loss_diff:.2e} < {loss_tol:.2e}")
                    break
            if test_fn is not None and test_tol > 0:
                if len(history["test"]) > 1:
                    if (test_score := max(history["test"][-loss_window:])) < test_tol:
                        self.logger.info(f"Termination criteria reached: test score {test_score:.2e} < {test_tol:.2e}")
                        break
            
            if save_interval > 0 and (curr_step % save_interval == 0):
                if save is not None:
                    self.logger.info(f"Saving results at iter={curr_step}")
                    with open(Path(save) / f"{prefix}opt-iter-{curr_step}.pkl", "wb") as fd:
                        d = {"params": params, "opt_state": opt_state, "loss": loss, "grads": grads, "history": history}
                        pickle.dump(d, fd)
                    
                    if plot_interval > 0:
                        loss_fig.savefig(Path(save) / f"{prefix}opt-loss.pdf", bbox_inches='tight')

                        if test_fn is not None:
                            test_fig.savefig(Path(save) / f"{prefix}opt-test.pdf", bbox_inches='tight')

            curr_step += 1

        # Save and exit
        self.logger.info("Finishing Optimization")

        if live_plot:
            plt.waitforbuttonpress()
            plt.ioff()
        
        if save is not None:
            self.logger.info("Saving final results...")
            with open(Path(save) / f"{prefix}opt-results.pkl", "wb") as fd:
                d = {"params": params, "opt_state": opt_state, "loss": loss, "grads": grads, "history": history}
                pickle.dump(d, fd)

            # History
            np.savetxt(
                Path(save) / f"{prefix}opt-history.csv", 
                np.array(list(history.values())).T, 
                fmt=["%d", "%8.3e", "%12.6e"] + (["%12.6e"] if test_fn is not None else []), 
                delimiter=",", 
                header=",".join(list(history.keys()))
            )

            # Plots
            if plot_interval > 0:
                loss_fig.savefig(Path(save) / f"{prefix}opt-loss.pdf", bbox_inches='tight')

                if test_fn is not None:
                    test_fig.savefig(Path(save) / f"{prefix}opt-test.pdf", bbox_inches='tight')
        
        return params

    def set_logger(self,
                   log_file: str | Path = None,
                   stdout: bool = None,
                   logger: logging.Logger = None,
                   level: int = logging.INFO):
        """Set a new `logging.Logger` object.

        :param log_file: log to file (if provided)
        :param stdout: whether to connect the logger to console (defaults to whatever is currently set or False)
        :param logger: the logging object to use (if None, then a new logger is created; this will override
                       the `log_file` and `stdout` arguments if set)
        :param level: the logging level to set (default is `logging.INFO`)
        """
        if stdout is None:
            stdout = False
            if self.logger is not None:
                for handler in self.logger.handlers:
                    if isinstance(handler, logging.StreamHandler):
                        stdout = True
                        break
        self.logger = logger or get_logger(self.__class__.__name__, log_file=log_file, stdout=stdout, level=level)
