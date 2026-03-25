from typing import Any, Callable, TypedDict
import time

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import optax
from pydantic import Field

from romtools.model import Model
from romtools.optimization import Optimizer

import os
from pathlib import Path


if (Path(os.getcwd()).name) != "scripts":
    os.chdir("scripts")

class LinearInputs(TypedDict):
    A: jnp.ndarray
    b: jnp.ndarray


class LinearOutputs(TypedDict):
    x: jnp.ndarray


class LinearResiduals(TypedDict):
    r: jnp.ndarray


class NonlinearInputs(TypedDict):
    c: jnp.ndarray


class NonlinearOutputs(TypedDict):
    x: jnp.ndarray


class NonlinearResiduals(TypedDict):
    r: jnp.ndarray


class LinearSystem(Model):
    """Simple linear system: residual = A x - b."""

    solver: Any = Field(default_factory=lambda: lx.QR())

    def evaluate(self, inputs: LinearInputs, outputs: LinearOutputs) -> LinearResiduals:
        A = jnp.asarray(inputs["A"])
        b = jnp.asarray(inputs["b"])
        x = jnp.asarray(outputs["x"])
        return {"r": A @ x - b}

    def solve(self, inputs: LinearInputs, residuals: LinearResiduals) -> LinearOutputs:
        A = jnp.asarray(inputs["A"])
        b = jnp.asarray(inputs["b"])
        target = jnp.asarray(residuals["r"])
        rhs = b + target

        def op(x: jnp.ndarray) -> jnp.ndarray:
            return A @ x

        struct = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
        operator = lx.FunctionLinearOperator(op, struct)
        solution = lx.linear_solve(operator, rhs, solver=self.solver)
        return {"x": solution.value}


class NonlinearSystem(Model):
    """Simple nonlinear system: residual = x^2 - c."""

    solver: Any = Field(default_factory=lambda: optx.Newton(rtol=1e-10, atol=1e-10, linear_solver=lx.QR()))
    adjoint: Any = Field(default_factory=lambda: optx.ImplicitAdjoint(linear_solver=lx.QR()))
    initial_guess: jnp.ndarray = Field(default_factory=lambda: jnp.array(1.0))
    max_steps: int = 100
    throw: bool = False

    def evaluate(self, inputs: NonlinearInputs, outputs: NonlinearOutputs) -> NonlinearResiduals:
        c = jnp.asarray(inputs["c"])
        x = jnp.asarray(outputs["x"])
        return {"r": x * x - c}

    def solve(self, inputs: NonlinearInputs, residuals: NonlinearResiduals) -> NonlinearOutputs:
        c = jnp.asarray(inputs["c"])
        target = jnp.asarray(residuals["r"])

        def residual_fn(x: jnp.ndarray, args: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            c_val, target_val = args
            return x * x - c_val - target_val

        solution = optx.root_find(
            residual_fn,
            solver=self.solver,
            y0=jnp.sqrt(jnp.clip(c, a_min=1e-6)),
            args=(c, target),
            max_steps=self.max_steps,
            adjoint=self.adjoint,
            throw=self.throw,
        )
        return {"x": solution.value}


def solve_known_systems() -> None:
    linear_model = LinearSystem()
    nonlinear_model = NonlinearSystem()

    A = jnp.array([[3.0, 1.0], [0.0, 2.0]])
    x_true = jnp.array([1.0, 2.0])
    b = A @ x_true
    linear_inputs: LinearInputs = {"A": A, "b": b}
    linear_residuals: LinearResiduals = {"r": jnp.zeros_like(b)}
    linear_solution = linear_model.solve(linear_inputs, linear_residuals)["x"]
    linear_residual_eval = linear_model.evaluate(linear_inputs, {"x": linear_solution})["r"]

    c = jnp.array(4.0)
    nonlinear_inputs: NonlinearInputs = {"c": c}
    nonlinear_residuals: NonlinearResiduals = {"r": jnp.array(0.0)}
    nonlinear_solution = nonlinear_model.solve(nonlinear_inputs, nonlinear_residuals)["x"]
    nonlinear_residual_eval = nonlinear_model.evaluate(nonlinear_inputs, {"x": nonlinear_solution})["r"]

    print("Linear solve: x =", linear_solution, "expected =", x_true)
    print("Linear residual check:", linear_residual_eval)
    print("Nonlinear solve: x =", nonlinear_solution, "expected =", jnp.sqrt(c))
    print("Nonlinear residual check:", nonlinear_residual_eval)


def optimize_linear_parameter() -> None:
    model = LinearSystem()
    alpha_true = jnp.array(3.0)
    b = jnp.array([1.0, 0.5])

    def make_A(alpha: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[alpha, 1.0], [1.0, 2.0]])

    target = model.solve({"A": make_A(alpha_true), "b": b}, {"r": jnp.zeros_like(b)})["x"]

    def loss(alpha: jnp.ndarray, args: dict[str, jnp.ndarray]) -> jnp.ndarray:
        b_val = args["b"]
        target_val = args["target"]
        x = model.solve({"A": make_A(alpha), "b": b_val}, {"r": jnp.zeros_like(b_val)})["x"]
        return jnp.sum((x - target_val) ** 2)

    solver = optx.OptaxMinimiser(optax.adam(0.2), rtol=1e-10, atol=1e-10)
    solution = optx.minimise(
        loss,
        solver=solver,
        y0=jnp.array(1.0),
        args={"b": b, "target": target},
        max_steps=200,
        throw=False,
    )

    print("Optimized linear alpha:", solution.value, "target:", alpha_true)


def optimize_nonlinear_parameter() -> None:
    model = NonlinearSystem()
    c_true = jnp.array(9.0)
    target = model.solve({"c": c_true}, {"r": jnp.array(0.0)})["x"]

    def loss(theta: jnp.ndarray) -> jnp.ndarray:
        c = jnp.exp(theta)
        x = model.solve({"c": c}, {"r": jnp.array(0.0)})["x"]
        return (x - target) ** 2

    opt = Optimizer()
    theta_hat = opt.run_debug(
        loss,
        jnp.array(0.0),
        optax.adam(0.2),
        loss_tol=-1,
        param_tol=-1,
        grad_tol=-1,
        max_steps=500,
        log_interval=20,
        plot_interval=20,
        hist_interval=1,
        save="res",
        prefix="test_",
        save_interval=0,
    )

    print("Optimized nonlinear c:", jnp.exp(theta_hat), "target:", c_true)


def optimize_linear_parameter_with_logging() -> None:
    model = LinearSystem()
    alpha_true = jnp.array(3.0)
    b = jnp.array([1.0, 0.5])
    b_test = jnp.array([3.0, 1.5])

    def make_A(alpha: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([[alpha, 1.0], [1.0, 2.0]])

    target = model.solve({"A": make_A(alpha_true), "b": b}, {"r": jnp.zeros_like(b)})["x"]
    test = model.solve({"A": make_A(alpha_true), "b": b_test}, {"r": jnp.zeros_like(b_test)})["x"]

    def loss(alpha: jnp.ndarray) -> jnp.ndarray:
        x = model.solve({"A": make_A(alpha), "b": b}, {"r": jnp.zeros_like(b)})["x"]
        return jnp.sum((x - target) ** 2)
    
    @jax.jit
    def test_score(alpha: jnp.ndarray):
        x = model.solve({"A": make_A(alpha), "b": b_test}, {"r": jnp.zeros_like(b_test)})["x"]
        return jnp.sum((x - test) ** 2)

    optimizer = optax.adam(0.2)
    alpha0 = jnp.array(1.0)
    opt = Optimizer()
    alpha_hat = opt.run_debug(
        loss,
        alpha0,
        optimizer,
        loss_tol=-1,
        param_tol=-1,
        grad_tol=1e-20,
        max_steps=1500,
        log_interval=100,
        plot_interval=50,
        hist_interval=1,
        save="res",
        prefix="test_",
        save_interval=0,
        test_fn=test_score
    )

    print("Optimized linear alpha:", alpha_hat, "target:", alpha_true)


def main() -> None:
    # solve_known_systems()
    # optimize_linear_parameter()
    optimize_linear_parameter_with_logging()
    # optimize_nonlinear_parameter()


if __name__ == "__main__":
    main()
