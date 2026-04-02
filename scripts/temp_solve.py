from typing import Any, Callable, TypedDict
import time

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import optax
from pydantic import Field
import matplotlib.pyplot as plt

from romtools.model import Model
from romtools.optimization import Optimizer
from romtools.utils import tree_l2_norm
from romtools.typing import PyTree
from romtools.plotting import gridplot
from romtools.solvers.utils import homogeneous_boundary, BoundarySpec
from romtools.solvers.poisson import const_initial_guess

import os
from pathlib import Path

# jax.config.update("jax_platforms", "cpu")
# jax.config.update("jax_enable_x64", True)

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
    initial_guess: Callable = Field(default_factory=lambda: const_initial_guess(1.0))
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


def optimize_newton_debug():
    def residual_callback(residual: PyTree) -> None:
        err = tree_l2_norm(residual).astype(float)
        # print(f"Residual norm: {err:.3e}")
        return
    
    # alpha = jnp.array([1., 1.2, 3., 3.2])
    # A = jnp.array([[3., 10., 30.], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    # P = 1e-4 * jnp.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]])
    # def hartmann_fn(params):
    #     x = params
    #     sum = jnp.array(0.0)
    #     for i in range(4):
    #         for j in range(3):
    #             sum = sum - alpha[i] * jnp.exp(-(A[i, j] * (x[j] - P[i, j])**2))
    #     return sum
    # params0 = jnp.array([0.5, 0.5, 0.5])
    # expected = jnp.array([0.114614, 0.555649, 0.852547])

    def rosenbrock(params, *args):
        x = params
        sum = jnp.array(0.0)
        for i in range(len(x) - 1):
            sum = sum + 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
        return sum

    # params0 = jnp.array([0.5, 0.5])
    # expected = jnp.ones_like(params0)
    # init_residual = rosenbrock(params0)

    def bohachevsky(params, *args):
        x1 = params[0]
        x2 = params[1]
        f1 = x1**2 + 2*x2**2 - 0.3*jnp.cos(3*jnp.pi*x1) - 0.4*jnp.cos(4*jnp.pi*x2) + 0.7
        f2 = x1**2 + 2*x2**2 - 0.3*jnp.cos(3*jnp.pi*x1)*jnp.cos(4*jnp.pi*x2) + 0.3
        f3 = x1**2 + 2*x2**2 - 0.3*jnp.cos(3*jnp.pi*x1 + 4*jnp.pi*x2) + 0.3
        res = {'f1': f1, 'f2': f2, 'f3': f3}
        return res

    params0 = jnp.array([-25, 57])
    init_residual = bohachevsky(params0)
    expected = jnp.array([0.0, 0.0])
    # solver = NewtonDebug(rtol=1e-3, atol=1e-5, callback=residual_callback)
    solver = optx.Newton(rtol=1e10, atol=1e-17)

    solution = optx.root_find(
        bohachevsky,
        solver=solver,
        y0=params0,
        max_steps=800,
        throw=False
    )
    # print(solution)
    print(f"Result: {optx.RESULTS[solution.result]}")
    print(f"Stats: {solution.stats}")
    print(f"Initial residual: {init_residual}")
    print(f"Final residual: {solution.state.f}")
    # print(f"Reduction final/init: {solution.state.f / init_residual}")
    print(f"Diff: {solution.state.diff}")
    print(f"Diff size: {solution.state.diffsize}")
    # error = tree_l2_norm(jax.tree.map(lambda x, y: x - y, solution.value, expected)) / tree_l2_norm(expected)
    error = jnp.linalg.norm(solution.value - expected) / max(1., jnp.linalg.norm(expected))
    print(f"Solution: {solution.value}, Expected: {expected}, Error: {error}")


def optimize_poisson():
    from romtools.solvers import Poisson2D

    neumann_bc = BoundarySpec(type='neumann', value=0.0)
    boundary = homogeneous_boundary(ndim=2, value=0.0)
    # boundary.boundary[1] = (neumann_bc, neumann_bc)  # y-bds
    # boundary.boundary[0] = (neumann_bc, neumann_bc)

    poisson = Poisson2D(
        config={
            "solver": optx.Newton(rtol=1e2, atol=1e-6),
            "grid": {"shape": (50, 50), "bounds": ((0, 1), (0, 1))},
            "throw": False,
            "max_steps": 50,
            # "initial_guess": lambda coords: jnp.ones_like(coords[0]) * 3
        },
        forcing_defaults={"const": 0.5},
        boundary_defaults=boundary
    )

    init_residual = poisson.evaluate({}, {"phi": jnp.zeros_like(poisson.config.grid.coords[0])})["phi_residual"]
    solution = poisson.solve(return_sol=True)
    print(f"Solution range: {solution.value.min()}, {solution.value.max()}")
    print(f"Solution shape: {solution.value.shape}")
    print(f"Result: {optx.RESULTS[solution.result]}")
    print(f"Stats: {solution.stats}")
    print(f"Diff: {jnp.linalg.norm(solution.state.diff)}")
    print(f"Diff size: {solution.state.diffsize}")
    print(f"Initial residual: {jnp.linalg.norm(init_residual)}")
    print(f"Final residual: {jnp.linalg.norm(solution.state.f)}")

    solve_spec = {
        'kind': 'pcolor',
        'data': (*poisson.config.grid.coords, solution.value),
        'opts': {'clim': 'auto', 'xlabel': "$x$", 'ylabel': "$y$", 'cbar_label': r"$\phi(x,y)$"},
        'kwargs': {'shading': 'gouraud'}
    }
    gridplot(solve_spec, subplot_size_in=(4, 3), scheme='dark')
    plt.show()


def main() -> None:
    # solve_known_systems()
    # optimize_linear_parameter()
    # optimize_linear_parameter_with_logging()
    # optimize_nonlinear_parameter()
    # optimize_newton_debug()
    optimize_poisson()


if __name__ == "__main__":
    main()
