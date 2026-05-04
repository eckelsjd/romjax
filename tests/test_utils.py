import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from romjax.utils import load_h5, save_h5


# def test_lineax_gmres_matrix_free():
#     A = jnp.array([[3.0, 1.0], [0.0, 2.0]])
#     b = jnp.array([4.0, 2.0])

#     def op(x: jnp.ndarray) -> jnp.ndarray:
#         return A @ x

#     struct = jax.ShapeDtypeStruct(b.shape, b.dtype)
#     operator = lx.FunctionLinearOperator(op, struct)
#     solver = lx.GMRES(rtol=1e-10, atol=1e-10, max_steps=10, restart=10)
#     sol = lx.linear_solve(operator, b, solver=solver)

#     x_true = jnp.linalg.solve(A, b)
#     assert jnp.allclose(sol.value, x_true, atol=1e-6)


# def test_optimistix_implicit_adjoint_grad():
#     solver = optx.Newton(rtol=1e-10, atol=1e-10, linear_solver=lx.QR())
#     adjoint = optx.ImplicitAdjoint(linear_solver=lx.QR())

#     def solve_root(a: jnp.ndarray) -> jnp.ndarray:
#         def F(y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
#             return y - (1.0 + args)

#         sol = optx.root_find(F, solver, y0=jnp.array(0.0), args=a, max_steps=10, adjoint=adjoint)
#         return sol.value

#     grad = jax.grad(lambda a: solve_root(a))(jnp.array(0.5))
#     eps = 1e-4
#     fd = (solve_root(0.5 + eps) - solve_root(0.5 - eps)) / (2 * eps)
#     assert jnp.allclose(grad, fd, atol=1e-3, rtol=1e-3)


def test_save_load_h5(tmp_path):
    data = {
        "scalars": {"a": jnp.array(1.5), "b": jnp.array(2.5, dtype=jnp.float32)},
        "vectors": {"x": jnp.arange(5.0), "y": jnp.linspace(0.0, 1.0, 6)},
        "matrix": {"M": jnp.arange(12.0).reshape(3, 4)},
    }

    filename = tmp_path / "roundtrip.h5"
    save_h5(data, filename, mode="w")

    loaded: dict[str, ArrayLike] = {}
    load_h5(loaded, filename, mode="r", jax=True)

    orig_leaves, orig_def = jax.tree_util.tree_flatten(data)
    loaded_leaves, loaded_def = jax.tree_util.tree_flatten(loaded)

    assert orig_def == loaded_def
    assert len(orig_leaves) == len(loaded_leaves)
    for orig, got in zip(orig_leaves, loaded_leaves):
        assert np.asarray(orig).shape == np.asarray(got).shape
        assert np.asarray(orig).dtype == np.asarray(got).dtype
        assert np.allclose(np.asarray(orig), np.asarray(got))
