import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import numpy as np
from jax.typing import ArrayLike

from romjax import DictModel
from romjax.utils import merge_pytrees, to_pytree, save_h5, load_h5


def test_to_pytree():
    class Inner(DictModel):
        value: jnp.ndarray

    class Outer(DictModel):
        inner: Inner
        entries: list
        pair: tuple

    arr = jnp.array([1.0, 2.0])
    model = Outer(
        inner=Inner(value=arr),
        entries=[{"a": jnp.array(3.0)}, (jnp.array(4.0),)],
        pair=(jnp.array(5.0), {"b": jnp.array(6.0)}),
    )

    tree = to_pytree(model)

    assert isinstance(tree, dict)
    assert isinstance(tree["entries"], list)
    assert isinstance(tree["pair"], tuple)
    assert tree["inner"]["value"] is arr


def test_merge_pytrees():
    class Data(DictModel):
        data: ArrayLike
    
    data = Data(data=jnp.linspace(0, 1, 10))
    more_data = Data(data=jnp.array([1, 2, 3]))

    shared = jnp.array([1.0, 2.0])
    defaults = {
        "a": {"x": shared, "y": jnp.array(2.0)},
        "b": (jnp.array(3.0), {"z": jnp.array(4.0)}),
        "d": data
    }
    overrides = {
        "a": {"y": jnp.array(5.0), "new": jnp.array(6.0)},
        "b": (jnp.array(7.0),),
        "c": jnp.array(8.0),
        "e": more_data
    }

    merged = merge_pytrees(to_pytree(defaults), to_pytree(overrides))

    assert merged is not defaults           # a new dict is made
    assert merged["a"]["x"] is shared       # arrays are reused
    assert merged["d"]["data"] is data.data
    assert merged["e"]["data"] is more_data.data
    assert float(merged["a"]["y"]) == 5.0
    assert float(merged["a"]["new"]) == 6.0 # new paths are blazed
    assert float(merged["b"][0]) == 7.0
    assert float(merged["b"][1]["z"]) == 4.0
    assert float(merged["c"]) == 8.0


def test_to_pytree_merge_and_jit_grad():
    """We can still do jit/grad/vmap through all of the pytree merging shenanigans."""
    class Inner(DictModel):
        value: jnp.ndarray

    class Outer(DictModel):
        inner: Inner
        extra: dict

    defaults = Outer(inner=Inner(value=jnp.array(1.0)), extra={"scale": jnp.array(2.0)})
    overrides = Outer(inner=Inner(value=jnp.array(3.0)), extra={"bias": jnp.array(4.0)})

    def merged_sum(scale: jnp.ndarray) -> jnp.ndarray:
        override = {"inner": {"value": scale}}
        tree = merge_pytrees(to_pytree(defaults), merge_pytrees(to_pytree(overrides), override))
        return 2 * tree["inner"]["value"] + tree["extra"]["scale"] + tree["extra"]["bias"]

    true_val = merged_sum(jnp.array(5.0))
    true_vmap_val = jnp.array([merged_sum(1.), merged_sum(2.)])
    jit_val = jax.jit(merged_sum)(jnp.array(5.0))
    grad_val = jax.grad(merged_sum)(jnp.array(5.0))
    vmap_val = jax.vmap(merged_sum)(jnp.array([1.0, 2.0]))

    assert jnp.allclose(true_val, jit_val)
    assert jnp.allclose(true_vmap_val, vmap_val)
    assert jnp.allclose(grad_val, 2.0)  # only thing that matters is operations applied to the array


# def test_optimistix_fixed_point_iteration():
#     A = jnp.array([[2.0, 0.0], [0.0, 4.0]])
#     b = jnp.array([2.0, 8.0])
#     diag = jnp.diag(A)

#     def jacobi_update(x: jnp.ndarray, _: None) -> jnp.ndarray:
#         residual = A @ x - b
#         return x - residual / diag

#     solver = optx.FixedPointIteration(rtol=1e-10, atol=1e-10, damp=0.0)
#     sol = optx.root_find(jacobi_update, solver, y0=jnp.zeros_like(b), args=None, max_steps=50)

#     assert jnp.allclose(sol.value, jnp.array([1.0, 2.0]), atol=1e-6)
#     jit_val = jax.jit(lambda: optx.root_find(jacobi_update, solver, y0=jnp.zeros_like(b), args=None, max_steps=50).value)()
#     assert jnp.allclose(jit_val, jnp.array([1.0, 2.0]), atol=1e-6)


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


def test_optimistix_implicit_adjoint_grad():
    solver = optx.Newton(rtol=1e-10, atol=1e-10, linear_solver=lx.QR())
    adjoint = optx.ImplicitAdjoint(linear_solver=lx.QR())

    def solve_root(a: jnp.ndarray) -> jnp.ndarray:
        def F(y: jnp.ndarray, args: jnp.ndarray) -> jnp.ndarray:
            return y - (1.0 + args)

        sol = optx.root_find(F, solver, y0=jnp.array(0.0), args=a, max_steps=10, adjoint=adjoint)
        return sol.value

    grad = jax.grad(lambda a: solve_root(a))(jnp.array(0.5))
    eps = 1e-4
    fd = (solve_root(0.5 + eps) - solve_root(0.5 - eps)) / (2 * eps)
    assert jnp.allclose(grad, fd, atol=1e-3, rtol=1e-3)


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
