import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from romjax.typing import DictModel
from romjax.tree import (
    UnaryOperator, 
    ErrorOperator, 
    TreeErrorOperator, 
    to_pytree, 
    pytree_merge
)


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

    merged = pytree_merge(to_pytree(defaults), to_pytree(overrides))

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
        tree = pytree_merge(to_pytree(defaults), pytree_merge(to_pytree(overrides), override))
        return 2 * tree["inner"]["value"] + tree["extra"]["scale"] + tree["extra"]["bias"]

    true_val = merged_sum(jnp.array(5.0))
    true_vmap_val = jnp.array([merged_sum(1.), merged_sum(2.)])
    jit_val = jax.jit(merged_sum)(jnp.array(5.0))
    grad_val = jax.grad(merged_sum)(jnp.array(5.0))
    vmap_val = jax.vmap(merged_sum)(jnp.array([1.0, 2.0]))

    assert jnp.allclose(true_val, jit_val)
    assert jnp.allclose(true_vmap_val, vmap_val)
    assert jnp.allclose(grad_val, 2.0)  # only thing that matters is operations applied to the array


def test_unary_operator():
    # Validation from hyphen-string
    # Composite op works with grad,vmap,jit per usual
    # Saving op_str
    pass


def test_error_operator():
    # From error_fn
    # From op and norm
    pass


def test_tree_error_operator():
    # From single unary string
    # From ops and norm
    # With override mean/norm
    # Make sure we get floats
    pass


def test_pytree_reduce():
    # Compare mean/norm with reduce
    # See how array_like filtering works
    pass


def test_get_operators_by_alias():
    # get unary operator
    # get binary operator
    # get tree operator
    pass


def test_pytree_iter():
    # pytree iter
    # pytree at and size
    pass
