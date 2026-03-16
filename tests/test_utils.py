import jax
import jax.numpy as jnp
import pytest
from jax.typing import ArrayLike

from romtools.utils import to_pytree, merge_pytrees
from romtools.solvers.utils import homogeneous_boundary, UniformGrid
from romtools.typing import DictModel


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


def test_merge_boundary_conditions():
    defaults = homogeneous_boundary(type="dirichlet", value=0.0, ndim=2)

    overrides = {
        "boundary": (
            (
                {"value": jnp.array(1.0)},
                {"value": jnp.array(2.0)},
            ),
            (
                {"type": "neumann", "value": jnp.array(0.5)},
                {"value": jnp.array(3.0)},
            ),
        )
    }

    merged = merge_pytrees(defaults, overrides)

    assert merged["boundary"][0][0]["type"] == "dirichlet"
    assert merged["boundary"][1][0]["type"] == "neumann"
    assert float(merged["boundary"][0][1]["value"]) == 2.0


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


def test_uniform_grid():
    # 1) Specifying bounds and shape and checking that spacing and coords are correct
    grid = UniformGrid(bounds=((0.0, 1.0), (0.0, 2.0)), shape=(2, 4))
    assert grid.spacing == (0.5, 0.5)
    assert grid.coords is not None
    assert grid.coords[0].shape == (2, 4)
    assert jnp.allclose(grid.coords[0][:, 0], jnp.array([0.25, 0.75]))
    assert jnp.allclose(grid.coords[1][0, :], jnp.array([0.25, 0.75, 1.25, 1.75]))

    # 2) Specifying bounds and spacing and checking that shape and coords are correct
    grid = UniformGrid(bounds=((0.0, 1.0), (0.0, 2.0)), spacing=(0.5, 0.5))
    assert grid.shape == (2, 4)
    assert grid.coords is not None
    assert grid.coords[0].shape == (2, 4)

    # 3) Specifying 1d coords and checking the resulting meshgrid (and shape, spacing, and bounds)
    x = jnp.array([0.25, 0.75])
    y = jnp.array([0.25, 0.75, 1.25, 1.75])
    grid = UniformGrid(coords=(x, y))
    assert grid.shape == (2, 4)
    assert grid.coords is not None
    assert jnp.allclose(jnp.array(grid.bounds[0]), jnp.array((0., 1.)))  # cell-centered
    assert jnp.allclose(jnp.array(grid.bounds[1]), jnp.array((0., 2.)))

    # 4) Specifying 2d coords and checking shape, spacing, and bounds
    xg, yg = jnp.meshgrid(x, y, indexing="ij")
    grid = UniformGrid(coords=(xg, yg))
    assert grid.shape == (2, 4)
    assert grid.coords is not None
    assert jnp.allclose(jnp.array(grid.bounds[0]), jnp.array((0., 1.)))
    assert jnp.allclose(jnp.array(grid.bounds[1]), jnp.array((0., 2.)))

    # 5) Making sure we get validation errors for misspecified coords or shape/spacing + bounds
    with pytest.raises(ValueError):
        UniformGrid(bounds=((0.0, 1.0),), shape=(2,), spacing=(0.25,))

    with pytest.raises(ValueError):
        UniformGrid(coords=(jnp.array([0.0, 1.0]), jnp.array([[0.0, 1.0], [2.0, 3.0]])))

    # 6) Make sure we don't serialize big coords array
    d = grid.model_dump()
    assert 'coords' not in d
