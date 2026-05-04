import jax.numpy as jnp
import pytest

from romjax.pde import BoundaryType, UniformGrid, homogeneous_boundary
from romjax.tree import pytree_merge


def test_merge_boundary_conditions():
    defaults = homogeneous_boundary(type="dirichlet", value=0.0, ndim=2)

    overrides = {
        "boundary": [
            (
                {"value": jnp.array(1.0)},
                {"value": jnp.array(2.0)},
            ),
            (
                {"type": BoundaryType.neumann, "value": jnp.array(0.5)},
                {"value": jnp.array(3.0)},
            ),
        ]
    }

    merged = pytree_merge(defaults, overrides)

    assert merged["boundary"][0][0]["type"] == BoundaryType.dirichlet
    assert merged["boundary"][1][0]["type"] == BoundaryType.neumann
    assert float(merged["boundary"][0][1]["value"]) == 2.0


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
    