import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from romjax.nn import Affine


def test_affine_last_layer_var_overwrites_jacobian_mlp_final_weights_only() -> None:
    affine = Affine(
        inputs_rank=2,
        outputs_rank=3,
        key=jax.random.key(0),
        last_layer_var=0.25,
    )

    assert affine.solution is not None
    assert affine.lower is not None
    assert affine.upper is not None
    assert affine.diagonal is not None
    solution_key, lower_key, upper_key, diagonal_key = jax.random.split(jax.random.key(0), 4)
    expected_solution = eqx.nn.MLP(
        in_size=2,
        out_size=3,
        width_size=2,
        depth=2,
        activation=jax.nn.swish,
        key=solution_key,
    )
    assert eqx.tree_equal(affine.solution, expected_solution)

    for module, key in zip(
        (affine.lower, affine.upper, affine.diagonal),
        (lower_key, upper_key, diagonal_key),
    ):
        expected = jax.random.normal(jax.random.split(key)[1], module.layers[-1].weight.shape) * 0.5
        assert jnp.array_equal(module.layers[-1].weight, expected)


def test_affine_identity_jac_init_initializes_jacobian_mlp_final_layers() -> None:
    affine = Affine(inputs_rank=2, outputs_rank=3, key=jax.random.key(1), identity_jac="init")

    assert affine.lower is not None
    assert affine.upper is not None
    assert affine.diagonal is not None
    assert jnp.array_equal(affine.lower.layers[-1].bias, jnp.zeros(3))
    assert jnp.array_equal(affine.upper.layers[-1].bias, jnp.zeros(3))
    assert jnp.array_equal(affine.diagonal.layers[-1].bias, jnp.ones(3))


def test_affine_default_initialization_is_unchanged() -> None:
    key = jax.random.key(2)
    default = Affine(inputs_rank=2, outputs_rank=3, key=key)
    explicit_default = Affine(inputs_rank=2, outputs_rank=3, key=key, last_layer_var=None, identity_jac=False)

    assert eqx.tree_equal(default, explicit_default)


def test_affine_rejects_negative_last_layer_var() -> None:
    with pytest.raises(ValueError, match="last_layer_var"):
        Affine(inputs_rank=1, outputs_rank=1, key=jax.random.key(3), last_layer_var=-1.0)
