import jax.numpy as jnp
import jax
import equinox as eqx

from romjax.utils import validate_array_reducer, validate_error_reducer


key = jax.random.key(0)
k1, k2, k3 = jax.random.split(key, 3)
n1, n2 = jax.random.split(k3, 2)

phi = jax.random.normal(k1, (2, 3))
xi = jax.random.normal(k2, (5,))

tree = {
    "phi": phi,
    "xi": [xi, 6., (3.3, 2.)],
    "z": "hello",
}

tree_hat = {
    "phi": phi + jax.random.normal(n1, (2, 3)) * 0.05,
    "xi": [xi + jax.random.normal(n2, (5,)) * 0.05, 6.1, (3., 1.8)],
    "z": "goodbye"
}

a = eqx.filter(tree_hat, eqx.is_array_like)

leaf_error = validate_error_reducer("absolute")
err_tree = jax.tree.map(leaf_error, eqx.filter(tree, eqx.is_array_like), eqx.filter(tree_hat, eqx.is_array_like))
eqx.tree_pprint(err_tree, short_arrays=False)

leaves = jnp.concatenate([leaf.ravel() for leaf in jax.tree.leaves(err_tree)])
print(leaves)
print(jnp.mean(leaves))