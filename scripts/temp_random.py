import jax
import jax.numpy as jnp

seed = 0
base_key = jax.random.key(seed)

def sample_one(i):
    key_i = jax.random.fold_in(base_key, i)
    return jax.random.normal(key_i)

sample_many = jax.vmap(sample_one)

n1 = jax.random.normal(base_key)

x1 = sample_many(jnp.arange(0, 10))
x2 = sample_many(jnp.arange(10, 20))
x3 = sample_many(jnp.arange(0, 20))

assert jnp.allclose(x3[:10], x1)
assert jnp.allclose(x3[10:], x2)
