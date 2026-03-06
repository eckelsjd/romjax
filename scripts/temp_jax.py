import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

def sum_of_squares(x):
    return jnp.sum(x**2)

grad_sum = jax.grad(sum_of_squares)
x = jnp.asarray([1., 2., 3., 4.])
print(sum_of_squares(x))
print(grad_sum(x))