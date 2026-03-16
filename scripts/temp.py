from pydantic import Field

from romtools.typing import DictModel

from jax.typing import ArrayLike
import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt


class MyModel(DictModel):
    alpha: ArrayLike = 1.
    beta: ArrayLike = Field(default_factory=lambda: jnp.linspace(0, 1, 10))

def func(a, *overrides):
    for d in overrides:
        a.update(d)

a = MyModel()
b = a.model_dump()
print(b['beta'] is a['beta'])
print(b)
# a = {'1': 1}

# print(a)

# func(a, {'2': 2, '3': 3}, {'hello': 'goodbye'}, {'1': 4})

# print(a)

# a = MyModel()
# arr = jnp.linspace(0.5, 1.5, 20)

# b = MyModel(alpha=jnp.linspace(1, 2, 30))
# a.update(b)
# a.update({'beta': arr})

# def f(x):
#     return x ** 3

# df = jax.vmap(jax.grad(f))


# fig, ax = plt.subplots(layout='tight', figsize=(4,3))
# ax.plot(a['beta'], df(a['beta']), label='Beta')
# ax.plot(a['alpha'], df(a['alpha']), label='alpha')
# ax.legend()
# plt.show()
