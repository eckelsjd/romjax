import jax

def one(sample):
    return sample['x'] ** 2

many = [{'x': i} for i in range(5)]
sol = jax.vmap(one)(many)
print(sol)
