import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from pathlib import Path

import romjax as rox
from romjax.poisson import Poisson2D
from romjax.model import FilterModel
from romjax.nn import LinearProjection
from romjax.utils import pytree_at

if Path(os.getcwd()).name != 'demo':
    os.chdir('demo')

nsamples = 5
n_latent = 20

key = jax.random.key(0)
in_key, out_key, init_key = jax.random.split(key, 3)

graph: rox.FunctionGraph = rox.YamlLoader.load("poisson_graph.yml")

poisson: Poisson2D = graph.edges['poisson']
x, y = poisson.config.grid.coords

n_full = int(jnp.prod(jnp.asarray(x.shape)))
transform: FilterModel = graph.edges['coordinate transform']
pod = LinearProjection(n_latent=n_latent, n_full=n_full, key=init_key)

sample_inputs = jax.jit(jax.vmap(poisson.sample_inputs))
sample_outputs = jax.jit(jax.vmap(lambda key, solution: poisson.sample_outputs(key, solution=solution), in_axes=(0,0)))
solve = jax.jit(jax.vmap(poisson.solve))
evaluate = jax.jit(jax.vmap(poisson.evaluate, in_axes=(0, 0)))

sample_keys = jax.random.split(in_key, nsamples)
inputs = sample_inputs(sample_keys)
outputs = solve(inputs)
residuals = evaluate(inputs, outputs)

output_keys = jax.random.split(out_key, nsamples)
output_samples = sample_outputs(output_keys, outputs)
residual_samples = evaluate(inputs, output_samples)

in_tree = {'inputs': pytree_at(inputs, 0), 'outputs': pytree_at(output_samples, 0), 'filters': [pod]}
out_tree, aux_forward = transform.forward_aux(in_tree)
back_tree, aux_back = transform.backward_aux(out_tree, aux_forward)

# input_clim = (jnp.min(inputs['conductivity']['k0']), jnp.max(inputs['conductivity']['k0']))
# output_clim = (jnp.min(outputs['phi']), jnp.max(outputs['phi']))
# sample_clim = (jnp.min(output_samples['phi']), jnp.max(output_samples['phi']))
# residual_clim = (jnp.min(residual_samples['phi_residual']), jnp.max(residual_samples['phi_residual']))

# in_specs = [
#     {'kind': 'pcolor', 'data': (x, y, inputs['conductivity']['k0'][i]), 'opts': {'clim': input_clim}}
#     for i in range(nsamples)
# ]
# out_specs = [
#     {'kind': 'pcolor', 'data': (x, y, outputs['phi'][i]), 'opts': {'clim': 'auto'}}
#     for i in range(nsamples)
# ]
# sample_specs = [
#     {'kind': 'pcolor', 'data': (x, y, output_samples['phi'][i]), 'opts': {'clim': 'auto'}}
#     for i in range(nsamples)
# ]
# res_specs = [
#     {'kind': 'pcolor', 'data': (x, y, residual_samples['phi_residual'][i]), 'opts': {'clim': 'auto'}}
#     for i in range(nsamples)
# ]

# fig, ax = rox.gridplot(in_specs + out_specs + sample_specs + res_specs, shape=(4, nsamples), scheme='dark')
# plt.show()
