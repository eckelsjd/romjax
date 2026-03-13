import jax
import jax.numpy as jnp
import flax
from flax import nnx
import optax
import timeit

jax.config.update('jax_platform_name', 'cpu')


## JIT
key = jax.random.key(0)
k1, k2, k3 = jax.random.split(key, 3)

def compute_stuff(x, w, b):
    y = x @ w
    y = y + b
    y = jnp.tanh(y)
    result = jnp.sum(y)
    return result

fast_compute_stuff = jax.jit(compute_stuff)

dim1, dim2, dim3 = 500, 1000, 500
xdata = jax.random.normal(k1, (dim1, dim2))
wdata = jax.random.normal(k2, (dim2, dim3))
bdata = jax.random.normal(k3, (dim3,))

# time_regular = timeit.timeit(
#     lambda: compute_stuff(xdata, wdata, bdata),
#     number=100
# )

# time_jit = timeit.timeit(
#     lambda: fast_compute_stuff(xdata, wdata, bdata),
#     number=100
# )

# print(f"compute_stuff: {time_regular:.4f}s")
# print(f"fast_compute_stuff: {time_jit:.4f}s")
# print(f"Speedup: {time_regular/time_jit:.2f}x")

## grad
def scalar_loss(params: dict[str, jnp.ndarray], x, y_true):
    y_pred = params['w'] * x + params['b']
    loss = jnp.mean((y_pred - y_true)**2)
    return loss

scalar_grad = jax.grad(scalar_loss)

params_init = {'w': jnp.array(2.0), 'b': jnp.array(1.0)}
x = jnp.array([1., 2., 3.])
y = jnp.array([7., 9., 11.])

grads = scalar_grad(params_init, x, y)
print(f'Init params: {params_init}')
print(f'Gradients:\n{grads}')

## vmap
def apply_affine(vector, matrix, bias):
    result = jnp.dot(matrix, vector) + bias
    return result

batch_size = 4
input_features = 3
output_features = 2

vectors = jax.random.normal(k1, (batch_size, input_features))
matrix = jax.random.normal(k2, (output_features, input_features))
bias = jax.random.normal(k3, (output_features,))

batch_affine = jax.vmap(apply_affine, in_axes=(0, None, None), out_axes=0)
print(f'batch result shape: {batch_affine(vectors, matrix, bias).shape}')

# flax nnx
class FNN(nnx.Module):

    def __init__(self, din, dout):
        self.fcl = nnx.Linear(din, dout)
    
    def __call__(self, x):
        return self.fcl(x)

model = FNN(3, 2)
lr = 0.001
opt = optax.adam(lr)
nn_opt = nnx.Optimizer(model, opt, wrt=nnx.Param)

@nnx.jit
def train(model, opt, x, y):
    def loss_fn(mod):
        ypred = mod(x)
        loss = jnp.mean((ypred-y)**2)
        return loss
    
    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    opt.update(model, grads)

xbatch = jax.random.normal(k1, (batch_size, 3))
ybatch = jax.random.normal(k2, (batch_size, 2))
