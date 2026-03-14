"""Small JAX demo of vmap, grad, and jit on a GPU.

Run with:

	uv run python scripts/temp_cuda.py
"""

from __future__ import annotations

from time import perf_counter

import jax
import jax.numpy as jnp


def scalar_linear_model(params: jax.Array, x: jax.Array) -> jax.Array:
	"""Compute a scalar linear model prediction, $y=wx+b$.

	:param params: model parameters as ``[w, b]``
	:param x: scalar input
	:return: scalar prediction
	"""
	w, b = params
	return w * x + b


# Vectorize the scalar model over a batch of x values.
batched_linear_model = jax.vmap(scalar_linear_model, in_axes=(None, 0))


def mse_loss(params: jax.Array, x_batch: jax.Array, y_batch: jax.Array) -> jax.Array:
	"""Compute mean squared error loss.

	:param params: model parameters as ``[w, b]``
	:param x_batch: batch of scalar inputs
	:param y_batch: batch of scalar targets
	:return: scalar mean squared error
	"""
	preds = batched_linear_model(params, x_batch)
	return jnp.mean((preds - y_batch) ** 2)


loss_grad = jax.grad(mse_loss)


@jax.jit
def train_step(params: jax.Array, x_batch: jax.Array, y_batch: jax.Array, lr: jax.Array) -> jax.Array:
	"""Single gradient-descent step, JIT-compiled by JAX.

	:param params: model parameters as ``[w, b]``
	:param x_batch: batch of scalar inputs
	:param y_batch: batch of scalar targets
	:param lr: learning rate
	:return: updated model parameters
	"""
	grads = loss_grad(params, x_batch, y_batch)
	return params - lr * grads


@jax.jit
def batched_predict(params: jax.Array, x_batch: jax.Array) -> jax.Array:
	"""JIT-compiled batched prediction using the vmapped model.

	:param params: model parameters as ``[w, b]``
	:param x_batch: batch of scalar inputs
	:return: batch of predictions
	"""
	return batched_linear_model(params, x_batch)


def pick_device() -> jax.Device:
	"""Pick GPU if available, otherwise fall back to CPU.

	:return: selected JAX device
	"""
	gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
	if gpu_devices:
		return gpu_devices[0]
	return jax.devices("cpu")[0]


def main() -> None:
	"""Run a tiny regression example with vmap, grad, and jit."""
	device = pick_device()
	print(f"Using JAX device: {device}")
	if device.platform != "gpu":
		print("GPU not found. Running on CPU fallback.")

	with jax.default_device(device):
		key = jax.random.key(0)
		x = jnp.linspace(-1.0, 1.0, 2048, dtype=jnp.float32)
		true_params = jnp.array([2.0, -0.5], dtype=jnp.float32)
		noise = 0.02 * jax.random.normal(key, shape=x.shape, dtype=jnp.float32)
		y = batched_linear_model(true_params, x) + noise

		params = jnp.array([0.0, 0.0], dtype=jnp.float32)
		lr = jnp.array(0.1, dtype=jnp.float32)

		initial_grads = loss_grad(params, x, y)
		print(f"Initial grads from grad: {initial_grads}")

		# Warm up JIT compilation before timing steady-state execution.
		params = train_step(params, x, y, lr)
		params.block_until_ready()

		start = perf_counter()
		for _ in range(300):
			params = train_step(params, x, y, lr)
		params.block_until_ready()
		elapsed_ms = 1_000.0 * (perf_counter() - start)

		preds = batched_predict(params, x)
		final_loss = mse_loss(params, x, y)
		preds.block_until_ready()
		final_loss.block_until_ready()

		print(f"Learned params: {params}")
		print(f"Final MSE: {float(final_loss):.6f}")
		print(f"JIT-ed 300 training steps took: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
	main()
