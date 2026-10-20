---
name: jax
description: Use when writing, reviewing, testing, or profiling JAX code in romjax, including transformed numerical functions, PyTrees, model training, or differentiable PDE solvers.
---

# JAX in romjax

## Core requirements

- Keep numerical functions pure: pass arrays, keys, and changing state explicitly; avoid mutation, hidden randomness, and Python side effects in transformed code. Use `jax.random` with explicit keys and split or fold keys deterministically.
- Write functions for single samples and PyTrees where practical. Add batching with `jax.vmap` (or `eqx.filter_vmap` for mixed array and non-array inputs); use `jax.lax.map` when sequential mapping or bounded memory is needed. Define `in_axes` explicitly when arguments have different batching behavior.
- JIT stable numerical computations with `jax.jit` or `eqx.filter_jit`. Keep shapes and control flow traceable, and separate configuration/static metadata from changing array values. Use `jax.lax` control-flow primitives for data-dependent loops or branches inside compiled functions. Avoid converting traced values to Python scalars or using them in Python conditionals.
- Use `jax.grad` or `jax.value_and_grad` for differentiable objectives. Preserve gradients through the intended path; use `jax.lax.stop_gradient` only when deliberately stopping them. Check gradients on small, representative inputs.
- Treat nested parameter, state, batch, and result structures as PyTrees. Use `jax.tree_util` for standard flatten/map/path operations and `romjax.tree` helpers when they match the needed operation. Register custom container types as PyTrees (or use `eqx.Module`) so transformations see their array leaves correctly.
- Annotate public array and PyTree interfaces with `jaxtyping` types such as `ArrayLike`, `Key`, and `PyTree` where useful. These annotations clarify shapes and roles; they do not replace runtime validation when inputs need validation.

## Repository library choices

- **Model training:** represent differentiable model state with `equinox` modules and use `eqx.filter_jit` / `eqx.filter_vmap` for mixed PyTrees. Use `optax` transformations for optimizer state and updates. Use Orbax for checkpoint save and restore; keep checkpoint boundaries explicit and test restoration against the expected PyTree structure.
- **Differentiable PDEs:** use `diffrax` for ODE/time integration, `lineax` for linear solves, and `optimistix` for root finding or nonlinear solves. Choose solver and adjoint deliberately, expose solver settings through existing configuration patterns, and check solver status/residuals as well as gradients.
- Prefer these libraries for their matching problem domain; do not add a dependency when JAX primitives or an existing romjax abstraction suffice.

## Verification and performance

- Test eager and transformed behavior on small deterministic examples. Cover expected values, output PyTree structure and shapes, and gradients for differentiable paths. Where relevant, compare eager and JIT results within numerical tolerance and exercise `vmap` with more than one sample.
- Debug tracing failures by reducing the function and checking for Python work on tracers, shape changes, or incompatible PyTree structures. Use `jax.debug.print` temporarily for values inside compiled code; ordinary `print` runs at trace time.
- Profile before optimizing. Use `jax.profiler` or the helpers in `romjax.profiling`; synchronize measured results with `block_until_ready()` so timings include device execution. Separate compilation time from steady-state execution and warm up compiled functions before comparing runs.

## Repository examples

- Training and Optax/Orbax patterns: `src/romjax/train.py`
- Diffrax, Lineax, and Optimistix integration: `src/romjax/pde.py`
- PyTree helpers: `src/romjax/tree.py`
- Profiling helpers: `src/romjax/profiling.py`
- Focused tests: `tests/test_train.py`, `tests/test_pde.py`, `tests/test_tree.py`, and `tests/test_profiling.py`
