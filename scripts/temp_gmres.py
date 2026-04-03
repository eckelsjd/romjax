import lineax as lx

import jax.numpy as jnp
import jax

import time
import os

# jax.config.update("jax_enable_x64", False)
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


N = 256
h = 1.0 / (N + 1)
diag = (-4.0 / h**2) * jnp.ones(N*N)

@jax.jit
def laplace_2d(u):
    u = u.reshape(N, N)
    
    u_pad = jnp.pad(u, 1)
    lap = (
        u_pad[1:-1, :-2] +
        u_pad[1:-1, 2:] +
        u_pad[:-2, 1:-1] +
        u_pad[2:, 1:-1] -
        4 * u_pad[1:-1, 1:-1]
    ) / h**2
    
    return lap.reshape(-1)

shape = jax.ShapeDtypeStruct((N*N,), jnp.float32)

A = lx.FunctionLinearOperator(laplace_2d, shape)
b = jnp.ones(N*N)

# Seems like rtol ~ 1e-3 to 1e-4 is best possible for float32
# Can maybe add atol ~ 1e-5 for small numbers
rtol = 1e-2
atol = 1e-20
max_steps = 300
restart = 30
stag = 30

M = lx.IdentityLinearOperator(shape)
# M = lx.FunctionLinearOperator(lambda v: v / diag, shape)

use_multigrid = False
mg_levels = 4
mg_pre_smooth = 2
mg_post_smooth = 2
mg_coarse_iters = 20
mg_omega = 0.8

def _laplace_2d_n(u, n, h):
    u = u.reshape(n, n)
    u_pad = jnp.pad(u, 1)
    lap = (
        u_pad[1:-1, :-2] +
        u_pad[1:-1, 2:] +
        u_pad[:-2, 1:-1] +
        u_pad[2:, 1:-1] -
        4 * u_pad[1:-1, 1:-1]
    ) / (h * h)
    return lap.reshape(-1)

def _restrict_full_weighting(r, n):
    r = r.reshape(n, n)
    r_pad = jnp.pad(r, 1)
    rc = (
        4 * r_pad[1:-1:2, 1:-1:2] +
        2 * (
            r_pad[1:-1:2, :-2:2] +
            r_pad[1:-1:2, 2::2] +
            r_pad[:-2:2, 1:-1:2] +
            r_pad[2::2, 1:-1:2]
        ) +
        (
            r_pad[:-2:2, :-2:2] +
            r_pad[:-2:2, 2::2] +
            r_pad[2::2, :-2:2] +
            r_pad[2::2, 2::2]
        )
    ) / 16.0
    return rc.reshape(-1)

def _prolong_piecewise_constant(e_c, n_coarse):
    e_c = e_c.reshape(n_coarse, n_coarse)
    e_f = jnp.repeat(jnp.repeat(e_c, 2, axis=0), 2, axis=1)
    return e_f.reshape(-1)

def _build_multigrid_preconditioner(n, levels, pre_smooth, post_smooth, coarse_iters, omega):
    ns = [n // (2**k) for k in range(levels)]
    hs = [1.0 / (ni + 1) for ni in ns]
    diags = [(-4.0 / (hi * hi)) for hi in hs]

    def _jacobi_smooth(u, f, n, h, diag, iters):
        def _body(_, u):
            r = f - _laplace_2d_n(u, n, h)
            return u + omega * (r / diag)
        return jax.lax.fori_loop(0, iters, _body, u)

    def _v_cycle(level, f):
        n = ns[level]
        h = hs[level]
        diag = diags[level]
        u = jnp.zeros_like(f)
        u = _jacobi_smooth(u, f, n, h, diag, pre_smooth)
        if level == levels - 1:
            return _jacobi_smooth(u, f, n, h, diag, coarse_iters)
        r = f - _laplace_2d_n(u, n, h)
        r_c = _restrict_full_weighting(r, n)
        e_c = _v_cycle(level + 1, r_c)
        e_f = _prolong_piecewise_constant(e_c, ns[level + 1])
        u = u + e_f
        return _jacobi_smooth(u, f, n, h, diag, post_smooth)

    return jax.jit(lambda f: _v_cycle(0, f))

# if use_multigrid:
#     mg_apply = _build_multigrid_preconditioner(
#         N,
#         mg_levels,
#         mg_pre_smooth,
#         mg_post_smooth,
#         mg_coarse_iters,
#         mg_omega,
#     )
#     M = lx.FunctionLinearOperator(mg_apply, shape)

# Very roughly, we need || b - Ax || < || atol + rtol|b| ||  *and*  ||x_(n+1) - x_(n)|| < || atol + rtol|xn| ||
solver = lx.GMRES(rtol, atol, max_steps=max_steps, restart=restart, stagnation_iters=stag)

t0 = time.perf_counter()
sol = lx.linear_solve(A, b, solver, options=dict(preconditioner=M))
sol_time = time.perf_counter() - t0

# t0 = time.perf_counter()
# true_sol = lx.linear_solve(A, b)
# true_time = time.perf_counter() - t0

# diff = jnp.max(jnp.abs(sol.value - true_sol.value))

print(f"Solution range: {sol.value.min()}, {sol.value.max()}")
print(f"Solution shape: {sol.value.shape}")
print(f"Result: {lx.RESULTS[sol.result]}")
print(f"Stats: {sol.stats}")
print(f"Solve time: {sol_time}")
print(" ")
# print(f"QR results: {lx.RESULTS[true_sol.result]}")
# print(f"QR range: {true_sol.value.min()}, {true_sol.value.max()}")
# print(f"QR time: {true_time}")
# print(f" ")
# print(f"Max difference: {diff}")
