from typing import Literal, Callable
import copy
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, OdeSolver
from scipy.integrate._ivp.rk import RungeKutta
from numpy.typing import ArrayLike


class RK4(RungeKutta):
    """Fixed-step classical Runge-Kutta 4th order method for solve_ivp. Pass dt to set constant time step."""
    order = 4
    n_stages = 4
    error_estimator_order = 4
    C = np.array([0, 1/2, 1/2, 1])      # for incrementing time steps
    A = np.array([                      # Butcher tableau (for combining previous slopes)
        [0, 0, 0, 0],
        [1/2, 0, 0, 0],
        [0, 1/2, 0, 0],
        [0 , 0, 1, 0]
    ])
    B = np.array([1/6, 1/3, 1/3, 1/6])  # for combining all k's in final state update
    E = np.zeros(n_stages+1)            # No error estimation (RK4 is not adaptive)

    def __init__(self, fun, t0, y0, t_bound, dt=np.inf, **kwargs):
        super().__init__(fun, t0, y0, t_bound, **kwargs)
        self.dt = dt
        self.h_abs = min(dt, self.h_abs)
    
    def _step_impl(self):
        self.h_abs = self.dt  # enforce constant dt
        return super()._step_impl()


class Euler(OdeSolver):
    """Fixed-step classical Euler method for solve_ivp. Pass dt to set constant time step."""

    def __init__(self, fun, t0, y0, t_bound, dt=np.inf, **kwargs):
        super().__init__(fun, t0, y0, t_bound, **kwargs)
        self.dt = min(dt, t_bound - t0)  # fixed step size

    def _step_impl(self):
        y = self.y
        t = self.t
        y_new = y + self.dt * self.fun(self.t, self.y)
        t_new = t + self.dt

        self.y = y_new
        self.t = t_new
        return True, None


def solve_ode(fun: callable,
              tspan: tuple[float, float],
              u0: ArrayLike, 
              method: Literal['RK4', 'Euler'] | str | OdeSolver = 'RK45', 
              fun_opts: dict = None, 
              verbose: bool = False,
              return_sol: bool = False,
              num_steps: int = None,
              time_step: float = None,
              eps: float = 1e-8,
              **ivp_opts
              ):
    """
    Small wrapper of scipy.integrate.solve_ivp.

    Modifications:
        - allow dict kwargs for derivative function
        - a few custom time-steppers (just RK4 and Euler)
        - some error checking and parsing of scipy return values

    :param fun: Function f(t, u, **kwargs) representing the RHS of the ODE
    :param tspan: Span of integration (s)
    :param u0: Initial state vector (1D array of length N)
    :param method: Integration method to use (local explicit RK4 and Euler supported)
    :param fun_opts: extra kwargs to pass to fun
    :param ivp_opts: extra opts passed through to solve_ivp
    :param verbose: whether to include extra print info/warning information
    :param return_sol: whether to return the scipy OdeSolution object instead (default False)
    :param num_steps: if provided, overrides t_eval with linearly spaced points of this length
    :param time_step: if provided, overrides t_eval with linearly spaced points over tspan with this dt
    :param eps: nugget for floating point comparison

    :return t_vals: Array of time values of shape (Nt,)
    :return u_vals: Array of state vectors of shape (Nt, N)
    :return sol: the scipy OdeSolution object (if requested, optional)
    """
    METHODS = {
        'RK4': RK4,
        'Euler': Euler
    }

    if fun_opts is None:
        fun_opts = {}
    
    if method in METHODS:
        method = METHODS.get(method)
    
    if num_steps is not None and num_steps > 0:
        ivp_opts = copy.deepcopy(ivp_opts)
        ivp_opts['t_eval'] = np.linspace(tspan[0], tspan[1], num_steps)
    elif time_step is not None and time_step > 0:
        ivp_opts = copy.deepcopy(ivp_opts)
        # Make sure floor works for very close boundaries (will cause an issue if larger than the time step)
        if eps > time_step:
            warnings.warn(f"Time step {time_step:.2E} < floating eps {eps:.2E}. May have unexpected results.")
        num_steps = int(np.floor((tspan[1] - tspan[0] + eps)/time_step))  # Ensure constant dt over tspan
        ivp_opts['t_eval'] = np.linspace(tspan[0], tspan[0] + num_steps*time_step, num_steps + 1)

    if verbose:
        print(f"Running solve_ivp...")
    
    sol = solve_ivp(lambda t, u, *args: fun(t, u, *args, **fun_opts),
                    tspan,
                    u0,
                    method=method,
                    **ivp_opts)

    if verbose:
        print(f"Finished solve_ivp.")
        print(f"Success: {sol.success}")
        print(f"Number of function evaluations: {sol.nfev}")
        print(f"Number of Jacobian evaluations: {sol.njev}")
        print(f"Termination status: {sol.status} -- {sol.message}")
    
    if return_sol:
        return sol
    else:
        return sol.t, sol.y.T  # (Nt, N)


def duffing_oscillator(t: float, u: ArrayLike, 
                       zeta: float = 0.05,
                       omega0: float = 1.0,
                       beta: float = 1.0,
                       delta: float = 1.0,
                       omega: float = 0.9,
                       envelope: Callable[[float], float] | Literal['smoothstep'] = None,
                       envelope_opts: dict = None):
    """RHS of Duffing oscillator in 1st-order form.
    u = [x, v] where x is displacement and v is velocity.

    xdot = v
    vdot = -2*zeta*omega0*v - omega0**2*x - s*beta*x**3 + s*delta*cos(omega*t)

    where s = envelope(t) controls the nonlinear and forcing terms.

    :param t: the current time (s)
    :param u: the current state (2,)
    :param envelope: Callable s = f(t) that controls the nonlinear/forcing terms. Defaults to constant 1 (i.e. no env),
                     Can also specify as 'smoothstep' for a quintic smoothstep.
    :param evelope_opts: Extra opts to pass to the envelope function
    
    :return dudt: the derivative of the state vector (2,)
    """
    if envelope is None:
        envelope = lambda t, **kwargs: 1
    elif envelope == 'smoothstep':
        envelope = _quintic_smoothstep
    if envelope_opts is None:
        envelope_opts = {}
    x, v = u[0], u[1]
    dxdt = v
    s = envelope(t, **envelope_opts)
    # dvdt = -gamma * v - alpha * x - s * beta * x**3 + s * delta * np.cos(omega * t)
    dvdt = -2*zeta*omega0*v - omega0**2*x - s*beta*x**3 + s*delta*np.cos(omega*t)
    return np.array([float(dxdt), float(dvdt)])


def _quintic_smoothstep(t: ArrayLike, 
                        start_val:float = 1, end_val: float = 0, 
                        start_t: float = 0, end_t:float = 1):
    """Smoothly step between (start_t, start_val) -> (end_t, end_val) over `t` following a 
    quintic smoothstep function, which is smooth in C^2 at the end points.
    """
    t = np.atleast_1d(t)
    if np.isclose(start_t, end_t):
        return np.where(t < start_t, start_val, end_val)  # Step function
    
    tau = np.clip((t - start_t) / (end_t - start_t), 0.0, 1.0)
    p = 10*tau**3 - 15*tau**4 + 6*tau**5  # 0->1, C^2 at ends

    return start_val + (end_val - start_val) * p


def _sine_forcing(t, x, x0=0.0, a=1.0, omegac=1.0, A=1.0, sigma=1.0, omegaa=1.0):
    """Forcing function for the Burgers equation. Gives a Gaussian bump with oscillating center
    and oscillating amplitude.
    
    F(t, x) = A * exp(-((x-xc) / sigma)^2) * sin(omegaa * t) with oscillating center 
      xc(t) = x0 + a*sin(omegac * t)
    """
    xc = x0 + a*np.sin(omegac * t)
    return A * np.exp(-((x - xc) / sigma)**2) * np.sin(omegaa * t)


def burgers(t: float, u: ArrayLike, 
            xspan: tuple[float, float] = (0, 1), 
            bc_type: Literal['dirichlet', 'periodic', 'neumann'] = 'neumann',
            bc_opts: dict = None,
            alpha: float = 1.0, 
            nu: float = 0.01,
            forcing: callable | Literal['sine'] = None,
            forcing_opts: dict = None,
            envelope: Callable[[float], float] | Literal['smoothstep'] = None,
            envelope_opts: dict = None,
            ):
    """RHS of Burger's equation with central-difference spatial derivatives and forcing.

    du/dt = -s*alpha*u*(du/dx) + nu*(d^2u/dx^2) + s*F(t, x)

    where s = envelope(t) controls the nonlinear and forcing terms.
    
    :param t: the current time (s)
    :param u: the current state (N,) for N spatial locations
    :param xspan: the 1d spatial domain
    :param bc_type: dirichlet or periodic
    :param bc_opts: opts for BC (not used)
    :param alpha: set to 1 for Burgers advection, set to 0 for diffusion only
    :param nu: the diffusion coefficient
    :param forcing: if provided, a forcing function F(t, x, **opts), can use provided 'sine' function
    :param forcing_opts: extra kwargs for the forcing function
    :param envelope: Callable s = f(t) that controls the nonlinear/forcing terms. Defaults to constant 1 (i.e. no env),
                     Can also specify as 'smoothstep' for a quintic smoothstep.
    :param evelope_opts: Extra opts to pass to the envelope function
    
    :return dudt: the derivative of the state vector (N,)
    """
    if bc_opts is None:
        bc_opts = {}
    if forcing_opts is None:
        forcing_opts = {}
    
    if forcing == 'sine':
        forcing = _sine_forcing
    elif forcing == 'none':
        forcing = None
    
    if envelope is None:
        envelope = lambda t, **kwargs: 1
    elif envelope == 'smoothstep':
        envelope = _quintic_smoothstep
    if envelope_opts is None:
        envelope_opts = {}

    N = len(u)
    L = xspan[1] - xspan[0]
    x = np.linspace(*xspan, N)
    dx = L / (N - 1)

    if forcing is not None:
        f = forcing(t, x, **forcing_opts)
    else:
        f = np.zeros_like(u)

    dudt = np.zeros_like(u)
    s = envelope(t, **envelope_opts)

    # Central nodes
    u_x = (u[2:] - u[:-2]) / (2 * dx)
    u_xx = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
    dudt[1:-1] = -s * alpha * u[1:-1] * u_x + nu * u_xx + s * f[1:-1]

    # BCs
    if bc_type == 'dirichlet':
        dudt[0] = 0.0
        dudt[-1] = 0.0
    elif bc_type == 'periodic':
        u_x0 = (u[1] - u[-1]) / (2 * dx)
        u_xx0 = (u[1] - 2*u[0] + u[-1]) / dx**2
        dudt[0] = -s * alpha * u[0] * u_x0 + nu * u_xx0 + s * f[0]

        u_xN = (u[0] - u[-2]) / (2 * dx)
        u_xxN = (u[0] - 2*u[-1] + u[-2]) / dx**2
        dudt[-1] = -s * alpha * u[-1] * u_xN + nu * u_xxN + s * f[-1]
    elif bc_type == 'neumann':
        dudt[0] = dudt[1]
        dudt[-1] = dudt[-2]
    else:
        raise NotImplementedError(f"Boundary condition '{bc_type}' not implemented.")
    
    return dudt


# def simulate_burgers():
    # vmin, vmax = np.min(u), np.max(u)
    # cmap = 'viridis'
    # imshow_args = {'extent': [t[0], t[-1], x[0], x[-1]], 'origin': 'lower', 'vmin': vmin, 'vmax': vmax, 'cmap': cmap, 'aspect': 'auto'}

    # fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
    # im = ax.imshow(u.T, **imshow_args)
    # im.cmap.set_bad(im.cmap.get_under())
    # im_ratio = Nx / Nt
    # cb = fig.colorbar(im, label='$u(x, t)$', fraction=0.046 * im_ratio, pad=0.04)
    # ax.set_xlabel('Time $t$')
    # ax.set_ylabel('Position $x$')
     # params = dict(x0=0.7, a=0.07, omegac=10*np.pi, A=5, sigma=0.08, omegaa=15*np.pi)
    # t = np.linspace(0, 1, 200)
    # x = np.linspace(0, 1, 200)
    # f = np.zeros((200, 200))
    # for i in range(200):
    #     f[i] = _burgers_forcing(t[i], x, **params)
    
    # d = from_numpy(t, x, ['f'], np.expand_dims(f, axis=-1))
    # fig, ax = plot1d(d['frames'], x, t, subplot_size_in=(6, 5), save='f.mp4', 
    #                  animate_opts={'fps' :20, 'blit': True, 'dpi': 100},
    #                  )

    # plt.show()


def simulate_duffing():
    # === Simulation Parameters ===
    u0 = np.array([1.0, 0.0])  # Initial state: [x0, v0]
    ti = 0.0
    tf = 200
    dt = 0.01
    params = dict(alpha=1, beta=1, delta=0.05, gamma=0.2, omega=1.1)
    params_lin = params.copy()
    params_lin['beta'] = 0

    # t45, y45 = solve_ode(duffing_oscillator, (ti, tf), u0, method='RK45', deriv_kwargs=params, verbose=True)
    # te, ye = solve_ode(duffing_oscillator, (ti, tf), u0, method='Euler', deriv_kwargs=params, verbose=True, dt=0.05)
    # t4, y4 = solve_ode(duffing_oscillator, (ti, tf), u0, method='RK4', deriv_kwargs=params, verbose=True, dt=dt)
    # print(f'45: {t45.shape}')
    # print(f'euler: {te.shape}')
    # print(f'rk4: {t4.shape}')

    t45, y45 = solve_ode(duffing_oscillator, (ti, tf), u0, method='RK45', fun_opts=params, dt=dt)
    tlin, ylin = solve_ode(duffing_oscillator, (ti, tf), u0, method='RK45', fun_opts=params_lin, dt=dt)

    fig, ax = plt.subplots(2, 1, layout='tight', figsize=(11, 6))
    ax[0].plot(t45, y45[:, 0], '-k', label='Duffing')
    ax[0].plot(tlin, ylin[:, 0], '--r', label='Linear')
    ax[1].plot(t45, y45[:, 1], '-k', label='Duffing')
    ax[1].plot(tlin, ylin[:, 1], '--r', label='Linear')
    # fig, ax = plt.subplots(2, 1, layout='tight', figsize=(11, 6))
    # ax[0].plot(t45, y45[:, 0], '-k', label='RK45')
    # ax[0].plot(te, ye[:, 0], '--r', label='Euler')
    # ax[0].plot(t4, y4[:, 0], '--b', label='RK4')
    # ax[1].plot(t45, y45[:, 1], '-k', label='RK45')
    # ax[1].plot(te, ye[:, 1], '--r', label='Euler')
    # ax[1].plot(t4, y4[:, 1], '--b', label='RK4')
    ax[0].set_ylabel('Position $x$')
    ax[1].set_ylabel('Velocity $v$')
    ax[0].set_xlabel('Time (s)')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend()

    fig, ax = plt.subplots(layout='tight', figsize=(6, 5))
    ax.plot(y45[:,0], y45[:, 1], '-k')
    ax.set_xlabel('Position $x$')
    ax.set_ylabel('Velocity $v$')

    plt.show()
