"""Script for running the sliding-window data-driven training algorithm.

Usage:
  python sliding-window.py config.json -o output_dir

Some possible improvements:
  - More robust comparison window relative error to handle misaligned discrete time steps (i.e. interpolate)
  - Abstract classes for models, roms, and solutions
  - More consistent plotting interface
  - Constant offset term to DMD or external forcing
  - Run from a config.json file with CLI and .h5 saving
"""
from typing import Any, Callable, Literal
from pathlib import Path
from dataclasses import dataclass, field
from functools import wraps
import warnings

import numpy as np
from numpy.typing import ArrayLike
from pydmd import DMD
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from ode import solve_ode, duffing_oscillator, burgers

from romtools.utils import relative_error, normalize, denormalize
from romtools.dataloader import load_h5, load_numpy, filter_data
from romtools.plotting import PlotMetadata, _get_scheme


CONFIG_PATH = Path(".")


@dataclass
class SlidingSolution:
    """Result of the sliding-window algorithm.
    
    :ivar sol: (Nstates, Ntime) The ground truth simulation snapshots
    :ivar t: (Ntime,) The times of the solution
    :ivar rom: The final trained ROM object upon termination
    :ivar window_index: The final window index upon termination (can recreate all train/learn window with this)
    :ivar epsilon: The history of ROM comparison errors
    :ivar exit_msg: Contains reason for algorithm termination (either 'success' or 'failed to converge before tf')
    :ivar debug_info: contains history of rom predictions for debugging purposes (can be massive, so careful)
    :ivar opts: Options passed to the sliding_window algorithm (i.e. hyperparams)
    """
    sol: ArrayLike
    t: ArrayLike
    rom: Any
    window_index: int
    epsilon: ArrayLike
    exit_msg: str
    debug: dict = field(default_factory=dict) # Contains 'rom_obj' -> trained ROMs, 't_rom' -> pred times, and 'u_rom' -> predictions
    opts: dict = field(default_factory=dict)


@dataclass
class ModelInformation:
    """Information required to run a model.
    
    :ivar coords: The 1d/2d grid coordinates of a PDE solution (not used for scalar models)
    :ivar variables: List of variable names (single for scalar models)
    :ivar initial_condition: The ODE initial conditions
    :ivar func: Callable model as t, u = f(u0, tspan) -> solves ODE from IC u0 over the provided tspan
    :ivar data_opts: Information (mesh, nodes, etc.) needed for post-processing
    """
    coords: ArrayLike
    variables: list[str]
    initial_condition: ArrayLike
    func: Callable[[tuple[float, float], ArrayLike], tuple[ArrayLike, ArrayLike]]
    data_opts: PlotMetadata


@dataclass
class ROMInformation:
    """Information required to run a reduced-order model.
    
    :ivar fit: The fitting function, callable as rom_obj = fit(t, u, prev_rom_obj)
    :ivar predict: The prediction function, callable as t, u = predict(tspan, u0, t0, rom_obj)
    """
    fit: Callable[[ArrayLike, ArrayLike, Any], Any]
    predict: Callable[[tuple[float, float], ArrayLike, float, Any], tuple[ArrayLike, ArrayLike]]


def select(t: ArrayLike, u: ArrayLike, t_window: tuple[float, float], copy: bool = False):
    """Sub-select times from `t` and data from `u` within the provided `t_window`.
    
    :param t: (Nt,) Array of simulation times
    :param u: (Nt, ...) Array of simulation data
    :param t_window: the start and end times to select data
    :param copy: whether to return copy arrays or just views (default)
    :return t_sub, u_sub: the subselected times and data
    """
    start_idx = np.argmin(np.abs(t - t_window[0]))
    end_idx = np.argmin(np.abs(t - t_window[1]))

    t_sub = t[start_idx:end_idx + 1]
    u_sub = u[start_idx:end_idx + 1]

    if copy:
        t_sub = t_sub.copy()
        u_sub = u_sub.copy()

    return t_sub, u_sub


def apply_denorm(variables: list[str] = None, 
                 methods: dict[str, str] = None, 
                 norm_consts: dict[str, tuple[float, float]] = None
                 ):
    """Decorator factory for applying denormalization to data from a function of the form t, u = func(tin, uin, *args),
    where `u` is the data to denormalize of shape `(..., Nvar)` with `Nvar` matching the length of variables.
    Uses the provided norm constants for denormalization (which were returned by the apply_norm function).

    No normalization is performed if variables or methods is empty.
    
    :param variables: List of variable names (keys in methods dict and matches last dim of data `u`)
    :param methods: Dictionary mapping variable -> normalization method
    :param norm_consts: Dictionary mapping variable -> normalization constants
    :return decorator: A decorator to wrap functions of the form t, u = func(tin, uin, *args)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(t_in, u_in, *args, **kwargs):
            # Normalize u_in (like initial conditions)
            if variables is None:
                u_in_norm = u_in
            else:
                norm = {v: methods.get(v, None) if isinstance(methods, dict) else methods for v in variables}
                consts = {v: None for v in variables} if norm_consts is None else norm_consts

                u_in_norm = u_in.copy()
                for i, (v, norm_method) in enumerate(norm.items()):
                    u_in_norm[..., i], _ = normalize(u_in_norm[..., i], method=norm_method, consts=consts[v])

            # Call underyling function
            t, u = func(t_in, u_in_norm, *args, **kwargs)

            # Denormalize output
            if variables is None:
                u_denorm = u
            else:
                u_denorm = u.copy()
                for i, (v, norm_method) in enumerate(norm.items()):
                    u_denorm[..., i] = denormalize(u_denorm[..., i], method=norm_method, consts=consts[v])
            
            return t, u_denorm

        return wrapper
    return decorator


def apply_norm(variables: list[str] = None, 
               methods: dict[str, str] = None
               ):
    """Decorator factory for applying normalization to a function of the form f(t, u, *args),
    where `u` is the data to normalize of shape `(..., Nvar)` with `Nvar` matching the length of variables.

    No normalization is performed if variables or methods is None (default).

    :param variables: List of variable names (keys in methods dict and matches last dim of data `u`)
    :param methods: Dictionary mapping variable -> normalization method.
    :return decorator: A decorator to wrap functions of the form res = (t, u, *args)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(t, u, *args, **kwargs):
            # --- Preprocess (normalize inputs) ---
            norm_consts = {}
            if variables is None:
                norm_u = u
            else:
                norm = {v: methods.get(v, None) if isinstance(methods, dict) else methods for v in variables}
                norm_u = u.copy()
                for i, (v, norm_method) in enumerate(norm.items()):
                    norm_u[..., i], norm_consts[v] = normalize(norm_u[..., i], method=norm_method)
            
            # --- Call underlying function ---
            result = func(t, norm_u, *args, **kwargs)

            return result, norm_consts
            
        return wrapper
    return decorator


def parse_rom(rom: Literal['exact_dmd'],
              rom_opts: dict = None
              ):
    """Get callable fit and predict functions for a given rom method and options.
    
    !!! Note
        I am realizing this is just starting to look like class inheritance, so a good future move would be to
        have a `ROM` abstract base class with a `from_dict` method that constructs an appropriate child class based
        on the provided options. The child classes would implement a specific rom method (like DMD) and would have
        standard fit/predict methods.
    """
    if rom_opts is None:
        rom_opts = {}

    match rom.lower():
        case 'exact_dmd':
            def fit(t: ArrayLike, u: ArrayLike, prev_rom: Any) -> Any:
                """Fit a DMD model. Assume training data is already normalized when passed here.
                
                :param t: (Ntime,) the array of times associated with training data `u`
                :param u: (Ntime, Nx, Nvar) the array of simulation training data
                :param prev_rom: a trained DMD object from a previous iteration (not used)
                """
                rom_obj = DMD(**rom_opts)
                rom_obj.fit(u.reshape((u.shape[0], -1)).T)  # (Nstates, Ntime)
                rom_obj.dt = t[1] - t[0]  # Monkey-patch the time step (needed for prediction later)
                return rom_obj
            
            def predict(tspan: tuple[float, float], u0: ArrayLike, t0: float, rom_obj: Any, eps=1e-8) -> tuple[ArrayLike, ArrayLike]:
                """Predict with a DMD model.
                
                :param tspan: the time horizon to predict over
                :param u0: the initial condition (not used here, its buried in the DMD amplitudes)
                :param t0: the initial time
                :param rom_obj: the trained DMD object (includes time step dt)
                :return t: the times of prediction over tspan
                :return u: the dmd predictions over the tspan
                """
                # DMD assumes start at t0, so shift the prediction window by t0
                ti, tf = tspan[0] - t0, tspan[1] - t0
                dt = rom_obj.dt
                if eps > dt:
                    warnings.warn(f"Time step {dt:.2E} < floating eps {eps:.2E}. May have unexpected results.")
                max_steps = int(np.floor((tf-ti+eps)/dt))  # Ensure the same constant dt is used over (ti, tf)
                t = np.linspace(ti, ti + max_steps*dt, max_steps + 1)

                # Analytical linear ODE prediction in continuous time u(t)
                b, lamb, phi = rom_obj.amplitudes, rom_obj.eigs, rom_obj.modes  # (r,), (r,) and (Nstates, r)
                omega = np.log(lamb) / dt  # Continuous time eigenvalues
                u = (phi @ np.diag(b) @ np.exp(t[np.newaxis, :] * omega[:, np.newaxis])).real  # (Nstates, Nt)
                u = u.T.reshape(t.shape + u0.shape)  # (Nt, Nx, Nvar)

                return t + t0, u

        case _:
            raise NotImplementedError(f"Requested rom method '{rom}' not supported.")
    
    return ROMInformation(fit=fit, predict=predict)


def flat_states(func):
    """Decorator to take a func of the form `t, uout = func(t, uin)` and flatten `uin` then reshape `uout` to original.
    Useful for models where there is no distinction between `(Npoints, Nvar)` dimensions (i.e. scalar ODEs or 1d PDEs).
    """
    @wraps(func)
    def wrapper(t, u, *args, **kwargs):
        og_shape = u.shape
        tout, uout = func(t, u.reshape((-1,)), *args, **kwargs)
        return tout, uout.reshape(tout.shape + og_shape)
    return wrapper


def parse_model(model: Literal['duffing', 'burgers', 'hall2de'], 
                model_opts: dict = None,
                ic_opts: dict = None
                ):
    """Get callable t, u = f(tspan, u0) for a supported model name and extra data required by the model.

    Supported:
    - duffing - nonlinear duffing oscillator
    - burgers - nonlinear 1d burgers PDE
    - hall2de - offline load hall2de simulation results

    !!! Note
        Like with the RomInformation, the ModelInformation can likely be generalized to an abstract class hierarchy,
        with each implementing model following a set of protocols. That would highly reduce the boilerplate here and 
        all the match/cases. But this works for now. Things change too fast.
    
    :param model: The name of a supported model
    :param model_opts: Extra opts for calling the model (ivp opts for duffing/burgers, results_file for Hall2de),
                       May also pass 'exclude' for variables to exclude in Hall2De analysis
    :param ic_opts: Initial condition specs (see each model below).
    
    :return ModelInformation: contains variable names, data coordinates, the callable model, etc. See `ModelInformation`.
    """
    if model_opts is None:
        model_opts = {}
    if ic_opts is None:
        ic_opts = {}

    match model.lower():
        case 'duffing':
            if 'ic' not in ic_opts:
                raise ValueError("Must pass 'ic' with duffing oscillator starting position and velocity")
            
            model_func = flat_states(lambda tspan, u0: solve_ode(duffing_oscillator, tspan, u0, **model_opts))
            variables = ['x (m)', 'v (m/s)']
            coords = np.array([0.0])  # not used
            data_opts = {}  # not used
            ic = np.atleast_1d(ic_opts.get('ic')).reshape((-1, len(variables))) 

        case 'burgers':
            model_func = flat_states(lambda tspan, u0: solve_ode(burgers, tspan, u0, **model_opts))
            variables = ['u']

            if 'u0' in ic_opts:
                ic = np.atleast_1d(ic_opts.get('u0'))
                coords = ic_opts.get('coords', np.linspace(0, 1, len(ic)))
            else:
                match ic_opts.get('type'):
                    case 'gaussian':
                        xspan = model_opts.get('fun_opts', {}).get('xspan', (0, 1))
                        Nx = ic_opts.get('num_points', 100)
                        coords = np.linspace(*xspan, Nx)
                        mu = ic_opts.get('mu', 0)
                        sigma = ic_opts.get('sigma', 1)
                        mag = ic_opts.get('mag', 1)
                        ic = mag * np.exp(-0.5 * ((coords-mu)/sigma)**2)
                    case _:
                        raise NotImplementedError(f"Initial condition '{ic_opts.get('type')}' not recognized for burgers equation")
            
            ic = np.reshape(ic, (-1, len(variables)))
            data_opts = {'coord': coords}

        case 'hall2de':
            # Just load data (don't run online currently)
            if 'results_file' not in model_opts:
                raise ValueError("Must pass a 'results_file' in model_opts for Hall2De as a path to processed h5 data (relative to config file).")

            d = load_h5(Path(CONFIG_PATH) / Path(model_opts.get("results_file")))
            exclude_vars = model_opts.get("exclude", [])
            time, coords, variables, data_array = load_numpy(d['filtered']['hist'])
            var_idx = [i for i, v in enumerate(variables) if v not in exclude_vars]
            data_array = data_array[..., var_idx]
            variables = [variables[i] for i in var_idx]
            ic = data_array[0, ...]

            def model_func(tspan, u0, time=time, data_array=data_array):
                ti_idx, tf_idx = np.argmin(np.abs(time - tspan[0])), np.argmin(np.abs(time - tspan[1]))
                t = time[ti_idx:tf_idx+1]
                u = data_array[ti_idx:tf_idx+1, ...]
                return t, u

            _temp = filter_data(d, loc='node', select_frames=[0])
            vertices = _temp['coords']
            _temp = filter_data(d, loc='cell', select_frames=[0])
            cells = _temp['coords']
            conn = d['connectivity']
            data_opts = dict(cells=cells, vertices=vertices, connectivity=conn)
        
        case _:
            raise NotImplementedError(f"Requested model '{model}' not supported.")
    
    return ModelInformation(func=model_func, coords=coords, variables=variables, initial_condition=ic, data_opts=data_opts)


def sliding_window(run_model: Callable[[tuple[float, float], ArrayLike], tuple[ArrayLike, ArrayLike]],
                   run_rom: Callable[[tuple[float, float], ArrayLike, float, Any], tuple[ArrayLike, ArrayLike]],
                   train_rom: Callable[[ArrayLike, ArrayLike, Any], Any],
                   u0: ArrayLike,
                   train_length: float,
                   t0: float = 0.0,
                   tf: float = None,
                   pred_length: float = None,
                   pred_offset: float | int = 10,
                   skip_length: float | int = 1,
                   eps_target: float = 1e-3,
                   repeat_target: int = 1,
                   variables: list[str] = None,
                   norm: dict[str, str] = None,
                   debug_opts: dict = None
                   ):
    """Train a data-driven reduced-order model with a sliding-window termination algorithm.
    The main idea is to run a simulation, incrementally training a ROM, until the ROM converges, then terminate.

    !!! Note
        User functions should handle shaping/reshaping states `u` so that they are input/output as `(Nstates, Nvar)`.
        This is for compatibility with `romtools` and per-variable normalization methods. For example, if you are 
        solving an IVP, then you can do `u.reshape((-1,))` to collapse the `Nstates` v. `Nvar` distinction. See the 
        `flat_states` decorator for a quick way to do this.
    
    !!! Info "Assumptions"
        1. Simulation data is returned by the model on a constant uniform time step grid (even if it solves adaptively, 
           it should be interpolated to a constant time step).
        2. The skip_length should be a constant integer multiple of the time step, and should be >= 1*dt. This is to ensure
           consistent advancement of continuous time windows on a discrete grid, and for consistent comparisons between iterations.

        Some implications of these assumptions:
        1. Training windows always start at a new discrete time step between iterations.
        2. The training window may be slightly longer or shorter than the specified length (it will snap to nearest discrete time step),
           but it will always be of the same constant length throughout the simulation
        3. The prediction window will be a constant length and floating in continuous time. ROMs should account for this when
           making discrete predictions. One solution (implemented by DMD here) is to snap the pred window to the nearest
           length that is an integer multiple of the simulation time step. This ensures predictions are consistent between
           window iterations.
    
    :param run_model: Function for obtaining simulation data, callable as t, u = f(tspan, u0), like solve_ivp
    :param run_rom: Function for obtaining ROM predictions over tspan, callable as t, u = fhat(tspan, u0, t0, rom_obj), where 
                    rom_obj contains rom parameters, information, etc. necessary for evaluating the rom
    :param train_rom: Function for training a ROM, callable as rom_obj = train(t, u, prev rom_obj), where (t, u) is the
                      simulation training data and prev rom_obj may contain previous weights/params
    :param u0: (Nstates, Nvar) the initial conditions
    :param train_length: the length of the sliding training window (in seconds)
    :param t0: The initial time (default 0)
    :param tf: If provided, the final simulation time to terminate the algorithm with status=failed.
    :param pred_length: The length of the sliding prediction window (defaults to same as train_length)
    :param pred_offset: The offset length from the end of the training window to the start of the prediction window, if 
                        an integer, it is interpreted as a multiple of the train_length (default 10*train_length)
    :param skip_length: The distance to slide the training/prediction windows at each iteration, if an integer, it is
                        interpreted as a multiple of the simulation time step (default 1*dt)
    :param eps_target: The target relative error between successive ROM predictions
    :param repeat_target: The number of iterations to obtain eps_target before terminating (increases robustness)
    :param variables: The variable names matching the last dim of simulation data (used for normalization)
    :param norm: Dictionary mapping variable names -> normalization methods
    :param debug_opts: If provided, save extra debug information in the output. Options are 'save_iter' for how often to
                       to save (in window indices), 'save_obj' to save the trained rom objects, and 'save_pred' to save
                       rom predictions. Off by default.
    
    :return sol: The SlidingSolution object, with simulation data, times, the trained ROM, and other info (see SlidingSolution)
    """
    # Defaults
    if pred_length is None:
        pred_length = train_length
    if isinstance(pred_offset, np.integer | int):
        pred_offset = pred_offset * train_length
    if skip_length is None:
        skip_length = 1
    debug_info = {}
    exit_msg = 'unknown'
    t0_init = t0

    # INITIALIZE
    w = 0
    t_train = t0
    t_pred = t_train + train_length + pred_offset
    train_window = (t_train, t_train + train_length)
    pred_window = (t_pred, t_pred + pred_length)
    t_true, u_true = run_model(train_window, u0)
    time_step = t_true[1] - t_true[0]   # Assumed constant (in seconds)

    # Force skip length to be constant integer multiple >=1 of simulation time step
    if isinstance(skip_length, np.integer | int):
        skip_length = skip_length * time_step
    skip_length = int(np.round(skip_length / time_step))
    if skip_length == 0:
        raise ValueError(f"Must specify a skip length larger than simulation time step of {time_step:.2E} s")
    skip_length = skip_length * time_step
    
    rom_obj_prev = None  # ROM predictor/parameters/weights etc. from previous window
    u_rom_prev = None    # ROM predictions from previous window
    t_rom_prev = None    # Times associated with previous predictions
    epsilon = []         # Track comparison window errors

    norm_train = apply_norm(variables=variables, methods=norm)(train_rom)

    while True:
        # TRAIN ROM AND PREDICT
        t_sub, u_train = select(t_true, u_true, train_window)
        t0, u0 = t_sub[0], u_train[0]
        rom_obj, norm_consts = norm_train(t_sub, u_train, rom_obj_prev)
        denorm_rom = apply_denorm(variables=variables, methods=norm, norm_consts=norm_consts)(run_rom)
        t_rom, u_rom = denorm_rom(pred_window, u0, t0, rom_obj)

        if debug_opts is not None and w % debug_opts.get('save_iter', 1) == 0:
            if debug_opts.get('save_obj'):
                debug_info.setdefault('rom_obj', [])
                debug_info['rom_obj'].append(rom_obj)
            if debug_opts.get('save_pred'):
                debug_info.setdefault('t_rom', [])
                debug_info.setdefault('u_rom', [])
                full_t_rom, full_u_rom = denorm_rom((t0, pred_window[-1]), u0, t0, rom_obj)
                debug_info['t_rom'].append(full_t_rom)  # Prediction from train to pred window
                debug_info['u_rom'].append(full_u_rom)

        # COMPUTE ERROR
        if w > 0:
            compare_window = (t_pred, t_rom_prev[-1])
            t_eps_curr, u_eps_curr = select(t_rom, u_rom, compare_window)            # (num_states, num_compare_snapshots)
            t_eps_prev, u_eps_prev = select(t_rom_prev, u_rom_prev, compare_window)  # (num_states, num_compare_snapshots)

            if not np.allclose(t_eps_curr, t_eps_prev):
                raise RuntimeError(f"Comparison window times are not aligned.")  # Might be able to relax this with a more robust error func
            
            epsilon.append(relative_error(u_eps_curr, u_eps_prev))
            
            if w >= repeat_target:
                if all([epsilon[-1 - i] < eps_target for i in range(repeat_target)]):
                    exit_msg = 'success'
                    break  # TERMINATE SIMULATION

        # COLLECT MORE DATA
        u0, t0 = u_true[-1], t_true[-1]
        new_t, new_u = run_model((train_window[-1], train_window[-1] + skip_length), u0)
        u_true = np.concatenate((u_true, new_u[1:]), axis=0)  # Don't duplicate initial condition at first index
        t_true = np.concatenate((t_true, new_t[1:]), axis=0)

        if tf is not None and t_true[-1] > tf:
            exit_msg = f'failed to converge before simulation reached tf={tf:.2E} s'
            break

        # SLIDE TRAINING/PREDICTION WINDOWS
        t_train = t_train + skip_length
        t_pred = t_pred + skip_length
        train_window = (t_train, t_train + train_length)
        pred_window = (t_pred, t_pred + pred_length)
        rom_obj_prev, u_rom_prev, t_rom_prev = rom_obj, u_rom, t_rom
        w += 1
    
    return SlidingSolution(sol=u_true, t=t_true, rom=rom_obj, window_index=w, epsilon=epsilon, exit_msg=exit_msg, debug=debug_info,
                           opts=dict(t0=t0_init, tf=tf, train_length=train_length, pred_length=pred_length, pred_offset=pred_offset,
                                     skip_length=float(skip_length), eps_target=eps_target, repeat_target=repeat_target, variables=variables,
                                     norm=norm, debug_opts=debug_opts))


def iter_windows(sol: SlidingSolution):
    """Iterate sliding train/predict windows from a sliding window result object."""
    t0 = float(sol.opts['t0'])
    train_length = float(sol.opts['train_length'])
    pred_length = float(sol.opts['pred_length'])
    skip_length = float(sol.opts['skip_length'])
    pred_offset = float(sol.opts['pred_offset'])

    t_train = t0
    t_pred = t_train + train_length + pred_offset
    train_window = (t_train, t_train + train_length)
    pred_window = (t_pred, t_pred + pred_length)

    yield train_window, pred_window  # w=0

    for w in range(1, sol.window_index+1):
        t_train = t_train + skip_length
        t_pred = t_pred + skip_length
        train_window = (t_train, t_train + train_length)
        pred_window = (t_pred, t_pred + pred_length)
        
        yield train_window, pred_window


def animate_scalar_window(sol: SlidingSolution,
                        var_idx: int = 0,
                        full_sol: tuple[ArrayLike, ArrayLike] = None,
                        train_window_opts: dict = None,
                        train_vline_opts: dict = None,
                        pred_window_opts: dict = None,
                        pred_vline_opts: dict = None,
                        animate_opts: dict = None,
                        legend_opts: dict = None,
                        bg_line_opts: dict = None,
                        fg_line_opts: dict = None,
                        true_color: str = 'k',
                        curr_color: str = 'green',
                        prev_color: str = 'brown',
                        figsize: tuple[float, float] = (5, 4),
                        coord_labels: list[str, str] = ('Time (s)', 'State'),
                        yscale: str = 'linear',
                        scheme: Literal['white', 'dark'] = 'white',
                        interactive: bool = False,
                        save: str | Path = None,
                        cutoff_line: float = None,
                        cutoff_line_opts: dict = None
                        ):
    """Show sliding solution progression for a scalar variable."""
    text_color, bg_color = _get_scheme(scheme)

    # Set default plot options
    if train_window_opts is None:
        train_window_opts = {"color": "c", "alpha": 0.3}
    if train_vline_opts is None:
        train_vline_opts = {"color": "c", "lw": 0.8, "ls": "--"}
    if pred_window_opts is None:
        pred_window_opts = {"color": "orange", "alpha": 0.3}
    if pred_vline_opts is None:
        pred_vline_opts = {"color": "orange", "lw": 0.8, "ls": "--"}
    if animate_opts is None:
        animate_opts = {"fps": 2, "dpi": 150, "frame_skip": 1, "blit": False}
    if legend_opts is None:
        legend_opts = {'fancybox': True}
    if bg_line_opts is None:
        bg_line_opts = {"ls": "--", "lw": 2, "alpha": 0.6}   # Background full predictions
    if fg_line_opts is None:
        fg_line_opts = {"ls": "-", "lw": 3, "alpha": 1}  # Forground window predictions
    if true_color == 'k':
        true_color = text_color
    legend_opts.update(dict(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color))

    if interactive:
        plt.ion()

    fig, ax = plt.subplots(layout='tight', figsize=figsize)
    fig.patch.set_facecolor(bg_color)
    axes_visible = coord_labels is not None
    ax.tick_params(axis='both', which='both', top=False, bottom=axes_visible, left=axes_visible, right=False, direction='in',
                   labelleft=axes_visible, labelbottom=axes_visible, color=text_color, labelcolor=text_color)
    ax.set_facecolor(bg_color)
    ax.set_yscale(yscale)
    for spine in ['bottom', 'left', 'top', 'right']:
        ax.spines[spine].set_visible(axes_visible)
        ax.spines[spine].set_color(text_color)
    if axes_visible:
        ax.set_xlabel(coord_labels[0], color=text_color)
        ax.set_ylabel(coord_labels[1], color=text_color)

    # Extract solution and windows
    t, u, epsilon = sol.t, sol.sol, sol.epsilon
    epsilon.insert(0, np.nan)  # no error at first window iteration
    debug_info = sol.debug
    windows = [_ for _ in iter_windows(sol)]

    # Plot true solution
    if full_sol is not None:
        t_full, u_full = full_sol
        ax.plot(t_full, u_full[:, 0, var_idx], c=true_color, **bg_line_opts)
    t_true, u_true = select(t, u, windows[0][0])
    true_line, = ax.plot(t_true, u_true[:, 0, var_idx], c=true_color, label="True", **fg_line_opts)

    # Train window
    train_win = windows[0][0]
    train_left = ax.axvline(train_win[0], **train_vline_opts)
    train_right = ax.axvline(train_win[1], label="Train", **train_vline_opts)
    train_rect = ax.axvspan(*train_win, **train_window_opts)

    # Predict window
    pred_win = windows[0][1]
    pred_left = ax.axvline(pred_win[0], **pred_vline_opts)
    pred_right = ax.axvline(pred_win[1], label="Test", **pred_vline_opts)
    pred_rect = ax.axvspan(*pred_win, **pred_window_opts)

    # Prev rom prediction (just placeholder for first window)
    bg_prev_rom, = ax.plot(np.nan, np.nan, c=prev_color, **bg_line_opts)
    fg_prev_rom, = ax.plot(np.nan, np.nan, c=prev_color, label="Previous ROM", **fg_line_opts)

    # Current rom prediction
    t_rom, u_rom = debug_info['t_rom'][0], debug_info['u_rom'][0]
    t_rom_pred, u_rom_pred = select(t_rom, u_rom, pred_win)
    bg_curr_rom, = ax.plot(t_rom, u_rom[:, 0, var_idx], c=curr_color, **bg_line_opts)
    fg_curr_rom, = ax.plot(t_rom_pred, u_rom_pred[:, 0, var_idx], c=curr_color, label="Current ROM", **fg_line_opts)

    if cutoff_line is not None:
        ax.axvline(cutoff_line, **cutoff_line_opts)

    ax.legend(**legend_opts)

    num_frames = int(len(windows) / animate_opts.get('frame_skip', 1)) + 1

    def update(idx):
        """Update plot with next plot window idx."""
        idx_use = idx * animate_opts.get('frame_skip', 1) if idx < num_frames - 1 else len(windows) - 1
        train_win, pred_win = windows[idx_use]
        title_str = f"w={idx_use}"
        if not np.isnan(epsilon[idx_use]):
            title_str += r", $\varepsilon ={}$".format(f"{epsilon[idx_use]:.1e}")
        fig.suptitle(title_str, color=text_color)
        
        # True solution
        t_true, u_true = select(t, u, (t[0], train_win[-1]))
        true_line.set_data(t_true, u_true[:, 0, var_idx])

        # Train window
        train_left.set_xdata([train_win[0], train_win[0]])
        train_right.set_xdata([train_win[1], train_win[1]])
        train_rect.set_x(train_win[0])
        train_rect.set_width(train_win[1] - train_win[0])

        # Prediction window
        pred_left.set_xdata([pred_win[0], pred_win[0]])
        pred_right.set_xdata([pred_win[1], pred_win[1]])
        pred_rect.set_x(pred_win[0])
        pred_rect.set_width(pred_win[1] - pred_win[0])

        # Previous rom prediction
        compare_win = pred_win if idx_use == 0 else (pred_win[0], windows[idx_use-1][1][1]) # (tpred, tf_prev)
        if idx_use > 0:
            t_rom, u_rom = debug_info['t_rom'][idx_use-1], debug_info['u_rom'][idx_use-1]
            t_rom_pred, u_rom_pred = select(t_rom, u_rom, compare_win)
            bg_prev_rom.set_data(t_rom, u_rom[:, 0, var_idx])
            fg_prev_rom.set_data(t_rom_pred, u_rom_pred[:, 0, var_idx])

        # Current rom prediction
        t_rom, u_rom = debug_info['t_rom'][idx_use], debug_info['u_rom'][idx_use]
        t_rom_pred, u_rom_pred = select(t_rom, u_rom, compare_win)
        bg_curr_rom.set_data(t_rom, u_rom[:, 0, var_idx])
        fg_curr_rom.set_data(t_rom_pred, u_rom_pred[:, 0, var_idx])
        
        return [true_line, train_left, train_right, train_rect, pred_left, pred_right, pred_rect,
                bg_curr_rom, fg_curr_rom, bg_prev_rom, fg_prev_rom]
    
    if interactive:
        for i in range(num_frames):
            update(i)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            plt.pause(0.001)
            plt.waitforbuttonpress()
    
    else:
        ani = FuncAnimation(fig, update, frames=num_frames, blit=animate_opts.get('blit', False), interval=1/animate_opts.get('fps', 2))

        if save is not None:
            print(f"Saving animation to '{save}'")
            def _progress(i, n):
                if np.mod(i, int(0.1 * n)) == 0 or i == 0 or i == n - 1:
                    print(f'Saving frame {i+1}/{n}...')
            ani.save(Path(save), writer='ffmpeg', progress_callback=_progress, dpi=animate_opts.get('dpi', 100), fps=animate_opts.get('fps', 2))
        else:
            plt.show()
    
    return fig, ax


def animate_1d_window(sol: SlidingSolution,
                    var_idx: int = 0,
                    full_sol: tuple[ArrayLike, ArrayLike] = None,
                    data_opts: PlotMetadata = None,
                    train_window_opts: dict = None,
                    train_vline_opts: dict = None,
                    pred_window_opts: dict = None,
                    pred_vline_opts: dict = None,
                    animate_opts: dict = None,
                    legend_opts: dict = None,
                    cmap: str = 'viridis',
                    figsize: tuple[float, float] = (5, 4),
                    coord_labels: list[str, str] = ('Time (s)', 'Position (m)'),
                    var_label: str = 'State',
                    titles: list[str, str] = ('Truth', 'Current ROM'),
                    scheme: Literal['white', 'dark'] = 'white',
                    interactive: bool = False,
                    save: str | Path = None,
                    t_skip: int = 1
                    ):
    """Show sliding solution progression for a 1d variable."""
    text_color, bg_color = _get_scheme(scheme)

    # Set default plot options
    if train_window_opts is None:
        train_window_opts = {"color": "c", "alpha": 0}
    if train_vline_opts is None:
        train_vline_opts = {"color": "c", "lw": 1.2, "ls": "--"}
    if pred_window_opts is None:
        pred_window_opts = {"color": "orange", "alpha": 0}
    if pred_vline_opts is None:
        pred_vline_opts = {"color": "orange", "lw": 1.2, "ls": "--"}
    if animate_opts is None:
        animate_opts = {"fps": 2, "dpi": 150, "frame_skip": 1, "blit": False}
    if legend_opts is None:
        legend_opts = {'fancybox': True}
    legend_opts.update(dict(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color))
    tsl = slice(None, None, t_skip)

    # Extract solution and windows
    t, u, epsilon = sol.t, sol.sol, sol.epsilon
    epsilon.insert(0, np.nan)  # no error at first window iteration
    debug_info = sol.debug
    windows = [_ for _ in iter_windows(sol)]
    coords = data_opts['coord'] if data_opts is not None else np.linspace(0, 1, u.shape[1])
    vmin, vmax = np.min(u[..., var_idx]), np.max(u[..., var_idx])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    bg_alpha = 0.5
    fg_alpha = 1

    if interactive:
        plt.ion()

    fig, axs = plt.subplots(1, 2, layout='tight', figsize=(figsize[0]*2, figsize[1]), sharey='row')
    fig.patch.set_facecolor(bg_color)
    axes_visible = coord_labels is not None
    for i, ax in enumerate(axs.flatten()):
        ax.tick_params(axis='both', which='both', top=False, bottom=axes_visible, left=axes_visible, right=False, direction='in',
                       labelleft=axes_visible, labelbottom=axes_visible, color=text_color, labelcolor=text_color)
        ax.set_facecolor(bg_color)
        for spine in ['bottom', 'left', 'top', 'right']:
            ax.spines[spine].set_visible(axes_visible)
            ax.spines[spine].set_color(text_color)
        if axes_visible:
            ax.set_xlabel(coord_labels[0], color=text_color)
            ax.set_ylabel(coord_labels[1], color=text_color)
        
        cb = plt.colorbar(mappable, ax=ax)
        cb.ax.set_ylabel(var_label, color=text_color)
        cb.ax.tick_params(labelcolor=text_color, color=text_color)
        cb.ax.tick_params(which='minor', color=(0,0,0,0), width=0, size=0)
        cb.ax.minorticks_off()
        cb.outline.set_edgecolor(text_color)
        ax.set_title(titles[i])
    
    train_win, pred_win = windows[0]

    # Plot true solution
    if full_sol is not None:
        t_full, u_full = full_sol
        axs[0].pcolormesh(t_full[tsl], coords, u_full[tsl, :, var_idx].T, cmap=cmap, norm=norm, alpha=bg_alpha)
        axs[1].pcolormesh(t_full[tsl], coords, u_full[tsl, :, var_idx].T, cmap=cmap, alpha=0)  # just to get ax size right
        t_true, u_true = select(t_full, u_full, pred_win)
        true_pcm_pred = [axs[0].pcolormesh(t_true[tsl], coords, u_true[tsl, :, var_idx].T, cmap=cmap, norm=norm, alpha=fg_alpha)]
    t_true, u_true = select(t, u, train_win)
    true_pcm = [axs[0].pcolormesh(t_true[tsl], coords, u_true[tsl, :, var_idx].T, cmap=cmap, norm=norm, alpha=fg_alpha)]

    # Train/pred windows
    train_left, train_right, train_rect = [], [], []
    pred_left, pred_right, pred_rect = [], [], []
    for i, ax in enumerate(axs.flatten()):
        train_left.append(ax.axvline(train_win[0], **train_vline_opts))
        train_right.append(ax.axvline(train_win[1], label="Train", **train_vline_opts))
        train_rect.append(ax.axvspan(*train_win, **train_window_opts))

        pred_left.append(ax.axvline(pred_win[0], **pred_vline_opts))
        pred_right.append(ax.axvline(pred_win[1], label="Test", **pred_vline_opts))
        pred_rect.append(ax.axvspan(*pred_win, **pred_window_opts))

    # Current rom prediction
    t_rom, u_rom = debug_info['t_rom'][0], debug_info['u_rom'][0]
    t_rom_pred, u_rom_pred = select(t_rom, u_rom, pred_win)
    bg_curr_rom = [axs[1].pcolormesh(t_rom[tsl], coords, u_rom[tsl, :, var_idx].T, cmap=cmap, norm=norm, alpha=bg_alpha)]
    fg_curr_rom = [axs[1].pcolormesh(t_rom_pred[tsl], coords, u_rom_pred[tsl, :, var_idx].T, cmap=cmap, norm=norm, alpha=fg_alpha)]

    axs[1].legend(**legend_opts)

    num_frames = int(len(windows) / animate_opts.get('frame_skip', 1)) + 1

    def update(idx):
        """Update plot with next plot window idx."""
        idx_use = idx * animate_opts.get('frame_skip', 1) if idx < num_frames - 1 else len(windows) - 1
        train_win, pred_win = windows[idx_use]
        title_str = f"w={idx_use}"
        if not np.isnan(epsilon[idx_use]):
            title_str += r", $\varepsilon ={}$".format(f"{epsilon[idx_use]:.1e}")
        fig.suptitle(title_str, color=text_color)

        artists = []
        
        # True solution
        t_true, u_true = select(t, u, (t[0], train_win[-1]))
        # del axs[0].collections[-1]
        true_pcm[0].remove()
        true_pcm[0] = axs[0].pcolormesh(t_true[tsl], coords, u_true[tsl, :, var_idx].T, cmap=cmap, norm=norm, alpha=fg_alpha)
        artists.append(true_pcm[0])

        if full_sol is not None:
            t_true, u_true = select(t_full, u_full, pred_win)
            # del axs[0].collections[-2]
            true_pcm_pred[0].remove()
            true_pcm_pred[0] = axs[0].pcolormesh(t_true[tsl], coords, u_true[tsl, :, var_idx].T, cmap=cmap, norm=norm, alpha=fg_alpha)
            artists.append(true_pcm_pred[0])

        # Train/prediction windows
        for i in range(len(train_left)):
            train_left[i].set_xdata([train_win[0], train_win[0]])
            train_right[i].set_xdata([train_win[1], train_win[1]])
            train_rect[i].set_x(train_win[0])
            train_rect[i].set_width(train_win[1] - train_win[0])

            pred_left[i].set_xdata([pred_win[0], pred_win[0]])
            pred_right[i].set_xdata([pred_win[1], pred_win[1]])
            pred_rect[i].set_x(pred_win[0])
            pred_rect[i].set_width(pred_win[1] - pred_win[0])
        
        for l in (train_left, train_right, train_rect, pred_left, pred_right, pred_rect):
            artists.extend(l)

        # Current rom prediction
        compare_win = pred_win if idx_use == 0 else (pred_win[0], windows[idx_use-1][1][1]) # (tpred, tf_prev)
        t_rom, u_rom = debug_info['t_rom'][idx_use], debug_info['u_rom'][idx_use]
        t_rom_pred, u_rom_pred = select(t_rom, u_rom, compare_win)
        # del axs[1].collections[-2]
        # del axs[1].collections[-1]
        bg_curr_rom[0].remove()
        fg_curr_rom[0].remove()
        bg_curr_rom[0] = axs[1].pcolormesh(t_rom[tsl], coords, u_rom[tsl, :, var_idx].T, cmap=cmap, norm=norm, alpha=bg_alpha)
        fg_curr_rom[0] = axs[1].pcolormesh(t_rom_pred[tsl], coords, u_rom_pred[tsl, :, var_idx].T, cmap=cmap, norm=norm, alpha=fg_alpha)
        artists.extend([bg_curr_rom[0], fg_curr_rom[0]])
        
        return artists
    
    if interactive:
        for i in range(num_frames):
            update(i)
            fig.canvas.draw()
            plt.pause(0.001)
            plt.waitforbuttonpress()
    
    else:
        ani = FuncAnimation(fig, update, frames=num_frames, blit=animate_opts.get('blit', False), interval=1/animate_opts.get('fps', 2))

        if save is not None:
            print(f"Saving animation to '{save}'")
            def _progress(i, n):
                if np.mod(i, int(0.1 * n)) == 0 or i == 0 or i == n - 1:
                    print(f'Saving frame {i+1}/{n}...')
            ani.save(Path(save), writer='ffmpeg', progress_callback=_progress, dpi=animate_opts.get('dpi', 100), fps=animate_opts.get('fps', 2))
        else:
            plt.show()
    
    return fig, ax


def run_duffing():
    model = 'duffing'
    cutoff = 30
    model_opts = {
        "time_step": 0.1,
        "fun_opts": {
            "zeta": 0.06,
            "omega0": 1.0,
            "beta": 5.0,
            "delta": 9.0,
            "omega": 0.9,
            "envelope": "smoothstep",
            "envelope_opts": {
                "start_t": 0,
                "end_t": cutoff,
            }
        }
    }
    u0 = [0, 0]
    ic_opts = {"ic": u0}

    rom = 'exact_dmd'
    rom_opts = {"svd_rank": 2, "exact": True, "opt": True}

    model_info = parse_model(model, model_opts, ic_opts)
    rom_info = parse_rom(rom, rom_opts)

    sliding_opts = {
        "train_length": 10.0,
        "pred_offset": 20.0,
        "skip_length": 2.0,
        "eps_target": 1e-8,
        "repeat_target": 2,
        "variables": model_info.variables,
        "tf": 70,
        "debug_opts": {"save_pred": True}
    }

    plot_opts = {
        'interactive': True,
        # 'save': 'duffing-sliding-window-2.mp4',
        'figsize': (6, 5),
        'legend_opts': {'loc': 'lower right'},
        'scheme': 'dark',
        'animate_opts': {'fps': 10, 'dpi': 150, 'blit': True, 'frame_skip': 1},
        'cutoff_line': cutoff,
        'cutoff_line_opts': {"ls": "-.", "lw": 2, "c": "r", "label": "Nonlinear cutoff"}
    }

    sol = sliding_window(model_info.func, 
                         rom_info.predict,
                         rom_info.fit,
                         model_info.initial_condition,
                         **sliding_opts)
    full_sol = model_info.func((0, 100), model_info.initial_condition)

    fig, ax = animate_scalar_window(sol, full_sol=full_sol, **plot_opts)


def run_burgers():
    model = 'burgers'
    model_opts = {
        "time_step": 0.01,
        "fun_opts": {
            "nu": 0.05,
            "xspan": [-5, 5],
            "bc_type": "neumann",
            "alpha": 0.7,
            "forcing": "sine",
            "forcing_opts": {
                "x0": 2,
                "a": 0,
                "omegac": 3.14,
                "A": 2,
                "sigma": 1,
                "omegaa": 6.28
            },
            "envelope": "smoothstep",
            "envelope_opts": {
                "start_t": 0,
                "end_t": 8
            }
        }
    }
    ic_opts = {
        "type": "gaussian",
        "xspan": [-5, 5],
        "num_points": 200,
        "mu": 0,
        "sigma": 0.5,
        "mag": 1
    }

    rom = 'exact_dmd'
    rom_opts = {
        "svd_rank": 0, 
        "exact": False, 
        "opt": True, 
        "tikhonov_regularization": 1e-5
    }

    model_info = parse_model(model, model_opts, ic_opts)
    rom_info = parse_rom(rom, rom_opts)

    train_length = 3.0
    pred_offset = 6.0
    tf = 12
    sliding_opts = {
        "train_length": train_length,
        "pred_offset": pred_offset,
        "skip_length": 0.5,
        "eps_target": 1e-8,
        "repeat_target": 4,
        "tf": tf,
        "debug_opts": {"save_pred": True},
        "variables": model_info.variables,
        "norm": {
            "u": "zscore"
        }
    }

    plot_opts = {
        "interactive": True,
        "save": "burgers-sliding-window.mp4",
        "figsize": (5, 7),
        "legend_opts": {"loc": "lower right"},
        "scheme": "dark",
        "t_skip": 5
        # "animate_opts": {}
    }

    sol = sliding_window(model_info.func,
                         rom_info.predict,
                         rom_info.fit,
                         model_info.initial_condition,
                         **sliding_opts)
    full_sol = model_info.func((0, tf+pred_offset+train_length), model_info.initial_condition)

    fig, ax = animate_1d_window(sol, full_sol=full_sol, data_opts=model_info.data_opts, **plot_opts)


def main():
    run_duffing()
    # run_burgers()


if __name__ == '__main__':
    main()
