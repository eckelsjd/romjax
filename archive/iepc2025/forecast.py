"""Run data-driven reduced-order modeling forecast analyses.

Usage:
  python forecast.py config.json -o output_dir

Steps:
  1) Run true models and collect data
  2) Train data-driven ROMs and forecast
  3) Show comparison plots

For each 'case' in the config file, you may specify more detailed information of 'rom' or any of the 'plot' routines.
Any 'rom' or 'plot' options placed at the top-level of the config file will be treated as global and apply to every case.
That means if you specify 'rom': {'exact_dmd': **opts} at the top-level, every case in 'cases': {} will have the 'exact_dmd'
ROM analysis applied to it with the given opts.

For plotting, everything passed to "global_plot" will be used in all plotting routines (unless overriden).
All arguments of `romtools.plotting.compare` may be passed to all of the plotting routines to configure the output.
Please see the `compare` and `gridplot` documentations for these options (not listed here).
In addition to `gridplot` arguments, you may also pass "skip" to skip a step, or "animate" to animate a step.

All additional configurations are described below.

Config file format:
{
  "cases": {               List of rom/model analyses to run
      "case_name": {       Name of the case to run
          "skip":          Whether to skip this case
          "model":         The name of the model to use
          "u0":            Details of the model initial conditions
          "ode_opts":      Extra options for solving the ode with scipy solve_ivp
      }
  }
  "rom": {                 List of rom analyses to apply
      "global_opts":       Everything in this dict will be applied to all subsequent rom analyses (e.g. train_window, time_fraction, etc.)
      rom_method: {        Supported rom methods include "exact_dmd"
          **rom_opts       Additional options to pass to the rom method. E.g. for DMD, these will be passed to the DMD constructor
      }
  }
  "global_plot": {         Provides global plotting options (useful defaults are set for all), can also override in each plot routine below
      "skip":              Whether to skip a plotting routine (will skip all if used in global_plot)
      "animate":           Whether to animate plots (where applicable), default False
  }
  "plot_total_error": {    Plots total relative error between truth and rom over time
      "time_multiplier":   How to scale time for error plots (default 1)
      "window_opts":       Options for the training window plot (defaults to green)
      "window_vline_opts": Extra opts passed to the vertical lines of the training window
      "animated_bar":      Options to put an animated vertical bar on the plot (see `LineMetadata`)
  }
  "plot_scalar_error": {   Plots scalar error on y-axis over time on x-axis
      "time_multiplier":   Same as total error
      "window_opts":       Same as total error
      "window_vline_opts": Same as total error
      "variable":          Pass a single string or None to display multiple ROMs over rows (default). Otherwise use a list of variables
      "show_rom":          The single ROM to show if variable is a list (defaults to first available)
      "reduce":            The name of a reduce function to generate scalar data from higher dimensional data
      "reduce_opts":       Options for the reduce function
  }
  "plot_1d_error": {       Plots 1d error both as (x,t) contour (optional) and as 1d animation
      "window_opts":       Same, training window only shown when x-axis is time
      "window_vline_opts": Same
      "reduce":            Same as scalar error, except these functions should reduce data to 1d
      "reduce_opts":       Same
      "show_rom":          Same
      "snapshot_time":     The time snapshot at which to compare truth and rom (defaults to the last time step)
      "time_fraction":     If true, interpret snapshot_time as a fraction of total time (default true)
      "contour": {         If specified, plot (x,t) contour with these override plotting options
        "contour_window":  If true, add training window to the contour plot (default false)
        "downsample":      A list of two integers, the rate to downsample the mesh contour for [x, t], default [1, 1]
      }
  }
  "plot_2d_error": {       Plots 2d error at specified prediction time and as 2d animation
      "show_rom":          All same as 1d error
      "snapshot_time":     Same
      "time_fraction":     Same
      "data_opts":         Extra options for PcolorMetadata or CellMetadata 2d plotting options (i.e. shading, mesh, etc.)
  }
}
"""
import argparse
import pickle
import json
import os 
import shutil
from pathlib import Path
from typing import Literal, Callable
import copy
import warnings
from functools import partial

import numpy as np
from numpy.typing import ArrayLike
from pydmd import DMD
import matplotlib.pyplot as plt

from romtools.dataloader import load_numpy, save_numpy, load_h5, save_h5, to_numpy, from_numpy, FRAME_KEY, filter_data
from romtools.plotting import _get_scheme, gridplot, compare, GRID_OPTS, PlotMetadata
from romtools.utils import normalize, denormalize, relative_error

from ode import duffing_oscillator, burgers, solve_ode
from postprocess import METRICS, SLICE_OPTS
from performance import slice_centerline, get_boundary, get_channel


# ================== DEFAULTS =======================================
# Everything is overrideable in the input config.json file as described above.
_default_file = Path(__file__).parent / "defaults.json"

if _default_file.exists():
    with open(_default_file, "r") as fd:
        _defaults = json.load(fd)
else:
    warnings.warn(f"Couldn't find default config file '{_default_file}'. Please place this next "
                  f"to the postprocess.py file. Falling back on hard-coded options...", UserWarning)
    _defaults = {}

# Only used in post-processing, not here
if "variables" in _defaults:
    _variables = _defaults.pop("variables") 

_default_bar = {"color": "r", "linewidth": 0.5}  # for animations

plt.style.use(Path(__file__).parent / 'iepc.mplstyle')
# ====================================================================

CONFIG_PATH = Path(".")  # Global for specifying paths relative to config path


def reduce(truth_tuple: tuple,
           rom_tuples: dict[str, tuple],
           truth_data_opts: PlotMetadata,
           model: Literal['duffing', 'burgers', 'hall2de'],
           method: str,
           h5_path: str | Path,
           plot_config: dict,
           tmult=1,
           **opts
           ):
    """Reduce higher-dimensional data to lower dimension for plotting/analysis. Each model may handle this differently,
    and each may have several supported reduction methods. Update plot_config and h5_path where necessary.
    
    :return truth: reduced true simulation data
    :return rom: reduced rom data
    :return data_opts: new coordinate/mesh details for plotting
    """
    saved_data = load_h5(h5_path, groups=['reduce'])  # reduce/method/rom structure
    new_data_opts = None  # Overwrite this and return to update mesh/coordinate plotting details

    # Update name of save file
    _save = Path(plot_config['save'])
    plot_config['save'] = _save.parent / f"{_save.stem}_reduce-{method.lower()}{_save.suffix}"

    # Helper for centerline reduction
    def _centerline_reduce(method, time_avg: bool = False):
        rom = []
        for r in rom_tuples:
            need_rom = True
            if 'reduce' in saved_data and method.lower() in saved_data['reduce']:
                if r in saved_data['reduce'][method.lower()]:
                    slice_data = from_numpy(*load_numpy(saved_data['reduce'][method.lower()][r]))[FRAME_KEY]
                    need_rom = False
            
            if need_rom:
                rom_dict_data = from_numpy(*rom_tuples[r])
                if time_avg:
                    avg = {}
                    for v in rom_tuples[r][-2]:  # variables
                        arr = np.zeros_like(rom_dict_data[FRAME_KEY][0][v])
                        for i in range(len(rom_dict_data[FRAME_KEY])):
                            arr += rom_dict_data[FRAME_KEY][i][v]
                        avg[v] = arr / len(rom_dict_data[FRAME_KEY])
                    rom_dict_data[FRAME_KEY] = [avg]
                
                slice_opts = copy.deepcopy(SLICE_OPTS)
                slice_opts.update(opts.get('slice_opts', {}))
                slice_coords, slice_data = slice_centerline(rom_dict_data[FRAME_KEY], truth_data_opts['cells'], 
                                                            truth_data_opts['vertices'], truth_data_opts['connectivity'], verbose=True,
                                                            **slice_opts)
                slice_dict = {'time': rom_dict_data['time'], 'coords': slice_coords, FRAME_KEY: slice_data}
                
                rom_result = {}
                save_numpy(*to_numpy(slice_dict), rom_result)
                save_h5({'reduce': {method.lower(): {r: rom_result}}}, h5_path)
            
            rom.append(slice_data)
        
        return rom

    # Helper for performance metric reduction
    def _performance_reduce(method, species):
        rom = []
        for r in rom_tuples:
            need_rom = True
            if 'reduce' in saved_data and method.lower() in saved_data['reduce']:
                if r in saved_data['reduce'][method.lower()]:
                    rom_perf = saved_data['reduce'][method.lower()][r]
                    need_rom = False
            
            # Reduce rom data
            if need_rom:
                print(f"Generating performance metrics for rom '{r}'...")
                if len(metrics := opts.get('metrics', list(METRICS.keys()))) > 0:
                    for m in metrics:
                        if m not in METRICS:
                            raise NotImplementedError(f"Requested metric '{m}' not supported.")
                        
                rom_dict_data = from_numpy(*rom_tuples[r])  # standard romtools dict format
                rom_perf = {
                    metric: METRICS[metric](rom_dict_data[FRAME_KEY], truth_data_opts['cells'], 
                                            truth_data_opts['vertices'], truth_data_opts['connectivity'], 
                                            integrand_opts=dict(species=species)) 
                    for metric in metrics
                }
                save_h5({'reduce': {method.lower(): {r: rom_perf}}}, h5_path)
            
            if 'thrust' in rom_perf:
                rom_perf['thrust'] = rom_perf['thrust'][:, 0]  # (Nt,) axial only

            rom.append([rom_perf])
        
        return rom

    match model.lower():
        case 'hall2de':
            if 'results_file' not in opts:
                raise TypeError(f"Need a 'results_file' for loading and reducing Hall2De postprocess data. "
                                f"This is the .h5 file generated by postprocess.py.")
            
            d = load_h5(Path(CONFIG_PATH) / opts['results_file'], groups=['performance', 'centerline'])
            species = d['species']
            boundary_edges, _ = get_boundary(truth_data_opts['connectivity'])
            channel_dims = get_channel(boundary_edges, truth_data_opts['vertices'])

            match method.lower():
                # Reduction to scalar performance metrics
                case 'performance':
                    truth_perf = d['performance']
                    if 'thrust' in truth_perf:
                        truth_perf['thrust'] = truth_perf['thrust'][:, 0]  # (Nt,) axial only
                    truth = [[truth_perf]]
                    rom = _performance_reduce(method, species)

                    plot_config['data_labels'] = plot_config.get("data_labels", {'thrust': 'Thrust (N)', 'discharge_current': "$I_d$", 'beam_current': "$I_b$"})
                    new_data_opts = {'share_plot': plot_config.pop("share_plot", {"Current (A)": ['discharge_current', 'beam_current']}),
                                     'coord': truth_tuple[0]*tmult}

                # Reduction to 1d channel centerline (avg)
                case 'centerline_avg':
                    truth_tup = load_numpy(d['centerline']['avg'])
                    truth = [from_numpy(*truth_tup)[FRAME_KEY]]
                    rom = _centerline_reduce(method, time_avg=True)

                    plot_config["coord_labels"] = plot_config.get("coord_labels", ["Channel centerline (channel lengths)"])
                    new_data_opts = {'coord': truth_tup[1][:, 0] / channel_dims['channel_length']}

                # Reduction to 1d channel centerline (time-resolved)
                case 'centerline_hist':
                    truth_tup = load_numpy(d['centerline']['hist'])
                    truth = [from_numpy(*truth_tup)[FRAME_KEY]]
                    rom = _centerline_reduce(method, time_avg=False)

                    plot_config["coord_labels"] = plot_config.get("coord_labels", ["Channel centerline (channel lengths)"])
                    new_data_opts = {'coord': truth_tup[1][:, 0] / channel_dims['channel_length']}
        case _:
            raise NotImplementedError(f"Reduction not supported for model '{model}'.")
    
    return truth, rom, new_data_opts
    


def run_model(model: Literal['duffing', 'burgers', 'hall2de'], 
              tspan: tuple[float, float] = (None, None), 
              u0: ArrayLike = None, 
              ode_opts: dict = None
              ):
    """Helper to run ivp-based models for testing roms. The return tuple is compatible with `romtools`.

    Note: The 'hall2de' model actually just loads pre-computed data from an h5 file obtained from postprocess.py.
    
    :param model: the model to run
    :param tspan: time span of integration
    :param u0: the initial conditions (each model may use this slightly differently)
    :param ode_opts: additional kwargs for solve_ode

    :return time: (Nt,) array of solution times
    :return coords: (Nx,) array of solution coordinates (for PDEs)
    :return variables: list of variable names in the solution
    :return data_array: (Nt, Nx, Nvar) array of simulation data (this is the "numpy" format from romtools)
    :return data_opts: Info about mesh/coordinates (for plotting)
    """
    if ode_opts is None:
        ode_opts = {}
    
    data_opts = {}  # Fill this with plotting information (not required)
    
    match model.lower():
        case 'duffing':
            if None in tspan:
                raise ValueError(f'Invalid tspan: {tspan}')
            if u0 is None:
                raise ValueError(f'Invalid u0: {u0}')
            time, u = solve_ode(duffing_oscillator, tspan, u0, **ode_opts)
            data_array = np.expand_dims(u, axis=1)  # (Nt, 1, 2)
            variables = ['x (m)', 'v (m/s)']
            coords = np.array([0.0])  # not used for scalars

        case 'burgers':
            if None in tspan:
                raise ValueError(f'Invalid tspan: {tspan}')
            if u0 is None:
                raise ValueError(f'Invalid u0: {u0}')
            if isinstance(u0, dict):
                # Build initial conditions from specs (use u0)
                match u0.get('type', 'gaussian'):
                    case 'gaussian':
                        xspan = u0.get('xspan', (0, 1))
                        Nx = u0.get('num_points', 100)
                        coords = np.linspace(*xspan, Nx)
                        mu = u0.get('mu', 0)
                        sigma = u0.get('sigma', 1)
                        mag = u0.get('mag', 1)
                        u0 = mag * np.exp(-0.5 * ((coords-mu)/sigma)**2)
                    case _:
                        raise NotImplementedError(f"Initial condition '{u0.get('type')}' not recognized for burgers equation")
            else:
                u0 = np.atleast_1d(u0)
                coords = np.linspace(0, 1, len(u0))
            
            time, u = solve_ode(burgers, tspan, u0, **ode_opts)
            data_array = np.expand_dims(u, axis=-1)  # (Nt, Nx, 1)
            variables = ['u']

        case 'hall2de':
            # Should already have filtered Hall2De data saved, load this from file
            # Use u0 as a path to an .h5 file (relative to cwd) with the filtered data (best I can do, sorry)
            if not isinstance(u0, dict):
                raise ValueError("For Hall2De, pass u0 as {'save_path', 'exclude'} for a path to the processed h5 data "
                                 " and optionally a list of variables to exclude from the ROM analysis")
            if 'results_file' not in u0:
                raise ValueError("Must pass a 'results_file' for u0 in Hall2De as a path to processed h5 data (relative to config file).")

            d = load_h5(Path(CONFIG_PATH) / Path(u0.get("results_file")))
            exclude_vars = u0.get("exclude", [])

            time, coords, variables, data_array = load_numpy(d['filtered']['hist'])
            ti = tspan[0] or time[0]
            tf = tspan[1] or time[-1]
            ti_idx, tf_idx = np.argmin(np.abs(time - ti)), np.argmin(np.abs(time - tf))

            time = time[ti_idx:tf_idx+1]
            data_array = data_array[ti_idx:tf_idx+1, ..., [i for i, v in enumerate(variables) if v not in exclude_vars]]
            variables = [v for v in variables if v not in exclude_vars]

            _temp = filter_data(d, loc='node', select_frames=[0])
            vertices = _temp['coords']
            _temp = filter_data(d, loc='cell', select_frames=[0])
            cells = _temp['coords']
            conn = d['connectivity']
            data_opts = dict(cells=cells, vertices=vertices, connectivity=conn)

        case _:
            raise NotImplementedError(f"Model '{model}' not implemented.")
        
    return (time, coords, variables, data_array), data_opts


def run_rom(method: Literal['exact_dmd'],
            sim_tuple: tuple[ArrayLike, ArrayLike, list[str], ArrayLike],
            train_window: tuple[float, float] = (0, -1),
            time_fraction: bool = True,
            norm: dict | str = None,
            **rom_opts
            ):
    """Helper to train/run roms on simulation data. The return tuple is compatible with `romtools` and
    directly comparable to the input simulation tuple.

    :param method: the rom method to use
    :param sim_tuple: (time, coords, variables, data_array) from the true simulation (romtools "numpy" format)
    :param train_window: tuple of times that bound the training window (use all by default)
    :param time_fraction: whether to interpret values in train_window as fractions of total time (default True)
    :param norm: If dict, then the norm method to use for each variable. If string, then same method applied to all
    :param rom_opts: additional options for training/running the ROMs (each method may take different opts)

    :return time: (Nt,) array of solution times
    :return coords: (Nx,) array of solution coordinates (for PDEs)
    :return variables: list of variable names in the solution
    :return data_array: (Nt, Nx, Nvar) array of ROM predictions (this is the "numpy" format from romtools)
    :return rom_obj: the Python object containing the reduced order model
    :return train_time: (Nt,) array of training window times
    """
    time, coords, variables, sim_data = sim_tuple
    num_steps, num_points, num_var = sim_data.shape
    dt = np.diff(time)

    # Train window
    tlen = time[-1] - time[0]
    if time_fraction:
        tstart = train_window[0] * tlen + time[0]
        tend = (train_window[1] if train_window[1] is not None and train_window[1] > 0 else 1) * tlen + time[0]
        train_window = (tstart, tend)
    if train_window[1] is None or train_window[1] < 0:
        train_window = (train_window[0], time[-1])
    
    start_idx = np.argmin(np.abs(time - train_window[0]))
    end_idx = np.argmin(np.abs(time - train_window[1]))

    train_time = time[start_idx:end_idx + 1]
    train_data = sim_data[start_idx:end_idx + 1].copy()

    # Normalize
    norm = {v: norm.get(v, None) if isinstance(norm, dict) else norm for v in variables}
    norm_consts = {}
    for i, (v, norm_method) in enumerate(norm.items()):
        train_data[..., i], norm_consts[v] = normalize(train_data[..., i], method=norm_method)

    # Train and predict
    match method.lower():
        case 'exact_dmd':
            dt_const = dt[0]  # Assume constant for dmd
            rom_obj = DMD(**rom_opts)
            rom_obj.fit(train_data.reshape((train_data.shape[0], -1)).T)    # (Nstates, Ntime)
            b, lamb, phi = rom_obj.amplitudes, rom_obj.eigs, rom_obj.modes  # (r,), (r,) and (Nstates, r)
            omega = np.log(lamb) / dt_const  # Continuous time eigenvalues
            rom_data = (phi @ np.diag(b) @ np.exp(time[np.newaxis, :] * omega[:, np.newaxis])).real  # (Nstates, Nt)
            rom_data = rom_data.T.reshape(sim_data.shape)
            print(f"DMD rank: {b.shape[0]}")
        case _:
            raise NotImplementedError(f"ROM method '{method}' not recognized.")
    
    # Denormalize
    for i, (v, norm_method) in enumerate(norm.items()):
        rom_data[..., i] = denormalize(rom_data[..., i], method=norm_method, consts=norm_consts[v])

        # if norm_method is not None and 'log' in norm_method:  # Values less than 0 will cause problems during plotting for log variables
        #     rom_data[rom_data[..., i] < 0, i] = np.nan

    np.nan_to_num(rom_data, nan=np.nan, posinf=np.nan, neginf=np.nan, copy=False)

    return (time, coords, variables, rom_data), rom_obj, train_time


def run_case(case: str, case_info: dict, out_dir: str | Path, overwrite: bool = False):
    """Run a given analysis with configs in case_info, write outputs, optionally overwriting existing results."""
    overwrite = overwrite or case_info.get('overwrite', False)
    print('='*30)
    print(f'CASE: {case}')
    print('='*30)

    need_model = overwrite
    case_dir = Path(out_dir) / case

    tspan = case_info.get('tspan', (None, None))
    u0 = case_info.get('u0', None)

    print("----------TRUE MODEL----------")

    if overwrite and case_dir.exists():
        shutil.rmtree(case_dir)
    if not case_dir.exists():
        os.mkdir(case_dir)
        need_model = True
    
    h5_path = case_dir / f'{case}.h5'
    if not need_model and h5_path.exists():
        d = load_h5(h5_path, groups=['truth'])
        if 'truth' in d:
            truth_data_opts = d['truth'].pop("data_opts", {})
            truth_tuple = load_numpy(d['truth'])  # (time, coords, variables, data)
            print(f"Loaded model data from '{h5_path}'.")
        else:
            need_model = True
    
    if need_model:
        if 'model' not in case_info:
            raise KeyError("Need to specify a model name for each case. Options: [duffing, burgers, hall2de]")
        
        print(f"Running '{case_info["model"]}' model...")
        truth_tuple, truth_data_opts = run_model(case_info['model'], tspan, u0, case_info.get('ode_opts', {}))  # (time, coords, variables, data)
        d = {'data_opts': truth_data_opts}
        save_numpy(*truth_tuple, d)
        save_h5({'truth': d}, h5_path, overwrite=True)

    print("----------REDUCED ORDER MODEL----------")

    roms = case_info['rom']
    rom_tuples = {}     # (time, coords, variables, data)
    rom_objs = {}       # the Python objects containing the "ROM" (i.e. pydmd, fno, etc.)
    train_windows = {}  # training window for the ROM
    rom_errors = {}     # Relative errors with truth model
    global_opts = roms.pop('global_opts', {})

    for r, rom_opts in roms.items():
        opts = copy.deepcopy(global_opts)
        opts.update(rom_opts)
        if opts.get("skip", False):
            continue

        need_rom = True

        if h5_path.exists():
            d = load_h5(h5_path, groups=[r])
            if r in d:
                if "rom_obj" in d[r]:
                    rom_objs[r] = pickle.loads(d[r].pop("rom_obj").tobytes())
                if "train_window" in d[r]:
                    train_windows[r] = d[r].pop("train_window")
                if "error" in d[r]:
                    rom_errors[r] = d[r].pop("error")
                rom_tuples[r] = load_numpy(d[r])
                print(f"Loaded data for rom '{r}' from '{h5_path}'.")
                need_rom = False
        
        if need_rom:
            print(f"Running rom '{r}'...")
            rom_tuples[r], rom_objs[r], train_windows[r] = run_rom(r, truth_tuple, **opts)
            rom_errors[r] = relative_error(truth_tuple[-1], rom_tuples[r][-1], axis=(1, 2))  # (Nt,)
            d = {'rom_obj': np.frombuffer(pickle.dumps(rom_objs[r]), dtype=np.uint8),
                 'train_window': train_windows[r], 'error': rom_errors[r]}
            save_numpy(*rom_tuples[r], d)
            save_h5({r: d}, h5_path, overwrite=True)

    print("----------PLOTS----------")

    global_plot = copy.deepcopy(case_info.get("global_plot", {}))
    for k, v in _defaults.items():
        global_plot[k] = global_plot.get(k, v)
    exclude_opts = {'animate', 'skip'}  # don't pass these on to gridplot

    def _extract_truth_and_rom(selector: Callable[[ArrayLike], ArrayLike], truth_tuple, rom_tuples):
        truth = [[{v: selector(truth_tuple[-1][..., i]) for i, v in enumerate(truth_tuple[-2])}]]  # v: (Nt, Nx)
        rom = [[{v: selector(tup[-1][..., i]) for i, v in enumerate(tup[-2])}] for tup in rom_tuples.values()]  # v: (Nt, Nx)
        return truth, rom
    
    def _update_from_global(plot_config: dict, fname_tag: str):
        """Update a specific plot config using global options (and update filename)"""
        for opt in global_plot:
            if opt not in exclude_opts and opt not in plot_config:
                plot_config[opt] = global_plot[opt]
        plot_config['save'] = (case_dir/(plot_config.get("save") or f"{case}_{fname_tag}")).with_suffix(".pdf")
    
    def _add_train_window(fig, axs, yscale=None, grid=False, plot_config=None, 
                          window_opts=None, window_vline_opts=None, tmult=1, update_leg=False):
        if plot_config is None:
            plot_config = {}
        if window_opts is None:
            window_opts = {'color': 'g', 'alpha': 0.3}
        if window_vline_opts is None:
            window_vline_opts = {'c': 'g', 'lw': 1.5, 'ls': '--'}

        text_color, bg_color = _get_scheme(plot_config.get("scheme", "white"))

        if train_windows:
            window = next(iter(train_windows.values()))[[0, -1]]  # Assume the same for all ROMs for comparison
            for i in range(axs.shape[0]):
                for j in range(axs.shape[1]):
                    ax = axs[i, j]
                    if grid:
                        ax.grid(**plot_config.get("error_grid_opts", GRID_OPTS))
                    if yscale:
                        ax.set_yscale(yscale)

                    ax.axvline(window[0]*tmult, **window_vline_opts, label='Train window')
                    ax.axvline(window[1]*tmult, **window_vline_opts)
                    ax.axvspan(window[0]*tmult, window[1]*tmult, **window_opts)

                    if update_leg and j == 0:  # Only put on first plot
                        leg = dict(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color, fancybox=True)
                        leg.update(plot_config.get("legend_opts", {}))
                        ax.legend(**leg)
    
    # ------------------------------TOTAL ERROR OVER TIME----------------------------------------
    k = 'plot_total_error'
    plot_config = case_info.get(k)
    if k in case_info and not plot_config.pop("skip", False):
        print(f"Plotting total error for case '{case}'")
        animate = plot_config.pop('animate', global_plot.get('animate', False))
        _update_from_global(plot_config, 'total_error')

        tmult = plot_config.pop("time_multiplier", 1)
        data_opts = {'coord': truth_tuple[0] * tmult,
                     'share_plot': {"Relative error": list(rom_errors.keys())}}
        plot_config['adjust'] = partial(_add_train_window, yscale='log', grid=True, plot_config=plot_config,
                                        window_opts=plot_config.pop("window_opts", None), tmult=tmult,
                                        window_vline_opts=plot_config.pop("window_vline_opts", None),
                                        update_leg=True)
        gridplot([rom_errors], data_opts, **plot_config)
        
        if animate:
            print(f"Generating animation of ROM total error for case '{case}'")
            data_opts['animated_bar'] = plot_config.pop("animated_bar", _default_bar)
            plot_config['save'] = plot_config['save'].with_suffix(".mp4")
            gridplot([rom_errors]*len(truth_tuple[0]), data_opts, time=truth_tuple[0], **plot_config)
    
    # ------------------------------SCALAR ERROR LINE PLOTS----------------------------------------
    k = 'plot_scalar_error'
    plot_config = case_info.get(k)
    if k in case_info and not plot_config.pop("skip", False):
        print(f"Plotting scalar error for case '{case}'")
        _update_from_global(plot_config, 'scalar_error')

        tmult = plot_config.pop("time_multiplier", 1)
        data_opts = {'coord': truth_tuple[0] * tmult}
        plot_config['col_labels'] = plot_config.get("col_labels", ["Truth", "ROM"])
        plot_config['adjust'] = partial(_add_train_window, plot_config=plot_config, tmult=tmult,
                                        window_opts=plot_config.pop("window_opts", None), update_leg=True,
                                        window_vline_opts=plot_config.pop("window_vline_opts", None))
        
        # Convert higher-dimensional data to scalar
        if (reduce_funcs := plot_config.pop("reduce", [])):
            reduce_opts = plot_config.pop("reduce_opts", {})

            def _iter_truth_and_rom():
                for f in reduce_funcs:
                    pconfig = copy.deepcopy(plot_config)
                    truth, rom, new_data_opts = reduce(truth_tuple, rom_tuples, truth_data_opts, 
                                                       case_info['model'], f, h5_path, pconfig, tmult=tmult, **reduce_opts)
                    yield truth, rom, pconfig, data_opts if new_data_opts is None else new_data_opts

        # Otherwise data is already scalar and should use directly
        else:
            def _iter_truth_and_rom():
                truth, rom = _extract_truth_and_rom(lambda arr: arr[:, 0], truth_tuple, rom_tuples)  # v: (Nt,)
                yield truth, rom, plot_config, data_opts

        for truth, rom, pconfig, dopts in _iter_truth_and_rom():
            if isinstance(v := pconfig.get("variable", None), str) or v is None:
                # Display multiple ROMs for a single variable (done by default)
                # pconfig['row_labels'] = pconfig.get("row_labels", [pconfig.get("data_labels", {}).get(k, k) for k in rom_tuples.keys()])
                pass
            else:
                # Display multiple variables for only the chosen ROM
                show_rom = pconfig.pop("show_rom", next(iter(rom_tuples.keys())))
                rom = [rom[list(rom_tuples.keys()).index(show_rom)]]

            compare(truth, rom, dopts, **pconfig)
    
    # ------------------------------1D FIELD ERROR LINE/CONTOUR----------------------------------------
    k = 'plot_1d_error'
    plot_config = case_info.get(k)
    if k in case_info and not plot_config.pop("skip", False):
        print(f"Plotting 1d error for case '{case}'")
        animate = plot_config.pop('animate', global_plot.get('animate', False))
        _update_from_global(plot_config, '1d_error')
        snapshot_time = plot_config.pop("snapshot_time", -1)
        time_fraction = plot_config.pop("time_fraction", True)
        contour_opts = plot_config.pop("contour", {})
        contour_window = contour_opts.pop("contour_window", False)
        contour_downsample = contour_opts.pop("downsample", (1, 1))
        tmult = plot_config.pop("time_multiplier", 1)

        # Get snapshot time index
        time = truth_tuple[0]
        tlen = time[-1] - time[0]
        if time_fraction:
            if snapshot_time < 0:
                snapshot_time = 1
            snapshot_time = snapshot_time * tlen + time[0]
        else:
            if snapshot_time < 0:
                snapshot_time = time[-1]
        snapshot_idx = np.argmin(np.abs(time - snapshot_time))

        data_opts = {'coord': truth_tuple[1]}
        plot_config['col_labels'] = plot_config.get("col_labels", ["Truth", "ROM"])
        _adjust_option = partial(_add_train_window, plot_config=plot_config, tmult=tmult,
                                 window_opts=plot_config.pop("window_opts", None), update_leg=True,
                                 window_vline_opts=plot_config.pop("window_vline_opts", None))
        
        # Convert higher-dimensional data to 1d
        if (reduce_funcs := plot_config.pop("reduce", [])):
            reduce_opts = plot_config.pop("reduce_opts", {})

            def _iter_truth_and_rom():
                for f in reduce_funcs:
                    pconfig = copy.deepcopy(plot_config)
                    truth, rom, new_data_opts = reduce(truth_tuple, rom_tuples, truth_data_opts, 
                                                       case_info['model'], f, h5_path, pconfig, tmult=tmult, **reduce_opts)
                    yield truth, rom, pconfig, data_opts if new_data_opts is None else new_data_opts

        # Otherwise data is already 1d and should use directly
        else:
            def _iter_truth_and_rom():
                truth_dict = from_numpy(*truth_tuple)[FRAME_KEY]
                rom_dicts = {r: from_numpy(*rom_tuples[r])[FRAME_KEY] for r in rom_tuples}
                yield [truth_dict], list(rom_dicts.values()), plot_config, data_opts

        for truth, rom, pconfig, dopts in _iter_truth_and_rom():
            if isinstance(v := pconfig.get("variable", None), str) or v is None:
                # Display multiple ROMs for a single variable (done by default)
                # pconfig['row_labels'] = pconfig.get("row_labels", [pconfig.get("data_labels", {}).get(k, k) for k in rom_tuples.keys()])
                pass
            else:
                # Display multiple variables for only the chosen ROM
                show_rom = pconfig.pop("show_rom", next(iter(rom_tuples.keys())))
                rom = [rom[list(rom_tuples.keys()).index(show_rom)]]
            
            if len(truth[0]) > 1:
                # Single snapshot
                print(f"Plotting 1d error at time snapshot t={time[snapshot_idx]:.1E} s for case '{case}'")
                compare([[truth[i][snapshot_idx]] for i in range(len(truth))],
                        [[rom[i][snapshot_idx]] for i in range(len(rom))],
                        dopts, **pconfig)
                
                # 1d animation
                if animate:
                    print(f"Generating 1d error animation for case '{case}'")
                    pconfig['save'] = pconfig['save'].with_suffix('.mp4')
                    compare(truth, rom, dopts, time=truth_tuple[0], **pconfig)
                
                # Contour (x, t) plot
                if contour_opts:
                    print(f"Plotting 1d error contour for case '{case}'")
                    truth_tup = to_numpy({FRAME_KEY: truth[0], 'time': truth_tuple[0], 'coords': dopts['coord']})
                    rom_tups = {r: to_numpy({FRAME_KEY: rom[i], 'time': rom_tuples[r][0], 'coords': dopts['coord']}) 
                                for i, r in enumerate(rom_tuples.keys())}
                    truth_2d, rom_2d = _extract_truth_and_rom(lambda arr: arr.T, truth_tup, rom_tups)  # v: (Nx, Nt) single 2d frame

                    x_skip, t_skip = contour_downsample
                    truth_2d = [[{v: arr[0][v][::x_skip, ::t_skip] for v in arr[0]}] for arr in truth_2d]
                    rom_2d = [[{v: arr[0][v][::x_skip, ::t_skip] for v in arr[0]}] for arr in rom_2d]

                    if contour_window:
                        pconfig['adjust'] = _adjust_option
                    pconfig.update(contour_opts)
                    pconfig['combine_opts'] = ()
                    _path = Path(pconfig['save'])
                    pconfig['save'] = _path.parent / f"{_path.stem}_contour.pdf"
                    compare(truth_2d, rom_2d, {'X': truth_tuple[0][::t_skip]*tmult, 'Y': dopts['coord'][::x_skip]}, **pconfig)

            # Static 1d (maybe from a reduction, like time-avg)
            else: 
                compare(truth, rom, dopts, **pconfig)

    # ------------------------------2D FIELD ERROR ANIMATION----------------------------------------
    k = 'plot_2d_error'
    plot_config = case_info.get(k)
    if k in case_info and not plot_config.pop("skip", False):
        animate = plot_config.pop('animate', global_plot.get('animate', False))
        _update_from_global(plot_config, '2d_error')
        snapshot_time = plot_config.pop("snapshot_time", -1)
        time_fraction = plot_config.pop("time_fraction", True)
        extra_data_opts = plot_config.pop("data_opts", {})
        truth_data_opts.update(extra_data_opts)

        # Get snapshot time index
        time = truth_tuple[0]
        tlen = time[-1] - time[0]
        if time_fraction:
            if snapshot_time < 0:
                snapshot_time = 1
            snapshot_time = snapshot_time * tlen + time[0]
        else:
            if snapshot_time < 0:
                snapshot_time = time[-1]
        snapshot_idx = np.argmin(np.abs(time - snapshot_time))

        plot_config['col_labels'] = plot_config.get("col_labels", ["Truth", "ROM"])

        # No reduction from higher dims to 2d (but could follow similar procedure as 1d error)
        truth = [from_numpy(*truth_tuple)[FRAME_KEY]]
        rom = [from_numpy(*rom_tuples[r])[FRAME_KEY] for r in rom_tuples]

        if isinstance(v := plot_config.get("variable", None), str) or v is None:
            pass
        else:
            show_rom = plot_config.pop("show_rom", next(iter(rom_tuples.keys())))  # Display multiple variables for only the chosen ROM
            rom = [rom[list(rom_tuples.keys()).index(show_rom)]]

        print(f"Plotting 2d error at time snapshot t={time[snapshot_idx]:.1E} s for case '{case}'")
        compare([[truth[i][snapshot_idx]] for i in range(len(truth))],
                [[rom[i][snapshot_idx]] for i in range(len(rom))],
                truth_data_opts, **plot_config)  # use truth data opts for 2d (could be cell or pcolor info)
        
        if animate:
            print(f"Generating 2d error animation for case '{case}'")
            plot_config['save'] = plot_config['save'].with_suffix('.mp4')
            compare(truth, rom, truth_data_opts, time=truth_tuple[0], **plot_config)


def main():
    """General idea is:
    
    1) Parse command line (read config file)
    2) Merge global options for each case
    3) Run each case
       a) Run true model
       b) Run reduced-order model(s)
       c) Show comparison plots. Choose from any/all of:
          i)   Total relative error time
          ii)  Scalar error over time
          iii) 1D field error (animated line plot or contour over time)
          iv)  2D field error (contour plot, optionally animated over time)
    """
    parser = argparse.ArgumentParser(description="Run reduced-order modeling forecasting analyses.")
    parser.add_argument("config", type=Path,
                        help="Path to the.json file with configuration details")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Path to output directory where data should be saved (default: same dir as input)")
    parser.add_argument("-O", "--overwrite", action="store_true",
                        help="Overwrite all existing post-processing data if found (default: false)")
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise ValueError(f"Config file '{config_path}' not found.")
    with open(config_path, 'r') as fd:
        config = json.load(fd)

    out_dir = Path(args.output) if args.output is not None else config_path.parent
    CONFIG_PATH = config_path.parent

    if not out_dir.exists():
        os.mkdir(out_dir)

    if 'cases' not in config:
        raise TypeError("Must specify 'cases' to run in config file")
    
    cases = config['cases']
    roms = config.get('rom', {})
    plot_opts = {k: v for k, v in config.items() if 'plot' in k}

    for case, case_info in cases.items():
        if case_info.get('skip', False):
            continue
        
        # Add global rom settings to each case (but cases individually can override)
        if 'rom' not in case_info:
            case_info['rom'] = roms
        
        if roms:
            for r in roms:
                if r not in case_info['rom']:
                    case_info['rom'][r] = roms[r]
                else:
                    for ropt in roms[r]:
                        if ropt not in case_info['rom'][r]:
                            case_info['rom'][r][ropt] = roms[r][ropt]
        
        # Update plot settings for each case (cases can individually override)
        for k, v in plot_opts.items():
            if k not in case_info:
                case_info[k] = v
            else:
                for popt in v:
                    if popt not in case_info[k]:
                        case_info[k][popt] = v[popt]

        run_case(case, case_info, out_dir, args.overwrite)


if __name__ == '__main__':
    main()
