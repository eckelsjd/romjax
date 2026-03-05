"""Postprocess Hall2De simulation data to quickly visualize performance and prep ROM analyses.

Usage:
  python postprocess.py config.json -o output_dir

Steps:
  1) Load              -- Load Hall2De simulation data
  2) Filter            -- Condense simulation data to be post-processed
  3) Postprocess       -- The rest of the postprocess steps are optional and are documented below. Options are:
     - plot_2d         -- 2D z-r plots of field quantities
     - plot_centerline -- 1D channel centerline plots 
     - plot_metrics    -- Time history plots of selected performance metrics

A full working example of the config file is provided as example-post-config.json. 
**NOTE** nearly every option except for "load" is optional with good defaults.

For plotting, everything passed to "global_plot" will be used in all plotting routines (unless overriden).
All arguments of `romtools.plotting.gridplot` may be passed to all of the plotting routines to configure the output.
Please see the `gridplot` documentation for these options (not listed here).
In addition to `gridplot` arguments, you may also pass "skip" to skip a step, or "animate" to animate a step.

All additional configurations are described below.

Config file format:
{
  "load": {
      "dat_file":  (Required) A path to the output Tecplot file from Hall2De (relative to the config file),
      "h5_file":   (Optional) A path to the postprocess binary hdf5 save file (defaults to same name/location as dat_file),
                   Will be loaded if it exists, otherwise will be created.
      "absolute_path": Whether the files are provided as absolute path (default is relative to the config file)
      "overwrite": Whether to completely reload tecplot data and overwrite h5, otherwise will only try
                   to load new data and append to existing data (default).
  }
  "filter": {
      "tspan":         Tuple of start and end times for filtering simulation data for postprocess (defaults to full span)
      "time_fraction": If true, interpret values in tspan as fractions of total time rather than absolute seconds, (default false)
      "downsample":    Rate to downsample simulation data (defaults to 1, which does not downsample)
      "overwrite":     Whether to rewrite with the new filtered data, or try to load existing (default)
      "variables":     List of variable names to keep
  }
  "global_plot": {     Provides global plotting options (useful defaults are set for all), can also override in each plot routine below
      "skip":          Whether to skip a plotting routine (will skip all if used in global_plot)
      "animate":       Whether to animate plots (where applicable), default False
  }
  "plot_2d": {         See `gridplot` documentation (no additional configurations)
  }
  "plot_centerline": {
      "slice_opts":    Options for slicing the centerline
  }
  "plot_metrics": {
      "metrics":       List of metrics to compute and plot
      "share_plot":    Put metrics on the same plot (see `romtools.plotting.LineMetadata`)
      "time_multiplier": Amount to multiply by time to plot on the x-axis
      "animated_bar":  Options to put an animated vertical bar on the plot (see `LineMetadata`)
  }
}
"""
import argparse
from pathlib import Path
import json
import os
import warnings
import copy

import numpy as np

from romtools.dataloader import load_tecplot, load_h5, save_h5, filter_data, FRAME_KEY
from romtools.dataloader import to_numpy, from_numpy, save_numpy, load_numpy
from romtools.plotting import gridplot
from romtools.utils import get_boundary

from performance import thrust, beam_current, discharge_current, slice_centerline, get_channel


# ================== DEFAULTS =======================================
# Everything is overrideable in the input config.json file as described above.
_default_file = Path(__file__).parent / "defaults.json"

if _default_file.exists():
    with open(_default_file, "r") as fd:
        _defaults = json.load(fd)
else:
    warnings.warn(f"Couldn't find default config file '{_default_file}'. Please place this next "
                  f"to the postprocess.py file. Falling back on hard-coded options...", UserWarning)

    _defaults = {
        "data_labels": {
            "nn (m^-3)" : "Neutral density ($m^{-3}$)",
            "un_z (m/s)": "Neutral axial velocity (m/s)",
            "un_r (m/s)": "Neutral radial velocity (m/s)",
            "ni(1,1) (m^-3)": "Ion density ($m^{-3}$)",
            "ui_z(1,1) (m/s)": "Ion axial velocity (m/s)",
            "ui_r(1,1) (m/s)": "Ion radial velocity (m/s)",
            "je_z (A/m^2)": "Electron axial current (A/$m^2$)",
            "je_r (A/m^2)": "Radial electron current density (A/$m^2$)",
            "Te (eV)": "Electron temperature (eV)",
            "Phi (V)": "Potential (V)"
        },
        "data_plot_opts": {
            "nn (m^-3)": {"cmap": "viridis", "norm": "log"},
            "un_z (m/s)": {"cmap": "viridis"}, 
            "un_r (m/s)": {"cmap": "viridis"},
            "ni(1,1) (m^-3)": {"cmap": "jet", "norm": "log"},
            "ui_z(1,1) (m/s)": {"cmap": "jet"},
            "ui_r(1,1) (m/s)": {"cmap": "jet"},
            "je_z (A/m^2)": {"cmap": "plasma", "norm": "symlog"},
            "Te (eV)": {"cmap": "plasma"},
            "Phi (V)": {"cmap": "plasma"}
        },
        "exclude": ["je_r (A/m^2)"]
    }

METRICS = {
    "thrust": thrust,
    "discharge_current": discharge_current,
    "beam_current": beam_current
}

SLICE_OPTS = {
    'num_points': 100,
    'zlim': 2  # num of channel lengths
}
# ========================================================================


def main():
    parser = argparse.ArgumentParser(description="Process Hall2De data and generate diagnostics.")
    parser.add_argument("config", type=Path, help="Path to the postprocess config file")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Path to output directory where outputs should be saved (default: same dir as config)")
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        raise ValueError(f"Config file '{config_path}' not found.")
    with open(config_path, 'r') as fd:
        config = json.load(fd)

    in_dir = config_path.parent
    out_dir = Path(args.output) if args.output is not None else in_dir

    if not out_dir.exists():
        os.mkdir(out_dir)

    print("----------LOAD----------")

    load_config = config.get("load", {"overwrite": False})
    is_absolute = load_config.get("absolute_path", False)

    if "dat_file" not in load_config:
        raise TypeError(f"Need to specify 'dat_file' for loading Hall2De data.")

    if is_absolute:
        dat_path = Path(load_config['dat_file'])
        h5_path = Path(load_config.get('h5_file', dat_path.with_suffix('.h5')))
    else:
        dat_path = in_dir / Path(load_config['dat_file'])
        h5_path = in_dir / load_config.get('h5_file', Path(load_config['dat_file']).with_suffix('.h5'))

    if not dat_path.exists():
        raise FileNotFoundError(f"Hall2De output tecplot file '{dat_path}' not found. Exiting...")
    
    if not h5_path.exists() or load_config.get('overwrite', False):
        print("Loading tecplot .dat file...")
        dict_data = load_tecplot(dat_path)

        print(f"Generating .h5 file...")
        save_h5(dict_data, h5_path, overwrite=load_config.get('overwrite', False))
    else:
        print(f"Loading from existing h5: '{h5_path}'")
        dict_data = load_h5(h5_path)
        t_last = dict_data['time'][-1]

        print(f"Trying to update with any new tecplot data...")
        new_data = load_tecplot(dat_path, t_start=t_last)

        # Remove duplicate time step at beginning that got reloaded
        new_data[FRAME_KEY] = new_data[FRAME_KEY][1:]
        new_data['time'] = new_data['time'][1:]

        if len(new_data[FRAME_KEY]) > 0:
            print(f"Appending new tecplot data past t={t_last:.2E} s, total of {len(new_data[FRAME_KEY])} new frames...")
            dict_data[FRAME_KEY].extend(new_data[FRAME_KEY])
            dict_data['time'] = np.concatenate((dict_data['time'], new_data['time']))
            save_h5(new_data, h5_path)
    
    _temp = filter_data(dict_data, loc='node', select_frames=[0])
    vertices = _temp['coords']
    _temp = filter_data(dict_data, loc='cell', select_frames=[0])
    cells = _temp['coords']
    conn = dict_data['connectivity']
    species = dict_data['species']

    print("----------FILTER----------")

    dict_data.setdefault('filtered', {})
    c = dict_data['filtered']
    filter_config = config.get("filter", {"tspan": (0, -1), "time_fraction": False})
    tspan = filter_config.get("tspan", (0, -1))
    fields = filter_config.get("variables", _defaults.pop("variables", None))

    if filter_config.get("time_fraction", False):
        tlen = dict_data['time'][-1] - dict_data['time'][0]
        tstart =  tspan[0] * tlen + dict_data['time'][0]
        tend = (tspan[1] if tspan[1] is not None and tspan[1] > 0 else 1) * tlen + dict_data['time'][0]
        tspan = (tstart, tend)

    filter_kwargs = dict(variables=fields, loc='cell', tlim=tspan,
                         select_frames=range(0, len(dict_data['time']), filter_config.get("downsample", 1)))
    
    if len(c) == 0 or filter_config.get("overwrite", False):
        print("Filtering simulation data...")
        time_avg_data = filter_data(dict_data, time_avg=True, **filter_kwargs)
        time_res_data = filter_data(dict_data, time_avg=False, **filter_kwargs)

        avg, hist = {}, {}
        save_numpy(*to_numpy(time_avg_data), avg)
        save_numpy(*to_numpy(time_res_data), hist)

        save_h5({'filtered': {'avg': avg, 'hist': hist}}, h5_path, overwrite=True)
    
    else:
        print("Loading existing filtered simulation data...")
        time_avg_data = from_numpy(*load_numpy(c['avg']))
        time_res_data = from_numpy(*load_numpy(c['hist']))
    
    # Get global plot options (or defaults)
    global_plot = config.get("global_plot", {})
    for k, v in _defaults.items():
        global_plot[k] = global_plot.get(k, v)
    if 'data_labels' not in global_plot:
        global_plot['data_labels'] = {v: v for v in time_avg_data[FRAME_KEY][0].keys()}
    exclude_opts = {'animate', 'skip'}  # don't pass these on to gridplot
    
    def _update_from_global(plot_config: dict, fname_tag: str):
        """Update a specific plot config using global options (and update filename)"""
        for opt in global_plot:
            if opt not in exclude_opts and opt not in plot_config:
                plot_config[opt] = global_plot[opt]
        plot_config['save'] = (out_dir/(plot_config.get("save") or f"{dat_path.stem}_{fname_tag}")).with_suffix(".pdf")
        
    plot_config = config.get("plot_2d")
    if "plot_2d" in config and not plot_config.pop("skip", False):
        print("----------PLOT 2D----------")
        animate = plot_config.pop('animate', global_plot.get('animate', False))
        extra_data_opts = plot_config.pop('data_opts', {})
        _update_from_global(plot_config, '2d')
        
        print("Generating 2d time-average plots")
        data_opts = {'cells': cells, 'vertices': vertices, 'connectivity': conn}
        data_opts.update(extra_data_opts)
        gridplot(time_avg_data[FRAME_KEY], data_opts, **plot_config)
        
        if animate:
            print("Generating 2d time-history animation")
            plot_config['save'] = plot_config['save'].with_suffix(".mp4")
            gridplot(time_res_data[FRAME_KEY], data_opts, time=time_res_data['time'], **plot_config)
    
    plot_config = config.get("plot_centerline")
    if "plot_centerline" in config and not plot_config.pop("skip", False):
        print("----------PLOT CENTERLINE----------")
        dict_data.setdefault('centerline', {})
        c = dict_data['centerline']
        boundary_edges, _ = get_boundary(conn)
        channel_dims = get_channel(boundary_edges, vertices)

        if len(c) == 0 or filter_config.get("overwrite", False):
            print("Generating centerline slice...")
            slice_opts = copy.deepcopy(SLICE_OPTS)
            slice_opts.update(plot_config.pop("slice_opts", {}))
            slice_avg_coords, slice_avg_data = slice_centerline(time_avg_data[FRAME_KEY], cells, vertices, conn, 
                                                                **slice_opts)
            slice_hist_coords, slice_hist_data = slice_centerline(time_res_data[FRAME_KEY], cells, vertices, conn, 
                                                                  verbose=True, **slice_opts)
            slice_avg = {'time': time_avg_data['time'], 'coords': slice_avg_coords, FRAME_KEY: slice_avg_data}
            slice_hist = {'time': time_res_data['time'], 'coords': slice_hist_coords, FRAME_KEY: slice_hist_data}
            
            avg, hist = {}, {}
            save_numpy(*to_numpy(slice_avg), avg)
            save_numpy(*to_numpy(slice_hist), hist)

            save_h5({'centerline': {'avg': avg, 'hist': hist}}, h5_path, overwrite=True)

        else:
            print("Using existing centerline slice")
            slice_avg = from_numpy(*load_numpy(c['avg']))
            slice_hist = from_numpy(*load_numpy(c['hist']))
            if 'slice_opts' in plot_config:
                del plot_config['slice_opts']
        
        animate = plot_config.pop('animate', global_plot.get('animate', False))
        _update_from_global(plot_config, 'centerline')

        data_opts = {'coord': slice_avg['coords'][:, 0] / channel_dims['channel_length']}
        plot_config["coord_labels"] = plot_config.get("coord_labels", ["Channel centerline (channel lengths)"])
        gridplot(slice_avg[FRAME_KEY], data_opts, **plot_config)

        if animate:
            print("Generating centerline time-history animation")
            plot_config['save'] = plot_config['save'].with_suffix(".mp4")
            gridplot(slice_hist[FRAME_KEY], data_opts, time=slice_hist['time'], **plot_config)
        
    plot_config = config.get("plot_metrics")
    if "plot_metrics" in config and not plot_config.pop("skip", False):
        print("----------PLOT METRICS----------")
        dict_data.setdefault('performance', {})
        c = dict_data['performance']

        if len(c) == 0 or filter_config.get("overwrite", False):
            print("Generating performance metrics...")
            if len(metrics := plot_config.pop('metrics', list(METRICS.keys()))) > 0:
                for m in metrics:
                    if m not in METRICS:
                        raise NotImplementedError(f"Requested metric '{m}' not supported.")
            perf = {
                metric: METRICS[metric](time_res_data[FRAME_KEY], cells, vertices, conn, 
                                        integrand_opts=dict(species=species)) 
                for metric in metrics
            }
            save_h5({'performance': perf}, h5_path, overwrite=True)
            
        else:
            print("Using existing performance metrics")
            perf = c

        if 'thrust' in perf:
            perf['thrust'] = perf['thrust'][:, 0]  # Plot axial thrust only
        
        animate = plot_config.pop('animate', global_plot.get('animate', False))
        _update_from_global(plot_config, 'perf')

        plot_config['grid'] = plot_config.get("grid", (1, 2) if 'discharge_current' in perf or 'beam_current' in perf else (1, 1))
        plot_config['data_labels'] = plot_config.get("data_labels", {'thrust': 'Thrust (N)', 'discharge_current': "$I_d$", 'beam_current': "$I_b$"})
        data_opts = {'share_plot': plot_config.pop("share_plot", {"Current (A)": ['discharge_current', 'beam_current']}),
                     'coord': time_res_data['time']*plot_config.pop("time_multiplier", 1)}
        gridplot([perf], data_opts, **plot_config)
        
        if animate:
            print("Generating performance metrics time-history animation")
            data_opts['animated_bar'] = plot_config.pop("animated_bar", {"color": "r", "linewidth": 0.5})
            plot_config['save'] = plot_config['save'].with_suffix(".mp4")
            gridplot([perf]*len(time_res_data['time']), data_opts, time=time_res_data['time'], **plot_config)


def _plot_just_mesh(h5_file: str | Path):
    """Not used, but provided for convenience in case you are debugging.
    
    :param h5_file: path to the postprocess h5 file
    """
    import matplotlib.pyplot as plt
    from performance import thruster_edges

    d = load_h5(Path(h5_file))
    _temp = filter_data(d, loc='node', select_frames=[0])
    vertices = _temp['coords']
    _temp = filter_data(d, loc='cell', select_frames=[0])
    cells = _temp['coords']
    conn = d['connectivity']

    empty_frame = {'mesh': np.full(len(cells), np.nan)}
    data_opts = {'cells': cells, 'vertices': vertices, 'connectivity': conn, 'show_mesh': True,
                 'group_boundary': thruster_edges, 'boundary_colors': {'anode': 'g', 'outflow': 'b', 'thruster': 'r'}}
    gridplot([empty_frame], data_opts, coord_labels=['z', 'r'], data_labels={'mesh': ''})
    boundary_edges, _ = get_boundary(conn)
    channel_dims = get_channel(boundary_edges, vertices)
    print(f"Channel dims: {channel_dims}")

    plt.show()


def _validate_metrics(h5_file: str | Path, circuit_file: str | Path):
    """Not used, but provided for convenience for debugging performance metric calculations.
    
    :param h5_file: path to the h5 postprocess file
    :param circuit_file: the circuit history.txt file output by Hall2De with performance metrics
    """
    # Compare hall2de thrust/current to computed
    d = np.loadtxt(Path(circuit_file), skiprows=1)
    t_circuit = d[:, 0]
    thrust_circuit = d[:, 3]
    id_circuit = np.abs(d[:, 1])
    ib_circuit = d[:, 2]

    data_dict = load_h5(Path(h5_file), groups=['performance', 'filtered'])
    thrust = data_dict['performance']['thrust']
    id = data_dict['performance']['discharge_current']
    ib = data_dict['performance']['beam_current']
    t = data_dict['filtered']['hist']['time']

    t = t * 1000
    t_circuit = t_circuit * 1000

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 5), layout='tight')
    ax[0].plot(t, thrust[:, 0], '-k', label='Local')
    ax[0].plot(t_circuit, thrust_circuit, '--r', label='Hall2De')
    ax[0].set_ylabel('Thrust (N)')
    ax[0].set_xlabel('Time (ms)')
    ax[0].legend()
    ax[1].plot(t, id, '-k', label='Local $I_d$')
    ax[1].plot(t_circuit, id_circuit ,'--r', label='Hall2De $I_d$')
    ax[1].plot(t, ib, '-b', label='Local $I_b$')
    ax[1].plot(t_circuit, ib_circuit, '--g', label='Hall2De $I_b$')
    ax[1].set_ylabel('Current (A)')
    ax[1].set_xlabel('Time (ms)')
    ax[1].legend()
    
    plt.show()


if __name__ == '__main__':
    main()
