"""Plotting utilities.

Includes:
  - gridplot - Plot simulation data (1d or 2d) in a grid (with animation utilities)
  - compare  - Helper to compare two sets of simulation data in a 3-column (truth, rom, error) plot
"""
# ruff: noqa
import copy
from abc import ABC
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, NotRequired, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.tri import Triangulation
from numpy.typing import ArrayLike

from romtools.utils import edge_normal, get_boundary, relative_error

__all__ = ['gridplot', 'compare']

ANIMATE_DEFAULT = {
    'blit': True, 
    'fps': 10, 
    'frame_skip': 1, 
    'dpi': 200
}

GRID_OPTS = {
    "color": (0.5, 0.5, 0.5, 0.2),
    "lw": 0.5
}


class Frame(TypedDict):
    """A single frame of simulation data. Maps several variable names `v` to numpy arrays.
    The arrays can be any shape, so long as all variables have the same shape.
    
    For example:
       (N,)    - 1d data
       (N, 2)  - 2d unstructured mesh data
       (N, M)  - 2d structured mesh data
       etc.
    """
    v: NotRequired[ArrayLike]


class PlotMetadata(ABC):
    """Base class for providing extra required info for generating plots. 
    
    Currently supported plot types:

    line - standard plt.plot lines
    pcolor - pcolormesh from structured 2d mesh data
    cell - use PolyCollection for quadrilaterals (such as from unstructured finite-volume mesh)
    """
    _supported = ['line', 'pcolor', 'cell']
    
    @classmethod
    def from_dict(cls, d):
        if 'type' not in d:
            raise TypeError(f"Must give a 'type' to construct PlotMetadata from a dict. Options are: {cls._supported}")
        
        plot_type = d.pop('type')

        if plot_type not in cls._supported:
            raise TypeError(f"Unsupported plot type '{plot_type}'. Options are: {cls._supported}")

        return {'line': LineMetadata, 'pcolor': PcolorMetadata, 'cell': CellMetadata}.get(plot_type)(**d)


@dataclass(frozen=True)
class LineMetadata(PlotMetadata):
    """Options for standard line plots.
    
    :ivar coord: (N,) 1d array of the horizontal coordinate
    :ivar animated_bar: if provided, an animated vertical bar will move left->right across the plot,
                        this dict will be passed to a plt line plot to change how the line looks.
    :ivar share_plot: sets of variables to display on same subplot, dict keys give the ylabels for the subplots,
                      each variable should only be shown on exactly one subplot
    """
    coord: ArrayLike
    animated_bar: Optional[dict] = None
    share_plot: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class CellMetadata(PlotMetadata):
    """Options for treating 2d data as cell-centered quadrilaterals.
    
    Required options are:
    :ivar cells: (N, 2) array of cell-center coordinates corresponding to the arrays in `data`. Order is (x,y)
    :ivar vertices: (M, 2) array of vertex coordinates. Order for coordinates is (x,y)
    :ivar connectivity: (N, 4) array specifying vertex indices for each cell
    
    Optional options are:
    :ivar shading: If flat (default), just color cells with avg value. If gouraud, use tripcolor to shade
    :ivar show_mesh: Show grid lines of mesh (must have node data and connectivity)
    :ivar cell_center_opts: If provided, options for plotting cell centers. Not plotted if None
    :ivar boundary_cell_color: If provided, the color to highlight boundary cells. Not highlighted if None
    :ivar boundary_colors: colors for boundaries, specify as dict for different values for each boundary group
    :ivar group_boundary: Sorts edges by index into groups (from which boundary is selected), defaults to selecting all
    """
    cells: ArrayLike
    vertices: ArrayLike
    connectivity: ArrayLike
    shading: Literal['flat', 'gouraud'] = 'flat'
    show_mesh: bool = False
    cell_center_opts: Optional[dict] = None
    boundary_cell_color: Optional[str] = None
    boundary_colors: Optional[dict | str] = None
    group_boundary: Optional[callable] = None


@dataclass(frozen=True)
class PcolorMetadata(PlotMetadata):
    """Options for treating 2d data as 2d regular mesh for pcolormesh (see plt docs for details).
    
    Required options are:
    :ivar X: (Nx,) 1d array of horizontal coordinates
    :ivar Y: (Ny,) 1d array of vertical coordinates
    """
    X: ArrayLike
    Y: ArrayLike


def _format_time_engineering(seconds: float):
    """Helper to format times in common engineering magnitudes."""
    prefixes = [
        (1e-12, "ps"),
        (1e-9, "ns"),
        (1e-6, "μs"),
        (1e-3, "ms"),
        (1.0,  "s"),
        (1e3, "ks")
    ]

    # Find the appropriate prefix and scale
    for factor, suffix in prefixes:
        scaled = seconds / factor
        if 1 <= scaled < 1000:
            return f"{scaled:.1f} {suffix}"
    
    # Fall back to scientific notation if out of normal range
    return f"{seconds:.1e} s"


def _get_scheme(scheme: Literal['white', 'dark']):
    """Return text and background colors for given scheme."""
    match scheme.lower():
        case 'white':
            text_color = 'black'
            bg_color = 'white'
        case 'dark' | 'black':
            text_color = 'white'
            bg_color = 'black'
        case _:
            text_color = 'black'
            bg_color = 'white'
    
    return text_color, bg_color


def compare(first: list[list[Frame]],
            second: list[list[Frame]],
            data_opts: PlotMetadata | dict,
            show_error: bool = True, 
            variable: str | list[str] = None,
            row_labels: list[str] = None,
            col_labels: list[str] = None,
            tag_data_labels: bool = False,
            text_offset: float = 0.05,
            error_func: Literal['absolute', 'relative'] | Callable[[ArrayLike, ArrayLike], ArrayLike] = 'relative',
            error_func_opts: dict = None,
            error_plot_opts: dict = None,
            error_grid_opts: dict = GRID_OPTS,
            combine_opts: tuple[dict, dict] = (),
            **gridplot_opts
            ):
    """Helper to compare two sets of simulation data using the gridplot function.

    Each element of `first` and `second` is frame data for a single row of comparison (see `gridplot`). If
    this frame data has several items (i.e. multiple frames), then it will be animated.
    If either `first` or `second` has only one element, then this data will be repeated on each row.
    A third column will show the relative error (pointwise) between the first two columns.
    
    :param first: The frame data for the first column. One set of frames per row.
    :param second: The frame data for the second column. One set of frames per row.
    :param data_opts: Contains mesh/coord information, assumed to be the same for all rows and both columns.
    :param show_error: Whether to include pointwise error in a third column.
    :param variable: The variable to plot from each frame (defaults to first). If multiple are provided in a list, 
                     then only the first item of first/second will be selected for plotting, and each row will 
                     instead contain the data for each of these variables.
    :param row_labels: Labels for rows. Rows not labeled if None.
    :param col_labels: Labels for first two columns (optionally third label for error column). Columns not labeled if None.
    :param tag_data_labels: If true, will add col_labels to data labels when merging (by default, will overwrite existing data label)
    :param text_offset: for row labels on the left - the amount (in figure coords) to push row labels to the left
    :param error_func: Callable as error_func(second_col, first_col, **opts), computes error between columns (default is relative error)
    :param error_func_opts: Options to pass to the error func
    :param error_plot_opts: different options to use for error column (defaults to using same as variables)
    :param error_grid_opts: If provided, options for showing a grid in the error column (shows small gray grid by default), set to None to turn off
    :param combine_opts: If provided, the first and second columns will be combined, with the provided options overriding
                         any existing plot options for variables in each column. This is only recommended for 1d data.
    :param gridplot_opts: Rest of the options are passed to gridplot
    """
    if variable is None:
        variable = next(iter(first[0][0]))
    
    num_frames = len(first[0])  # same for all
    
    data_opts = copy.deepcopy(data_opts)
    opts = copy.deepcopy(gridplot_opts)
    pre_adjust = opts.get("adjust", None)
    data_plot_opts = {}
    data_labels = {}

    if error_func_opts is None:
        error_func_opts = {}

    if isinstance(error_func, str):
        match error_func:
            case 'relative': 
                error_func = relative_error
                error_func_opts.update({"pointwise": True})
                error_tag = 'relative error'
            case 'absolute':
                error_func = lambda pred, targ, **kwargs: np.abs(pred - targ)
                error_tag = 'absolute error'
                error_title = 'Absolute error'
            case _:
                raise NotImplementedError(f"Error function '{error_func}' not recognized.")
    else:
        error_tag = 'error'
    
    if col_labels is not None and show_error and len(col_labels) < 3:
        col_labels.append(error_tag.capitalize())

    # Figure out how to iterate variables (depending on which plots are sharing)
    if isinstance(data_opts, PlotMetadata):
        data_opts = asdict(data_opts)

    share_plot = data_opts.get('share_plot', {})
    if isinstance(variable, str):
        # Only one variable, no sharing, just pick frames and go
        num_rows = max(len(first), len(second))
        def _iter_rows():
            for r in range(num_rows):
                frame1 = first[r] if len(first) > 1 else first[0]
                frame2 = second[r] if len(second) > 1 else second[0]
                yield r, frame1, frame2, variable
    else:
        if share_plot:
            # Decide row based on share plots
            all_vars = copy.deepcopy(variable)
            set_vars = set(all_vars)
            for s in share_plot.values():
                set_vars = set_vars - set(s)
            num_single_vars = len(set_vars)
            num_rows = num_single_vars + len(share_plot)

            # Divide plots up into groups
            shared_groups = list(share_plot.values())
            shared_groups_map = {}  # shared group_idx -> index in all groups
            all_groups = []
            for v in all_vars:
                v_in_shared_group = False
                for group_idx, group in enumerate(shared_groups):
                    if v in group:
                        v_in_shared_group = True
                        if group_idx in shared_groups_map:
                            all_groups[shared_groups_map[group_idx]].append(v)
                        else:
                            all_groups.append([v])
                            shared_groups_map[group_idx] = len(all_groups) - 1
                        break  # each variable should only be in exactly one group
                
                if not v_in_shared_group:  # single vars get their own plot
                    all_groups.append([v])

            def _iter_rows():
                for r, group in enumerate(all_groups):
                    for v in group:
                        yield r, first[0], second[0], v
        else:
            # No sharing, just iterate all variables (only use first items for each column)
            num_rows = len(variable)
            def _iter_rows():
                for r in range(num_rows):
                    yield r, first[0], second[0], variable[r]

    num_cols = 3 if show_error else 2
    opts['grid'] = (num_rows, num_cols - 1 if len(combine_opts) > 0 else num_cols)  # One less column if combining
    combined_frames = [{} for _ in range(num_frames)]

    vmin = []
    vmax = []

    new_share_plot = {}
    share_plot_tags = col_labels if col_labels is not None else ['', ' ', '  ']  # need distinct group names for new share plot
    if share_plot:
        for group in share_plot:
            for j in range(num_cols):
                if len(combine_opts) > 0:
                    if j == 0:
                        new_share_plot.setdefault(group, [])
                    elif j == 1:
                        new_share_plot.setdefault(f"{group} {error_tag}", [])
                else:
                    new_share_plot.setdefault(f"{group} {share_plot_tags[j]}", [])

    # Combine frames and plot options
    for r, frame1, frame2, curr_var in _iter_rows():
        # Will only work on single static frames (animations update clims over time)
        vmin.append(np.nanmin([np.nanmin(frame1[0][curr_var]), np.nanmin(frame2[0][curr_var])]))
        vmax.append(np.nanmax([np.nanmax(frame1[0][curr_var]), np.nanmax(frame2[0][curr_var])]))

        og_label = opts.get('data_labels', {}).get(curr_var, curr_var)
        share_curr_var = len(combine_opts) > 0
        if share_curr_var:
            for group in share_plot:
                if curr_var in share_plot[group]:
                    share_curr_var = False  # Prevent redundant sharing
                    break

        if share_curr_var:
            new_share_plot.setdefault(og_label, [])
        
        for j in range(num_cols):
            var_key = f'{r}_{j}_{curr_var}'  # unique id for variable data and plotting options
            
            data_plot_opts[var_key] = copy.deepcopy(opts.get('data_plot_opts', {}).get(curr_var, {}))
            data_labels[var_key] = og_label

            # Update plot options for first two columns if combining
            if len(combine_opts) > 0 and j < 2:
                data_plot_opts[var_key].update(combine_opts[j])
                if tag_data_labels:
                    data_labels[var_key] += f' {share_plot_tags[j]}'
                elif col_labels is not None:
                    data_labels[var_key] = col_labels[j]

            # Update error plot opts
            if j >= 2:
                if error_plot_opts is not None:
                    if var_key in data_plot_opts:
                        data_plot_opts[var_key].update(error_plot_opts)
                    else:
                        data_plot_opts[var_key] = error_plot_opts
                if share_curr_var:
                    data_labels[var_key] = error_tag.capitalize()
                    # data_labels[var_key] += f' {error_tag}'
            
            # Share data for 1d plots
            if share_plot:
                for group in share_plot:
                    if curr_var in share_plot[group]:
                        if len(combine_opts) > 0:
                            if j < 2:  # combine first two columns
                                new_share_plot[group].append(var_key)
                            else:
                                new_share_plot[f"{group} {error_tag}"].append(var_key)
                        else:
                            new_share_plot[f"{group} {share_plot_tags[j]}"].append(var_key)
            
            # Combine first two columns even if not originally shared
            if share_curr_var and j < 2:
                new_share_plot[og_label].append(var_key)

            # Merge the frames
            for i in range(num_frames):
                row_data = [frame1[i][curr_var], frame2[i][curr_var]]
                
                if show_error and j >= 2:
                    row_data.append(error_func(row_data[1], row_data[0], **error_func_opts))
                    
                combined_frames[i][var_key] = row_data[j]
    
    def _adjust(fig, axs):
        text_color, bg_color = _get_scheme(opts.get("scheme", "white"))
        text_opts = opts.get("text_opts", {})
        if pre_adjust is not None:
            pre_adjust(fig, axs)

        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                # Merge first two column color bars (won't work on animations, just for single frames)
                if j < axs.shape[1] - 1:
                    if len(axs[i, j].collections) > 0:  # 2d data
                        axs[i, j].collections[0].set_clim(vmin[i], vmax[i])
                    elif len(axs[i, j].lines) > 0:      # 1d data
                        axs[i, j].set_ylim([vmin[i], vmax[i]])
                else:
                    if show_error and error_grid_opts:
                        axs[i, j].grid(**error_grid_opts)
                
                if i == 0 and col_labels is not None and len(combine_opts) == 0:
                    axs[i, j].set_title(col_labels[j] if j < len(col_labels) else error_tag.capitalize(), color=text_color, **text_opts)
                
                if j == 0 and row_labels is not None:
                    pos = axs[i, 0].get_position()
                    x_pos = pos.x0 - text_offset
                    y_pos = 0.5 * (pos.y1 + pos.y0)
                    fig.text(x_pos, y_pos, row_labels[i], va='center', ha='right', rotation=90, color=text_color, **text_opts)
                
    if new_share_plot:
        data_opts['share_plot'] = new_share_plot
    
    opts['data_labels'] = data_labels
    opts['data_plot_opts'] = data_plot_opts
    opts['adjust'] = _adjust

    fig, axs = gridplot(combined_frames, data_opts, **opts)

    return fig, axs


def gridplot(data: list[Frame],
             data_opts: PlotMetadata | dict,
             data_plot_opts: dict = None,
             global_plot_opts: dict = None,
             text_opts: dict = None,
             animate_opts: dict = None,
             legend_opts: dict = None,
             data_labels: dict = None,
             coord_labels: list[str] = None,
             time: ArrayLike = None,
             scheme: Literal['white', 'dark'] = 'white',
             exclude: list[str] = None,
             grid: tuple[int, int] = None,
             subplot_size_in: tuple[float, float] = (3, 2.5),
             save: str | Path = None,
             adjust: callable = None
             ):
    """Plots simulation data in a grid of subplots. Can do both 1d and 2d plots, and optionally animate over time.
    
    :param data: List of frames of simulation data to plot. If several frames are provided, the result will be
                 animated. If a single frame is provided, the result will be a static plot. Each frame is a dict
                 with field variable names mapped to arrays of data to plot.
    :param data_opts: Options for plotting data (CellMetadata, LineMetadata, and PcolorMetadata supported, see their
                 docstrings for details). Most importantly, this will contain information about the 1d or 2d grid coordinates.
    :param data_plot_opts: A dict mapping variable names to plot options. All plot options are passed to the underlying
                 matplotlib plot function for the given data type (e.g. ax.plot(**opts) for line plots)
    :param global_plot_opts: Options to use on all subplots in the grid (data_plot_opts takes priority for a given variable)
    :param text_opts: Options to pass to axis labels
    :param animate_opts: dict with animation options alternate for (blit=True, fps=10, frame_skip=1, dpi=200)
    :param legend_opts: Options to pass to axis legend construction
    :param data_labels: Labels to show on colorbars or y axes for each variable (defaults to just using variable names)
    :param coord_labels: Will show axis labels for (x,y) if provided, otherwise will hide axes (default)
    :param time: (Nt,) array of simulation time values (only used for labeling animation plot), must be same length as data
    :param scheme: Either white (default) or dark, for setting text and background colors
    :param exclude: variables to exclude from plotting, (default none)
    :param grid: The shape of subplots for multiple variables. By default, will make the best square grid.
    :param subplot_size_in: Tuple (W, H) of each subplot size (inches), all subplots are set to this size
    :param save: Name of file to save to (won't save if None)
    :param adjust: a catch-all func for applying additional changes to the plot before animating/saving,
                   callable as adjust(fig, axs)
    """
    if isinstance(data_opts, dict):
        if 'type' not in data_opts:
            # Try to infer plot type
            if 'cells' in data_opts:
                data_opts['type'] = 'cell'
            elif 'X' in data_opts and 'Y' in data_opts:
                data_opts['type'] = 'pcolor'
            elif 'coord' in data_opts:
                data_opts['type'] = 'line'
        data_opts = PlotMetadata.from_dict(data_opts)

    if exclude is None:
        exclude = {}
    if data_labels is None:
        data_labels = {}
    if animate_opts is None:
        animate_opts = {}
    if data_plot_opts is None:
        data_plot_opts = {}
    if global_plot_opts is None:
        global_plot_opts = {}
    if text_opts is None:
        text_opts = {}
    if legend_opts is None:
        legend_opts = {}
    
    all_vars = [v for v in data[0].keys() if v not in exclude]

    text_color, bg_color = _get_scheme(scheme)
    labels = {v: data_labels.get(v, v) for v in all_vars}
    a_opts = {k: animate_opts.get(k, v) for k, v in ANIMATE_DEFAULT.items()}
    plot_opts = copy.deepcopy(data_plot_opts)

    # Set defaults for plot options for all variables
    for v in all_vars:
        d = plot_opts.setdefault(v, global_plot_opts)
        for k in global_plot_opts:
            if k not in d:
                d[k] = global_plot_opts[k]

    # Plot sharing likely only ever needed for line plots
    share_plot = data_opts.share_plot if hasattr(data_opts, 'share_plot') else {}
    set_vars = set(all_vars)
    for s in share_plot.values():
        set_vars = set_vars - set(s)
    num_single_vars = len(set_vars)
    num_plots = num_single_vars + len(share_plot)

    # Divide plots up into groups
    shared_groups = list(share_plot.values())
    shared_groups_map = {}  # shared group_idx -> index in all groups
    all_groups = []
    for v in all_vars:
        v_in_shared_group = False
        for group_idx, group in enumerate(shared_groups):
            if v in group:
                v_in_shared_group = True
                if group_idx in shared_groups_map:
                    all_groups[shared_groups_map[group_idx]].append(v)
                else:
                    all_groups.append([v])
                    shared_groups_map[group_idx] = len(all_groups) - 1
                break  # each variable should only be in exactly one group
        
        if not v_in_shared_group:  # single vars get their own plot
            all_groups.append([v])
    shared_groups_inverse = {v: k for k, v in shared_groups_map.items()}
    
    if grid is None:
        c = int(np.ceil(np.sqrt(num_plots)))
        r = int(np.ceil(num_plots / c))
        grid = (r, c)

    # Setup figure, axis subplots
    fig, axs = plt.subplots(*grid, layout='tight', squeeze=False, sharex='col', 
                            sharey='none' if isinstance(data_opts, LineMetadata) else 'row', 
                            figsize=(subplot_size_in[0]*grid[1], subplot_size_in[1]*grid[0]))
    fig.patch.set_facecolor(bg_color)

    # Animation objects {variable: plot object}
    xdata = {}       # set_xdata
    ydata = {}       # set_ydata
    collections = {} # set_array

    # Do some extra stuff before plotting
    def _pre_plot():
        if isinstance(data_opts, CellMetadata):
            quads = [[data_opts.vertices[i] for i in cell] for cell in data_opts.connectivity]
            boundary_edges, boundary_cells = get_boundary(data_opts.connectivity)
            edge_groups = data_opts.group_boundary(boundary_edges, data_opts.vertices) if data_opts.group_boundary is not None else {}

            boundary_colors = {} if data_opts.boundary_colors is None else data_opts.boundary_colors
            boundary_colors = {b: boundary_colors.get(b, text_color) if isinstance(boundary_colors, dict) else boundary_colors for 
                               b in edge_groups}

            if data_opts.show_mesh:
                edgecolors = []
                for cell_idx in range(len(quads)):
                    if data_opts.boundary_cell_color is not None and cell_idx in boundary_cells:
                        edgecolors.append(data_opts.boundary_cell_color)
                    else:
                        edgecolors.append((0.5, 0.5, 0.5, 0.3))
            else:
                edgecolors='face'
            
            if data_opts.shading == 'gouraud':
                # Use tripcolor for shading
                triangles = []
                for quad in data_opts.connectivity:
                    v0, v1, v2, v3 = quad
                    triangles.append([v0, v1, v2])
                    triangles.append([v0, v2, v3])
                triangles = np.array(triangles)
                triang = Triangulation(data_opts.vertices[:, 0], data_opts.vertices[:, 1], triangles)
            else:
                triang = None
        
            return (quads, boundary_edges, boundary_cells, edge_groups, boundary_colors, edgecolors, triang)
        
        else:
            return ()

    def _iter_all_vars():
        """Iterate over variables and plot indices."""
        # Group plots in the order variables appear, accounting for shared plots where applicable        
        for plot_idx, group in enumerate(all_groups):
            for v in group:
                yield v, plot_idx

    pre_plot_info = _pre_plot()  # in case more things are needed for plotting
    skip_first_clims = False

    for v, plot_idx in _iter_all_vars():
        ax = axs.flatten()[plot_idx]
        arr = data[0].get(v)

        if np.all(np.isnan(arr)):
            skip_first_clims = True
            arr = np.full_like(arr, 1e-6)  # small workaround to avoid cbar issues

        axes_visible = coord_labels is not None
        ax.tick_params(axis='both', which='both', top=False, bottom=axes_visible, left=axes_visible, right=False, direction='in',
                       labelleft=axes_visible, labelbottom=axes_visible, color=text_color, labelcolor=text_color)
        ax.set_facecolor(bg_color)
        for spine in ['bottom', 'left', 'top', 'right']:
            ax.spines[spine].set_visible(axes_visible)
            ax.spines[spine].set_color(text_color)
        
        if axes_visible and plot_idx // grid[1] == grid[0] - 1:  # last row on grid
            ax.set_xlabel(coord_labels[0], color=text_color, **text_opts)
        if axes_visible and plot_idx % grid[1] == 0 and len(coord_labels) > 1:  # first column
            ax.set_ylabel(coord_labels[1], color=text_color, **text_opts)
        
        # Simple line plots
        if isinstance(data_opts, LineMetadata):
            line_opts = copy.deepcopy(plot_opts.get(v, {}))
            if 'cmap' in line_opts:
                if 'c' not in line_opts and 'color' not in line_opts:
                    line_opts['color'] = plt.get_cmap(line_opts['cmap'])(0)  # Use first color in cmap
                
                del line_opts['cmap']  # Can't use this in line plots
            
            if 'norm' in line_opts:
                ax.set_yscale(line_opts['norm'])
                del line_opts['norm']  # Can't use this in line plots

            l = ax.plot(data_opts.coord, arr, label=labels[v], **line_opts)
            ydata[v] = l[0]
            
            if data_opts.animated_bar is not None:
                lvert = ax.axvline(x=data_opts.coord[0], **data_opts.animated_bar)
                xdata[v] = lvert

            if plot_idx in shared_groups_inverse:
                leg = dict(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color, fancybox=True)
                leg.update(legend_opts)
                ax.legend(**leg)
                ylabel = list(share_plot.keys())[shared_groups_inverse[plot_idx]]
            else:
                ylabel = labels[v]
            ax.set_ylabel(ylabel, color=text_color, **text_opts)
        
        # Pcolor structured mesh 2d plot
        elif isinstance(data_opts, PcolorMetadata):
            pcm = ax.pcolormesh(data_opts.X, data_opts.Y, arr, **plot_opts.get(v, {}))
            collections[v] = pcm

            cb = plt.colorbar(pcm, ax=ax)
            cb.ax.set_ylabel(labels.get(v, v), color=text_color, **text_opts)
            cb.ax.tick_params(labelcolor=text_color, color=text_color)
            cb.ax.tick_params(which='minor', color=(0,0,0,0), width=0, size=0)
            cb.ax.minorticks_off()
            cb.outline.set_edgecolor(text_color)
        
        # Polycollection quadrilateral 2d plot
        elif isinstance(data_opts, CellMetadata):
            coords_min = np.min(data_opts.vertices, axis=0)
            coords_max = np.max(data_opts.vertices, axis=0)
            ax.autoscale(enable=False)
            ax.set_xlim([coords_min[0], coords_max[0]])
            ax.set_ylim([coords_min[1], coords_max[1]])
            quads, boundary_edges, boundary_cells, edge_groups, boundary_colors, edgecolors, triang = pre_plot_info

            if triang is None:
                # Use poly collection for 'flat' shading (default)
                pc = PolyCollection(quads, array=arr, edgecolors=edgecolors, **plot_opts.get(v, {}))
                ax.add_collection(pc)
                collections[v] = pc
            else:
                # Use triangles for shading otherwise
                vertex_arr = np.zeros(data_opts.vertices.shape[0])
                counts = np.zeros_like(vertex_arr)
                for center_val, verts in zip(arr, data_opts.connectivity):
                    for vidx in verts:
                        vertex_arr[vidx] += center_val
                        counts[vidx] += 1
                vertex_arr /= counts

                tpc = ax.tripcolor(triang, vertex_arr, shading='gouraud', edgecolors=edgecolors, **plot_opts.get(v, {}))
                collections[v] = tpc

            # Outline the boundary
            for i, edge in enumerate(boundary_edges):
                x_vals, y_vals = data_opts.vertices[edge, 0], data_opts.vertices[edge, 1]
                c = text_color
                for b in edge_groups:
                    if i in edge_groups[b]:
                        c = boundary_colors[b]
                        break
                ax.plot(x_vals, y_vals, color=c, linewidth=1.5)
            
            # Show normal boundary vectors
            if data_opts.boundary_cell_color is not None:
                pos = np.zeros((len(boundary_edges), 2))
                vel = np.zeros((len(boundary_edges), 2))
                for i, edge in enumerate(boundary_edges):
                    p1 = data_opts.vertices[edge[0]]
                    p2 = data_opts.vertices[edge[1]]
                    pos[i, :] = (p1 + p2) / 2
                    vel[i, :] = edge_normal(p1, p2, data_opts.cells[boundary_cells[i]])
                ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], color=data_opts.boundary_cell_color)

            if data_opts.cell_center_opts is not None:
                ax.scatter(data_opts.cells[:, 0], data_opts.cells[:, 1], **data_opts.cell_center_opts)
            
            cb = plt.colorbar(collections[v], ax=ax)
            cb.ax.set_ylabel(labels.get(v, v), color=text_color, **text_opts)
            cb.ax.tick_params(labelcolor=text_color, color=text_color)
            cb.ax.tick_params(which='minor', color=(0,0,0,0), width=0, size=0)
            cb.ax.minorticks_off()
            cb.outline.set_edgecolor(text_color)

    if adjust is not None:
        adjust(fig, axs)
    
    # Iterate over frames to animate
    if len(data) > 1:
        # Get global (ymin, ymax) for ylims/clims
        for v, plot_idx in _iter_all_vars():
            ax = axs.flatten()[plot_idx]
            if v in ydata:
                curr_min, curr_max = ax.get_ylim()
            elif v in collections:
                curr_min, curr_max = collections[v].get_clim()
            else:
                curr_min, curr_max = (np.nan, np.nan)

            ymin, ymax = [curr_min], [curr_max]
            for i in range(len(data)):
                if i == 0 and skip_first_clims:  # Only from an all-nan workaround
                    continue

                arr = data[i][v]
                ymin.append(np.nanmin(arr))
                ymax.append(np.nanmax(arr))
            
            if v in ydata:
                ax.set_ylim([np.nanmin(ymin), np.nanmax(ymax)])
            elif v in collections:
                collections[v].set_clim(np.nanmin(ymin), np.nanmax(ymax))
            else:
                pass # Shouldn't be here
        
        num_frames = int(len(data) / a_opts['frame_skip']) + 1  # use last one to show last time step

        def update(idx):
            idx_use = idx * a_opts['frame_skip'] if idx < num_frames - 1 else len(data) - 1
            if time is not None:
                fig.suptitle(f"t={_format_time_engineering(time[idx_use])}", color=text_color, **text_opts)
            
            xret, ret = None, None
            for v, plot_idx in _iter_all_vars():
                if v in ydata:
                    ydata[v].set_ydata(data[idx_use][v])
                    if ret is None:
                        ret = list(ydata.values())
                elif v in collections:
                    cell_arr = data[idx_use][v]

                    # Need to average cell data to vertices for tri mesh
                    if isinstance(collections[v], TriMesh):
                        vertex_arr = np.zeros(data_opts.vertices.shape[0])
                        counts = np.zeros_like(vertex_arr)
                        for center_val, verts in zip(cell_arr, data_opts.connectivity):
                            for vidx in verts:
                                vertex_arr[vidx] += center_val
                                counts[vidx] += 1
                        vertex_arr /= counts
                        collections[v].set_array(vertex_arr)
                    else:
                        collections[v].set_array(cell_arr)

                    if ret is None:
                        ret = list(collections.values())
                
                if v in xdata:
                    xdata[v].set_xdata([data_opts.coord[idx_use], data_opts.coord[idx_use]])  # Vertical line
                    if xret is None:
                        xret = list(xdata.values())
            
            return ret + (xret if xret is not None else [])
        
        ani = FuncAnimation(fig, update, frames=num_frames, blit=a_opts['blit'], interval=1/a_opts['fps'])

        if save is not None:
            print(f"Saving animation to '{save}'")
            def _progress(i, n):
                if np.mod(i, int(0.1 * n)) == 0 or i == 0 or i == n - 1:
                    print(f'Saving frame {i+1}/{n}...')
            ani.save(Path(save), writer='ffmpeg', progress_callback=_progress, dpi=a_opts['dpi'], fps=a_opts['fps'])
        else:
            plt.show()
    
    # Static figure
    else:
        if save is not None:
            fig.savefig(Path(save), bbox_inches='tight')
    
    return fig, axs
