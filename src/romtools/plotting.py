"""Plotting utilities.

Includes:
  - gridplot - Plot simulation data (1d or 2d) in a grid (with animation utilities)
  - get_scheme - Get a plotting color scheme
"""
# ruff: noqa
import copy
from abc import ABC
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, NotRequired, Optional, TypedDict, Any, Generator, Iterable
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PolyCollection, TriMesh
from matplotlib.tri import Triangulation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from numpy.typing import ArrayLike


__all__ = ['gridplot', 'PlotOpts', 'PlotSpec', 'SupportedPlots', 'get_scheme']


def _default_progress(i, n):
    """For printing animation save progress."""
    if n is not None:
        if np.mod(i, int(0.1 * n)) == 0 or i == 0 or i == n - 1:
            print(f'Saving frame {i+1}/{n}...')
    else:
        if np.mod(i+1, 20) == 0 or i == 0:
            print(f'Saving frame {i+1}...')


ANIMATE_DEFAULT = {
    'blit': False, 
    'fps': 10,
    'dpi': 200,
    'writer': 'ffmpeg',
    'progress_callback': _default_progress
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


def get_scheme(scheme: Literal['white', 'dark']):
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


type SupportedPlots = Literal["line", "pcolor", "contour", "contourf", "hist", "hist2d"] # TODO: tri, quad
type PlotName = str  # just a short, memorable key name

@dataclass
class PlotOpts:
    """A few particular options common to all types of plots.
    
    :ivar xlabel: the x-axis label
    :ivar ylabel: the y-axis label
    :ivar xscale: scale for x-axis
    :ivar yscale: scale for y-axis
    :ivar xlim: limits for x-axis (defaults to autoscale if None)
    :ivar ylim: limits for y-axis (defaults to autoscale if None)
    :ivar clim: limits for colorbar (if None, will not show a colorbar), can also set to 'auto'
    :ivar cbar_label: label for the colorbar (must set clim to show colorbar)
    :ivar leg_label: label for legend (legend only shown if all artists on an axis have a label)
    :ivar ax_visible: whether to show axes, ticks, and spines (default True)
    :ivar animate: whether to animate data for this plot (default False)
    """
    xlabel: str = ""
    ylabel: str = ""
    xscale: str | None = None
    yscale: str | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    clim: tuple[float, float] | Literal['auto'] | None = None
    cbar_label: str | None = None
    leg_label: str | None = None
    ax_visible: bool = True
    animate: bool = False

@dataclass
class PlotSpec:
    """Specs for a matplotlib plot.

    !!! Example "Animation"
        You can generate an animation simply like:
        ```python
        spec = PlotSpec(
            kind='line',
            data=[(x0, y0), (x1, y1), ...],
            opts={'animate': True},
            kwargs={'color': 'red', 'ls': '--'}
        )
        
        fig, ax = gridplot(spec)
        ```
    
    :ivar kind: the type of plot (see `SupportedPlots`)
    :ivar data: the data to plot. For animations use an iterable over data to generate frame data.
    :ivar opts: extra specialized plot options (see `PlotOpts`)
    :ivar kwargs: extra kwargs passed directly to the matplotlib plotting routine (e.g. plot, contour, etc.)
    :ivar name: optional short name for specifying local option overrides
    """
    kind: SupportedPlots
    data: Any | Iterable[Any]
    opts: PlotOpts = field(default_factory=lambda: PlotOpts())
    kwargs: dict = field(default_factory=dict)
    name: PlotName | None = None

type PlotSpecs = PlotSpec | tuple[PlotSpec, ...]  # multiple on same graph
type DataFrame = list[list[tuple[Any, ...]]]


def _fill_plot_specs_grid(plots, local_opts, global_opts, local_kwargs, global_kwargs, shape):
    """Fill a 2d grid of plot specs, merging global and local options with the default.
       The order of precedence is: local > global > default.
       opts are specialized plot options. kwargs are everything else that go into plt functions.
    """
    def fill_spec(spec: dict | PlotSpec):
        """For a single plot spec"""
        # Get default opts and kwargs
        spec = copy.copy(spec) if isinstance(spec, PlotSpec) else PlotSpec(**spec)
        opts = spec.opts
        if isinstance(opts, PlotOpts):
            opts = copy.deepcopy(asdict(opts))
        kwargs = copy.deepcopy(spec.kwargs)

        # Overwrite with global
        d = global_opts
        if isinstance(d, PlotOpts):
            d = asdict(d)
        opts.update(d)
        kwargs.update(global_kwargs)

        # Overwrite with local
        if spec.name is not None:
            d = local_opts.get(spec.name, {})
            if isinstance(d, PlotOpts):
                d = asdict(d)
            opts.update(d)
            kwargs.update(local_kwargs.get(spec.name, {}))

        spec.opts = PlotOpts(**opts)
        spec.kwargs = kwargs
        return spec

    def fill_specs(specs: dict | PlotSpec | tuple):
        """Handle multiple specs per plot"""
        if isinstance(specs, tuple):
            return tuple(fill_spec(spec) for spec in specs)
        else:
            return (fill_spec(specs),)
    
    # 1 subplot
    if isinstance(plots, dict | PlotSpec | tuple):
        return [[fill_specs(plots)]]
    
    # 2d grid
    if all(isinstance(row, list) for row in plots):
        return [[fill_specs(spec) for spec in row] for row in plots]
    
    # Convert 1d sequence to 2d
    if shape is None:
        n = len(plots)
        c = int(math.ceil(math.sqrt(n)))
        r = int(math.ceil(n / c))
        shape = (r, c)

    grid = []
    nrows, ncols = shape
    for i in range(nrows):
        row = [fill_specs(spec) for spec in plots[i*ncols:(i+1)*ncols]]
        row += [None] * (ncols - len(row))  # pad
        grid.append(row)

    return grid


def gridplot(
    plots: PlotSpecs | list[PlotSpecs] | list[list[PlotSpecs]],
    scheme: Literal['white', 'dark'] = 'white',
    subplot_size_in: tuple[float, float] = (3, 2.5),
    shape: tuple[int, int] | None = None,
    title: Iterable[str] | None = None,
    save: str | Path | None = None,
    adjust: Callable[[Figure, Axes, Iterable[Artist], list[list[Colorbar]]], None] | None = None,
    animate_opts: dict | None = None,
    legend_opts: dict | None = None,
    plot_opts: dict[PlotName, PlotOpts] | None = None,
    plot_kwargs: dict[PlotName, dict[str, Any]] | None = None,
    global_plot_opts: PlotOpts | None = None,
    global_plot_kwargs: dict[str, Any] | None = None,
    **subplot_kwargs
) -> tuple[Figure, Axes, Optional[FuncAnimation]]:
    """Generate a grid of plt subplots with easy formatting and animation.
    
    All you need to specify are plot specs (the data, colors, linestyles, etc.). The plots will be arranged nicely
    and a lot of the animation/formatting routines are automated. Also provides a nice way to apply similar formats
    across different subplots.
    
    !!! Example "A static line plot and an animated contour"
        ```python
        def generate_contour():
            for t in range(50):
                yield X, Y, np.sin(X * t) * np.cos(Y * t)

        sin_spec = dict(kind='line', data=(x, np.sin(x)))
        cos_spec = dict(kind='line', data=(x, np.cos(x)))
        contour_spec = dict(
            kind='contour', 
            data=generate_contour(), 
            opts=dict(animate=True)
        )

        fig, axs = gridplot([(sin_spec, cos_spec), contour_spec]) -> (1,2) animated subplot with sin/cos on first axis
        ```
    
    :param plots: grid of PlotSpecs with plot data and style options. If a single PlotSpec, then a (1,1) figure will
                  be generated. If a 1D list of PlotSpecs, then the grid will be shaped to the nearest square. If a 
                  2D list of PlotSpecs, then will use this grid directly. Use tuples of PlotSpecs to specify multiple
                  plots per axis. See `PlotSpec` for details on specifying plot data, styles, and supported plot types.
    :param scheme: the color scheme (dark or white)
    :param subplot_size_in: the size of each subplot in inches (W, H)
    :param shape: the shape of the subplot grid. If None, the shape will be inferred
    :param title: for animations, an iterable to update the figure title (such as showing the time step)
    :param save: name of file to save (use .gif or .mp4 for animations, use .pdf, .png, or similar for static)
    :param adjust: catch-all func for applying changes before saving/animating. Call as adjust(fig, axs, artists, cbars)
    :param animate_opts: options for animating/saving movie. Defaults to 10 fps, 200 dpi, and blit=False with ffmpeg
    :param legend_opts: extra options for legends (same used for all subplots if applicable)
    :param plot_opts: local overrides for plot options. Specify as plot.name -> { override_opts }. See `PlotOpts`.
    :param plot_kwargs: local ovverides for plot kwargs. Specify as plot.name -> { override_kwargs }. See `PlotSpec`.
    :param global_plot_opts: global overrides applied to all subplot options. See `PlotOpts`.
    :param global_plot_kwargs: global overrides applied to all subplot kwargs. See `PlotSpec`.
    :param **subplot_kwargs: all extra arguments are passed to plt.subplots
    :return: the Figure and Axes objects, optionally the FuncAnimation object if plot is animated
    """
    if plot_opts is None:
        plot_opts = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if animate_opts is None:
        animate_opts = {}
    if legend_opts is None:
        legend_opts = {}
    if global_plot_opts is None:
        global_plot_opts = {}
    if global_plot_kwargs is None:
        global_plot_kwargs = {}

    plots = _fill_plot_specs_grid(plots, plot_opts, global_plot_opts, plot_kwargs, global_plot_kwargs, shape)
    shape = (len(plots), len(plots[0]))
    a_opts = copy.deepcopy(ANIMATE_DEFAULT)
    a_opts.update(animate_opts)
    text_color, bg_color = get_scheme(scheme)
    
    fig, axs = plt.subplots(*shape, squeeze=False, layout='constrained',
                            figsize=(subplot_size_in[0]*shape[1], subplot_size_in[1]*shape[0]), **subplot_kwargs)
    fig.patch.set_facecolor(bg_color)

    def _iter_plot_specs():
        """Iterate over all axes (i,j) and plot specs (k) per axis."""
        for i in range(shape[0]):
            for j in range(shape[1]):
                if plots[i][j] is not None:
                    for k, spec in enumerate(plots[i][j]):
                        yield i, j, k, spec
    
    def _get_iterable_data():
        """For animation. Return new iterators over all data."""
        data = [[list() for _ in range(shape[1])] for _ in range(shape[0])]

        for i, j, k, spec in _iter_plot_specs():
            data[i][j].append(iter(spec.data) if spec.opts.animate else iter([spec.data]))
        
        return data
 
    def _setup_plotting_area():
        """Setup axis colors, ticks, spines, legend, etc. Return cbars, animate status, and legend(s) status"""
        animate = False
        cbars = [[None for _ in range(shape[1])] for _ in range(shape[0])]
        legends = [[False for _ in range(shape[1])] for _ in range(shape[0])]
        for i, j, k, spec in _iter_plot_specs():
            if k > 0:  # Just set up each i,j axis once
                continue

            ax = axs[i, j]

            if spec.opts.clim is not None:
                sm = ScalarMappable(norm=spec.kwargs.pop("norm", "linear"), cmap=spec.kwargs.pop("cmap", "viridis"))
                sm.set_array([])

                if spec.opts.clim != 'auto':
                    sm.norm.vmin = spec.opts.clim[0]
                    sm.norm.vmax = spec.opts.clim[1]

                cb = fig.colorbar(sm, ax=ax)
                cb.ax.set_ylabel(spec.opts.cbar_label or "", color=text_color)
                cb.ax.tick_params(labelcolor=text_color, color=text_color)
                cb.ax.tick_params(which='minor', color=(0, 0, 0, 0), width=0, size=0)
                cb.ax.minorticks_off()
                cb.outline.set_edgecolor(text_color)
                cbars[i][j] = cb

            xlabel, ylabel, xscale, yscale, xlim, ylim = None, None, None, None, None, None
            for s in plots[i][j]:
                if xlabel is None or xlabel == "":
                    xlabel = s.opts.xlabel
                if ylabel is None or ylabel == "":
                    ylabel = s.opts.ylabel
                if xscale is None or xscale == "":
                    xscale = s.opts.xscale
                if yscale is None or yscale == "":
                    yscale = s.opts.yscale
                if xlim is None:
                    xlim = s.opts.xlim
                if ylim is None:
                    ylim = s.opts.ylim
            ax_visible = any(s.opts.ax_visible for s in plots[i][j])
            animate = animate or any(s.opts.animate for s in plots[i][j])
            legends[i][j] = all(s.opts.leg_label is not None for s in plots[i][j])

            ax.tick_params(axis='both', which='both', top=False, bottom=ax_visible, 
                           left=ax_visible, right=False, direction='in', labelleft=ax_visible, 
                           labelbottom=ax_visible, color=text_color, labelcolor=text_color)
            ax.set_facecolor(bg_color)
            for spine in ['bottom', 'left', 'top', 'right']:
                ax.spines[spine].set_visible(ax_visible)
                ax.spines[spine].set_color(text_color)
            
            if ax_visible and xlabel is not None:
                ax.set_xlabel(xlabel, color=text_color)
            if ax_visible and ylabel is not None:
                ax.set_ylabel(ylabel, color=text_color)
            if xscale is not None:
                ax.set_xscale(xscale)
            if yscale is not None:
                ax.set_yscale(yscale)
            if xlim is not None:
                ax.set_xlim(xlim)
            else:
                ax.autoscale(enable=True, axis="x")
            if ylim is not None:
                ax.set_ylim(ylim)
            else:
                ax.autoscale(enable=True, axis="y")
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                if plots[i][j] is None:
                    axs[i, j].axis('off')

        return animate, cbars, legends

    animate, cbars, legends = _setup_plotting_area()

    def _draw_empty_plots():
        """Initialize plots and gather all artists."""
        artists = []

        def _empty_structured_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            x = np.array([[0.0, 1.0], [0.0, 1.0]])
            y = np.array([[0.0, 0.0], [1.0, 1.0]])
            z = np.zeros((2, 2))
            return x, y, z

        for i, j, k, spec in _iter_plot_specs():
            ax = axs[i, j]
            cbar = cbars[i][j]

            def _get_kwargs():
                kwargs = copy.deepcopy(spec.kwargs)
                if cbar is not None:
                    kwargs["norm"] = copy.deepcopy(cbar.norm)
                    kwargs["cmap"] = cbar.cmap
                return kwargs

            p = None
            match spec.kind.lower():
                case "line":
                    p, = ax.plot([], [], label=spec.opts.leg_label, **spec.kwargs)
                case "pcolor":
                    p = ax.pcolormesh(*_empty_structured_mesh(), **_get_kwargs())
                case "contour":
                    p = ax.contour(*_empty_structured_mesh(), **_get_kwargs())
                case "contourf":
                    p = ax.contourf(*_empty_structured_mesh(), **_get_kwargs())
                case "hist":
                    kwargs = copy.deepcopy(spec.kwargs)
                    if "bins" not in kwargs:
                        kwargs["bins"] = 10
                    _, bin_edges, container = ax.hist([0.0, 1.0], label=spec.opts.leg_label, **kwargs)
                    container._hist_bins = bin_edges  # monkey-patch for some reason
                    p = container
                case "hist2d":
                    kwargs = _get_kwargs()
                    if "bins" not in kwargs:
                        kwargs["bins"] = 10
                    _, xedges, yedges, mesh = ax.hist2d([0.0, 1.0], [0.0, 1.0], **kwargs)
                    mesh._hist_bins = (xedges, yedges) # monkey-patch for some reason
                    p = mesh
                case other:
                    raise ValueError(f"Plot kind '{other}' not recognized")
            
            artists.append(p)

            if legends[i][j]:
                leg = dict(facecolor=bg_color, edgecolor=text_color, labelcolor=text_color, fancybox=True)
                leg.update(legend_opts)
                ax.legend(**leg)

        return artists
    
    all_artists_og = _draw_empty_plots()
    all_artists = copy.copy(all_artists_og)
    fig.suptitle("")
    fig.canvas.draw()

    if adjust is not None:
        adjust(fig, axs, all_artists, cbars)
    
    def _update(frame_and_title: tuple[DataFrame, str | None]):
        """Update the plot with new data."""
        frame, title_str = frame_and_title
        updated_artists = []  # only the artists that get updated

        if title_str is not None:
            fig.suptitle(title_str, color=text_color)
        else:
            fig.suptitle("")

        for flat_idx, ((i, j, k, spec), artist) in enumerate(zip(_iter_plot_specs(), list(all_artists))):
            ax = axs[i, j]
            cbar = cbars[i][j]
            data = frame[i][j][k]

            if data is None:
                continue  # No data left
            
            def _update_clim():
                z = data[-1] if isinstance(data, tuple) else data
                if cbar is not None and spec.opts.clim == 'auto':
                    cbar.norm.vmin = np.nanmin(z)
                    cbar.norm.vmax = np.nanmax(z)
            
            def _get_kwargs():
                kwargs = copy.deepcopy(spec.kwargs)
                if cbar is not None:
                    kwargs["norm"] = copy.deepcopy(cbar.norm)
                    kwargs["cmap"] = cbar.cmap
                return kwargs

            match spec.kind.lower():
                case "line":
                    if isinstance(data, tuple):  # ([X], [Y])
                        if data[0] is not None:
                            artist.set_xdata(data[0])
                        if data[1] is not None:
                            artist.set_ydata(data[1])
                    else:
                        artist.set_ydata(data)   # [Y]
                    updated_artists.append(artist)

                case "pcolor":
                    _update_clim()
                    if isinstance(data, tuple):  # (X, Y, Z)
                        artist.remove()
                        new_artist = ax.pcolormesh(*data, **_get_kwargs())
                        all_artists[flat_idx] = new_artist
                        updated_artists.append(new_artist)
                        if data[0] is not None and data[1] is not None:
                            ax.set_xlim(np.nanmin(data[0]), np.nanmax(data[0]))
                            ax.set_ylim(np.nanmin(data[1]), np.nanmax(data[1]))
                    else:
                        artist.set_array(data)    # [Z]
                        updated_artists.append(artist)

                case "contour":
                    _update_clim()
                    artist.remove()
                    new_artist = ax.contour(*data, **_get_kwargs())
                    all_artists[flat_idx] = new_artist
                    updated_artists.append(new_artist)

                case "contourf":
                    _update_clim()
                    artist.remove()
                    new_artist = ax.contourf(*data, **_get_kwargs())
                    all_artists[flat_idx] = new_artist
                    updated_artists.append(new_artist)
                
                case "hist":
                    kwargs = _get_kwargs()
                    orientation = kwargs.get("orientation", "vertical")
                    stacked = kwargs.get("stacked", False)
                    if spec.opts.leg_label is not None and "label" not in kwargs:
                        kwargs["label"] = spec.opts.leg_label

                    if isinstance(data, tuple):
                        x = data[0]
                        bins_override = data[1] if len(data) > 1 else None
                        weights_override = data[2] if len(data) > 2 else None
                    else:
                        x = data
                        bins_override = None
                        weights_override = None

                    is_multiset = np.ndim(x) > 1
                    if isinstance(x, list) and len(x) > 0 and not np.isscalar(x[0]):
                        is_multiset = True

                    if stacked or is_multiset:
                        # Fallback to re-plot for stacked/multi-dataset histograms.
                        if bins_override is not None:
                            kwargs["bins"] = bins_override
                        if weights_override is not None:
                            kwargs["weights"] = weights_override
                        for patch in artist.patches:
                            patch.remove()
                        n, bin_edges, new_container = ax.hist(x, **kwargs)
                        new_container._hist_bins = bin_edges
                        all_artists[flat_idx] = new_container
                        updated_artists.extend(new_container.patches)
                    else:
                        bins = bins_override if bins_override is not None else kwargs.get("bins")
                        if bins is None:
                            bins = getattr(artist, "_hist_bins", 10)
                        weights = weights_override if weights_override is not None else kwargs.get("weights")
                        range_ = kwargs.get("range")
                        density = kwargs.get("density", False)

                        hist, bin_edges = np.histogram(x, bins=bins, range=range_, density=density, weights=weights)

                        if len(artist.patches) != len(hist):
                            if bins_override is not None:
                                kwargs["bins"] = bins_override
                            if weights_override is not None:
                                kwargs["weights"] = weights_override
                            for patch in artist.patches:
                                patch.remove()
                            n, bin_edges, new_container = ax.hist(x, **kwargs)
                            new_container._hist_bins = bin_edges
                            all_artists[flat_idx] = new_container
                            updated_artists.extend(new_container.patches)
                        else:
                            artist._hist_bins = bin_edges
                            bottom = kwargs.get("bottom", 0.0)
                            if np.ndim(bottom) == 0:
                                bottom_vals = np.full_like(hist, float(bottom), dtype=float)
                            else:
                                bottom_vals = np.asarray(bottom, dtype=float)
                            for patch, count, left, right, base in zip(
                                artist.patches, hist, bin_edges[:-1], bin_edges[1:], bottom_vals
                            ):
                                width = right - left
                                if orientation == "horizontal":
                                    patch.set_x(base)
                                    patch.set_width(count)
                                    patch.set_y(left)
                                    patch.set_height(width)
                                else:
                                    patch.set_x(left)
                                    patch.set_width(width)
                                    patch.set_y(base)
                                    patch.set_height(count)
                            updated_artists.extend(artist.patches)
                
                case "hist2d":
                    kwargs = _get_kwargs()

                    if isinstance(data, tuple):
                        if len(data) < 2:
                            raise ValueError("hist2d expects (x, y) data.")
                        x, y = data[0], data[1]
                        weights_override = data[2] if len(data) > 2 else None
                    else:
                        raise ValueError("hist2d expects (x, y) data.")

                    bins = kwargs.get("bins")
                    if bins is None:
                        bins = getattr(artist, "_hist_bins", 10)
                    range_ = kwargs.get("range")
                    density = kwargs.get("density")
                    weights = weights_override if weights_override is not None else kwargs.get("weights")

                    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range_, density=density, weights=weights)
                    if cbar is not None and spec.opts.clim == 'auto':
                        cbar.norm.vmin = np.nanmin(hist)
                        cbar.norm.vmax = np.nanmax(hist)
                        kwargs["norm"] = copy.deepcopy(cbar.norm)
                        kwargs["cmap"] = cbar.cmap

                    artist.remove()
                    for key in ["bins", "range", "density", "weights", "cmin", "cmax"]:
                        kwargs.pop(key, None)
                    mesh = ax.pcolormesh(xedges, yedges, hist.T, **kwargs)
                    mesh._hist_bins = (xedges, yedges)
                    all_artists[flat_idx] = mesh
                    updated_artists.append(mesh)

                case other:
                    raise ValueError(f"Plot kind '{other}' not recognized")
            
            if ax.get_autoscalex_on() or ax.get_autoscaley_on():
                ax.relim()
                ax.autoscale_view()
                # fig.canvas.draw_idle()
                # fig.canvas.flush_events()

        return updated_artists

    def _frames() -> Generator[tuple[DataFrame, str | None], None, None]:
        """Return the next data from all plot specs (if available), and a title str."""
        iterable_data = _get_iterable_data()
        iterable_title = iter(title) if title is not None else None
        
        while True:
            frame = [[list() for _ in range(shape[1])] for _ in range(shape[0])]
            keep_going = False
            for i, j, k, _ in _iter_plot_specs():
                try:
                    next_data = next(iterable_data[i][j][k])
                    keep_going = True
                except StopIteration:
                    next_data = None
                frame[i][j].append(next_data)
            
            if not keep_going:
                break
            
            title_str = None
            if title is not None:
                try:
                    title_str = next(iterable_title)
                except StopIteration:
                    title_str = None
   
            yield frame, title_str
    
    if animate:
        for ax in axs.flatten():
            ax.set_position(ax.get_position().frozen())
            ax.set_in_layout(False)

        ani = FuncAnimation(fig, _update, frames=_frames, init_func=lambda: all_artists_og, repeat=False, 
                            cache_frame_data=False, blit=a_opts['blit'], interval=int(1000/a_opts['fps']))

        if save is not None:
            print(f"Saving animation to '{save}'")
            del a_opts['blit']
            ani.save(Path(save), **a_opts)

        return fig, axs, ani
    
    # Static figure
    else:
        _update(next(_frames()))
        if save is not None:
            fig.savefig(Path(save), bbox_inches='tight')
    
        return fig, axs


### DEPRECATED ###
def _old_gridplot(data: list[Frame],
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

    text_color, bg_color = get_scheme(scheme)
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
