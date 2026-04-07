"""Plotting utilities.

Includes:
  - gridplot - Plot simulation data (1d or 2d) in a grid (with animation utilities)
  - PlotOpts - Extra options for plotting (see gridplot)
  - PlotSpec - Spec for a single subplot (see gridplot)
  - SupportedPlots - plt plots supported by gridplot
  - get_scheme - Get a plotting color scheme
"""
import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Any, Generator, Iterable
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.artist import Artist
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar


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
                        if len(data) == 3 and data[0] is not None and data[1] is not None:
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
    