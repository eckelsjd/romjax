"""Plotting utilities.

Update global configuration for gridplot with `romjax.plotting.set_global(**opts)`.

Includes:
  - gridplot - Plot simulation data (1d or 2d) in a grid (with animation utilities)
  - set_global - default global options for gridplot
  - PlotSpec - spec for a single subplot (see gridplot)
  - AxisOptions - extra axis options
  - AnimateOptions - extra animate options
  - GridplotConfig - config object for gridplot
  - get_scheme - Get a plotting color scheme
"""
import copy
import math
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterable, Literal, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar
from loguru import logger
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from pydantic import Field, SkipValidation

from romjax.typing import DictModel

__all__ = ['gridplot', 'set_global', 'PlotSpec', 'AxisOptions', 'AnimateOptions', 'GridplotConfig', 'get_scheme']


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


type PlotName = str  # just a short, memorable key name


class AnimateOptions(DictModel):
    """A few particular animate options. All extra options will be passed to FuncAnimation.save.
    
    :ivar blit: blitting for FuncAnimation
    :ivar num_frames: estimate of total number of frames for the progress bar
    :ivar log_interval: how often to log while saving animation (by default will not log)
    :ivar fps: frames per second
    :ivar progress_callback: callable override for custom progress while saving animation
    """
    blit: bool | None = None
    num_frames: int | None = None
    log_interval: int | None = None
    fps: int | None = None
    progress_callback: Callable[[int ,int], None] | Literal["log", "bar"] | None = None

    def get_save_kwargs(self):
        """Return kwargs that can be passed directly to FuncAnimation.save."""
        d = copy.deepcopy(self.model_extra)
        d["fps"] = self.fps

        if self.progress_callback == "log":
            interval = self.log_interval or 10
            num_frames = self.num_frames
            def save_progress(i, n):
                n = n or num_frames
                if i % interval == 0:
                    logger.debug(f"Frame {i}/{n-1}" if n is not None else f"Frame {i}")
            d["progress_callback"] = save_progress
        elif self.progress_callback == "bar":
            d["progress_callback"] = lambda bar, i, n: bar()
        else:
            d["progress_callback"] = self.progress_callback
        
        return d


class AxisOptions(DictModel):
    """A few particular axis options common to all types of plots.
    
    :ivar xlabel: the x-axis label
    :ivar ylabel: the y-axis label
    :ivar title: the axis title
    :ivar xscale: scale for x-axis
    :ivar yscale: scale for y-axis
    :ivar cscale: scale for colorbar axis normalization
    :ivar xlim: limits for x-axis (defaults to autoscale if None)
    :ivar ylim: limits for y-axis (defaults to autoscale if None)
    :ivar clim: limits for colorbar (if None, will not show a colorbar), can also set to 'auto'
    :ivar cbar_label: label for the colorbar (must set clim to show colorbar)
    :ivar leg_label: label for legend (legend only shown if all artists on an axis have a label)
    :ivar ax_visible: whether to show axes, ticks, and spines (default True)
    :ivar animate: whether to animate data for this plot (default False)
    :ivar grid: options for showing axis grid
    """
    xlabel: str | None = None
    ylabel: str | None = None
    title: str | None = None
    xscale: str | None = None
    yscale: str | None = None
    cscale: str | None = None
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    clim: tuple[float, float] | Literal['auto'] | None = None
    cbar_label: str | None = None
    leg_label: str | None = None
    ax_visible: bool | None = None
    animate: bool | None = None
    grid: dict | bool | None = None


class PlotSpec(DictModel):
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
    kind: Literal["line", "pcolor", "contour", "contourf", "hist", "hist2d"] # TODO: tri, quad
    data: Annotated[Any | Iterable[Any], SkipValidation] = Field(exclude=True)
    opts: AxisOptions = Field(default_factory=AxisOptions)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    name: PlotName | None = None


type PlotSpecs = PlotSpec | tuple[PlotSpec, ...]  # multiple on same graph
type DataFrame = list[list[tuple[Any, ...]]]


class GridplotConfig(DictModel):
    """Configuration options for `gridplot`.
    
    :ivar scheme: the color scheme (dark or white)
    :ivar subplot_size_in: the size of each subplot in inches (W, H)
    :ivar shape: the shape of the subplot grid. If None, the shape will be inferred
    :ivar title: for animations, an iterable to update the figure title (such as showing the time step)
    :ivar save: name of file to save (use .gif or .mp4 for animations, use .pdf, .png, or similar for static)
    :ivar adjust: catch-all func for applying changes before saving/animating. Call as adjust(fig, axs, artists, cbars)
    :ivar animate_opts: options for animating/saving movie. Defaults to 10 fps, 200 dpi, and blit=False with ffmpeg
    :ivar legend_kwargs: extra options for legends (same used for all subplots if applicable)
    :ivar local_axis_opts: local overrides for plot options. Specify as plot.name->{ override_opts }. See `AxisOptions`.
    :ivar local_plot_kwargs: local ovverides for plot kwargs. Specify as plot.name->{ override_kwargs }. See `PlotSpec`.
    :ivar global_axis_opts: global overrides applied to all subplot options. See `AxisOptions`.
    :ivar global_plot_kwargs: global overrides applied to all subplot kwargs. See `PlotSpec`.
    :ivar subplots_kwargs: all extra arguments are passed to plt.subplots
    """
    scheme: Literal['white', 'dark'] | None = None
    subplot_size_in: tuple[float, float] | None = None
    shape: tuple[int, int] | None = None
    title: Iterable[str] | None = None
    save: str | Path | None = None
    adjust: Callable[[Figure, Axes, Iterable[Artist], list[list[Colorbar]]], None] | None = None
    animate_opts: AnimateOptions = Field(default_factory=AnimateOptions)
    legend_kwargs: dict = Field(default_factory=dict)
    local_axis_opts: dict[PlotName, AxisOptions] = Field(default_factory=dict)
    local_plot_kwargs: dict[PlotName, dict[str, Any]] = Field(default_factory=dict)
    global_axis_opts: AxisOptions = Field(default_factory=AxisOptions)
    global_plot_kwargs: dict = Field(default_factory=dict)
    subplots_kwargs: dict = Field(default_factory=dict)

    def fill_plot_grid(self, plots: PlotSpecs | list[PlotSpecs] | list[list[PlotSpecs]]) -> list[list[PlotSpecs]]:
        """
        Fill a 2d grid of plot specs, merging global and local options with the default.
        The order of precedence is: local > global > default.
        """
        def fill_spec(spec: PlotSpec):
            """Validate a single plot spec."""
            # Default
            spec = PlotSpec.model_validate(spec)

            # Global
            opts = self.merge(spec.opts, self.global_axis_opts)
            kwargs = self.merge(spec.kwargs, self.global_plot_kwargs)

            # Local
            if spec.name is not None:
                if spec.name in self.local_axis_opts:
                    opts = self.merge(opts, self.local_axis_opts[spec.name])
                if spec.name in self.local_plot_kwargs:
                    kwargs = self.merge(kwargs, self.local_plot_kwargs[spec.name])

            spec.opts = opts
            spec.kwargs = kwargs
            return spec

        def fill_specs(specs: PlotSpec | tuple):
            """Handle multiple specs per plot."""
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
        if self.shape is None:
            n = len(plots)
            c = int(math.ceil(math.sqrt(n)))
            r = int(math.ceil(n / c))
            self.shape = (r, c)

        grid = []
        nrows, ncols = self.shape
        for i in range(nrows):
            row = [fill_specs(spec) for spec in plots[i*ncols:(i+1)*ncols]]
            row += [None for _ in  range(ncols - len(row))]  # pad
            grid.append(row)

        return grid

    @staticmethod
    def merge(base: Mapping, override: Mapping, in_place: bool = False) -> Mapping:
        """Recursively merge two mappings into a new one.

        Values in ``override`` only replace corresponding values in ``base``
        when the override value is not ``None``. Nested mappings are merged
        recursively with the same rule.
        """
        merged = base if in_place else copy.deepcopy(base)

        for key, value in override.items():
            if value is None:
                continue

            if isinstance(value, Mapping):
                base_value = merged.get(key)
                if isinstance(base_value, Mapping):
                    merged[key] = GridplotConfig.merge(base_value, value)
                else:
                    merged[key] = copy.deepcopy(value)
            elif key == "title":
                merged[key] = value  # skip copying title which may be an iterable
            else:
                merged[key] = copy.deepcopy(value)

        return merged


global_config = GridplotConfig(
    scheme="white", 
    subplot_size_in=(3, 2.5), 
    animate_opts=dict(blit=False, progress_callback="bar", fps=10, dpi=200, writer="ffmpeg"),
    subplots_kwargs=dict(squeeze=False, layout="constrained")
)


def set_global(**settings):
    """Update global default gridplot settings."""
    GridplotConfig.merge(global_config, settings, in_place=True)


def gridplot(
    plots: PlotSpecs | list[PlotSpecs] | list[list[PlotSpecs]],
    **cfg: GridplotConfig
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
    
    See `GridplotConfig` for additional keyword options.
    
    :param plots: grid of PlotSpecs with plot data and style options. If a single PlotSpec, then a (1,1) figure will
                  be generated. If a 1D list of PlotSpecs, then the grid will be shaped to the nearest square. If a 
                  2D list of PlotSpecs, then will use this grid directly. Use tuples of PlotSpecs to specify multiple
                  plots per axis. See `PlotSpec` for details on specifying plot data, styles, and supported plot types.
    :return: the Figure and Axes objects, optionally the FuncAnimation object if plot is animated
    """
    cfg = GridplotConfig.merge(global_config, GridplotConfig(**cfg))  # allows maintaining a top-level global config
    plots = cfg.fill_plot_grid(plots)
    shape = (len(plots), len(plots[0]))
    text_color, bg_color = get_scheme(cfg.scheme)
    if "figsize" not in cfg.subplots_kwargs:
        cfg.subplots_kwargs["figsize"] = (cfg.subplot_size_in[0]*shape[1], cfg.subplot_size_in[1]*shape[0])
    
    fig, axs = plt.subplots(*shape, **cfg.subplots_kwargs)
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
                norm = spec.kwargs.pop("norm", spec.opts.cscale or "linear")
                sm = ScalarMappable(norm=norm, cmap=spec.kwargs.pop("cmap", "viridis"))
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

            xlabel, ylabel, xscale, yscale, xlim, ylim, title, grid = None, None, None, None, None, None, None, None
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
                if title is None or title == "":
                    title = s.opts.title
                if grid is None or not grid:
                    grid = s.opts.grid
            ax_visible = any(s.opts.ax_visible or s.opts.ax_visible is None for s in plots[i][j])
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
            if ax_visible and title is not None:
                ax.set_title(title, color=text_color)
            if ax_visible and grid:
                ax.grid(**grid) if isinstance(grid, Mapping) else ax.grid()
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

    def _set_hist_bins(artist, bins) -> None:
        """Store histogram bins when the matplotlib artist type supports attributes."""
        try:
            artist._hist_bins = bins
        except AttributeError:
            if isinstance(artist, list | tuple):
                for item in artist:
                    _set_hist_bins(item, bins)

    def _get_hist_bins(artist, default):
        """Return cached histogram bins, handling list-valued artists from step histograms."""
        if hasattr(artist, "_hist_bins"):
            return artist._hist_bins
        if isinstance(artist, list | tuple):
            for item in artist:
                bins = _get_hist_bins(item, None)
                if bins is not None:
                    return bins
        return default

    def _hist_artists(artist) -> list[Artist]:
        """Flatten matplotlib histogram return values into removable/updateable artists."""
        if hasattr(artist, "patches"):
            return list(artist.patches)
        if isinstance(artist, list | tuple):
            artists = []
            for item in artist:
                artists.extend(_hist_artists(item))
            return artists
        return [artist]

    def _remove_hist_artists(artist) -> None:
        """Remove all artists produced by a histogram call."""
        for hist_artist in _hist_artists(artist):
            hist_artist.remove()

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
                    _set_hist_bins(container, bin_edges)
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
                leg.update(cfg.legend_kwargs)
                ax.legend(**leg)

        return artists
    
    all_artists_og = _draw_empty_plots()
    all_artists = copy.copy(all_artists_og)
    fig.suptitle("")
    fig.canvas.draw()

    if cfg.adjust is not None:
        cfg.adjust(fig, axs, all_artists, cbars)
    
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

            def _hist_bins(x, bins, range_):
                """Return log-spaced histogram bins for log-x histograms when bins are inferred."""
                if spec.opts.xscale != "log" or not isinstance(bins, int | str):
                    return bins

                if range_ is None:
                    if isinstance(x, list | tuple) and len(x) > 0 and not np.isscalar(x[0]):
                        x_arr = np.concatenate([np.ravel(np.asarray(item, dtype=float)) for item in x])
                    else:
                        x_arr = np.ravel(np.asarray(x, dtype=float))
                    positive = x_arr[np.isfinite(x_arr) & (x_arr > 0.0)]
                    if positive.size == 0:
                        raise ValueError("Log-scaled histograms require positive finite data.")
                    log_values = np.log10(positive)
                    log_range = None
                else:
                    vmin, vmax = range_
                    if vmin <= 0.0 or vmax <= 0.0:
                        raise ValueError("Log-scaled histogram ranges must be positive.")
                    log_values = None
                    log_range = (np.log10(vmin), np.log10(vmax))

                if isinstance(bins, str):
                    sample = log_values if log_values is not None else np.asarray(log_range)
                    log_edges = np.histogram_bin_edges(sample, bins=bins, range=log_range)
                    return np.power(10.0, log_edges)

                if range_ is None:
                    vmin = np.nanmin(positive)
                    vmax = np.nanmax(positive)
                if vmin == vmax:
                    factor = np.sqrt(10.0)
                    vmin /= factor
                    vmax *= factor

                return np.geomspace(vmin, vmax, bins + 1)

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
                        kwargs["bins"] = _hist_bins(x, kwargs.get("bins", _get_hist_bins(artist, 10)),
                                                    kwargs.get("range"))
                        if weights_override is not None:
                            kwargs["weights"] = weights_override
                        _remove_hist_artists(artist)
                        n, bin_edges, new_container = ax.hist(x, **kwargs)
                        _set_hist_bins(new_container, bin_edges)
                        all_artists[flat_idx] = new_container
                        updated_artists.extend(_hist_artists(new_container))
                    else:
                        bins = bins_override if bins_override is not None else kwargs.get("bins")
                        if bins is None:
                            bins = _get_hist_bins(artist, 10)
                        weights = weights_override if weights_override is not None else kwargs.get("weights")
                        range_ = kwargs.get("range")
                        density = kwargs.get("density", False)
                        bins = _hist_bins(x, bins, range_)

                        hist, bin_edges = np.histogram(x, bins=bins, range=range_, density=density, weights=weights)

                        if not hasattr(artist, "patches") or len(artist.patches) != len(hist):
                            if bins_override is not None:
                                kwargs["bins"] = bins_override
                            kwargs["bins"] = bin_edges
                            if weights_override is not None:
                                kwargs["weights"] = weights_override
                            _remove_hist_artists(artist)
                            n, bin_edges, new_container = ax.hist(x, **kwargs)
                            _set_hist_bins(new_container, bin_edges)
                            all_artists[flat_idx] = new_container
                            updated_artists.extend(_hist_artists(new_container))
                        else:
                            _set_hist_bins(artist, bin_edges)
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

                    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range_, 
                                                          density=density, weights=weights)
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
        iterable_title = iter(cfg.title) if cfg.title is not None else None
        
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
            if cfg.title is not None:
                try:
                    title_str = next(iterable_title)
                except StopIteration:
                    title_str = None
   
            yield frame, title_str
    
    if animate:
        for ax in axs.flatten():
            ax.set_position(ax.get_position().frozen())
            ax.set_in_layout(False)

        a_opts = cfg.animate_opts
        blit = a_opts['blit'] or False
        interval = int(1000/fps) if (fps := a_opts['fps']) is not None else 200

        ani = FuncAnimation(fig, _update, frames=_frames, init_func=lambda: all_artists_og, repeat=False, 
                            cache_frame_data=False, blit=blit, interval=interval)

        if cfg.save is not None:
            logger.debug(f"Saving animation to '{cfg.save}'")
            save_kwargs = a_opts.get_save_kwargs()

            if a_opts.progress_callback == "bar":
                with alive_bar(a_opts["num_frames"]) as bar:
                    save_kwargs["progress_callback"] = partial(save_kwargs["progress_callback"], bar)
                    ani.save(Path(cfg.save), **save_kwargs)
            else:
                ani.save(Path(cfg.save), **save_kwargs)

        return fig, axs, ani
    
    # Static figure
    else:
        _update(next(_frames()))
        if cfg.save is not None:
            fig.savefig(Path(cfg.save), bbox_inches='tight')
    
        return fig, axs 
