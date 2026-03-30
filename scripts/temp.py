import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

import copy

def animate_line() -> FuncAnimation:
    fig, ax = plt.subplots(figsize=(6, 5), layout="tight")

    t = np.linspace(0.0, 20.0, 400)
    y = np.sin(t)

    line, = ax.plot([], [], "-b", lw=2)
    # ax.set_xlim(t.min(), t.max())
    # ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
    ax.set_xlabel("t")
    ax.set_ylabel("sin(t)")
    ax.grid(True, alpha=0.3)
    ax.autoscale(enable=True, axis='x')
    ax.set_ylim((-1.5, 1.5))
    ax.get_autoscalex_on

    def frames():
        """Yield successive (t, y) data slices for animation."""
        for i in range(1, t.size + 1):
            yield t[:i], y[:i]

    def init():
        """Initialize artists for blitting."""
        line.set_data([], [])
        return (line,)

    def update(frame):
        """Update artists with the latest frame data."""
        t_slice, y_slice = frame
        line.set_data(t_slice, y_slice)
        ax.relim()
        ax.autoscale_view()
        # fig.canvas.draw_idle()
        # fig.canvas.flush_events()
        return (line,)

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=False,
        interval=50,
        repeat=False,
        cache_frame_data=10
    )
    plt.show()


def animate_contourf() -> FuncAnimation:
    fig, ax = plt.subplots(figsize=(6, 5), layout="tight")

    t = np.linspace(0.0, 2.0 * np.pi, 120)

    def grid(phase: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(0.0, 1.0 + 0.3 * np.sin(phase), 80)
        y = np.linspace(0.0, 1.0 + 0.3 * np.cos(phase), 80)
        xg, yg = np.meshgrid(x, y)
        return x, y, xg, yg

    def field(xg: np.ndarray, yg: np.ndarray, phase: float) -> np.ndarray:
        amp = 1.0 + 0.4 * np.sin(2.0 * phase)
        return amp * np.sin(np.pi * xg + phase) * np.sin(np.pi * yg - phase)

    x0, y0, xg0, yg0 = grid(0.0)
    z0 = field(xg0, yg0, 0.0)
    quad = ax.contourf(xg0, yg0, z0, levels=21, cmap="viridis")
    quad.set_clim
    cbar = fig.colorbar(quad, ax=ax, label="amplitude")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    def frames():
        """Yield successive scalar phase values."""
        yield from t

    def init():
        """Initialize artists for blitting."""
        return [quad]

    def update(phase):
        """Update contourf, axes, and colorbar for the new frame."""
        nonlocal quad
        nonlocal cbar
        x, y, xg, yg = grid(phase)
        z = field(xg, yg, phase)

        quad.remove()
        quad = ax.contourf(xg, yg, z, levels=21, cmap="viridis")

        ax.relim()
        ax.autoscale_view()

        # ax.set_xlim(x.min(), x.max())
        # ax.set_ylim(y.min(), y.max())
        # cbar.update_normal(quad)
        return [quad]

    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=False,
        interval=50,
        repeat=False,
        save_count=10,
    )
    plt.show()


def animate_grid() -> FuncAnimation:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="tight")
    ax_line = axs[0]
    ax_contour=  axs[1]

    t = np.linspace(0.0, 20.0, 400)

    line, = ax_line.plot([], [], "-b", lw=2)
    ax_line.set_xlabel("t")
    ax_line.set_ylabel("sin(t)")
    ax_line.grid(True, alpha=0.3)

    phase = np.linspace(0.0, 2.0 * np.pi, 120)

    def grid(phase_value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(0.0, 1.0 + 0.3 * np.sin(phase_value), 80)
        y = np.linspace(0.0, 1.0 + 0.3 * np.cos(phase_value), 80)
        xg, yg = np.meshgrid(x, y)
        return x, y, xg, yg

    def field(xg: np.ndarray, yg: np.ndarray, phase_value: float) -> np.ndarray:
        amp = 1.0 + 0.4 * np.sin(2.0 * phase_value)
        return amp * np.sin(np.pi * xg + phase_value) * np.sin(np.pi * yg - phase_value)
    
    vmin, vmax = np.inf, -np.inf
    for p in phase:
        g = grid(p)
        z = field(g[2], g[3], p)
        vmin, vmax = float(np.nanmin([vmin, np.nanmin(z)])), float(np.nanmax([vmax, np.nanmax(z)]))
    print(f"vmin: {vmin} vmax: {vmax}")

    # cbar_norm = Normalize(-5, 5)
    cont_norm = Normalize(vmin, vmax)

    x0, y0, xg0, yg0 = grid(phase[0])
    z0 = field(xg0, yg0, phase[0])
    sm = plt.cm.ScalarMappable(cmap='jet', norm='linear')
    sm.set_array([])  # required dummy

    cbar = fig.colorbar(sm, ax=ax_contour)
    cbar_norm = cbar.norm
    cbar_norm.vmin = -10
    cbar_norm.vmax = 10

    quad = ax_contour.contourf(xg0, yg0, z0, levels=10, cmap="viridis")
    # cbar = fig.colorbar(quad, ax=ax_contour, label="amplitude")
    ax_contour.set_xlabel("x")
    ax_contour.set_ylabel("y")

    def frames():
        """Yield successive data slices for the line and contour plots."""
        for i, phase_value in enumerate(phase, start=1):
            t_slice = t[:i]
            y_slice = np.sin(t_slice)
            yield t_slice, y_slice, phase_value

    def init():
        """Initialize artists for animation."""
        line.set_data([], [])
        return [line, quad]

    def update(frame):
        """Update artists with the latest frame data."""
        nonlocal quad
        t_slice, y_slice, phase_value = frame

        line.set_data(t_slice, y_slice)
        # ax_line.relim()
        # ax_line.autoscale_view()

        x, y, xg, yg = grid(phase_value)
        z = field(xg, yg, phase_value)
        quad.remove()
        quad = ax_contour.contourf(xg, yg, z, levels=10, cmap="viridis")
        ax_contour.set_xlim(x.min(), x.max())
        ax_contour.set_ylim(y.min(), y.max())
        # cbar.update_normal(quad)

        return [line, quad]

    for ax in axs.flatten():
        ax.set_position(ax.get_position().frozen())
        ax.set_in_layout(False)

    print(cbar_norm is cont_norm)
    print(f"cbar: {cbar_norm.vmin}, {cbar_norm.vmax}")
    print(f"cont: {cont_norm.vmin} {cont_norm.vmax}")
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=False,
        interval=50,
        repeat=False,
        cache_frame_data=False
    )

    plt.show()


def debug_colorbar():
    nx, ny = 100, 100
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y)
    t = np.arange(50) * 0.2

    def generate_field(time_series):
        for t in time_series:
            yield X, Y, np.sin(X**2 + Y**2 + t) * np.exp(-0.1 * (X**2 + Y**2))

    vmin, vmax = np.inf, -np.inf
    for _, _, z in generate_field(t):
        vmin, vmax = np.min([vmin, np.nanmin(z)]), np.max([vmax, np.nanmax(z)])
    
    cont_norm = Normalize(vmin, vmax)

    sm = plt.cm.ScalarMappable(cmap='jet', norm=Normalize(0, 1))
    sm.set_array([])  # required dummy

    fig, ax = plt.subplots(figsize=(6, 5), layout='tight')

    cbar = fig.colorbar(sm, ax=ax)
    cbar_norm = cbar.norm
    cbar_norm.vmin = -10
    cbar_norm.vmax = 10

    z0 = next(generate_field(t))[-1]
    quad = ax.pcolormesh(X, Y, z0, cmap=cbar.cmap, norm=copy.deepcopy(cbar_norm))

    def init():
        """Initialize artists for animation."""
        return [quad]

    def frames():
        return generate_field(t)
    
    def update(frame):
        """Update artists with the latest frame data."""
        x, y, z = frame
        nonlocal quad
        quad.remove()
        quad = ax.pcolormesh(x, y, z, cmap=cbar.cmap, norm=copy.deepcopy(cbar_norm))

        return [quad]
    
    print(cbar_norm is cont_norm)
    print(f"cbar: {cbar_norm.vmin}, {cbar_norm.vmax}")
    print(f"cont: {cont_norm.vmin} {cont_norm.vmax}")
    ani = FuncAnimation(
        fig,
        update,
        frames=frames,
        init_func=init,
        blit=False,
        interval=50,
        repeat=True,
        cache_frame_data=False
    )

    plt.show()

if __name__ == "__main__":
    # animate_line()
    # animate_contourf()
    # animate_grid()
    debug_colorbar()