import numpy as np
import matplotlib.pyplot as plt

from romjax.plotting import gridplot
from romjax.utils import format_time_engineering

# -----------------------
# Create synthetic data
# -----------------------
nx, ny = 100, 100
x = np.linspace(-3, 3, nx)
y = np.linspace(-3, 3, ny)
X, Y = np.meshgrid(x, y)
t = np.arange(50) * 0.2

def generate_field(time_series):
    for t in time_series:
        yield X, Y, np.sin(X**2 + Y**2 + t) * np.exp(-0.1 * (X**2 + Y**2))

def generate_line(time_series, kind='sin'):
    for i in range(len(time_series)):
        yield t[:i+1], np.sin(t[:i+1]) if kind=='sin' else np.cos(t[:i+1])

def generate_title(time_series):
    for t in time_series:
        yield f"t={format_time_engineering(t)}"

vmin, vmax = np.inf, -np.inf
for _, _, z in generate_field(t):
    vmin, vmax = np.min([vmin, np.nanmin(z)]), np.max([vmax, np.nanmax(z)])

print(f"expected clim: {vmin}, {vmax}")

field_spec = {
    'kind': 'pcolor',
    'data': generate_field(t),
    'opts': {'clim': (vmin, vmax), 'xlabel': 'x', 'ylabel': 'y', 'cbar_label': 'Field f(x,y)', 'animate': True},
    'kwargs': {'cmap': 'jet'}
}

sin_spec = {
    'kind': 'line',
    'data': generate_line(t),
    'opts': {'animate': True, 'xlabel': 'x', 'ylabel': 'y', 'leg_label': 'Sine'},
    'kwargs': {'color': 'red'},
}

cos_spec = {
    'kind': 'line',
    'data': generate_line(t, 'cos'),
    'opts': {'animate': True, 'ylim': (-1, 1), 'xlim': (0, 10)},
    'kwargs': {'color': 'blue'},
}

const_spec = {
    'kind': 'line',
    'data': (t, np.sin(0.5*t)),
    'opts': {'leg_label': 'const'},
    'kwargs': {'lw':5, 'ls': '--', 'c': 'k'}
}

fig, axs = gridplot(
    [(cos_spec, sin_spec), const_spec, field_spec],
    scheme='dark',
    shape=(1, 3),
    subplot_size_in=(5, 4),
    animate_opts=dict(fps=15, blit=True, writer='ffmpeg', dpi=150),
    save="movie.mp4"
)

# -----------------------
# Custom colormap + norm
# -----------------------
# cmap = plt.get_cmap("plasma")  # custom cmap

# # Initialize with dummy range (will update dynamically)
# norm = Normalize(vmin=-1, vmax=1)

# # Persistent ScalarMappable (for colorbar ONLY)
# sm = ScalarMappable(norm='linear', cmap=cmap)
# norm = sm.norm
# sm.set_array([])  # required for colorbar

# # -----------------------
# # Set up figure
# # -----------------------
# fig, ax = plt.subplots()

# # Initial field
# Z0 = generate_field(0)

# # Initial contour
# # cont = ax.contourf(X, Y, Z0, cmap=cmap, norm=norm)
# cont = ax.pcolormesh(X, Y, Z0, cmap=cmap, norm=norm)

# # Colorbar tied to persistent ScalarMappable
# cbar = plt.colorbar(sm, ax=ax)
# norm = cbar.norm

# # -----------------------
# # Animation update
# # -----------------------
# def update(frame):
#     global cont

#     # Remove old contour artists
#     cont.remove()

#     # Generate new data
#     Z = generate_field(frame * 0.2)

#     # Compute new dynamic limits
#     vmin, vmax = Z.min(), Z.max()

#     # Update normalization (this is the key step)
#     norm.vmin = vmin
#     norm.vmax = vmax

#     # Recreate contour with SAME cmap + norm object
#     # cont = ax.contourf(X, Y, Z, cmap=cmap, norm=norm)
#     cont = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm)

#     # Update colorbar to reflect new norm
#     # cbar.update_normal(sm)

#     return [cont]

# # -----------------------
# # Run animation
# # -----------------------
# ani = FuncAnimation(
#     fig,
#     update,
#     frames=50,
#     interval=100,
#     blit=False
# )

# plt.show()