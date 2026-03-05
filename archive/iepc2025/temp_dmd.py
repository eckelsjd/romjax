"""Just playing around with different DMDs."""
import numpy as np
from pydmd import DMD, HankelDMD, EDMD
import matplotlib.pyplot as plt

from romtools.utils import normalize, denormalize
from romtools.plotting import compare

from forecast import run_model


rom_opts = {
    'svd_rank': 20,
    'opt': True,
    'kernel_metric': 'rbf',
    'kernel_params': {
        'gamma': 0.2
    },
    # 'tikhonov_regularization': 1e-4,
    # 'd': 4
}

Nx = 500
Nt = 1000
tspan = [0, 10]
xspan = [-5, 5]
train_window = [0, 0.3]
norm = 'none'
model = 'burgers'
u0 = {
    'type': 'gaussian',
    'mu': 0,
    'sigma': 0.5,
    'mag': 1,
    'xspan': xspan,
    'num_points': Nx
}
ode_opts = {
    "num_steps": Nt,
    "fun_opts": {
        "nu": 0.1,
        "xspan": xspan,
        "bc_type": "neumann",
        "alpha": 1
    }
}

def compute_dmd(snapshots):
    # rom_obj = DMD(**opts)
    num_states = snapshots.shape[0]
    # rom_obj = HankelDMD(**rom_opts)
    rom_obj = EDMD(**rom_opts)

    rom_obj.fit(snapshots)    
    b, lamb, phi = rom_obj.amplitudes, rom_obj.eigs, rom_obj.modes  # (r,), (r,) and (Nstates, r)
    omega = np.log(lamb) / dt_const  # Continuous time eigenvalues
    rom_data = (phi @ np.diag(b) @ np.exp(time[np.newaxis, :] * omega[:, np.newaxis])).real  # (Nstates, Nt)

    r = b.shape[0]
    print(f"DMD rank: {b.shape[0]}")

    return rom_data[:num_states], r

truth_tup, data_opts = run_model(model, tspan, u0, ode_opts)

time, coords, variables, sim_data = truth_tup
dt = np.diff(time)
dt_const = dt[0]  # Assume constant for dmd

tlen = time[-1] - time[0]
tstart = train_window[0] * tlen + time[0]
tend = (train_window[1] if train_window[1] is not None and train_window[1] > 0 else 1) * tlen + time[0]
train_window = (tstart, tend)

start_idx = np.argmin(np.abs(time - train_window[0]))
end_idx = np.argmin(np.abs(time - train_window[1]))

train_time = time[start_idx:end_idx + 1]
train_data = sim_data[start_idx:end_idx + 1].copy()  # (Nt, Nx, Nvar)

# Normalize
norm = {v: norm.get(v, None) if isinstance(norm, dict) else norm for v in variables}
norm_consts = {}
for i, (v, norm_method) in enumerate(norm.items()):
    train_data[..., i], norm_consts[v] = normalize(train_data[..., i], method=norm_method)

snapshots = train_data.reshape((train_data.shape[0], -1)).T  # (Nstates, Ntime)

rom_data, r = compute_dmd(snapshots)

rom_data = rom_data.T.reshape(sim_data.shape)  # (Nt, Nx, Nvar)

# Denormalize
for i, (v, norm_method) in enumerate(norm.items()):
    rom_data[..., i] = denormalize(rom_data[..., i], method=norm_method, consts=norm_consts[v])

# Contour plot
pconfig = {
    "subplot_size_in": [5, 4],
    "col_labels": ["Truth", "DMD"],
    "data_labels": {"u": "Field variable $u(x,t)$"},
    "coord_labels": ["Time ($t$)", "Position ($x$)"],
    "data_plot_opts": {"u": {"cmap": "viridis"}},
    "error_plot_opts": {"cmap": "bwr", "norm": "log"}
}

truth_2d = [[{'u': truth_tup[-1][..., 0].T}]]
rom_2d = [[{'u': rom_data[..., 0].T}]]
fig, ax = compare(truth_2d, rom_2d, {'X': time, 'Y': coords}, **pconfig)

fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
s = np.linalg.svd(snapshots, full_matrices=False, compute_uv=False)
frac = s ** 2 / np.sum(s ** 2)
# r = r if isinstance(r, int) else int(np.where(np.cumsum(frac) >= r)[0][0]) + 1
ax.plot(frac, '-ok', ms=3)
ax.plot(frac[:r], 'or', ms=5, label=r'{}'.format(f'SVD rank $r={r}$'))
ax.set_yscale('log')

plt.show()
