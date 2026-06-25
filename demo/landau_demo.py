"""Run a Landau damping visual verification for the Vlasov1D1V solver."""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from loguru import logger

from romjax import YamlLoader, gridplot
from romjax.utils import format_time_engineering

# jax.config.update('jax_platforms', 'cpu')


def electrostatic_energy(model, potential: jnp.ndarray) -> jnp.ndarray:
    """Compute saved-time electrostatic energy from a potential trajectory.

    :param model: configured Vlasov model
    :param potential: potential trajectory shaped ``(Nx, Nt)``
    :return: energy at each saved time
    """
    dx = model.grid.spacing[0]
    debye = jnp.asarray(model.params["debye"])
    electric = jnp.zeros_like(potential)
    electric = electric.at[1:-1].set(-(potential[2:] - potential[:-2]) / (2.0 * dx))
    electric = electric.at[0].set(-(potential[1] - potential[0]) / dx)
    electric = electric.at[-1].set(-(potential[-1] - potential[-2]) / dx)
    return 0.5 * debye**2 * jnp.sum(electric**2, axis=0) * dx


def main() -> None:
    """Load the benchmark config, solve it, and save the energy plot."""
    root = Path(__file__).resolve().parents[1]
    model = YamlLoader.load(root / "demo" / "landau.yml")
    solution, solution_obj = model.solve(return_sol=True)

    logger.info(solution_obj.result)
    logger.info(str(solution_obj.stats))

    times = model.solver.save_times()
    energy = electrostatic_energy(model, solution["fields"]["potential"])

    out_dir = root / "demo" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(times, energy)
    ax.set_xlabel("time")
    ax.set_ylabel("electrostatic energy")
    ax.set_title("1D1V Landau damping")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "landau_damping_energy.png", dpi=200)

    residual = model.evaluate(None, solution)
    xg, vg = model.grid.coords
    x = xg[:, 0]
    density, velocity, temperature = jax.vmap(model._density_moments, in_axes=2, out_axes=1)(solution["fields"]["vdf"])
    boundary = model._boundary_inputs(model._resolve_inputs(None))
    electric = -jax.vmap(lambda phi: model._potential_gradient(phi, boundary), in_axes=1, out_axes=1)(
        solution["fields"]["potential"]
    )

    sl = (0, len(times), 4)

    def gen_vdf():
        for i in range(*sl):
            yield xg, vg, solution["fields"]["vdf"][..., i]

    def gen_vdf_residual():
        for i in range(*sl):
            yield xg, vg, residual["fields"]["vdf"][..., i]

    def gen_potential():
        for i in range(*sl):
            yield x, solution["fields"]["potential"][:, i]
    
    def gen_potential_residual():
        for i in range(*sl):
            yield x, residual["fields"]["potential"][:, i]

    def gen_electric():
        for i in range(*sl):
            yield x, electric[:, i]

    def gen_density():
        for i in range(*sl):
            yield x, density[:, i]

    def gen_velocity():
        for i in range(*sl):
            yield x, velocity[:, i]

    def gen_temperature():
        for i in range(*sl):
            yield x, temperature[:, i]

    def gen_title():
        for i in range(*sl):
            yield format_time_engineering(times[i])

    vdf_scale = max(
        float(jnp.max(jnp.abs(solution["fields"]["vdf"]))),
        float(jnp.max(jnp.abs(residual["fields"]["vdf"]))),
    )
    vdf_clim = (-vdf_scale, vdf_scale)
    pot_ylim = (float(jnp.min(solution["fields"]["potential"])), float(jnp.max(solution["fields"]["potential"])))
    electric_ylim = (float(jnp.min(electric)), float(jnp.max(electric)))
    density_ylim = (float(jnp.min(density)), float(jnp.max(density)))
    velocity_ylim = (float(jnp.min(velocity)), float(jnp.max(velocity)))
    temperature_ylim = (float(jnp.min(temperature)), float(jnp.max(temperature)))
    vdf_spec = {
        "kind": "pcolor",
        "data": gen_vdf(),
        "opts": {
            "xlabel": "Position ($x$)",
            "ylabel": "Velocity ($v$)",
            "animate": True,
            "clim": vdf_clim,
            "cbar_label": "VDF",
        },
        "kwargs": {"cmap": "viridis"},
    }
    residual_vdf_spec = {
        "kind": "pcolor",
        "data": gen_vdf_residual(),
        "opts": {
            "xlabel": "Position ($x$)",
            "ylabel": "Velocity ($v$)",
            "animate": True,
            "clim": vdf_clim,
            "cbar_label": "VDF residual",
        },
        "kwargs": {"cmap": "viridis"},
    }
    pot_spec = {
        "kind": "line",
        "data": gen_potential(),
        "opts": {
            "xlabel": "Position ($x$)",
            "ylabel": "Potential ($\\phi$)",
            "animate": True,
            "ylim": pot_ylim,
        },
        "kwargs": {"color": plt.get_cmap("viridis")(0)},
    }
    residual_pot_spec = {
        "kind": "line",
        "data": gen_potential_residual(),
        "opts": {
            "animate": True,
            "ylim": pot_ylim
        },
        "kwargs": {"color": plt.get_cmap("viridis")(0), "ls": "--"},
    }
    electric_spec = {
        "kind": "line",
        "data": gen_electric(),
        "opts": {
            "xlabel": "Position ($x$)",
            "ylabel": "Electric field ($E$)",
            "animate": True,
            "ylim": electric_ylim,
        },
        "kwargs": {"color": plt.get_cmap("viridis")(0.125)},
    }
    density_spec = {
        "kind": "line",
        "data": gen_density(),
        "opts": {
            "xlabel": "Position ($x$)",
            "ylabel": "Density ($n$)",
            "animate": True,
            "ylim": density_ylim,
        },
        "kwargs": {"color": plt.get_cmap("viridis")(0.25)},
    }
    velocity_spec = {
        "kind": "line",
        "data": gen_velocity(),
        "opts": {
            "xlabel": "Position ($x$)",
            "ylabel": "Bulk velocity ($u$)",
            "animate": True,
            "ylim": velocity_ylim,
        },
        "kwargs": {"color": plt.get_cmap("viridis")(0.5)},
    }
    temperature_spec = {
        "kind": "line",
        "data": gen_temperature(),
        "opts": {
            "xlabel": "Position ($x$)",
            "ylabel": "Temperature ($T$)",
            "animate": True,
            "ylim": temperature_ylim,
        },
        "kwargs": {"color": plt.get_cmap("viridis")(0.75)},
    }
    cfg = {
        "scheme": "dark",
        "subplot_size_in": (4, 3),
        "title": gen_title(),
        "save": out_dir / "landau_damping.mp4",
        "animate_opts": {
            "blit": True,
            "num_frames": len(range(*sl)),
            "progress_callback": "bar",
            "fps": 15,
            "writer": "ffmpeg",
            "dpi": 150,
        },
        "shape": (2, 4),
    }

    fig, ax, ani = gridplot(
        [vdf_spec, residual_vdf_spec, (pot_spec, residual_pot_spec), electric_spec, density_spec, velocity_spec, temperature_spec],
        **cfg,
    )


if __name__ == "__main__":
    main()
