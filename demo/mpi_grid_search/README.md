# MPI GridSearch demo

Run these commands from the repository root after installing a working MPI distribution:

```sh
uv sync --extra mpi --extra cuda12 --all-groups
mpiexec -n 4 .venv/bin/romx run demo/mpi_grid_search/grid.yml
```

The four MPI ranks consist of one coordinator and three case workers. Each case runs in its worker's child process. The six candidates write to `demo/mpi_grid_search/output/cases/`; the manifest and best copy are in `demo/mpi_grid_search/output`.

On Slurm, submit `sbatch demo/mpi_grid_search/sbatch.sh` from the repository root. The script requests four MPI tasks and two CPUs per task, then uses `srun` to launch them. Check `srun --mpi=list` at your site and change `--mpi=pmix` if necessary. Slurm determines rank placement and CPU affinity; the executor uses the ranks it receives.

For GPU cases, request and bind one GPU per task with Slurm's `--gpus-per-task=1` where supported. Each worker's case subprocess inherits its rank's GPU visibility, so JAX sees only the GPU assigned to that task. The coordinator rank also receives a GPU under this simple allocation. Configure JAX options in the routine YAML as usual; GridSearch passes them to case subprocesses.
