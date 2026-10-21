#!/bin/bash
#SBATCH --job-name=romjax-mpi-grid
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00

set -euo pipefail

# Use the MPI mode supported by your site's Slurm and MPI installation.
srun --mpi=pmix .venv/bin/romx run demo/mpi_grid_search/grid.yml
