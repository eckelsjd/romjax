# TPE and ASHA with Train

Run from the repository root:

```sh
uv sync --extra optuna
uv run romx run demo/optuna_study/serial.yml
uv run romx run demo/optuna_study/process.yml
uv run romx run demo/optuna_study/conditional.yml
# On an MPI installation with the mpi extra:
mpiexec -n 5 uv run romx run demo/optuna_study/mpi.yml
```

The small deterministic linear regression example uses real JAX/Optax training,
held-out validation, TPE sampling, and ASHA pruning. It tunes learning rate,
regularization weight, and a string-valued penalty family. The conditional example
instead chooses Adam or SGD and only suggests momentum for SGD.

Reports use completed optimizer steps. Validation runs every ten steps even though
`host_interval` is null. ASHA first considers pruning at step 20; Train limits each
trial to 180 updates. The number pruned varies with ordering and sampled parameters.

Each run adds 40 attempts to its named journal-backed study. To make a short run,
create an `!overrides:...` YAML setting `n_trials: 4`; change `root` and `study_name`
for a new experiment. `write_policy: overwrite` resets only that owned study.
Serial/process/MPI examples have different roots and names.

Inspect `study_manifest.yml`, `best.json`, and `trials/trial_*/trial.yml` under the
configured root. Each trial retains `case.yml` and referenced YAML snapshots. The
best three completed trials retain their full artifacts; unsuccessful trials are
also retained by default. Child stdout/stderr are captured in their trial folders.

A process worker supervises one isolated training subprocess. Four workers can
train concurrently; choose capacity to fit CPU/GPU memory. The MPI example uses
rank zero for sampling and four worker ranks for training. All nodes must see the
same paths on a filesystem supporting atomic rename and POSIX advisory locks.
