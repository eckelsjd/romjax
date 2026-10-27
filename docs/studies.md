# Hyperparameter studies

`OptunaStudy` is a `Routine` that creates Optuna trials, applies sampled values to
raw Train YAML, and runs isolated training children through `ExecutorConfig`.
Install with `uv sync --extra optuna`. Optuna is optional for other romjax APIs;
this implementation is tested against Optuna 5.0.x.

See [the runnable TPE/ASHA demo](../demo/optuna_study/README.md) for complete serial,
process, conditional-space, and MPI examples.

```yaml
!romx:OptunaStudy
root: output/study
base: __parent__/train.yml
study_name: experiment
n_trials: 40
sampler:
  name: optuna.samplers.TPESampler
  kwargs: {seed: 42, n_startup_trials: 8, constant_liar: true}
pruner:
  name: optuna.pruners.SuccessiveHalvingPruner
  kwargs: {min_resource: 20, reduction_factor: 3}
metric: {name: test_score}
override:
  - name: learning_rate
    path: [optimizer, kwargs, learning_rate]
    distribution: {kind: float, low: 1.0e-5, high: 0.01, log: true}
executor: {kind: process, max_workers: 4, max_in_flight: 4}
```

`base` can be a path or a deferred inline `!overrides` source. Train is constructed
only inside its trial subprocess, after applying overrides and child environment.
The base must produce a `Train`; arbitrary routines with only final file metrics
cannot participate in live training pruning.

## Search spaces

Paths are nonempty lists of string mapping keys and integer sequence indices.
For example `[loss, terms, 1, weight]` edits one list entry; a key containing a dot
is a literal key. Existing ancestors and list indices must exist. A missing final
mapping key may be added, allowing parameters omitted from constructor defaults.
Duplicate/overlapping targets, negative/bool indices, and conflicts with injected
case roots are rejected.

Supported distributions forward all native suggestion arguments:

```yaml
distribution: {kind: float, low: 0.01, high: 1.0, step: 0.01}
# Or:
# distribution: {kind: int, low: 8, high: 64, step: 8}
# distribution: {kind: categorical, choices: [null, false, "false", "null", "1e-3"]}
```

Float/int `log` and `step` restrictions follow Optuna. Samples are typed YAML node
assignments, not string substitutions: a sampled null replaces a list entry even
though ordinary YAML sequence merging uses null as an inheritance placeholder.
Unchanged siblings and YAML tags survive.

To select entire objects, map primitive string labels to YAML values:

```yaml
override:
  - name: optimizer_family
    path: [optimizer]
    distribution: {kind: categorical, choices: [adam, sgd]}
    values:
      adam: {name: optax.adam, kwargs: {learning_rate: 0.001}}
      sgd: {name: optax.sgd, kwargs: {learning_rate: 0.01, momentum: 0.9}}
```

Default assignment mode `replace` removes obsolete fields. `mode: merge` opts into
existing YAML merge semantics. Optuna records the primitive label; trial metadata
records expanded assignments. Arbitrary objects are not native Optuna categories.

For conditional spaces or tied parameters, replace `override` with
`search_space: {callable: !!python/name:your.module.search, ...kwargs}`. The callable
receives a live trial and returns `Sequence[ConfigAssignment]`; it can suggest once
and assign that value to several paths. Build whole family-specific objects in one
assignment rather than returning overlapping parent/child assignments. See the
conditional demo. No expression evaluation or custom YAML mini-language is used.

## Samplers, pruners, storage

Sampler/pruner specifications use `ThirdPartyType` and check the resulting Optuna
base class. Every installed constructor argument is available through `args` and
`kwargs`, including nested objects and importable callables. For example:

```yaml
pruner:
  name: optuna.pruners.PatientPruner
  kwargs:
    wrapped_pruner: {name: optuna.pruners.MedianPruner}
    patience: 3
storage:
  name: optuna.storages.JournalStorage
  kwargs:
    log_storage:
      name: optuna.storages.journal.JournalFileBackend
      kwargs: {file_path: output/study/study.journal}
```

Omitted storage defaults to a journal under root. Explicit `storage: null` is
in-memory, including for process/MPI executors because only the controller accesses
storage. It cannot resume an earlier invocation. A URL string is also accepted.
Storage constructor specs are deferred until `run()`:

```yaml
storage:
  name: optuna.storages.RDBStorage
  kwargs:
    url: postgresql+psycopg://user:password@host/database
    engine_kwargs: {pool_pre_ping: true, pool_size: 4}
# Alternatively:
# storage:
#   name: optuna.storages.GrpcStorageProxy
#   kwargs: {host: storage-host, port: 13000}
```

Install the chosen backend's drivers separately. Prefer an importable storage
factory for credentials instead of embedding secrets in shared YAML. The library
neither starts a database/proxy server nor deletes a whole external database.
Backend constructor options remain available subject to the backend's own limits.

This engine uses public ask/tell. Native RDB heartbeat settings require `optimize`
and are rejected; use `feedback` liveness settings instead. `n_jobs` is not a second
pool: configure `executor`. Native worker-owned optimize loops and multi-objective
studies are not implemented. Samplers/callbacks that call `study.stop()` request
no further submissions; already allocated active work is drained. GridSampler is
supported, including exhaustion, but parallel grid execution can produce duplicate
parameter combinations and does not promise old GridSearch ordering.

## Training metrics and pruning

`Train.fit(observer=...)` returns a `TrainingResult`; its `params` field contains
trained parameters. `Train.run()` returns an exit code. Observers receive immutable
`TrainingReport(step, metrics, final)` snapshots outside JAX transformations.
`TrainingDecision.STOP` finalizes checkpoints and histories. Observer exceptions
propagate after cleanup.

Train's `diagnostics.report_interval` and `diagnostics.report_metrics` determine
when and what is reported. Set them in the base Train configuration or via
explicit overrides. A periodic report occurs only when its interval and
`host_interval` are both due. With `report_interval: null` or `host_interval: null`,
only a final report is sent and live pruning is unavailable. Final reports describe
returned parameters and cannot prune finished training. See [training diagnostics](training.md).

`metric` selects and reduces reports:

- `name`: `loss` (default) or `test_score`.
- `reduction`: `last` (default), `best` (direction-aware), or `mean` over `window`
  (default 10 reports). The same reduction is used for intermediate and final values.
- `invalid`: `fail` (default) or `skip`. A trial with no finite objective fails.

Configure Train to report the selected metric for periodic pruning; `test_score`
requires `Train.test`. A final report at a previously observed step replaces that
step's observation in the reducer.

ASHA is `SuccessiveHalvingPruner`; Train's termination settings impose maximum
resource. Pruning yields Optuna `PRUNED`, not a failed subprocess or failed exit
policy. Normal early training termination yields `COMPLETE` with its final score.

## Execution, cancellation, and persistence

All existing executor kinds are supported. Only the executor owns pools/futures
and progress. Set `executor.show_progress` and `executor.progress_bar` as for
GridSearch. Child progress remains controlled by Train; quiet children capture
stdout/stderr under each trial root.

MPI uses the existing top-level rank-zero master and borrowed worker pool. Each
worker supervises an isolated child. No nested MPI or additional communicator is
created. All workers must see the same paths on a shared filesystem supporting
atomic rename and POSIX advisory locks. The controller uses a separate thread to
service feedback, so serial execution can also report/prune while waiting.

`feedback` defaults (seconds): `poll_interval: 0.1`, `response_timeout: 120`,
`stale_after: 300`, `shutdown_grace: 10`. Independent controller/child leases cover
long compilation and batch evaluation. Supervisors terminate then kill children
when a controller lease expires or execution is canceled. A child-side watchdog also
exits on controller lease expiry if its supervisor has died. An MPI rank failure can
invalidate the pool; recovery requires a new launch, not transparent rank repair.

`n_trials` counts additional attempts in this invocation. `timeout` stops new
launches and drains active work. `callbacks` are importable `(Study, FrozenTrial)`
functions after terminalization; callback errors abort scheduling. `enqueue_trials`
forwards dictionaries with `params`, `user_attrs`, and `skip_if_exists` to Optuna.
`user_attrs` initializes study metadata; `romjax_` names are reserved.

Write policies operate during `run()`, never YAML validation:

- `reuse`: load the named study, verify the experiment fingerprint, and add trials.
  Changing base/search/metric settings requires a new study or overwrite.
- `error`: reject existing studies or owned run artifacts.
- `overwrite`: reset only the named owned study and its trial artifacts.

A POSIX root lock prevents competing controllers. A study records its root identity;
using another root for the same study is rejected. The root also records its study
and storage identity; changing either requires a new root, including with overwrite.
Fingerprints include referenced YAML contents in both the base and declarative
assignment values. Files chosen dynamically by a search callback are snapshotted
per trial but cannot be enumerated for the initial compatibility check.
After lease expiry, owned orphan
RUNNING trials become FAIL. Retrying means a new trial, optionally with enqueued old
parameters; checkpoint continuation of an interrupted Optuna trial is not automatic.

All trial artifacts are preserved, including unsuccessful trials. `best/` contains
a copy of the best COMPLETE trial directory, including its configuration and
checkpoints. It is updated as better trials finish and restored on reuse if missing.
`study_manifest.yml` mirrors the study's trial states and metadata, including all
completed cases. On `write_policy: overwrite`, existing trial and best directories
are moved into a uniquely named `archive/` directory before the study is reset.

`exit_policy: any` returns nonzero if any trial fails; `all` only when all attempts
fail; `ignore` ignores trial failures. Canceled queued trials become FAIL and invoke
terminal callbacks once. Infrastructure exceptions still propagate.
A nonempty directory is never treated as evidence of successful training.

`training_seed` optionally overrides `Train.init_seed`. `seed_policy: fixed`
(default) shares it across trials; `trial` adds the trial number. Sampler seeds are
independent. Parallel order changes adaptive sampling/pruning, and restoring study
storage does not restore sampler RNG state: identical seeds do not promise identical
parallel or interrupted-versus-uninterrupted studies.

## Migrating from GridSearch

| GridSearch concept | OptunaStudy equivalent |
| --- | --- |
| base / path overrides | same raw-YAML base, named distributions instead of cases |
| Cartesian enumeration | GridSampler with finite distributions, or adaptive sampler |
| case root paths / child environment | case_root_path / child_env |
| metric callable | metric reduces Train reports into an Optuna objective |
| reuse / overwrite / error | named-study lifecycle plus config fingerprint |
| serial/process/MPI/hybrid | same ExecutorConfig |
| exit policy | similar top-level form; PRUNED is distinct from FAIL |

GridSearch remains supported. Existing arbitrary non-Train GridSearch routines do
not automatically become prunable OptunaStudy objectives.
