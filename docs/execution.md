# Routine executors

`DataGeneration`, `CompositeRoutine`, and `GridSearch` use one
`romjax.execution.ExecutorConfig`. Serial execution is the default.

```yaml
executor:
  kind: thread
  max_workers: 4
  max_in_flight: 8
  show_progress: true
  progress_bar:
    title: Training cases
```

`executor: serial`, `executor: process`, and `executor: mpi` are also accepted.
Thread and process pool sizes default to Python's executor defaults. The default
submission budget is the configured local worker count, otherwise 32; serial
execution uses one, MPI uses its worker count, and hybrid uses its slot count.
`max_in_flight` overrides the budget, bounded by available hybrid slots.

Each routine prepares domain payloads and handles results:

- **DataGeneration:** one GenImplicitModel or GenSource input batch, including all
  output batches for those inputs. Numerical functions use JIT/lax.map with independent map_batch_size (default 0 for full vectorization). Workers return host arrays; only the controller reads/writes target dataset files.
  Datasets and expanded configurations run in sequence.
- **GridSearch:** one case specification, run in an isolated subprocess. Reused
  cases produce immediate results and still invoke result handlers.
- **CompositeRoutine:** one child routine. Remote execution requires YAML-backed
  child specifications; local execution also accepts live routine instances.

GenImplicitModel and GenSource inherit DataGeneration's executor unless they
specify their own `executor`. Standalone generators default to serial execution.
GenDataBenchmark uses its nested generator's executor, falling back to the parent
DataGeneration executor or serial execution when used alone. Other generator
kinds retain their independent execution and progress behavior.

## Data-generation benchmarks

```yaml
executor: {kind: process, max_workers: 4, show_progress: false}
datasets:
  sample:
    kind: benchmark
    generator: {kind: source, samples: 1, seed: 7, batch_size: 8}
    samples: 100
    warmup: 1
    repeat: 5
    hardware: workstation
```

The graph must contain the corresponding `sample` edge, as for normal generation.
The benchmark's `samples` sets the number of inputs in each repeat; the nested
generator supplies numerical settings and seeds. Implicit output batches use the
generator's `batch_size`, including partial input and output batches.

Each repeat records one wall-clock interval from submission of the first input
batch through completion of all batches. Dispatch, transport of task metadata,
and synchronization are included. Pool startup/shutdown, model reconstruction,
key preparation, warmup, progress rendering, and all file I/O are excluded.
Numerical results stay on their executing worker and are synchronized before
completion is reported. Generated samples are never saved.

One pool is retained across warmup and all repeats. Every participating worker
executes `warmup` complete untimed passes so that all batch shapes are compiled
before measurement. With `warmup: 0`, first-use compilation is measured. Progress
is updated between passes; either `show_progress: false` on the benchmark or
`executor.show_progress: false` suppresses it.

The HDF5 cache groups use `<hardware>.<executor-kind>.<fingerprint>`, for example
`workstation.process.<fingerprint>`. Explicit `hardware` and `DATA_GEN_HARDWARE`
values supply the prefix. The fingerprint identifies execution settings,
participating worker count, and inherited JAX settings; progress options are
excluded. Each group stores `total` seconds per repeat and `amortized` seconds
per input sample, plus compatibility metadata. Old cache groups are preserved
but are not reused for the new full-pass measurement schema.

`ExecutorConfig.execute_passes(tasks, worker, warmup=1, repeat=5)` provides the
shared pass driver and returns one duration per repeat. `TaskWorker.initialize`
may prepare worker-local state from all task payloads before warmup begins.
Each worker retains its own shallow copy of the prepared instance, so attributes
assigned during initialization persist across passes. The worker must synchronize
asynchronous work before `run` returns.
Task results are discarded, exceptions propagate, and cleanup retains ownership
of borrowed MPI pools. Hybrid execution is not supported for repeated passes.

## User-controlled progress

Executor-driven tasks use only `executor.show_progress` and
`executor.progress_bar`; GenImplicitModel and GenSource no longer have their own
`show_progress` field. Per-executor bar options override routine-provided defaults
and the existing global `RoutineConfig.progress_bar` defaults.

The driver forwards bar options unchanged, advances progress using task units,
and sets `bar.text` to the completed task's label. A generation batch contributes
its number of inputs; a case or child routine contributes one. Manual progress
mode is honored, including when configured globally.

Every execution independently honors its configuration. For example:

```yaml
!romx:CompositeRoutine
executor: {kind: serial, show_progress: false}
routines:
  - __parent__/generate.yml
  - __parent__/search.yml
```

Each child file can enable its own bar and choose its own title:

```yaml
executor:
  kind: mpi
  show_progress: true
  progress_bar: {title: Generating training samples}
```

There is no suppression based on nesting, active bars, threads, or MPI placement.
Enabling both parent and child bars is allowed; alive_progress determines how
those bars behave, and any errors propagate naturally.

## Native futures and the task API

Only the shared driver manages pools and futures. It calls native
`Executor.submit`, `wait(FIRST_COMPLETED)`, `Future.result`, `Future.cancel`, and
`Executor.shutdown`. MPI uses mpi4py's compatible executor. The only custom
executor is a small serial adapter returning native completed futures.

```python
from romjax.execution import ExecutorConfig, TaskContext, TaskSpec, TaskWorker

class SquareWorker(TaskWorker[int, int]):
    def run(self, payload: int, context: TaskContext) -> int:
        return payload * payload

if __name__ == '__main__':
    tasks = [TaskSpec(str(i), f'case {i}', i) for i in range(5)]
    outcomes = ExecutorConfig(kind='process', max_workers=2).execute(
        tasks, SquareWorker(),
    )
    print([outcome.value for outcome in outcomes])
```

`TaskSpec` contains a key, label, payload, progress units, and optional
`ImmediateResult`. `ImmediateResult(None)` is a completed success, distinct from
work requiring execution. `TaskOutcome` is a completed data record containing the
value or exception, timing, task, and resource context; it is not another future.
Execution duration is measured on the worker with a monotonic clock.

Use `on_result` for completion-order handling. Returning `ExecutionAction.STOP`
stops new submissions, cancels queued work where possible, and drains active
work while handling its actual results. Collected outcomes return in submission
order. Use `collect_results=False` when the handler consumes results without
retaining them, as data generation does.

Ordinary worker exceptions raise by default. With `error_policy='collect'`, the
handler receives error outcomes; CompositeRoutine uses this for its `force`
policy. Preparation, iteration, transport, submission, pool, and callback failures
are always fatal. If a callback raises, subsequent callbacks are skipped while
pending work is canceled/drained. Cleanup also covers KeyboardInterrupt.

Workers can implement `prepare(remote=...)` to retain live local state or build
serializable remote configuration once per execution. Worker contexts carry
settings and resources, never pools, futures, or progress objects.

## Hybrid subprocess slots

GridSearch can request shared CPU/GPU slot scheduling:

```yaml
executor:
  kind: hybrid
  gpu:
    devices: [0, 1]
    workers_per_device: 1
    memory_fraction: 0.4
    preallocate: false
  cpu: {max_workers: 2}
```

The common driver assigns and reuses slots through the same native-future loop.
Its task context supplies device metadata and environment settings to the
subprocess worker. GPU slots select the GPU platform explicitly; CPU slots select
CPU. Reserved device environment keys must be configured through the executor.

Hybrid execution requires a worker explicitly supporting isolated subprocess
environments. GenImplicitModel, GenSource, and CompositeRoutine do not support
hybrid execution. Slot settings never mutate the controller's environment.

## MPI ownership and reconstruction

Install with `uv sync --extra mpi`, then launch at least two ranks:

```sh
mpiexec -n 5 uv run romx run composite.yml
```

The top-level CLI owns one MPI pool. Serial or thread composite children can
borrow it for DataGeneration and GridSearch, successively or concurrently. A
borrower's cleanup only cancels/drains its own futures and never closes the pool.

An MPI composite distributes children across workers; those children may use
local serial/thread/process execution. MPI cannot be requested again inside a
remote task or spawned local process. No MPI request silently falls back to
serial execution.

Python scripts enter the same context on every launched rank:

```python
import romjax

if __name__ == '__main__':
    with romjax.mpi_context() as controller:
        if controller:
            raise SystemExit(romjax.load('composite.yml').run())
```

Process pools use spawn; define custom workers, models, and routines in importable
modules and protect script entry points with the normal main guard. JAX precision
and platform settings propagate to remote workers. Persistent MPI workers retain
their initialized platform/device environment; heterogeneous device settings belong
in isolated subprocess slots.

Remote generation rebuilds models from YAML and compiles locally, caching at most
four generation executions per worker. Keep compression artifacts referenced by
path; private caches and compiled functions are not serialized with batches.
Non-reconstructible configurations fail before submission. `__parent__/` paths
retain the declaring file's context, including separately loaded graph/sampler
YAML files. Workers need the same configuration, artifact, and dataset filesystem.

Each batch computes only its assigned families; the controller commits returned results to the indexed HDF5 shards. Completed data remains reusable after
a stopped run. With `throw=False`, per-sample failure handling remains inside each
generation task; infrastructure failures still propagate.

The old routine-specific executor config classes, `run_cases`, `open`,
`ExecutionSession`, and public completion/progress helpers have been removed.

See [data storage](data-storage.md) for the independent task, compute, and file capacities and controller-only writing.

## Adaptive Optuna studies

`OptunaStudy` also uses `ExecutorConfig`. It lazily asks for trials as capacity
becomes available, runs isolated Train children, and tells Optuna their terminal
states. Rank zero/controller retains all live Optuna objects and storage access.
A separate feedback thread exchanges metric/pruning messages through the shared
artifact filesystem; no additional MPI communicator is used. See [studies](studies.md).

`ExecutorConfig.execute(..., on_abort=callback)` can notify a domain's active
workers before the driver cancels/drains futures following an exception, including
KeyboardInterrupt. Normal `ExecutionAction.STOP` retains its existing drain
semantics. OptunaStudy uses this hook to cancel child supervisors promptly.
