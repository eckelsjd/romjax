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
  output batches for those inputs. Numerical functions retain JIT/vmap batching.
  Datasets and expanded configurations run in sequence.
- **GridSearch:** one case specification, run in an isolated subprocess. Reused
  cases produce immediate results and still invoke result handlers.
- **CompositeRoutine:** one child routine. Remote execution requires YAML-backed
  child specifications; local execution also accepts live routine instances.

GenImplicitModel and GenSource inherit DataGeneration's executor unless they
specify their own `executor`. Standalone generators default to serial execution.
Other generator kinds, including benchmarks, retain their independent execution
and progress behavior.

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

Each batch writes only its assigned samples. Completed data remains reusable after
a stopped run. With `throw=False`, per-sample failure handling remains inside each
generation task; infrastructure failures still propagate.

The old routine-specific executor config classes, `run_cases`, `open`,
`ExecutionSession`, and public completion/progress helpers have been removed.
