"""Task orchestration on native concurrent.futures executors.

Only this module owns pools, pending futures, resource slots, and task progress.
MPI pools are borrowed from the top-level context and are never closed here.
"""
from __future__ import annotations

import multiprocessing
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Barrier, Lock, get_ident
from typing import Any, ClassVar, Literal
from uuid import uuid4

from alive_progress import alive_bar, config_handler
from pydantic import ConfigDict, Field, NonNegativeInt, PositiveInt, TypeAdapter, field_validator, model_validator

from romjax.typing import DictModel, WriteStream
from romjax.utils import _NullProgress

__all__ = ['ExecutorConfig', 'ExecutionAction', 'ExecutionError', 'ImmediateResult', 'ProgressBarConfig',
           'TaskContext', 'TaskOutcome', 'TaskSpec', 'TaskWorker']

_HYBRID_ENV_KEYS = frozenset({'CUDA_VISIBLE_DEVICES', 'JAX_PLATFORMS',
                              'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_PREALLOCATE'})
_pass_barriers: dict[str, Any] = {}
_pass_workers: dict[tuple[str, int], TaskWorker] = {}
_mpi_pass_lock = Lock()


class ExecutionError(RuntimeError):
    """Preparation, transport, or execution infrastructure failed."""


class ExecutionAction(Enum):
    """A result handler's scheduling decision."""
    CONTINUE = 'continue'
    STOP = 'stop'


class ProgressBarConfig(DictModel):
    """Unmodified alive_bar keyword options for one execution."""

    @model_validator(mode='after')
    def _normalize_file(self):
        if hasattr(self, 'file'):
            object.__setattr__(self, 'file', TypeAdapter(WriteStream).validate_python(self.file))
        return self


@dataclass(frozen=True)
class ImmediateResult[R]:
    """A precomputed result, including a successful None value.

    :param value: result that bypasses worker submission
    """
    value: R


@dataclass(frozen=True)
class TaskSpec[P, R]:
    """Domain payload and display metadata for one independent task.

    :param key: task identifier within this execution
    :param label: text displayed when the task completes
    :param payload: worker input
    :param units: progress units represented by this task
    :param immediate: optional precomputed result
    """
    key: str
    label: str
    payload: P
    units: int = 1
    immediate: ImmediateResult[R] | None = None


@dataclass(frozen=True)
class TaskContext:
    """Worker settings without pools, futures, or progress objects.

    :param run_id: identity for worker-local initialization caches
    :param environment: inherited settings plus optional resource-slot overrides
    :param device: assigned resource metadata, if any
    """
    run_id: str
    environment: Mapping[str, str] = field(default_factory=dict)
    device: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TaskOutcome[P, R]:
    """Completed data record; asynchronous state remains on native Future objects.

    :param task: originating task
    :param value: worker result
    :param error: ordinary worker exception, if any
    :param started_at: worker wall-clock start timestamp
    :param finished_at: worker wall-clock finish timestamp
    :param elapsed_seconds: elapsed time measured on the executing worker
    :param context: execution and resource metadata
    """
    task: TaskSpec[P, R]
    value: R | None
    error: Exception | None
    started_at: str
    finished_at: str
    elapsed_seconds: float
    context: TaskContext


class TaskWorker[P, R](ABC):
    """Domain adapter that prepares work without managing its execution.

    Subclasses may prepare serializable remote specifications while retaining live
    Python objects for local execution. Hybrid workers must explicitly support
    passing resource environments to an isolated subprocess.
    """
    supports_hybrid: ClassVar[bool] = False
    environment: Mapping[str, str] = {}

    def prepare(self, *, remote: bool) -> TaskWorker[P, R]:
        """Prepare one worker representation before any tasks are submitted.

        :param remote: whether native transport must serialize the worker
        :return: prepared worker
        """
        return self

    def initialize(self, payloads: Sequence[P], context: TaskContext) -> None:
        """Prepare worker-local state before repeated passes are timed.

        The pass driver retains a shallow copy of the prepared worker on each
        executing thread/process. Attributes assigned here persist across passes.

        :param payloads: all payloads that may run on this worker
        :param context: stable context shared by initialization and measured passes
        """

    @abstractmethod
    def run(self, payload: P, context: TaskContext) -> R:
        """Execute a domain payload.

        :param payload: task input
        :param context: inherited settings and assigned resources
        :return: domain result
        """
        raise NotImplementedError


@dataclass(frozen=True)
class _TaskReturn:
    value: Any
    error: Exception | None
    started_at: str
    finished_at: str
    elapsed_seconds: float


@dataclass(frozen=True)
class _DeviceSlot:
    kind: Literal['cpu', 'gpu']
    index: int | None
    env: dict[str, str]

    def manifest(self) -> dict[str, Any]:
        return {'kind': self.kind, 'index': self.index}


def _configure_jax(environment: Mapping[str, str]) -> None:
    import jax

    if 'JAX_ENABLE_X64' in environment:
        jax.config.update('jax_enable_x64', environment['JAX_ENABLE_X64'].lower() in {'true', '1'})
    if 'JAX_PLATFORMS' in environment:
        jax.config.update('jax_platforms', environment['JAX_PLATFORMS'])


def _configure_process(environment: Mapping[str, str]) -> None:
    os.environ.update(environment)
    os.environ['ROMJAX_MPI_CASE_CHILD'] = '1'
    os.environ['MPI4PY_RC_INITIALIZE'] = '0'
    _configure_jax(environment)


def _configure_pass_process(environment: Mapping[str, str], run_id: str, barrier: Any) -> None:
    _configure_process(environment)
    _pass_barriers[run_id] = barrier


@dataclass
class _PassBootstrap(TaskWorker):
    worker: TaskWorker | None
    tasks: Sequence[TaskSpec]
    warmup: int
    mpi: bool = False
    cleanup: bool = False

    def run(self, payload: Any, context: TaskContext) -> None:
        identity = (context.run_id, get_ident())
        try:
            if self.cleanup:
                _pass_workers.pop(identity, None)
                return
            worker = copy(self.worker)
            _pass_workers[identity] = worker
            worker.initialize(tuple(task.payload for task in self.tasks), context)
            for _ in range(self.warmup):
                for task in self.tasks:
                    if task.immediate is None:
                        worker.run(task.payload, context)
        finally:
            # A blocking bootstrap occupies each native worker until every
            # worker has initialized. Even very short warmups cannot all run
            # on one worker while another worker starts cold during timing.
            if self.mpi:
                from mpi4py.futures import get_comm_workers

                get_comm_workers().Barrier()
            else:
                _pass_barriers[context.run_id].wait()


class _PassDispatch(TaskWorker):
    """Dispatch payloads to the instance initialized on the executing worker."""

    def run(self, payload: Any, context: TaskContext) -> Any:
        return _pass_workers[(context.run_id, get_ident())].run(payload, context)


def _run_task(worker: TaskWorker, payload: Any, context: TaskContext, remote: bool) -> _TaskReturn:
    previous = {key: os.environ.get(key) for key in (*context.environment, 'ROMJAX_MPI_CASE_CHILD')}
    try:
        if remote:
            os.environ.update(context.environment)
            os.environ['ROMJAX_MPI_CASE_CHILD'] = '1'
            _configure_jax(context.environment)
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        try:
            value, error = worker.run(payload, context), None
        except ExecutionError:
            raise
        except Exception as exc:
            value, error = None, exc
        return _TaskReturn(value, error, started_at, datetime.now(timezone.utc).isoformat(), time.monotonic() - start)
    finally:
        if remote:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class _SerialExecutor(Executor):
    """Inline adapter with standard Future result/exception behavior."""
    def __init__(self):
        self._shutdown = False

    def submit(self, fn, /, *args, **kwargs):
        if self._shutdown:
            raise RuntimeError('cannot schedule new futures after shutdown')
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        self._shutdown = True


class HybridGpuConfig(DictModel):
    """GPU resource slots for hybrid execution.

    :param devices: visible GPU indices to schedule, or ``"auto"`` to use JAX-visible GPUs
    :param workers_per_device: number of simultaneous child processes per GPU
    :param memory_fraction: optional JAX/XLA GPU memory fraction for child processes
    :param preallocate: optional JAX/XLA GPU preallocation flag for child processes
    """

    devices: Literal["auto"] | tuple[NonNegativeInt, ...] = "auto"
    workers_per_device: PositiveInt = 1
    memory_fraction: float | None = Field(default=None, gt=0.0, le=1.0)
    preallocate: bool | None = None

    @field_validator("devices", mode="before")
    @classmethod
    def _coerce_devices(cls, value: Any) -> Any:
        if value == "auto":
            return value
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return tuple(value)
        return value
    
    @model_validator(mode="before")
    @classmethod
    def _from_devices(cls, value):
        if not isinstance(value, Mapping | HybridGpuConfig):
            if isinstance(value, str | tuple | list):
                return {"devices": value}
            if isinstance(value, int):
                return {"workers_per_device": value}
        return value
    
    def get_env(self, device_index: int) -> dict[str, str]:
        """Return child environment overrides for one GPU slot."""
        env = {"CUDA_VISIBLE_DEVICES": str(device_index), "JAX_PLATFORMS": "gpu"}
        if self.memory_fraction is not None:
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.memory_fraction)
        if self.preallocate is not None:
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(self.preallocate).lower()
        return env


class HybridCpuConfig(DictModel):
    """CPU resource slots for hybrid execution.

    :param max_workers: CPU child-process slots, or ``None`` to use all local CPU cores
    """

    max_workers: NonNegativeInt | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_workers(cls, value):
        if ((isinstance(value, str) and value.lstrip("-").isdigit()) or 
            isinstance(value, int) or value is None):
            return {"max_workers": value}
        return value


class ExecutorConfig(DictModel):
    """One YAML-friendly configuration and driver for all routine tasks.

    :param kind: native executor backend or hybrid subprocess scheduling
    :param max_workers: local pool size; None uses Python's default
    :param max_in_flight: submission budget; None uses configured workers or backend capacity
    :param show_progress: whether to open this execution's progress bar
    :param progress_bar: alive_bar options, forwarded without rewriting
    :param gpu: hybrid GPU slots
    :param cpu: hybrid CPU slots
    """
    model_config = ConfigDict(extra='forbid')
    kind: Literal['serial', 'thread', 'process', 'mpi', 'hybrid'] = 'serial'
    max_workers: PositiveInt | None = None
    max_in_flight: PositiveInt | None = None
    show_progress: bool = True
    progress_bar: ProgressBarConfig = Field(default_factory=ProgressBarConfig)
    gpu: HybridGpuConfig | None = None
    cpu: HybridCpuConfig | None = None

    @model_validator(mode='after')
    def _validate_backend(self) -> ExecutorConfig:
        if self.max_workers is not None and self.kind not in {'thread', 'process'}:
            raise ValueError('max_workers is only configurable for thread and process executors.')
        if self.kind == 'hybrid':
            if self.gpu is None:
                object.__setattr__(self, 'gpu', HybridGpuConfig())
            if self.cpu is None:
                object.__setattr__(self, 'cpu', HybridCpuConfig())
        elif self.gpu is not None or self.cpu is not None:
            raise ValueError('gpu and cpu slot options require kind=hybrid.')
        return self

    @classmethod
    def from_dict(cls, value: Any) -> ExecutorConfig:
        """Normalize a backend name, mapping, or configuration.

        :param value: YAML-friendly execution configuration
        :return: validated configuration
        """
        if isinstance(value, cls):
            return value
        aliases = {'threads': 'thread', 'threadpool': 'thread', 'processes': 'process', 'ppool': 'process'}
        if isinstance(value, str):
            value = {'kind': aliases.get(value, value)}
        if not isinstance(value, Mapping):
            raise ValueError('Executor must be a string, mapping, or ExecutorConfig.')
        data = dict(value)
        data['kind'] = aliases.get(data.get('kind', 'serial'), data.get('kind', 'serial'))
        return cls.model_validate(data)

    def validate_environment(self, environment: Mapping[str, str]) -> None:
        """Reject hybrid-owned settings supplied outside its resource configuration.

        :param environment: domain worker's explicit environment overrides
        :raises ExecutionError: if reserved settings conflict with hybrid scheduling
        """
        if self.kind == 'hybrid' and (reserved := sorted(set(environment) & _HYBRID_ENV_KEYS)):
            message = f'Hybrid execution controls these environment keys through executor config: {reserved}'
            raise ExecutionError(message)

    def _build_slots(self) -> list[_DeviceSlot]:
        if self.gpu.devices == 'auto':
            import jax
            try:
                indices = range(len(jax.devices('gpu')))
            except RuntimeError:
                indices = ()
        else:
            indices = self.gpu.devices
        slots = [_DeviceSlot('gpu', index, self.gpu.get_env(index))
                 for index in indices for _ in range(self.gpu.workers_per_device)]
        workers = self.cpu.max_workers if self.cpu.max_workers is not None else os.cpu_count() or 1
        slots.extend(_DeviceSlot('cpu', None, {'JAX_PLATFORMS': 'cpu'}) for _ in range(workers))
        if not slots:
            raise ExecutionError('Hybrid execution has no available GPU or CPU slots.')
        return slots

    @contextmanager
    def _pool(self, environment: Mapping[str, str], slots: Sequence[_DeviceSlot]) -> Iterator[tuple[Executor, int]]:
        if self.kind == 'mpi':
            from romjax.mpi_runtime import _require_mpi, active_executor, active_size
            pool = active_executor()
            if os.environ.get('ROMJAX_MPI_CASE_CHILD') == '1':
                raise ExecutionError('Nested MPI execution is unavailable inside an MPI task or local process child.')
            if pool is None or active_size() < 2:
                _require_mpi()
                raise ExecutionError('MPI execution requires a launch with at least two ranks via mpiexec or srun '
                                     'and a top-level mpi_context().')
            yield pool, self.max_in_flight or active_size() - 1
            return
        if self.kind == 'serial':
            pool, capacity = _SerialExecutor(), 1
        elif self.kind == 'process':
            pool = ProcessPoolExecutor(self.max_workers, mp_context=multiprocessing.get_context('spawn'),
                                       initializer=_configure_process, initargs=(environment,))
            capacity = self.max_in_flight or self.max_workers or 32
        else:
            pool = ThreadPoolExecutor(len(slots) if slots else self.max_workers)
            capacity = min(len(slots), self.max_in_flight or len(slots)) if slots else (
                self.max_in_flight or self.max_workers or 32)
        try:
            yield pool, capacity
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    def execute[P, R](
        self, tasks: Iterable[TaskSpec[P, R]], worker: TaskWorker[P, R], *,
        total_units: int | None = None, progress_bar: Mapping[str, Any] | None = None,
        on_result: Callable[[TaskOutcome[P, R]], ExecutionAction | None] | None = None,
        error_policy: Literal['raise', 'collect'] = 'raise', collect_results: bool = True,
        jax_env: Mapping[str, str] | None = None,
        on_abort: Callable[[], None] | None = None,
    ) -> list[TaskOutcome[P, R]]:
        """Execute tasks using native submit, wait, result, cancel, and shutdown.

        :param tasks: lazy task specifications
        :param worker: domain adapter prepared once before scheduling
        :param total_units: progress total; inferred for task sequences when omitted
        :param progress_bar: routine-provided bar defaults, overridden by this config
        :param on_result: controller-thread outcome handler; STOP halts new submissions
        :param error_policy: raise task exceptions or deliver them as collected outcomes
        :param collect_results: retain outcomes in submission order
        :param on_abort: notify domain workers before draining after an exception or interruption
        :param jax_env: explicit inherited JAX settings overriding the current snapshot
        :return: completed outcomes, or an empty list when collection is disabled
        """
        import jax
        if error_policy not in {'raise', 'collect'}:
            raise ValueError('error_policy must be raise or collect.')
        self.validate_environment(worker.environment)
        if self.kind == 'hybrid' and not worker.supports_hybrid:
            raise ExecutionError('Hybrid execution requires a worker supporting isolated subprocess environments.')
        remote = self.kind in {'process', 'mpi'}
        prepared = worker.prepare(remote=remote)
        environment = {'JAX_ENABLE_X64': str(jax.config.jax_enable_x64).lower()}
        if jax.config.jax_platforms is not None:
            environment['JAX_PLATFORMS'] = jax.config.jax_platforms
        environment.update(jax_env or {})
        slots = self._build_slots() if self.kind == 'hybrid' else []
        if total_units is None and isinstance(tasks, Sequence):
            total_units = sum(task.units for task in tasks)
        options = {**(progress_bar or {}), **dict(self.progress_bar)}
        manual = config_handler(**options).manual if self.show_progress else False
        context = alive_bar(total_units, **options) if self.show_progress else _NullProgress()
        run_id = uuid4().hex
        with self._pool(environment, slots) as (pool, capacity), context as bar:
            return self._execute(pool, capacity, prepared, tasks, run_id, environment, slots, remote,
                                 bar, manual, total_units,
                                 on_result, error_policy, collect_results, on_abort)

    def _pass_worker_count(self, task_count: int) -> int:
        """Resolve native worker defaults and bound local pools by runnable work."""
        if self.kind == 'serial':
            return 1
        if self.kind == 'mpi':
            from romjax.mpi_runtime import active_size

            return max(1, active_size() - 1)
        cores = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
        default = min(32, cores + 4) if self.kind == 'thread' else cores
        budget = self.max_in_flight or self.max_workers or 32
        return max(1, min(self.max_workers or default, budget, task_count))

    @contextmanager
    def _pass_pool(
        self, environment: Mapping[str, str], run_id: str, workers: int,
    ) -> Iterator[tuple[Executor, int]]:
        """Open a native pool with an untimed rendezvous for worker bootstrap."""
        if self.kind == 'mpi':
            # MPI collective bootstrap must not interleave with another pass
            # runner borrowing the same top-level pool.
            with _mpi_pass_lock, self._pool(environment, ()) as (pool, capacity):
                yield pool, capacity
            return
        if self.kind == 'process':
            context = multiprocessing.get_context('spawn')
            barrier = context.Barrier(workers)
            _pass_barriers[run_id] = barrier
            pool = ProcessPoolExecutor(
                workers, mp_context=context, initializer=_configure_pass_process,
                initargs=(environment, run_id, barrier),
            )
        else:
            _pass_barriers[run_id] = Barrier(workers)
            pool = _SerialExecutor() if self.kind == 'serial' else ThreadPoolExecutor(workers)
        try:
            yield pool, self.max_in_flight or workers
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            _pass_barriers.pop(run_id, None)
            for identity in tuple(_pass_workers):
                if identity[0] == run_id:
                    _pass_workers.pop(identity, None)

    @staticmethod
    def _bootstrap_pass_workers(
        pool: Executor, workers: int, bootstrap: _PassBootstrap, context: TaskContext, remote: bool,
    ) -> list[_TaskReturn]:
        """Submit a complete rendezvous, draining futures even after submission fails."""
        futures = []
        try:
            for _ in range(workers):
                futures.append(pool.submit(_run_task, bootstrap, None, context, remote))
            return [future.result() for future in futures]
        except BaseException:
            barrier = _pass_barriers.get(context.run_id)
            if barrier is not None:
                barrier.abort()
            for future in futures:
                future.cancel()
            raise
        finally:
            wait(futures)

    def execute_passes[P, R](
        self, tasks: Sequence[TaskSpec[P, R]], worker: TaskWorker[P, R], *,
        warmup: int = 1, repeat: int = 5,
        progress_bar: Mapping[str, Any] | None = None,
        jax_env: Mapping[str, str] | None = None,
    ) -> list[float]:
        """Time complete passes through reusable tasks on one prepared pool.

        Every worker initializes and performs ``warmup`` untimed passes before
        measurement. Each duration spans submission through completion of all
        tasks, including dispatch and transport. Workers must synchronize any
        asynchronous numerical results before returning. Progress is rendered
        between passes. Results are discarded; ordinary task exceptions raise.

        :param tasks: nonempty sequence replayed unchanged for each pass
        :param worker: worker supporting initialization and repeated execution
        :param warmup: untimed passes on each worker; zero includes first-use compilation in timing
        :param repeat: number of measured passes
        :param progress_bar: progress defaults, overridden by executor options
        :param jax_env: explicit inherited JAX settings
        :return: full-pass wall times in seconds, in repeat order
        :raises ExecutionError: if hybrid execution is requested
        :raises ValueError: if tasks or repetition counts are invalid
        """
        import jax

        if not tasks or warmup < 0 or repeat < 1 or any(task.units < 0 for task in tasks):
            raise ValueError('Passes require tasks, nonnegative units/warmup, and a positive repeat count.')
        if self.kind == 'hybrid':
            raise ExecutionError('Repeated passes do not support hybrid execution.')
        self.validate_environment(worker.environment)
        remote = self.kind in {'process', 'mpi'}
        prepared = worker.prepare(remote=remote)
        environment = {'JAX_ENABLE_X64': str(jax.config.jax_enable_x64).lower()}
        if jax.config.jax_platforms is not None:
            environment['JAX_PLATFORMS'] = jax.config.jax_platforms
        environment.update(jax_env or {})
        workers = self._pass_worker_count(len(tasks))
        units = sum(task.units for task in tasks)
        total_units = units * (workers * warmup + repeat)
        options = {**(progress_bar or {}), **dict(self.progress_bar)}
        manual = config_handler(**options).manual if self.show_progress else False
        progress = alive_bar(total_units, **options) if self.show_progress else _NullProgress()
        run_id = uuid4().hex
        durations: list[float] = []
        with self._pass_pool(environment, run_id, workers) as (pool, capacity), progress as bar:
            # MPI needs one collective task per rank, independently of the
            # normal max_in_flight submission budget.
            bootstrap = _PassBootstrap(prepared, tasks, warmup, self.kind == 'mpi')
            context = TaskContext(run_id, environment)
            initialized = False
            try:
                results = self._bootstrap_pass_workers(pool, workers, bootstrap, context, remote)
                initialized = True
                for result in results:
                    if result.error is not None:
                        raise result.error
                completed = units * workers * warmup
                if warmup and self.show_progress:
                    bar.text = f'Warmup complete ({workers} workers)'
                    fraction = completed / total_units if total_units else 1.
                    bar(fraction if manual else completed)
                dispatcher = _PassDispatch()
                for index in range(repeat):
                    # alive_bar renders on a background thread in interactive
                    # terminals. Pausing stops that renderer as well as deferring
                    # updates, keeping its work outside the measured interval.
                    with bar.pause() if self.show_progress else nullcontext():
                        start = time.perf_counter()
                        self._execute(pool, capacity, dispatcher, tasks, run_id, environment, (), remote,
                                      None, False, units, None, 'raise', False)
                        durations.append(time.perf_counter() - start)
                    completed += units
                    if self.show_progress:
                        bar.text = f'Repeat {index + 1}/{repeat}; {len(tasks)} batches complete'
                        fraction = completed / total_units if total_units else 1.
                        bar(fraction if manual else units)
            finally:
                if self.kind == 'mpi' and initialized:
                    cleanup = _PassBootstrap(None, (), 0, mpi=True, cleanup=True)
                    results = self._bootstrap_pass_workers(pool, workers, cleanup, context, remote)
                    for result in results:
                        if result.error is not None:
                            raise result.error
        return durations

    @staticmethod
    def _execute[P, R](
        pool: Executor, capacity: int, worker: TaskWorker[P, R], tasks: Iterable[TaskSpec[P, R]],
        run_id: str, environment: Mapping[str, str], slots: Sequence[_DeviceSlot], remote: bool,
        bar: Any, manual: bool, total_units: int | None,
        on_result: Callable[[TaskOutcome[P, R]], ExecutionAction | None] | None,
        error_policy: Literal['raise', 'collect'], collect_results: bool,
        on_abort: Callable[[], None] | None = None,
    ) -> list[TaskOutcome[P, R]]:
        source = iter(tasks)
        pending: dict[Future[_TaskReturn], tuple[int, TaskSpec[P, R], TaskContext, _DeviceSlot | None]] = {}
        available = list(slots)
        outcomes: list[tuple[int, TaskOutcome[P, R]]] = []
        sequence = 0
        progress_units = 0
        stopped = exhausted = False

        def handle(index: int, task: TaskSpec[P, R], result: _TaskReturn, context: TaskContext) -> None:
            nonlocal stopped, progress_units
            outcome = TaskOutcome(task, result.value, result.error, result.started_at, result.finished_at,
                                  result.elapsed_seconds, context)
            if collect_results:
                outcomes.append((index, outcome))
            progress_units += task.units
            if bar is not None:
                bar.text = task.label
                if isinstance(bar, _NullProgress):
                    bar()
                elif manual:
                    bar(progress_units / total_units if total_units else 1.)
                else:
                    bar(task.units)
            if result.error is not None and error_policy == 'raise':
                raise result.error
            if on_result is not None and on_result(outcome) == ExecutionAction.STOP:
                stopped = True

        try:
            while True:
                while not stopped and not exhausted and len(pending) < capacity:
                    try:
                        task = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    if task.units < 0:
                        raise ValueError('Task progress units must be nonnegative.')
                    slot = available.pop(0) if slots and task.immediate is None else None
                    context = TaskContext(run_id, {**environment, **(slot.env if slot else {})},
                                          slot.manifest() if slot else None)
                    index, sequence = sequence, sequence + 1
                    if task.immediate is not None:
                        now = datetime.now(timezone.utc).isoformat()
                        handle(index, task, _TaskReturn(task.immediate.value, None, now, now, 0.), context)
                    else:
                        future = pool.submit(_run_task, worker, task.payload, context, remote)
                        pending[future] = (index, task, context, slot)
                if stopped:
                    for future in pending:
                        future.cancel()
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index, task, context, slot = pending.pop(future)
                    if slot is not None:
                        available.append(slot)
                    if not future.cancelled():
                        handle(index, task, future.result(), context)
        except BaseException:
            if on_abort is not None:
                on_abort()
            raise
        finally:
            for future in pending:
                future.cancel()
            if pending:
                wait(pending)
        return [outcome for _, outcome in sorted(outcomes, key=lambda item: item[0])]
