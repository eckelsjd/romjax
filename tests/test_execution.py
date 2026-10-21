"""Shared driver contracts using native concurrent futures."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event

import pytest

from romjax.execution import (
    ExecutionAction,
    ExecutorConfig,
    ImmediateResult,
    TaskSpec,
    TaskWorker,
    _SerialExecutor,
)


class SquareWorker(TaskWorker):
    def run(self, payload, context):
        return payload * payload


class FailingWorker(TaskWorker):
    def run(self, payload, context):
        raise ValueError('task failed')


class PrecisionWorker(TaskWorker):
    def run(self, payload, context):
        import jax
        return jax.config.jax_enable_x64


class NestedMpiWorker(TaskWorker):
    def run(self, payload, context):
        ExecutorConfig(kind='mpi', show_progress=False).execute([], SquareWorker())


def tasks(count):
    return [TaskSpec(str(i), f'task {i}', i) for i in range(count)]


@pytest.mark.parametrize('kind', ['serial', 'thread', 'process'])
def test_execute_uses_native_executors(kind):
    config = ExecutorConfig(kind=kind, show_progress=False, max_workers=2 if kind != 'serial' else None)
    outcomes = config.execute(tasks(5), SquareWorker())
    assert [outcome.value for outcome in outcomes] == [i * i for i in range(5)]
    assert all(outcome.error is None and outcome.elapsed_seconds >= 0 for outcome in outcomes)


def test_serial_adapter_returns_native_future():
    with _SerialExecutor() as pool:
        future = pool.submit(int, 'bad')
        assert isinstance(future, Future)
        with pytest.raises(ValueError):
            future.result()


def test_immediate_none_and_serial_callback_order():
    observed = []
    specs = [TaskSpec('a', 'a', 3), TaskSpec('b', 'b', None, immediate=ImmediateResult(None))]
    outcomes = ExecutorConfig(show_progress=False).execute(specs, SquareWorker(), on_result=observed.append)
    assert [outcome.value for outcome in outcomes] == [9, None]
    assert observed == outcomes


def test_stop_bounds_consumption_and_drains_running_results():
    started = Event()
    finished = Event()
    consumed = []
    seen = []

    class Worker(TaskWorker):
        def run(self, payload, context):
            if payload == 0:
                started.wait(5)
            else:
                started.set()
                finished.set()
            return payload

    def source():
        for i in range(10):
            consumed.append(i)
            yield TaskSpec(str(i), str(i), i)

    def handle(outcome):
        seen.append(outcome.value)
        return ExecutionAction.STOP

    config = ExecutorConfig(kind='thread', max_workers=2, show_progress=False)
    outcomes = config.execute(source(), Worker(), on_result=handle)
    assert consumed == [0, 1]
    assert finished.is_set()
    assert sorted(seen) == [0, 1]
    assert [outcome.value for outcome in outcomes] == [0, 1]


def test_worker_error_policy():
    config = ExecutorConfig(show_progress=False)
    with pytest.raises(ValueError, match='task failed'):
        config.execute(tasks(1), FailingWorker())
    results = config.execute(tasks(2), FailingWorker(), error_policy='collect')
    assert all(isinstance(result.error, ValueError) for result in results)


def test_callback_failure_stops_further_callbacks():
    seen = []

    def handle(outcome):
        seen.append(outcome)
        raise RuntimeError('callback failed')

    config = ExecutorConfig(kind='thread', max_workers=2, show_progress=False)
    with pytest.raises(RuntimeError, match='callback failed'):
        config.execute(tasks(10), SquareWorker(), on_result=handle)
    assert len(seen) == 1


def test_mpi_borrow_does_not_close_pool(monkeypatch):
    from romjax import mpi_runtime

    with ThreadPoolExecutor(2) as pool:
        monkeypatch.setattr(mpi_runtime, '_active_executor', pool)
        monkeypatch.setattr(mpi_runtime, '_active_size', 3)
        results = ExecutorConfig(kind='mpi', show_progress=False).execute(tasks(2), SquareWorker())
        assert [outcome.value for outcome in results] == [0, 1]
        assert pool.submit(int, '3').result() == 3


def test_process_inherits_effective_jax_precision():
    import jax

    original = jax.config.jax_enable_x64
    try:
        jax.config.update('jax_enable_x64', True)
        config = ExecutorConfig(kind='process', max_workers=1, show_progress=False)
        results = config.execute(tasks(1), PrecisionWorker())
        assert results[0].value is True
    finally:
        jax.config.update('jax_enable_x64', original)


def test_local_process_rejects_nested_mpi():
    with pytest.raises(Exception, match='Nested MPI execution'):
        ExecutorConfig(kind='process', max_workers=1, show_progress=False).execute(tasks(1), NestedMpiWorker())


def test_progress_options_and_nested_bars_are_not_suppressed(monkeypatch):
    opened = []
    updates = []

    class Bar:
        text = ''
        def __call__(self, units=1):
            updates.append((self.text, units))

    @contextmanager
    def factory(total, **kwargs):
        opened.append((total, kwargs))
        yield Bar()

    monkeypatch.setattr('romjax.execution.alive_bar', factory)
    monkeypatch.setenv('ROMJAX_MPI_CASE_CHILD', '1')
    child = ExecutorConfig(progress_bar={'title': 'child', 'refresh_secs': 0.2})
    parent = ExecutorConfig(progress_bar={'title': 'parent'})

    class Parent(TaskWorker):
        def run(self, payload, context):
            child.execute([TaskSpec('child', 'child label', 2, units=3)], SquareWorker(), total_units=3)

    parent.execute(tasks(1), Parent())
    assert opened == [(1, {'title': 'parent'}), (3, {'title': 'child', 'refresh_secs': 0.2})]
    assert updates == [('child label', 3), ('task 0', 1)]
    opened.clear()
    parent.show_progress = False
    parent.execute(tasks(1), Parent())
    assert opened == [(3, {'title': 'child', 'refresh_secs': 0.2})]


def test_hybrid_requires_subprocess_capability():
    config = ExecutorConfig(kind='hybrid', gpu={'devices': []}, cpu={'max_workers': 1}, show_progress=False)
    with pytest.raises(Exception, match='isolated subprocess'):
        config.execute(tasks(1), SquareWorker())


def test_hybrid_replenishes_slots_and_handles_immediate_results():
    class Worker(TaskWorker):
        supports_hybrid = True
        def run(self, payload, context):
            return context.device, context.environment['JAX_PLATFORMS']

    seen = []
    specs = [TaskSpec('reuse', 'reuse', None, immediate=ImmediateResult(None)), *tasks(5)]
    config = ExecutorConfig(kind='hybrid', gpu={'devices': []}, cpu={'max_workers': 2}, show_progress=False)
    results = config.execute(specs, Worker(), on_result=seen.append)
    assert len(seen) == 6 and results[0].value is None
    assert all(outcome.value == ({'kind': 'cpu', 'index': None}, 'cpu') for outcome in results[1:])


def test_task_source_failure_drains_owned_work():
    finished = Event()

    class Worker(TaskWorker):
        def run(self, payload, context):
            finished.set()

    def source():
        yield TaskSpec('one', 'one', None)
        raise RuntimeError('source failed')

    config = ExecutorConfig(kind='thread', max_workers=2, show_progress=False)
    with pytest.raises(RuntimeError, match='source failed'):
        config.execute(source(), Worker())
    # Either canceled before starting, or finished before execute propagated.
    assert not any(thread.name.startswith('ThreadPoolExecutor') and thread.is_alive()
                   for thread in __import__('threading').enumerate())


def test_global_manual_progress_setting_is_honored(monkeypatch):
    from alive_progress import config_handler

    values = []

    @contextmanager
    def factory(total, **kwargs):
        class Bar:
            text = ''
            def __call__(self, value):
                values.append(value)
        assert kwargs == {'title': 'configured title'}
        yield Bar()

    monkeypatch.setattr('romjax.execution.alive_bar', factory)
    original = config_handler().manual
    try:
        config_handler.set_global(manual=True)
        ExecutorConfig(progress_bar={'title': 'configured title'}).execute(tasks(2), SquareWorker())
        assert values == [.5, 1.]
    finally:
        config_handler.set_global(manual=original)


def test_native_process_serialization_failure_is_fatal_even_when_collecting():
    worker = SquareWorker()
    worker.nonserializable = lambda: None
    config = ExecutorConfig(kind='process', max_workers=1, show_progress=False)
    with pytest.raises(Exception, match='pickle|pickl'):
        config.execute(tasks(1), worker, error_policy='collect')


def test_keyboard_interrupt_is_not_collected_and_pool_is_drained():
    started = Event()
    finished = Event()

    class Worker(TaskWorker):
        def run(self, payload, context):
            if payload == 0:
                started.wait(5)
                raise KeyboardInterrupt()
            started.set()
            finished.set()

    config = ExecutorConfig(kind='thread', max_workers=2, show_progress=False)
    with pytest.raises(KeyboardInterrupt):
        config.execute(tasks(2), Worker(), error_policy='collect')
    assert finished.is_set()


def test_submission_failure_leaves_borrowed_mpi_pool_usable(monkeypatch):
    from romjax import execution, mpi_runtime

    class FailOncePool(ThreadPoolExecutor):
        submissions = 0
        def submit(self, *args, **kwargs):
            self.submissions += 1
            if self.submissions == 2:
                raise RuntimeError('submission failed')
            return super().submit(*args, **kwargs)

    # This pool stands in for transport only; real MPI bootstrap is covered by mpiexec tests.
    native_run = execution._run_task
    monkeypatch.setattr(execution, '_run_task', lambda worker, payload, context, remote:
                        native_run(worker, payload, context, False))
    with FailOncePool(2) as pool:
        monkeypatch.setattr(mpi_runtime, '_active_executor', pool)
        monkeypatch.setattr(mpi_runtime, '_active_size', 3)
        config = ExecutorConfig(kind='mpi', show_progress=False)
        with pytest.raises(RuntimeError, match='submission failed'):
            config.execute(tasks(5), SquareWorker(), error_policy='collect')
        assert pool.submit(int, '4').result() == 4
