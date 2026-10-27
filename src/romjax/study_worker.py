"""Isolated training children and bounded shared-filesystem study feedback."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any
from uuid import uuid4

from pydantic import ConfigDict, PositiveFloat

from romjax.execution import TaskContext, TaskWorker
from romjax.train import Train, TrainingDecision, TrainingReport, TrainingReportConfig
from romjax.typing import DictModel


class FeedbackConfig(DictModel):
    """Polling, response, lease, and child-shutdown deadlines in seconds."""

    model_config = ConfigDict(extra='forbid')
    poll_interval: PositiveFloat = 0.1
    response_timeout: PositiveFloat = 120
    stale_after: PositiveFloat = 300
    shutdown_grace: PositiveFloat = 10


class _Mailbox:
    @staticmethod
    def write(path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
        try:
            temporary.write_text(json.dumps(value), encoding='utf-8')
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def read(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None

    @staticmethod
    def stale(path: Path, threshold: float) -> bool:
        try:
            return time.time() - path.stat().st_mtime > threshold
        except FileNotFoundError:
            return True


class _FeedbackServer:
    def __init__(self, root: Path, config: FeedbackConfig) -> None:
        self.root, self.config = root, config
        self.root.mkdir(parents=True, exist_ok=True)
        self.handlers: dict[str, Callable] = {}
        self.sequences: dict[str, int] = {}
        self.lock = RLock()
        self.closed = Event()
        self.error: BaseException | None = None
        self.thread = Thread(target=self._serve, daemon=True, name='romjax-study-feedback')

    def register(self, key: str, handler: Callable[[TrainingReport], str]) -> None:
        with self.lock:
            self.handlers[key] = handler
            (self.root / key).mkdir(parents=True, exist_ok=True)

    def unregister(self, key: str) -> None:
        with self.lock:
            self.handlers.pop(key, None)

    def cancel(self, key: str, reason: str) -> None:
        _Mailbox.write(self.root / key / 'cancel.json', {'reason': reason})

    def cancel_all(self, reason: str) -> None:
        with self.lock:
            for key in self.handlers:
                self.cancel(key, reason)

    def _serve(self) -> None:
        try:
            while not self.closed.is_set():
                (self.root / 'controller.lease').touch()
                with self.lock:
                    handlers = list(self.handlers.items())
                for key, handler in handlers:
                    request = _Mailbox.read(self.root / key / 'request.json')
                    if request is None or request['sequence'] == self.sequences.get(key):
                        continue
                    sequence = request['sequence']
                    try:
                        if sequence != self.sequences.get(key, 0) + 1 or request['key'] != key:
                            raise ValueError('Feedback identity or sequence mismatch.')
                        report = TrainingReport(request['step'], request['metrics'], request['final'])
                        decision = handler(report)
                        response = {'sequence': sequence, 'key': key, 'decision': decision}
                    except Exception as exc:
                        response = {'sequence': sequence, 'key': key, 'error': str(exc)}
                    _Mailbox.write(self.root / key / 'response.json', response)
                    self.sequences[key] = sequence
                self.closed.wait(self.config.poll_interval)
        except BaseException as exc:
            self.error = exc
            self.cancel_all(f'Feedback service failed: {exc}')

    def __enter__(self) -> _FeedbackServer:
        (self.root / 'controller.lease').touch()
        self.thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.closed.set()
        self.thread.join()


class _FeedbackClient:
    def __init__(self, root: Path, config: FeedbackConfig) -> None:
        self.root, self.config = root, config
        self.sequence = 0
        self.pruned = False

    def observe(self, report: TrainingReport) -> TrainingDecision:
        self.sequence += 1
        _Mailbox.write(self.root / 'request.json', {
            'key': self.root.name, 'sequence': self.sequence, 'step': report.step,
            'metrics': dict(report.metrics), 'final': report.final,
        })
        deadline = time.monotonic() + self.config.response_timeout
        while time.monotonic() < deadline:
            cancel = _Mailbox.read(self.root / 'cancel.json')
            if cancel is not None:
                raise RuntimeError(cancel['reason'])
            response = _Mailbox.read(self.root / 'response.json')
            if response is not None and response['sequence'] == self.sequence:
                if response['key'] != self.root.name:
                    raise RuntimeError('Feedback identity mismatch.')
                if 'error' in response:
                    raise RuntimeError(response['error'])
                if response['decision'] == 'prune' and not report.final:
                    self.pruned = True
                    return TrainingDecision.STOP
                return TrainingDecision.CONTINUE
            time.sleep(self.config.poll_interval)
        raise TimeoutError('Controller feedback response timed out.')


@dataclass(frozen=True)
class TrialLaunchSpec:
    """Serializable launch configuration; no live Optuna or device objects.

    :param key: unique trial key within the invocation
    :param root: trial artifact directory
    :param control: invocation mailbox directory
    :param reporting: requested training metric configuration
    :param feedback: communication deadlines
    :param quiet: capture child stdout/stderr
    """

    key: str
    root: Path
    control: Path
    reporting: TrainingReportConfig
    feedback: FeedbackConfig
    quiet: bool = True


@dataclass(frozen=True)
class TrialExecutionResult:
    """Structured terminal child outcome with execution identity and diagnostics."""

    key: str
    state: str
    exit_code: int
    error: str | None = None
    device: Mapping[str, Any] | None = None
    step: int | None = None
    stop_reason: str | None = None


class _StudyWorker(TaskWorker):
    supports_hybrid = True

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        self.environment = dict(environment or {})

    def run(self, spec: TrialLaunchSpec, context: TaskContext) -> TrialExecutionResult:
        mailbox = spec.control / spec.key
        launch = mailbox / 'launch.json'
        _Mailbox.write(launch, {'key': spec.key, 'root': str(spec.root), 'control': str(mailbox),
                                'feedback': spec.feedback.model_dump(), 'reporting': spec.reporting.model_dump()})
        environment = {**os.environ, **self.environment, **context.environment, 'ROMJAX_MPI_CASE_CHILD': '1'}
        with ExitStack() as stack:
            streams = {}
            if spec.quiet:
                streams = {key: stack.enter_context((spec.root / f'{key}.log').open('w'))
                           for key in ('stdout', 'stderr')}
            # Start the lease clock at launch; queued tasks cannot become stale.
            (mailbox / 'child.lease').touch()
            process = subprocess.Popen([sys.executable, '-m', 'romjax.study_worker', str(launch)],
                                       env=environment, **streams)
            reason = None
            try:
                while process.poll() is None:
                    cancel = _Mailbox.read(mailbox / 'cancel.json')
                    if cancel is not None:
                        reason = cancel['reason']
                    elif _Mailbox.stale(spec.control / 'controller.lease', spec.feedback.stale_after):
                        reason = 'Controller lease expired.'
                    elif _Mailbox.stale(mailbox / 'child.lease', spec.feedback.stale_after):
                        reason = 'Child lease expired.'
                    if reason:
                        break
                    time.sleep(spec.feedback.poll_interval)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=spec.feedback.shutdown_grace)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            result = _Mailbox.read(mailbox / 'result.json')
            if reason or process.returncode != 0 or result is None:
                return TrialExecutionResult(spec.key, 'failed', process.returncode or 1,
                                            reason or (result or {}).get('error') or 'Child failed or result missing.',
                                            context.device)
            if result.get('key') != spec.key or result.get('state') not in ('complete', 'pruned', 'failed'):
                return TrialExecutionResult(spec.key, 'failed', 1,
                                            'Invalid child result identity/state.', context.device)
            return TrialExecutionResult(spec.key, result['state'], process.returncode, result.get('error'),
                                        context.device, result.get('step'), result.get('stop_reason'))

    @staticmethod
    def main(launch_path: Path) -> int:
        import romjax

        launch = _Mailbox.read(launch_path)
        root = Path(launch['control'])
        feedback = FeedbackConfig.model_validate(launch['feedback'])
        client = _FeedbackClient(root, feedback)
        stop = Event()

        def heartbeat() -> None:
            while not stop.is_set():
                # This thread survives supervisor/controller loss while the main
                # thread is blocked in loading, compilation, or a long JAX call.
                if _Mailbox.stale(root.parent / 'controller.lease', feedback.stale_after):
                    os._exit(1)
                (root / 'child.lease').touch()
                stop.wait(min(feedback.poll_interval, feedback.stale_after / 3))

        thread = Thread(target=heartbeat, daemon=True)
        thread.start()
        code = 0
        try:
            routine = romjax.load(Path(launch['root']) / 'case.yml')
            if not isinstance(routine, Train):
                raise TypeError('OptunaStudy base must construct a Train routine.')
            result = routine.fit(observer=client.observe,
                                 reporting=TrainingReportConfig.model_validate(launch['reporting']))
            outcome = {'key': launch['key'], 'state': 'pruned' if client.pruned else 'complete',
                       'step': result.step, 'metrics': dict(result.metrics), 'stop_reason': result.stop_reason}
        except BaseException as exc:
            outcome = {'key': launch['key'], 'state': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
            traceback.print_exc()
            code = 1
        finally:
            stop.set()
            thread.join()
        _Mailbox.write(root / 'result.json', outcome)
        return code


if __name__ == '__main__':
    raise SystemExit(_StudyWorker.main(Path(sys.argv[1])))
