
import pytest

from romjax.study_worker import FeedbackConfig, _FeedbackClient, _FeedbackServer
from romjax.train import TrainingDecision, TrainingReport


def test_feedback_round_trip_and_pruning(tmp_path):
    received = []
    config = FeedbackConfig(poll_interval=0.01, response_timeout=2)
    server = _FeedbackServer(tmp_path, config)
    server.register('trial', lambda r: received.append(r) or 'prune')
    with server:
        client = _FeedbackClient(tmp_path / 'trial', config)
        assert client.observe(TrainingReport(2, {'loss': 1.5})) == TrainingDecision.STOP
        assert client.pruned
    assert received[0].step == 2 and received[0].metrics['loss'] == 1.5


def test_feedback_timeout_and_handler_failure(tmp_path):
    config = FeedbackConfig(poll_interval=0.01, response_timeout=0.1)
    with pytest.raises(TimeoutError):
        _FeedbackClient(tmp_path / 'missing', config).observe(TrainingReport(1, {'loss': 1}))

    def fail(report):
        raise ValueError('invalid score')

    server = _FeedbackServer(tmp_path, config)
    server.register('trial', fail)
    with server:
        with pytest.raises(RuntimeError, match='invalid score'):
            _FeedbackClient(tmp_path / 'trial', config).observe(TrainingReport(1, {'loss': 1}))


def test_cancel_interrupts_client(tmp_path):
    config = FeedbackConfig(poll_interval=0.01, response_timeout=1)
    server = _FeedbackServer(tmp_path, config)
    server.register('trial', lambda r: 'continue')
    server.cancel('trial', 'aborted')
    with pytest.raises(RuntimeError, match='aborted'):
        _FeedbackClient(tmp_path / 'trial', config).observe(TrainingReport(1, {'loss': 1}))


def test_supervisor_terminates_child_when_controller_lease_expires(tmp_path, monkeypatch):
    import subprocess
    import sys

    from romjax.execution import TaskContext
    from romjax.study_worker import TrialLaunchSpec, _StudyWorker
    from romjax.train import TrainingReportConfig

    root = tmp_path / 'trial'
    control = tmp_path / 'control'
    root.mkdir()
    control.mkdir()
    launched = []
    original = subprocess.Popen

    def sleeping_child(command, **kwargs):
        process = original([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(subprocess, 'Popen', sleeping_child)
    result = _StudyWorker().run(TrialLaunchSpec('trial', root, control, TrainingReportConfig(),
                                              FeedbackConfig(poll_interval=0.01, shutdown_grace=0.1)),
                                TaskContext('test'))
    assert result.state == 'failed' and 'Controller lease' in result.error
    assert launched[0].poll() is not None


def test_feedback_duplicate_request_is_idempotent(tmp_path):
    from romjax.study_worker import _Mailbox

    calls = []
    config = FeedbackConfig(poll_interval=0.01, response_timeout=1)
    server = _FeedbackServer(tmp_path, config)
    server.register('trial', lambda r: calls.append(r.step) or 'continue')
    with server:
        client = _FeedbackClient(tmp_path / 'trial', config)
        client.observe(TrainingReport(1, {'loss': 1}))
        request = _Mailbox.read(tmp_path / 'trial' / 'request.json')
        _Mailbox.write(tmp_path / 'trial' / 'request.json', request)
        client.observe(TrainingReport(2, {'loss': 0.5}))
    assert calls == [1, 2]


def test_child_watchdog_exits_during_blocked_loading(tmp_path):
    import json
    import subprocess
    import sys

    control = tmp_path / 'control'
    mailbox = control / 'trial'
    mailbox.mkdir(parents=True)
    launch = tmp_path / 'launch.json'
    launch.write_text(json.dumps({'key': 'trial', 'root': str(tmp_path), 'control': str(mailbox),
                                 'feedback': {'poll_interval': 0.01, 'stale_after': 0.1,
                                              'shutdown_grace': 0.1}, 'reporting': {}}))
    # Run the actual child entry point with loading blocked before the first report.
    script = ('import time, romjax; from pathlib import Path; '
              'from romjax.study_worker import _StudyWorker; '
              'romjax.load = lambda *a: time.sleep(30); '
              f'_StudyWorker.main(Path({str(launch)!r}))')
    process = subprocess.Popen([sys.executable, '-c', script])
    try:
        assert process.wait(timeout=8) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
