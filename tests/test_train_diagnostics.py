import jax.numpy as jnp
import optax
import pytest

from romjax.train import DiagnosticsConfig, Train, TrainingDecision


def loss(params, batch):
    return params['w'] ** 2


def test_diagnostic_config_and_checkpoint_inference():
    config = DiagnosticsConfig(report_interval=10, report_metrics=('loss', 'test_score'), checkpoint_interval=7)
    assert config.report_interval == 10 and config.checkpoint_interval == 7
    with pytest.raises(ValueError):
        DiagnosticsConfig(report_metrics=('loss', 'loss'))
    train = Train(loss=loss, init_params={'w': jnp.array(1.)}, optimizer=optax.sgd(.1),
                  checkpointer={'save_decision_policy': 4})
    assert train.diagnostics.checkpoint_interval == 4


def test_reports_respect_host_gate_and_share_validation():
    validations, reports = [], []

    def test(params):
        validations.append(float(params['w']))
        return params['w'] ** 2

    train = Train(loss=loss, init_params={'w': jnp.array(1.)}, optimizer=optax.sgd(.01), test=test,
                  termination=41, host_interval=4,
                  diagnostics={'show_progress': False, 'test_interval': 6,
                               'report_interval': 10, 'report_metrics': ['test_score']})
    result = train.fit(observer=lambda report: reports.append(report) or TrainingDecision.CONTINUE)
    assert [(r.step, r.final) for r in reports] == [(20, False), (40, False), (41, True)]
    assert len(validations) == 7
    assert validations[-1] == pytest.approx(float(result.params['w']))


def test_final_metrics_without_observer():
    train = Train(loss=loss, init_params={'w': jnp.array(1.)}, optimizer=optax.sgd(.1),
                  termination=3, host_interval=None, diagnostics={'show_progress': False})
    result = train.fit()
    assert result.metrics['loss'] == pytest.approx(.8 ** 6)
    assert not callable(train)


def test_final_only_host_gate_with_observer():
    reports, validations = [], []

    def test(params):
        validations.append(float(params['w']))
        return params['w'] ** 2

    train = Train(loss=loss, init_params={'w': jnp.array(1.)}, optimizer=optax.sgd(.1), test=test,
                  termination=3, host_interval=None,
                  diagnostics={'show_progress': False, 'test_interval': 1, 'report_interval': 1,
                               'report_metrics': ['test_score']})
    result = train.fit(observer=lambda report: reports.append(report) or TrainingDecision.CONTINUE)
    assert [(report.step, report.final) for report in reports] == [(3, True)]
    assert validations == pytest.approx([float(result.params['w'])])


def test_relocated_checkpoint_field_is_rejected_before_overwrite(tmp_path):
    artifact = tmp_path / 'loss.csv'
    artifact.write_text('keep')
    with pytest.raises(ValueError, match='diagnostics.checkpoint_interval'):
        Train(loss=loss, init_params={'w': jnp.array(1.)}, optimizer=optax.sgd(.1), root=tmp_path,
              write_policy='overwrite', checkpoint_interval=2)
    assert artifact.read_text() == 'keep'


def test_stop_reports_final_once_without_revalidating():
    reports, validations = [], []

    def test(params):
        validations.append(float(params['w']))
        return params['w'] ** 2

    train = Train(loss=loss, init_params={'w': jnp.array(1.)}, optimizer=optax.sgd(.1), test=test,
                  termination=9, host_interval=2, diagnostics={'show_progress': False, 'test_interval': 4,
                                                               'report_interval': 4, 'report_metrics': ['test_score']})
    result = train.fit(observer=lambda report: reports.append(report) or TrainingDecision.STOP)
    assert result.step == 4 and result.stop_reason == 'observer'
    assert [(report.step, report.final) for report in reports] == [(4, False), (4, True)]
    assert len(validations) == 2
    assert result.metrics['test_score'] == pytest.approx(.8 ** 8)


@pytest.mark.parametrize('invalid', [
    {'reporting': {'interval': 1}},
    {'loss_interval': 2},
    {'termination_interval': 2},
    {'diagnostics': {'reporting': {'interval': 1}}},
    {'diagnostics': {'report_interval': 0}},
])
def test_invalid_diagnostics_never_overwrite_artifacts(tmp_path, invalid):
    artifact = tmp_path / 'loss.csv'
    artifact.write_text('keep')
    with pytest.raises(ValueError):
        Train(loss=loss, init_params={'w': jnp.array(1.)}, optimizer=optax.sgd(.1), root=tmp_path,
              write_policy='overwrite', **invalid)
    assert artifact.read_text() == 'keep'
