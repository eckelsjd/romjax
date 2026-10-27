from pathlib import Path

import pytest

try:
    import optuna
except ImportError:
    pytest.skip('requires the optuna extra', allow_module_level=True)

import romjax
from romjax.study import MetricConfig, OptunaStudy, ParameterConfig
from romjax.train import TrainingReport


def test_parameter_sampling_preserves_categories():
    trial = optuna.trial.FixedTrial({'label': 'off', 'rate': 0.01})
    spec = ParameterConfig(name='label', path=['optimizer'], distribution={'kind': 'categorical', 'choices': ['off']},
                           values={'off': None})
    assignment = spec.sample(trial)
    assert assignment.value is None and assignment.path == ('optimizer',)
    rate = ParameterConfig(name='rate', path=['optimizer', 'kwargs', 'learning_rate'],
                           distribution={'kind': 'float', 'low': 1e-4, 'high': 0.1, 'log': True})
    assert rate.sample(trial).value == 0.01


def test_study_yaml_defers_storage_and_base(tmp_path):
    journal = tmp_path / 'not-created' / 'journal.log'
    config = romjax.load(f'''!romx:OptunaStudy
root: {tmp_path / 'output'}
base: {tmp_path / 'missing.yml'}
override: []
sampler: {{name: optuna.samplers.TPESampler, kwargs: {{seed: 42}}}}
pruner:
  name: optuna.pruners.PatientPruner
  kwargs:
    wrapped_pruner: {{name: optuna.pruners.MedianPruner}}
    patience: 2
storage:
  name: optuna.storages.JournalStorage
  kwargs:
    log_storage:
      name: optuna.storages.journal.JournalFileBackend
      kwargs: {{file_path: {journal}}}
''')
    assert isinstance(config.sampler, optuna.samplers.TPESampler)
    assert isinstance(config.pruner, optuna.pruners.PatientPruner)
    assert not journal.exists() and not config.root.exists()
    restored = romjax.load(romjax.YamlLoader.dump(config, _preserve_model_types=True))
    assert isinstance(restored, OptunaStudy)


def test_invalid_distribution_and_conflicts(tmp_path):
    with pytest.raises(ValueError):
        ParameterConfig(name='x', path=['x'], distribution={'kind': 'float', 'low': 0, 'high': 1, 'log': True})
    with pytest.raises(ValueError, match='overlap'):
        OptunaStudy(root=tmp_path, base=Path('x.yml'), override=[
            {'name': 'x', 'path': ['root'], 'distribution': {'kind': 'categorical', 'choices': ['x']}}])


def test_metric_reduction_invalid_and_direction():
    reducer = MetricConfig(name='test_score', reduction='best').reducer('maximize')
    assert reducer.observe(TrainingReport(1, {'test_score': 2.0})) == 2.0
    assert reducer.observe(TrainingReport(2, {'test_score': 1.0})) == 2.0
    assert reducer.final_value() == 2.0
    with pytest.raises(ValueError):
        reducer.observe(TrainingReport(3, {'test_score': float('nan')}))


def test_rolling_reducer_replaces_final_step_and_skips_invalid():
    reducer = MetricConfig(reduction='mean', window=2, invalid='skip').reducer('minimize')
    assert reducer.observe(TrainingReport(1, {'loss': 3.0})) == 3.0
    assert reducer.observe(TrainingReport(2, {'loss': 1.0})) == 2.0
    assert reducer.observe(TrainingReport(2, {'loss': 5.0}, final=True)) == 4.0
    assert reducer.observe(TrainingReport(3, {'loss': float('nan')})) is None
    assert reducer.final_value() == 4.0
    with pytest.raises(ValueError):
        MetricConfig(invalid='skip').reducer('minimize').final_value()


@pytest.fixture
def base_train(tmp_path):
    path = tmp_path / 'base.yml'
    path.write_text('''!romx:Train
root: unused
loss: !!python/name:tests.study_helpers.loss
init_params: !pd:tests.study_helpers.Initial {}
optimizer: {name: optax.sgd, kwargs: {learning_rate: 0.1}}
test: !!python/name:tests.study_helpers.score
termination: {max_steps: 2}
diagnostics: {show_progress: false, report_interval: 1, report_metrics: [loss, test_score]}
''')
    return path


@pytest.mark.parametrize('executor', ['serial', {'kind': 'process', 'max_workers': 2}])
def test_study_runs_and_reuses_trials(base_train, tmp_path, executor):
    config = OptunaStudy(root=tmp_path / 'study', base=base_train, n_trials=2,
                         executor=executor, jax_platforms='cpu',
                         metric={'name': 'test_score'},
                         override=[{'name': 'lr', 'path': ['optimizer', 'kwargs', 'learning_rate'],
                                    'distribution': {'kind': 'float', 'low': 0.05, 'high': 0.15}}])
    config.executor.show_progress = False
    assert config.run() == 0
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(config.root / 'study.journal')))
    study = optuna.load_study(study_name=config.study_name, storage=storage)
    assert len(study.trials) == 2
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    for trial in study.trials:
        assert trial.value == pytest.approx((1 - 2 * trial.params['lr']) ** 4, rel=1e-5)
    config.n_trials = 1
    assert config.run() == 0
    assert len(study.get_trials()) == 3


def test_pruned_trials_are_not_failures(base_train, tmp_path):
    from tests.study_helpers import AlwaysPrune

    config = OptunaStudy(root=tmp_path / 'prune', base=base_train, n_trials=1, storage=None,
                         pruner=AlwaysPrune(), jax_platforms='cpu',
                         executor={'show_progress': False})
    assert config.run() == 0
    import yaml
    manifest = yaml.safe_load((config.root / 'study_manifest.yml').read_text())
    assert manifest['trials'][0]['state'] == 'PRUNED'


def test_final_only_training_without_diagnostics(base_train, tmp_path):
    import yaml

    from tests.study_helpers import AlwaysPrune

    base_train.write_text(base_train.read_text().split('diagnostics:')[0])
    config = OptunaStudy(root=tmp_path / 'final-only', base=base_train, n_trials=1, storage=None,
                         pruner=AlwaysPrune(), jax_platforms='cpu', executor={'show_progress': False})
    assert config.run() == 0
    case = (config.root / 'trials' / 'trial_00000' / 'case.yml').read_text()
    assert 'diagnostics:' not in case
    rows = yaml.safe_load((config.root / 'study_manifest.yml').read_text())['trials']
    assert rows[0]['state'] == 'COMPLETE'


def test_report_interval_removed_from_metric():
    with pytest.raises(ValueError):
        MetricConfig(report_interval=1)


def test_metric_missing_observation_policy():
    report = TrainingReport(1, {'loss': 1.0})
    with pytest.raises(ValueError, match='Missing or nonfinite'):
        MetricConfig(name='test_score').reducer('minimize').observe(report)
    reducer = MetricConfig(name='test_score', invalid='skip').reducer('minimize')
    assert reducer.observe(report) is None
    assert reducer.observe(TrainingReport(2, {'test_score': 0.5}, final=True)) == 0.5


def test_reuse_rejects_changed_config_and_overwrite_resets(base_train, tmp_path):
    config = OptunaStudy(root=tmp_path / 'reuse', base=base_train, n_trials=1, jax_platforms='cpu',
                         executor={'show_progress': False})
    assert config.run() == 0
    config.metric = MetricConfig(reduction='best')
    with pytest.raises(ValueError, match='fingerprint'):
        config.run()
    config.write_policy = 'overwrite'
    assert config.run() == 0
    archives = list((config.root / 'archive').glob('*/trials/trial_00000/case.yml'))
    assert len(archives) == 1
    import yaml
    manifest = yaml.safe_load((config.root / 'study_manifest.yml').read_text())
    assert len(manifest['trials']) == 1


@pytest.mark.parametrize('direction,values,best', [
    ('minimize', [2.0, 1.0, 3.0], 1),
    ('maximize', [2.0, 3.0, 1.0], 1),
    ('minimize', [1.0, 1.0, 2.0], 0),
])
def test_best_copy_reconciliation(tmp_path, direction, values, best):
    from romjax.study import _StudyCoordinator

    config = OptunaStudy(root=tmp_path, base=Path('unused'), direction=direction)
    coordinator = _StudyCoordinator(config)
    coordinator.study = optuna.create_study(direction=direction)
    for number, value in enumerate(values):
        trial = coordinator.study.ask()
        root = tmp_path / 'trials' / f'trial_{number:05d}'
        (root / 'checkpoint').mkdir(parents=True)
        (root / 'checkpoint' / 'data').write_text(str(number))
        (root / 'case.yml').write_text(str(number))
        coordinator.study.tell(trial, value)
        coordinator._manifest()
        coordinator._update_best()
    assert (tmp_path / 'best' / 'checkpoint' / 'data').read_text() == str(best)
    import shutil
    shutil.rmtree(tmp_path / 'best')
    coordinator._update_best()
    assert (tmp_path / 'best' / 'case.yml').read_text() == str(best)


def test_best_copy_failure_preserves_previous_copy(tmp_path, monkeypatch):
    import shutil

    from romjax.study import _StudyCoordinator

    coordinator = _StudyCoordinator(OptunaStudy(root=tmp_path, base=Path('unused')))
    coordinator.study = optuna.create_study()
    for number, value in enumerate((2.0, 1.0)):
        trial = coordinator.study.ask()
        root = tmp_path / 'trials' / f'trial_{number:05d}'
        root.mkdir(parents=True)
        (root / 'case.yml').write_text(str(number))
        coordinator.study.tell(trial, value)
        coordinator._manifest()
        if number == 0:
            coordinator._update_best()
    with monkeypatch.context() as patch:
        patch.setattr(shutil, 'copytree', lambda *args: (_ for _ in ()).throw(OSError('copy failed')))
        with pytest.raises(OSError, match='copy failed'):
            coordinator._update_best()
    assert (tmp_path / 'best' / 'case.yml').read_text() == '0'
    coordinator._update_best()
    assert (tmp_path / 'best' / 'case.yml').read_text() == '1'


def test_grid_sampler_and_callback_stop(base_train, tmp_path):
    from tests.study_helpers import stop_after_first

    for callbacks in ((), (stop_after_first,)):
        root = tmp_path / ('callback' if callbacks else 'grid')
        config = OptunaStudy(root=root, base=base_train, n_trials=5, jax_platforms='cpu',
                             sampler={'name': 'optuna.samplers.GridSampler',
                                      'kwargs': {'search_space': {'lr': [0.1, 0.2]}}},
                             callbacks=callbacks, executor={'show_progress': False},
                             override=[{'name': 'lr', 'path': ['optimizer', 'kwargs', 'learning_rate'],
                                        'distribution': {'kind': 'categorical', 'choices': [0.1, 0.2]}}])
        assert config.run() == 0
        import yaml
        rows = yaml.safe_load((root / 'study_manifest.yml').read_text())['trials']
        assert len(rows) == (1 if callbacks else 2)
        assert all(row['state'] == 'COMPLETE' for row in rows)


def test_all_artifacts_and_best_copy(base_train, tmp_path):
    config = OptunaStudy(root=tmp_path / 'retention', base=base_train, n_trials=2, jax_platforms='cpu',
                         executor={'show_progress': False})
    assert config.run() == 0
    import yaml
    rows = yaml.safe_load((config.root / 'study_manifest.yml').read_text())['trials']
    assert all('retained' not in row for row in rows)
    assert all((config.root / 'trials' / f'trial_{row["number"]:05d}' / 'case.yml').exists() for row in rows)
    assert (config.root / 'best' / 'case.yml').exists()
    assert (config.root / 'best' / 'trial.yml').exists()


@pytest.mark.parametrize('field', ['save_policy', 'retain_unsuccessful', 'final_metric'])
def test_removed_options_are_rejected(base_train, tmp_path, field):
    with pytest.raises(ValueError):
        OptunaStudy(root=tmp_path / 'removed', base=base_train, **{field: 'all'})


def test_failure_is_terminal_and_exit_policy_is_configurable(base_train, tmp_path):
    base_train.write_text(base_train.read_text().replace('tests.study_helpers.loss', 'tests.study_helpers.broken_loss'))
    config = OptunaStudy(root=tmp_path / 'failure', base=base_train, n_trials=1, jax_platforms='cpu',
                         executor={'show_progress': False})
    assert config.run() == 1
    config.exit_policy = 'ignore'
    assert config.run() == 0
    import yaml
    rows = yaml.safe_load((config.root / 'study_manifest.yml').read_text())['trials']
    assert all(row['state'] == 'FAIL' and row['user_attrs']['romjax_error'] for row in rows)


def test_callback_failure_reconciles_inflight_trials(base_train, tmp_path):
    from tests.study_helpers import broken_callback

    config = OptunaStudy(root=tmp_path / 'abort', base=base_train, n_trials=3, jax_platforms='cpu',
                         callbacks=(broken_callback,),
                         executor={'kind': 'process', 'max_workers': 2, 'max_in_flight': 2, 'show_progress': False})
    with pytest.raises(RuntimeError, match='callback failed'):
        config.run()
    import yaml
    rows = yaml.safe_load((config.root / 'study_manifest.yml').read_text())['trials']
    assert len(rows) == 2
    assert all(row['state'] in ('COMPLETE', 'FAIL') for row in rows)


def test_recovery_fails_owned_orphan_and_preserves_other_studies(base_train, tmp_path):
    config = OptunaStudy(root=tmp_path / 'recover', base=base_train, n_trials=1, jax_platforms='cpu',
                         executor={'show_progress': False})
    assert config.run() == 0
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(str(config.root / 'study.journal')))
    study = optuna.load_study(study_name=config.study_name, storage=storage)
    orphan = study.ask()
    orphan.set_user_attr('romjax_root', str(config.root))
    orphan.set_user_attr('romjax_invocation', 'dead')
    other = optuna.create_study(study_name='other', storage=storage)
    other.ask()
    assert config.run() == 0
    assert study.get_trials()[1].state == optuna.trial.TrialState.FAIL
    assert other.get_trials()[0].state == optuna.trial.TrialState.RUNNING


def test_runtime_assignment_failure_leaves_no_running_trial(base_train, tmp_path):
    config = OptunaStudy(root=tmp_path / 'badpath', base=base_train, n_trials=1, jax_platforms='cpu',
                         override=[{'name': 'x', 'path': ['missing', 'field'],
                                    'distribution': {'kind': 'categorical', 'choices': ['a']}}],
                         executor={'show_progress': False})
    with pytest.raises(ValueError, match='ancestor'):
        config.run()
    import yaml
    rows = yaml.safe_load((config.root / 'study_manifest.yml').read_text())['trials']
    assert [r['state'] for r in rows] == ['FAIL']


def test_hybrid_cpu_and_sqlite_storage(base_train, tmp_path):
    config = OptunaStudy(root=tmp_path / 'hybrid', base=base_train, n_trials=1,
                         storage={'name': 'optuna.storages.RDBStorage',
                                  'kwargs': {'url': f'sqlite:///{tmp_path / "study.db"}'}},
                         executor={'kind': 'hybrid', 'gpu': {'devices': []},
                                   'cpu': {'max_workers': 1}, 'show_progress': False})
    assert config.run() == 0
    import yaml
    row = yaml.safe_load((config.root / 'study_manifest.yml').read_text())['trials'][0]
    assert row['state'] == 'COMPLETE'
    assert row['user_attrs']['romjax_device']['kind'] == 'cpu'


def test_external_rdb_when_configured(base_train, tmp_path):
    import os
    from uuid import uuid4

    url = os.environ.get('ROMJAX_TEST_OPTUNA_RDB')
    if not url:
        pytest.skip('set ROMJAX_TEST_OPTUNA_RDB to run external RDB integration')
    config = OptunaStudy(root=tmp_path / 'rdb', base=base_train, n_trials=1,
                         study_name=f'romjax-test-{uuid4().hex}', storage=url, jax_platforms='cpu',
                         executor={'show_progress': False})
    try:
        assert config.run() == 0
    finally:
        optuna.delete_study(study_name=config.study_name, storage=url)


def test_root_cannot_be_reassigned_to_another_study(base_train, tmp_path):
    from romjax.configuration import YamlTemplate
    from romjax.study import _StudyCoordinator

    config = OptunaStudy(root=tmp_path / 'owned', base=base_train)
    coordinator = _StudyCoordinator(config)
    coordinator.template = YamlTemplate.from_source(base_train)
    with coordinator._ownership():
        coordinator._open_study()
        coordinator._manifest()
    artifact = config.root / 'trials' / 'checkpoint'
    artifact.parent.mkdir()
    artifact.write_text('preserve')
    config.study_name = 'another'
    config.write_policy = 'overwrite'
    with pytest.raises(ValueError, match='identity|another study'):
        with coordinator._ownership():
            coordinator._open_study()
    assert artifact.read_text() == 'preserve'


def test_assignment_yaml_contents_affect_fingerprint(base_train, tmp_path):
    from romjax.configuration import YamlTemplate
    from romjax.study import _StudyCoordinator

    graph = tmp_path / 'graph.yml'
    graph.write_text('width: 8')
    config = OptunaStudy(root=tmp_path / 'study', base=base_train, override=[
        {'name': 'graph', 'path': ['graph'], 'distribution': {'kind': 'categorical', 'choices': ['a']},
         'values': {'a': str(graph)}}])
    coordinator = _StudyCoordinator(config)
    coordinator.template = YamlTemplate.from_source(base_train)
    before = coordinator._fingerprint()
    graph.write_text('width: 16')
    assert coordinator._fingerprint() != before


def test_queued_cancellations_count_as_failures(base_train, tmp_path):
    from tests.study_helpers import stop_after_first

    observed = []

    def record(study, trial):
        observed.append(trial.number)
        stop_after_first(study, trial)

    config = OptunaStudy(root=tmp_path / 'queued', base=base_train, n_trials=5, jax_platforms='cpu',
                         callbacks=(record,), executor={'kind': 'thread', 'max_workers': 1,
                                                       'max_in_flight': 5, 'show_progress': False})
    assert config.run() == 1
    import yaml
    rows = yaml.safe_load((config.root / 'study_manifest.yml').read_text())['trials']
    assert any(row['state'] == 'FAIL' for row in rows)
    assert sorted(observed) == [row['number'] for row in rows]


def test_grpc_storage_when_configured(base_train, tmp_path):
    import os
    from uuid import uuid4

    host = os.environ.get('ROMJAX_TEST_OPTUNA_GRPC_HOST')
    if not host:
        pytest.skip('set ROMJAX_TEST_OPTUNA_GRPC_HOST to run gRPC storage integration')
    pytest.importorskip('grpc')
    config = OptunaStudy(root=tmp_path / 'grpc', base=base_train, n_trials=1,
                         study_name=f'romjax-test-{uuid4().hex}', jax_platforms='cpu',
                         storage={'name': 'optuna.storages.GrpcStorageProxy',
                                  'kwargs': {'host': host,
                                             'port': int(os.environ.get('ROMJAX_TEST_OPTUNA_GRPC_PORT', '13000'))}},
                         executor={'show_progress': False})
    storage = config.storage.build()
    try:
        assert config.run() == 0
        study = optuna.load_study(study_name=config.study_name, storage=storage)
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    finally:
        optuna.delete_study(study_name=config.study_name, storage=storage)


def test_generated_case_preserves_train_reporting_and_host_gate(base_train, tmp_path):
    import hashlib

    import yaml

    base_train.write_text(base_train.read_text().replace('diagnostics:', 'host_interval: 2\ndiagnostics:'))
    config = OptunaStudy(root=tmp_path / 'configured', base=base_train, n_trials=1, jax_platforms='cpu',
                         metric={'name': 'test_score'},
                         executor={'show_progress': False})
    assert config.run() == 0
    case = config.root / 'trials' / 'trial_00000' / 'case.yml'
    node = yaml.compose(case.read_text())
    fields = {key.value: value for key, value in node.value}
    diagnostics = {key.value: value for key, value in fields['diagnostics'].value}
    assert fields['host_interval'].value == '2'
    assert diagnostics['report_interval'].value == '1'
    assert [item.value for item in diagnostics['report_metrics'].value] == ['loss', 'test_score']
    assert diagnostics['show_progress'].value == 'false'
    manifest = yaml.safe_load((config.root / 'study_manifest.yml').read_text())
    assert manifest['trials'][0]['user_attrs']['romjax_config_sha256'] == hashlib.sha256(case.read_bytes()).hexdigest()
