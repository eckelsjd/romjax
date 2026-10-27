"""Adaptive YAML-configured Optuna studies using romjax executors."""
from __future__ import annotations

import gc
import hashlib
import math
import shutil
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from threading import RLock
from typing import Annotated, Any, Literal
from uuid import uuid4

try:
    import optuna
except ImportError as exc:
    raise ImportError('OptunaStudy requires the optuna extra: uv sync --extra optuna') from exc
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

import romjax
from romjax.configuration import ConfigAssignment, YamlTemplate
from romjax.execution import ExecutionAction, ExecutorConfig, TaskSpec
from romjax.grid_search import MetricCallable, _SavePolicy
from romjax.routine import Routine
from romjax.study_worker import FeedbackConfig, TrialLaunchSpec, _FeedbackServer, _Mailbox, _StudyWorker
from romjax.train import TrainingReport
from romjax.typing import CallableModel, DeferredThirdPartySpec, DictModel, ThirdPartyType, require_type


class FloatParameter(DictModel):
    """Arguments for Optuna's suggest_float, including logarithmic or stepped ranges."""

    model_config = ConfigDict(extra='forbid')
    kind: Literal['float'] = 'float'
    low: float
    high: float
    step: float | None = None
    log: bool = False

    @model_validator(mode='after')
    def _validate_distribution(self) -> FloatParameter:
        optuna.distributions.FloatDistribution(**self.model_dump(exclude={'kind'}))
        return self


class IntParameter(DictModel):
    """Arguments for Optuna's suggest_int."""

    model_config = ConfigDict(extra='forbid')
    kind: Literal['int'] = 'int'
    low: int
    high: int
    step: int = 1
    log: bool = False

    @model_validator(mode='after')
    def _validate_distribution(self) -> IntParameter:
        optuna.distributions.IntDistribution(**self.model_dump(exclude={'kind'}))
        return self


class CategoricalParameter(DictModel):
    """Native Optuna categorical primitives, preserved without string coercion."""

    model_config = ConfigDict(extra='forbid')
    kind: Literal['categorical'] = 'categorical'
    choices: tuple[Any, ...]

    @field_validator('choices')
    @classmethod
    def _choices(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        if not value or any(type(x) not in (str, int, float, bool, type(None)) for x in value):
            raise ValueError('Categorical choices must be nonempty native Optuna primitives.')
        if any(isinstance(x, float) and not math.isfinite(x) for x in value):
            raise ValueError('Categorical numeric choices must be finite.')
        return value


class ParameterConfig(ConfigAssignment):
    """A named distribution mapped to a raw YAML path.

    :param name: stable Optuna parameter name
    :param distribution: float, int, or categorical suggestion arguments
    :param values: optional string-category to YAML-fragment mapping
    """

    name: str
    value: Any = Field(default=None, exclude=True)
    distribution: Annotated[FloatParameter | IntParameter | CategoricalParameter, Field(discriminator='kind')]
    model_config = ConfigDict(serialize_by_alias=True, populate_by_name=True)
    mapped_values: dict[str, Any] | None = Field(default=None, alias='values')

    @model_validator(mode='after')
    def _validate_values(self) -> ParameterConfig:
        if self.mapped_values is not None:
            if not isinstance(self.distribution, CategoricalParameter) or any(
                type(label) is not str or label not in self.mapped_values for label in self.distribution.choices
            ):
                raise ValueError('values requires string categories with a value for every label.')
        return self

    def sample(self, trial: optuna.Trial) -> ConfigAssignment:
        """Suggest a primitive and return its typed YAML assignment.

        :param trial: active Optuna trial
        :return: assignment, expanded through values when configured
        """
        spec = self.distribution
        value = getattr(trial, f'suggest_{spec.kind}')(self.name, **spec.model_dump(exclude={'kind'}))
        if self.mapped_values is not None:
            value = self.mapped_values[value]
        return ConfigAssignment(path=self.path, value=value, mode=self.mode)


class MetricConfig(DictModel):
    """Live objective selection, reduction, cadence, and missing-value policy."""

    model_config = ConfigDict(extra='forbid')
    name: Literal['loss', 'test_score'] = 'loss'
    report_interval: PositiveInt = 10
    reduction: Literal['last', 'best', 'mean'] = 'last'
    window: PositiveInt = 10
    invalid: Literal['fail', 'skip'] = 'fail'

    def reducer(self, direction: Literal['minimize', 'maximize']) -> _MetricReducer:
        """Create independent reduction state for one trial.

        :param direction: optimization direction
        :return: stateful host reducer
        """
        return _MetricReducer(self, direction)


class _MetricReducer:
    def __init__(self, config: MetricConfig, direction: str) -> None:
        self.config, self.direction = config, direction
        self.values: dict[int, float] = {}

    def observe(self, report: TrainingReport) -> float | None:
        value = report.metrics.get(self.config.name)
        if value is None or not math.isfinite(value):
            if self.config.invalid == 'fail':
                raise ValueError(f'Missing or nonfinite {self.config.name} at step {report.step}.')
            return None
        self.values[report.step] = float(value)
        return self.final_value()

    def final_value(self) -> float:
        if not self.values:
            raise ValueError('Trial produced no finite objective metric.')
        values = list(self.values.values())
        if self.config.reduction == 'best':
            return (min if self.direction == 'minimize' else max)(values)
        if self.config.reduction == 'mean':
            values = values[-self.config.window:]
            return sum(values) / len(values)
        return values[-1]


type Sampler = Annotated[
    ThirdPartyType(default_modules='optuna.samplers'),
    AfterValidator(partial(require_type, optuna.samplers.BaseSampler)),
]
type Pruner = Annotated[
    ThirdPartyType(default_modules='optuna.pruners'),
    AfterValidator(partial(require_type, optuna.pruners.BasePruner)),
]


class OptunaStudy(Routine):
    """Run adaptive Train trials with YAML sampling and shared executor settings.

    :param root: study artifact directory
    :param base: unconstructed base Train YAML
    :param override: named parameter distributions and target paths
    :param executor: serial, thread, process, MPI, or hybrid execution configuration
    :param storage: URL, deferred storage constructor, null for memory, or omitted for journal
    :param n_trials: new trial attempts during this invocation
    """

    root: Path
    base: Path | romjax.YamlSource
    override: tuple[ParameterConfig, ...] = ()
    search_space: CallableModel | None = None
    study_name: str = 'romjax'
    direction: Literal['minimize', 'maximize'] = 'minimize'
    n_trials: PositiveInt = 20
    timeout: PositiveFloat | None = None
    sampler: Sampler = Field(default_factory=lambda: {'name': 'TPESampler', 'kwargs': {'seed': 0}})
    pruner: Pruner = Field(default_factory=optuna.pruners.NopPruner)
    storage: Any = 'journal'
    metric: MetricConfig = Field(default_factory=MetricConfig)
    final_metric: MetricCallable | None = None
    executor: Annotated[ExecutorConfig, BeforeValidator(ExecutorConfig.from_dict)] = Field(
        default_factory=ExecutorConfig
    )
    write_policy: Literal['reuse', 'overwrite', 'error'] = 'reuse'
    save_policy: _SavePolicy = Field(default_factory=_SavePolicy)
    exit_policy: Literal['any', 'all', 'ignore'] = 'any'
    retain_unsuccessful: bool = True
    case_root_path: tuple[tuple[str | int, ...], ...] = (('root',),)
    child_env: dict[str, str] = Field(default_factory=dict)
    child_quiet: bool = True
    callbacks: tuple[Callable, ...] = ()
    user_attrs: dict[str, Any] = Field(default_factory=dict)
    enqueue_trials: tuple[dict[str, Any], ...] = ()
    gc_after_trial: bool = False
    training_seed: int | None = None
    seed_policy: Literal['fixed', 'trial'] = 'fixed'
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)

    @field_validator('base', mode='before')
    @classmethod
    def _base(cls, value: Any) -> Any:
        return romjax.YamlLoader.resolve_parent_path(value) if isinstance(value, str | Path) else value

    @field_validator('storage', mode='before')
    @classmethod
    def _storage(cls, value: Any) -> Any:
        if value is None or isinstance(value, str) and (value == 'journal' or '://' in value):
            return value
        if isinstance(value, optuna.storages.BaseStorage):
            return value
        return DeferredThirdPartySpec.model_validate(value)

    @field_validator('save_policy', mode='before')
    @classmethod
    def _save_policy(cls, value: Any) -> _SavePolicy:
        return _SavePolicy.model_validate(value)

    @model_validator(mode='after')
    def _validate_study(self) -> OptunaStudy:
        if self.search_space is not None and self.override:
            raise ValueError('Use override or search_space, not both.')
        if len({p.name for p in self.override}) != len(self.override):
            raise ValueError('Parameter names must be unique.')
        roots = [ConfigAssignment(path=path, value='') for path in self.case_root_path]
        YamlTemplate.validate_paths([*self.override, *roots])
        self.executor.validate_environment(self.child_env)
        if (isinstance(self.storage, DeferredThirdPartySpec)
                and self.storage.kwargs.get('heartbeat_interval') is not None):
            raise ValueError('Native RDB heartbeat requires optimize; configure feedback.stale_after instead.')
        return self

    def run(self) -> int:
        """Run new trials, preserving study state and finalizing all owned attempts."""
        return _StudyCoordinator(self).run()


class _ControllerStudy(optuna.study.Study):
    """Adapt Optuna's public stop hook to the executor's drain semantics."""

    active = False
    stop_requested = False

    def stop(self) -> None:
        if not self.active:
            raise RuntimeError('Study.stop() requires an active OptunaStudy run.')
        self.stop_requested = True


class _StudyCoordinator:
    def __init__(self, config: OptunaStudy) -> None:
        self.config = config
        self.root = config.root.resolve()
        self.invocation = uuid4().hex
        self.control = self.root / '.control' / self.invocation
        self.lock = RLock()
        self.live: dict[int, optuna.Trial] = {}
        self.reducers: dict[int, _MetricReducer] = {}
        self.report_steps: dict[int, int] = {}
        self.failed = 0
        self.finished = 0
        self.study: _ControllerStudy | None = None
        self.server: _FeedbackServer | None = None

    @contextmanager
    def _ownership(self):
        # Advisory locks are released by the OS on controller death. On multi-node
        # filesystems the lock service must support POSIX advisory locking.
        import fcntl

        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / '.study.lock').open('a') as owner:
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('Study root has an active controller.') from exc
            try:
                yield
            finally:
                fcntl.flock(owner, fcntl.LOCK_UN)

    def _storage(self):
        value = self.config.storage
        if value == 'journal':
            value = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(str(self.root / 'study.journal')))
        elif isinstance(value, DeferredThirdPartySpec):
            value = value.build()
        if value is None:
            value = optuna.storages.InMemoryStorage()
        if not isinstance(value, str | optuna.storages.BaseStorage):
            raise ValueError('Storage factory must produce an Optuna BaseStorage.')
        if isinstance(value, optuna.storages.RDBStorage) and value.get_heartbeat_interval() is not None:
            raise ValueError('RDB heartbeat requires optimize, not ask/tell.')
        return value

    def _fingerprint(self) -> str:
        fields = {'override', 'search_space', 'metric', 'final_metric', 'direction', 'sampler', 'pruner',
                  'case_root_path', 'training_seed', 'seed_policy'}
        dumped = self.config.model_dump(include=fields)
        text = romjax.YamlLoader.dump(dumped, sort_keys=True, _preserve_yaml_sources=True)
        # Resolve declarative fragments just as render() does, including __parent__.
        import yaml

        node = yaml.compose(text, Loader=yaml.SafeLoader)
        dependencies = YamlTemplate.from_source(romjax.YamlSource(node, self.template.source.source_path))
        return hashlib.sha256((self.template.fingerprint() + dependencies.fingerprint()).encode()).hexdigest()

    def _open_study(self) -> None:
        config = self.config
        identity_path = self.root / 'study_identity.json'
        identity = {'study_name': config.study_name,
                    'storage': (None if isinstance(config.storage, optuna.storages.BaseStorage) else
                                hashlib.sha256(romjax.YamlLoader.dump(
                                    config.model_dump(include={'storage'}), sort_keys=True).encode()).hexdigest())}
        recorded = _Mailbox.read(identity_path)
        if recorded is not None and any(recorded.get(key) != value for key, value in identity.items()):
            raise ValueError('Study root identity belongs to another study or storage; use a new root.')
        manifest = self.root / 'study_manifest.yml'
        if recorded is None and (manifest.exists() or (self.root / 'trials').exists()):
            raise ValueError('Study root identity is missing; use a new root to preserve existing artifacts.')
        storage = self._storage()
        try:
            existing = optuna.load_study(study_name=config.study_name, storage=storage)
        except KeyError:
            existing = None
        if config.write_policy == 'error' and (existing is not None or manifest.exists()
                                                or (self.root / 'trials').exists()):
            raise ValueError('Study or owned artifacts already exist.')
        if config.storage is None and config.write_policy == 'reuse' and manifest.exists():
            raise ValueError('Cannot reuse an ephemeral in-memory study; configure persistent storage.')
        identity['token'] = recorded['token'] if recorded else uuid4().hex
        if recorded is not None and isinstance(config.storage, optuna.storages.BaseStorage):
            if existing is None or existing.user_attrs.get('romjax_owner') != identity['token']:
                raise ValueError('Study root identity does not match the supplied storage.')
        fingerprint = self._fingerprint()
        if existing is not None:
            attrs = existing.user_attrs
            if attrs.get('romjax_root') != str(self.root):
                raise ValueError('Existing study belongs to another root or is not owned by romjax.')
            if config.write_policy == 'reuse' and attrs.get('romjax_fingerprint') != fingerprint:
                raise ValueError('Study configuration fingerprint changed; use a new study or overwrite.')
            for trial in existing.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
                old = trial.user_attrs.get('romjax_invocation')
                lease = self.root / '.control' / str(old) / 'controller.lease'
                if old and not _Mailbox.stale(lease, config.feedback.stale_after):
                    raise RuntimeError('Previous controller lease has not expired.')
                if trial.user_attrs.get('romjax_root') != str(self.root):
                    raise RuntimeError('Study contains a running trial not owned by this root.')
                existing.tell(trial.number, state=optuna.trial.TrialState.FAIL)
            if config.write_policy == 'overwrite':
                optuna.delete_study(study_name=config.study_name, storage=storage)
                existing = None
        _Mailbox.write(identity_path, identity)
        if config.write_policy == 'overwrite':
            if (self.root / 'trials').exists():
                shutil.rmtree(self.root / 'trials')
            manifest.unlink(missing_ok=True)
            (self.root / 'best.json').unlink(missing_ok=True)
        created = optuna.create_study(study_name=config.study_name, storage=storage,
                                      sampler=config.sampler, pruner=config.pruner, direction=config.direction,
                                      load_if_exists=config.write_policy == 'reuse')
        self.study = _ControllerStudy(created.study_name, storage, config.sampler, config.pruner)
        self.study.active = True
        for key, value in config.user_attrs.items():
            if key.startswith('romjax_'):
                raise ValueError('romjax_ study attributes are reserved.')
            self.study.set_user_attr(key, value)
        self.study.set_user_attr('romjax_owner', identity['token'])
        self.study.set_user_attr('romjax_root', str(self.root))
        self.study.set_user_attr('romjax_fingerprint', fingerprint)
        self.study.set_user_attr('romjax_optuna_version', optuna.__version__)
        for enqueue in config.enqueue_trials:
            self.study.enqueue_trial(**enqueue)

    def _report(self, number: int, report: TrainingReport) -> str:
        with self.lock:
            trial = self.live[number]
            previous = self.report_steps.get(number, -1)
            if report.step < previous or report.step == previous and not report.final:
                raise ValueError('Training reports must have increasing optimizer steps.')
            value = self.reducers[number].observe(report)
            if value is not None and report.step > previous:
                trial.report(value, report.step)
                self.report_steps[number] = report.step
            if report.final or value is None:
                return 'continue'
            return 'prune' if trial.should_prune() else 'continue'

    def _tasks(self):
        started = time.monotonic()
        config = self.config
        try:
            for _ in range(config.n_trials):
                with self.lock:
                    if self.study.stop_requested or (config.timeout is not None
                                                     and time.monotonic() - started >= config.timeout):
                        break
                    if (isinstance(config.sampler, optuna.samplers.GridSampler)
                            and config.sampler.is_exhausted(self.study)):
                        break
                    trial = self.study.ask()
                    number = trial.number
                    self.live[number] = trial
                    self.reducers[number] = config.metric.reducer(config.direction)
                    trial.set_user_attr('romjax_root', str(self.root))
                    trial.set_user_attr('romjax_invocation', self.invocation)
                    root = self.root / 'trials' / f'trial_{number:05d}'
                    root.mkdir(parents=True, exist_ok=False)
                    raw = config.search_space(trial) if config.search_space is not None else [
                        parameter.sample(trial) for parameter in config.override]
                    assignments = [ConfigAssignment.model_validate(item) for item in raw]
                    assignments.extend(ConfigAssignment(path=path, value=str(root)) for path in config.case_root_path)
                    if config.training_seed is not None:
                        seed = config.training_seed + (number if config.seed_policy == 'trial' else 0)
                        assignments.append(ConfigAssignment(path=('init_seed',), value=seed))
                        trial.set_user_attr('romjax_training_seed', seed)
                    source = self.template.render(assignments)
                    template = YamlTemplate.from_source(source)
                    node = template.source.node
                    if not any(key.value == 'diagnostics' for key, _ in node.value):
                        template = YamlTemplate.from_source(template.render([
                            ConfigAssignment(path=('diagnostics',), value={})]))
                    source = template.render([
                        ConfigAssignment(path=('diagnostics', 'report_interval'), value=config.metric.report_interval),
                        ConfigAssignment(path=('diagnostics', 'report_metrics'), value=[config.metric.name]),
                    ])
                    source = YamlTemplate.from_source(source).snapshot(root / 'sources').source
                    text = romjax.YamlLoader._node_to_yaml(source.node)
                    (root / 'case.yml').write_text(text, encoding='utf-8')
                    trial.set_user_attr('romjax_config_sha256', hashlib.sha256(text.encode()).hexdigest())
                    # Store YAML text because assignments can contain tagged non-JSON fragments.
                    trial.set_user_attr('romjax_assignments', romjax.YamlLoader.dump(
                        [dict(item) for item in assignments], _preserve_yaml_sources=True))
                    key = f'trial_{number:05d}'
                    self.server.register(key, partial(self._report, number))
                    spec = TrialLaunchSpec(key, root, self.control, config.feedback, config.child_quiet)
                yield TaskSpec(str(number), key, spec)
        except BaseException:
            self.server.cancel_all('Study preparation interrupted.')
            raise

    def _result(self, outcome):
        number = int(outcome.task.key)
        try:
            with self.lock:
                trial = self.live[number]
                result = outcome.value
                error = str(outcome.error) if outcome.error is not None else None
                state = optuna.trial.TrialState.FAIL
                value = None
                if result is not None:
                    error = result.error
                    if result.state == 'pruned':
                        state = optuna.trial.TrialState.PRUNED
                    elif result.state == 'complete' and result.exit_code == 0:
                        try:
                            value = (self.config.final_metric(outcome.task.payload.root)
                                     if self.config.final_metric else self.reducers[number].final_value())
                            if not math.isfinite(value):
                                raise ValueError('Final objective is nonfinite.')
                            state = optuna.trial.TrialState.COMPLETE
                        except Exception as exc:
                            error = str(exc)
                    trial.set_user_attr('romjax_device', dict(result.device or {}))
                if result is not None:
                    trial.set_user_attr('romjax_step', result.step)
                    trial.set_user_attr('romjax_stop_reason', result.stop_reason)
                if error:
                    trial.set_user_attr('romjax_error', error)
                trial.set_user_attr('romjax_elapsed_seconds', outcome.elapsed_seconds)
                self._finish(number, state, value)
            if self.config.gc_after_trial:
                gc.collect()
            return ExecutionAction.STOP if self.study.stop_requested else ExecutionAction.CONTINUE
        except BaseException:
            self.server.cancel_all('Study result handling interrupted.')
            raise

    def _finish(self, number: int, state: optuna.trial.TrialState, value: float | None = None) -> None:
        """Record a terminal trial exactly once, including canceled queued tasks."""
        trial = self.live[number]
        frozen = self.study.tell(trial, values=value, state=state)
        self.live.pop(number)
        key = f'trial_{number:05d}'
        self.server.unregister(key)
        self.finished += 1
        self.failed += state == optuna.trial.TrialState.FAIL
        root = self.root / 'trials' / key
        if root.exists():
            _Mailbox.write(root / 'retained.json', {'retained': True})
        self._manifest()
        if self.config.save_policy.mode == 'rolling':
            self._retain()
        for callback in self.config.callbacks:
            callback(self.study, frozen)

    def _manifest(self) -> None:
        trials = []
        for trial in self.study.get_trials():
            root = self.root / 'trials' / f'trial_{trial.number:05d}'
            record = {'number': trial.number, 'state': trial.state.name, 'params': trial.params,
                      'value': trial.value, 'user_attrs': trial.user_attrs,
                      'retained': root.exists() and (root / 'retained.json').exists()}
            trials.append(record)
            if root.exists():
                # JSON is also valid YAML and permits atomic writes with one shared primitive.
                _Mailbox.write(root / 'trial.yml', record)
        _Mailbox.write(self.root / 'study_manifest.yml', {'study_name': self.study.study_name, 'trials': trials})

    def _retain(self) -> None:
        policy = self.config.save_policy
        complete = sorted((t for t in self.study.get_trials() if t.state == optuna.trial.TrialState.COMPLETE),
                          key=lambda t: ((t.value if self.config.direction == 'minimize' else -t.value), t.number))
        keep = {t.number for t in complete[:policy.count]} if policy.mode != 'all' else {t.number for t in complete}
        for trial in self.study.get_trials():
            if not trial.state.is_finished():
                continue
            if trial.state != optuna.trial.TrialState.COMPLETE and self.config.retain_unsuccessful:
                continue
            if trial.number in keep:
                continue
            root = self.root / 'trials' / f'trial_{trial.number:05d}'
            if root.exists():
                for child in root.iterdir():
                    if child.name in ('case.yml', 'trial.yml', 'sources'):
                        continue
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
        if complete:
            best = complete[0]
            _Mailbox.write(self.root / 'best.json', {'number': best.number, 'value': best.value, 'params': best.params,
                                                   'config': f'trials/trial_{best.number:05d}/case.yml'})
        self._manifest()

    def run(self) -> int:
        self.template = YamlTemplate.from_source(self.config.base)
        with self._ownership():
            self._open_study()
            self.server = _FeedbackServer(self.control, self.config.feedback)
            try:
                with self.server:
                    try:
                        self.config.executor.execute(self._tasks(),
                            _StudyWorker({**self.config._jax_child_env(), **self.config.child_env}),
                            total_units=self.config.n_trials, on_result=self._result,
                            error_policy='collect', collect_results=False, jax_env=self.config._jax_child_env(),
                            on_abort=lambda: self.server.cancel_all('Study execution interrupted.'))
                    finally:
                        self.server.cancel_all('Study controller stopped.')
                    if self.server.error is not None:
                        raise RuntimeError('Feedback service failed.') from self.server.error
            finally:
                with self.lock:
                    original_error = sys.exception()
                    cleanup_error = None
                    for trial in list(self.live.values()):
                        current = next(t for t in self.study.get_trials() if t.number == trial.number)
                        if not current.state.is_finished():
                            trial.set_user_attr('romjax_error', 'Trial canceled before terminal result.')
                            trial.set_user_attr('romjax_stop_reason', 'canceled')
                            try:
                                self._finish(trial.number, optuna.trial.TrialState.FAIL)
                            except BaseException as exc:
                                cleanup_error = cleanup_error or exc
                    self.study.active = False
                    self._retain()
                    if cleanup_error is not None and original_error is None:
                        raise cleanup_error
            failed = self.failed > 0 if self.config.exit_policy == 'any' else self.failed == self.finished > 0
            return int(failed and self.config.exit_policy != 'ignore')
