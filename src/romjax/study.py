"""Adaptive YAML-configured Optuna studies using romjax executors."""
from __future__ import annotations

import gc
import hashlib
import math
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from functools import partial
from glob import glob, has_magic
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
    """Objective selection, reduction, and missing-value policy for training reports."""

    model_config = ConfigDict(extra='forbid')
    name: Literal['loss', 'test_score'] = 'loss'
    reduction: Literal['last', 'best', 'mean'] = 'last'
    window: PositiveInt = 10
    invalid: Literal['fail', 'skip', 'prune'] = 'fail'

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
        self.invalid_history: dict[int, str] = {}
        self.final_invalid = False

    def observe(self, report: TrainingReport) -> float | None:
        value = report.metrics.get(self.config.name)
        if value is None or not math.isfinite(value):
            if self.config.invalid == 'fail':
                raise ValueError(f'Missing or nonfinite {self.config.name} at step {report.step}.')
            if self.config.invalid == 'skip':
                return None
            value = math.nan if value is None else float(value)
            self.values[report.step] = value
            self.invalid_history[report.step] = (
                'missing' if report.metrics.get(self.config.name) is None
                else 'nan' if math.isnan(value)
                else 'positive_infinity' if value > 0
                else 'negative_infinity'
            )
            self.final_invalid = self.final_invalid or report.final
            return value
        self.values[report.step] = float(value)
        return self.final_value()

    def final_value(self) -> float:
        if not self.values:
            raise ValueError('Trial produced no finite objective metric.')
        values = list(self.values.values())
        if self.config.reduction == 'best':
            finite = [value for value in values if math.isfinite(value)]
            return (min if self.direction == 'minimize' else max)(finite) if finite else math.nan
        if self.config.reduction == 'mean':
            values = values[-self.config.window:]
            if any(not math.isfinite(value) for value in values):
                return math.nan
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


def _journal_storage(path: Path) -> optuna.storages.JournalStorage:
    """Open a cross-platform Optuna journal storage.

    :param path: journal file path
    :return: Optuna storage backed by ``path``
    """
    journal = str(path)
    lock = optuna.storages.journal.JournalFileOpenLock(journal) if sys.platform == 'win32' else None
    return optuna.storages.JournalStorage(optuna.storages.journal.JournalFileBackend(journal, lock_obj=lock))


class StorageRef(DictModel):
    """A YAML-friendly Optuna journal, URL, or constructed storage reference.

    :param journal: Optuna journal file path
    :param storage: Optuna storage URL, deferred ``BaseStorage`` constructor, or runtime storage
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')
    journal: Path | None = None
    storage: str | DeferredThirdPartySpec | optuna.storages.BaseStorage | None = None

    @model_validator(mode='after')
    def _validate_reference(self) -> StorageRef:
        if (self.journal is None) == (self.storage is None):
            raise ValueError('StorageRef requires exactly one of journal or storage.')
        return self

    def open(self) -> str | optuna.storages.BaseStorage:
        """Open or construct this storage reference.

        :return: Optuna storage URL or object
        :raises ValueError: if a deferred factory produces the wrong type
        """
        if self.journal is not None:
            return _journal_storage(self.journal)
        value = self.storage.build() if isinstance(self.storage, DeferredThirdPartySpec) else self.storage
        if not isinstance(value, str | optuna.storages.BaseStorage):
            raise ValueError('Storage factory must produce an Optuna BaseStorage.')
        return value

    def journal_path(self) -> Path | None:
        """Return the resolved journal path when this is a journal reference.

        :return: resolved journal path, or null for non-journal storage
        """
        return None if self.journal is None else self.journal.resolve()


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
    executor: Annotated[ExecutorConfig, BeforeValidator(ExecutorConfig.from_dict)] = Field(
        default_factory=ExecutorConfig
    )
    write_policy: Literal['reuse', 'overwrite', 'error'] = 'reuse'
    exit_policy: Literal['any', 'all', 'ignore'] = 'any'
    case_root_path: tuple[tuple[str | int, ...], ...] = (('root',),)
    child_env: dict[str, str] = Field(default_factory=dict)
    child_quiet: bool = True
    callbacks: tuple[Callable, ...] = ()
    user_attrs: dict[str, Any] = Field(default_factory=dict)
    enqueue_trials: tuple[dict[str, Any], ...] = ()
    gc_after_trial: bool = False
    training_seed: int | None = None                    # To override Train.init_seed
    seed_policy: Literal['fixed', 'trial'] = 'fixed'    # Use fixed seed or add trial number to each init_seed
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
        if isinstance(value, StorageRef):
            return value
        if isinstance(value, Mapping) and 'journal' in value:
            return StorageRef.model_validate(value)
        if isinstance(value, optuna.storages.BaseStorage):
            return value
        return DeferredThirdPartySpec.model_validate(value)

    @model_validator(mode='before')
    @classmethod
    def _reject_removed_options(cls, value: Any) -> Any:
        if isinstance(value, dict):
            removed = {'save_policy', 'retain_unsuccessful', 'final_metric'} & value.keys()
            if removed:
                raise ValueError(f'Unsupported study fields: {sorted(removed)}')
        return value

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


class CopyStudy(Routine):
    """Copy studies from several Optuna storages into one destination.

    :param sources: stable labels and their source storage references
    :param destination: destination storage reference
    :param journal_globs: journal discovery patterns, processed when the routine runs
    :param name_policy: fail on existing names or namespace copied study names by source label
    :param write_policy: behavior when an individual destination study already exists
    """

    sources: dict[str, StorageRef] = Field(default_factory=dict)
    destination: StorageRef
    journal_globs: tuple[str, ...] = ()
    name_policy: Literal['fail', 'namespace'] = 'fail'
    write_policy: Literal['error', 'overwrite', 'reuse'] = 'error'

    @model_validator(mode='after')
    def _validate_sources(self) -> CopyStudy:
        if any(not label.strip() for label in self.sources):
            raise ValueError('Source labels must be nonempty.')
        return self

    @staticmethod
    def _glob_root(pattern: str) -> Path:
        """Return the static directory prefix of a journal glob pattern."""
        path = Path(pattern)
        root = Path(path.anchor) if path.anchor else Path()
        parts = path.parts[1:] if path.anchor else path.parts
        for part in parts:
            if has_magic(part):
                return root.absolute()
            root /= part
        return root.parent.absolute()

    @staticmethod
    def _discovery_label(path: Path, root: Path) -> str:
        """Derive a sanitized Optuna namespace from a discovered journal path."""
        relative = path.relative_to(root)
        parts = relative.parts[:-1] if path.name == 'study.journal' else (*relative.parts[:-1], path.stem)
        if not parts:
            parts = (path.parent.name if path.name == 'study.journal' else path.stem,)
        label = '/'.join(re.sub(r'[^A-Za-z0-9._-]+', '-', part).strip('._-') for part in parts)
        if not label or any(not part for part in label.split('/')):
            raise ValueError(f'Discovered journal has an invalid source path: {path}')
        return label

    def _source_refs(self) -> dict[str, StorageRef]:
        """Return explicit and discovered source references.

        :return: normalized source references keyed by stable labels
        :raises ValueError: if a glob has no files or a discovered label is ambiguous
        """
        sources = dict(self.sources)
        for label, reference in sources.items():
            path = reference.journal_path()
            if path is not None and not path.is_file():
                raise ValueError(f'Explicit source {label!r} must be an existing journal file: {path}')
        journals = {path for ref in sources.values() if (path := ref.journal_path()) is not None}
        for pattern in self.journal_globs:
            root = self._glob_root(pattern)
            matches = sorted(
                {Path(match).absolute() for match in glob(pattern, recursive=True) if Path(match).is_file()}
            )
            if not matches:
                raise ValueError(f'Journal glob matched no journal files: {pattern}')
            for matched in matches:
                path = matched.resolve()
                if path in journals:
                    continue
                label = self._discovery_label(matched, root)
                if not label or label in sources:
                    raise ValueError(f'Discovered journal has duplicate source label: {label!r}')
                sources[label] = StorageRef(journal=path)
                journals.add(path)
        return sources

    def _validate_destination(self, sources: dict[str, StorageRef]) -> None:
        """Reject known source and destination aliases.

        :param sources: normalized source references
        :raises ValueError: if a source and destination share a known backend identity
        """
        journal = self.destination.journal_path()
        if journal is not None and any(ref.journal_path() == journal for ref in sources.values()):
            raise ValueError('CopyStudy destination must not alias a source journal.')
        if isinstance(self.destination.storage, str) and any(
            ref.storage == self.destination.storage for ref in sources.values()
        ):
            raise ValueError('CopyStudy destination must not alias a source storage URL.')

    def _prepare_destination(self) -> None:
        """Create parent directories for file-backed destination storage."""
        journal = self.destination.journal_path()
        if journal is not None:
            journal.parent.mkdir(parents=True, exist_ok=True)
            return
        storage = self.destination.storage
        if isinstance(storage, str) and storage.startswith('sqlite:///'):
            database = storage.removeprefix('sqlite:///').split('?', maxsplit=1)[0]
            if database and database != ':memory:':
                Path(database).parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> int:
        """Copy all studies in each configured source storage.

        :return: zero after every study is copied
        :raises ValueError: if no sources are selected or a destination aliases a source
        """
        sources = self._source_refs()
        if not sources:
            raise ValueError('CopyStudy requires at least one source storage.')
        self._validate_destination(sources)
        self._prepare_destination()
        destination = self.destination.open()
        existing = {summary.study_name for summary in optuna.get_all_study_summaries(destination)}
        for label, reference in sorted(sources.items()):
            source = reference.open()
            summaries = sorted(optuna.get_all_study_summaries(source), key=lambda summary: summary.study_name)
            for summary in summaries:
                target = None if self.name_policy == 'fail' else f'{label}/{summary.study_name}'
                name = summary.study_name if target is None else target
                if name in existing:
                    if self.write_policy == 'reuse':
                        continue
                    if self.write_policy == 'overwrite':
                        optuna.delete_study(study_name=name, storage=destination)
                optuna.copy_study(
                    from_study_name=summary.study_name,
                    from_storage=source,
                    to_storage=destination,
                    to_study_name=target,
                )
                existing.add(name)
        return 0


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
        """Exclusively own the study root for this controller invocation.

        POSIX uses an advisory ``flock`` while Windows locks the first byte of
        the lock file with ``msvcrt.locking``. Both locks are released by the
        operating system if the controller process exits unexpectedly. On a
        multi-node filesystem, the underlying lock service must support the
        selected platform's file-locking primitive.

        :raises RuntimeError: if another controller owns this study root
        """

        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / '.study.lock').open('a+b') as owner:
            owner.seek(0, 2)
            if owner.tell() == 0:
                owner.write(b'\\0')
                owner.flush()
            owner.seek(0)
            try:
                if sys.platform == 'win32':
                    import msvcrt

                    msvcrt.locking(owner.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Study root has an active controller.') from exc
            try:
                yield
            finally:
                owner.seek(0)
                if sys.platform == 'win32':
                    import msvcrt

                    msvcrt.locking(owner.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(owner, fcntl.LOCK_UN)

    def _storage(self):
        value = self.config.storage
        if value == 'journal':
            value = _journal_storage(self.root / 'study.journal')
        elif isinstance(value, StorageRef):
            journal = value.journal_path()
            if journal is not None:
                journal.parent.mkdir(parents=True, exist_ok=True)
            value = value.open()
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
        fields = {'override', 'search_space', 'metric', 'direction', 'sampler', 'pruner',
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
            archive = self.root / 'archive' / uuid4().hex
            for path in (self.root / 'trials', self.root / 'best', manifest, self.root / 'best.json',
                         self.root / '.best-backup'):
                if path.exists():
                    archive.mkdir(parents=True, exist_ok=True)
                    path.rename(archive / path.name)
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
            reducer = self.reducers[number]
            value = reducer.observe(report)
            if reducer.invalid_history:
                trial.set_user_attr('romjax_invalid_metrics', [
                    {'step': step, 'kind': kind} for step, kind in reducer.invalid_history.items()
                ])
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
                    source = YamlTemplate.from_source(source).snapshot(root / 'sources').source
                    text = romjax.YamlLoader._node_to_yaml(source.node)
                    case_path = root / 'case.yml'
                    case_path.write_text(text, encoding='utf-8')
                    trial.set_user_attr('romjax_config_sha256', hashlib.sha256(case_path.read_bytes()).hexdigest())
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
                            reducer = self.reducers[number]
                            value = reducer.final_value()
                            if (
                                self.config.metric.invalid == 'prune'
                                and (reducer.final_invalid or not math.isfinite(value))
                            ):
                                value = (sys.float_info.max if self.config.direction == 'minimize'
                                         else -sys.float_info.max)
                            elif not math.isfinite(value):
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
        self._manifest()
        self._update_best()
        for callback in self.config.callbacks:
            callback(self.study, frozen)

    def _manifest(self) -> None:
        trials = []
        for trial in self.study.get_trials():
            root = self.root / 'trials' / f'trial_{trial.number:05d}'
            record = {'number': trial.number, 'state': trial.state.name, 'params': trial.params,
                      'value': trial.value, 'user_attrs': trial.user_attrs}
            trials.append(record)
            if root.exists():
                # JSON is also valid YAML and permits atomic writes with one shared primitive.
                _Mailbox.write(root / 'trial.yml', record)
        _Mailbox.write(self.root / 'study_manifest.yml', {'study_name': self.study.study_name, 'trials': trials})

    def _update_best(self) -> None:
        """Copy the best completed trial, preserving the previous copy on failure."""
        best_dir = self.root / 'best'
        backup = self.root / '.best-backup'
        if backup.exists():
            if best_dir.exists():
                shutil.rmtree(backup)
            else:
                backup.rename(best_dir)
        complete = self.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
        if not complete:
            return
        best = self.study.best_trial
        current = _Mailbox.read(best_dir / 'trial.yml')
        if current is not None and current.get('number') == best.number:
            return
        source = self.root / 'trials' / f'trial_{best.number:05d}'
        temporary = self.root / f'.best-{uuid4().hex}.tmp'
        try:
            shutil.copytree(source, temporary)
            if best_dir.exists():
                best_dir.rename(backup)
            try:
                temporary.rename(best_dir)
            except BaseException:
                if backup.exists():
                    backup.rename(best_dir)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def run(self) -> int:
        self.template = YamlTemplate.from_source(self.config.base)
        with self._ownership():
            self._open_study()
            self._manifest()
            self._update_best()
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
                    self._manifest()
                    self._update_best()
                    if cleanup_error is not None and original_error is None:
                        raise cleanup_error
            failed = self.failed > 0 if self.config.exit_policy == 'any' else self.failed == self.finished > 0
            return int(failed and self.config.exit_policy != 'ignore')
