"""Grid-search routine for YAML-configured ROM experiments."""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import yaml
from alive_progress import alive_bar
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

import romjax
from romjax.routine import Routine, RoutineError
from romjax.tree import TreePath, coerce_tree_path, coerce_tree_paths, pytree_merge, set_subtree
from romjax.typing import CallableModel, from_registry
from romjax.utils import _NullProgress

type WritePolicy = Literal["reuse", "overwrite", "error"]
type _DeviceKind = Literal["cpu", "gpu"]

_HYBRID_ENV_KEYS = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    }
)

__all__ = [
    "ExecutorConfig",
    "GridSearch",
    "MetricCallable",
]


class ExecutorConfig(BaseModel, ABC):
    """Abstract execution backend configuration for grid-search cases."""

    show_progress: bool = True

    @classmethod
    def from_dict(cls, value: Any) -> "ExecutorConfig":
        """Construct the concrete executor config requested by a YAML-friendly value.

        :param value: string, mapping, or already-validated executor config
        :return: concrete executor config
        """
        if isinstance(value, ExecutorConfig):
            return value

        aliases = {
            "serial": "serial",
            "thread": "thread",
            "threads": "thread",
            "threadpool": "thread",
            "process": "process",
            "processes": "process",
            "ppool": "process",
            "hybrid": "hybrid",
        }
        if isinstance(value, str):
            try:
                value = {"kind": aliases[value]}
            except KeyError as exc:
                raise ValueError(f"Unknown executor: {value!r}.") from exc
        if not isinstance(value, Mapping):
            raise ValueError("GridSearch executor must be a string, mapping, or ExecutorConfig.")

        kind = aliases.get(str(value.get("kind", "serial")))
        if kind is None:
            raise ValueError(f"Unknown executor: {value.get('kind')!r}.")

        if kind == "serial":
            return SerialExecutorConfig.model_validate(value)
        if kind in {"thread", "process"}:
            return ConcurrentExecutorConfig.model_validate({**value, "kind": kind})
        if kind == "hybrid":
            data = dict(value)
            if isinstance(data.get("hybrid"), Mapping):
                data = {**data, **data["hybrid"]}
            data.pop("hybrid", None)
            return HybridExecutorConfig.model_validate(data)
        raise RoutineError(f"Unsupported executor: {kind}")
    
    def progress_context(self, total: int):
        return alive_bar(total) if self.show_progress else _NullProgress()

    @abstractmethod
    def run_cases(
        self,
        specs: Sequence["_CaseSpec"],
        on_result: Callable[["GridSearchCaseResult"], None] | None = None,
    ) -> list["GridSearchCaseResult"]:
        """Run all prepared case specs.

        :param specs: prepared case specs
        :param on_result: optional callback invoked after each completed case result is available
        :return: case results
        """
        raise NotImplementedError

    def validate_child_env(self, child_env: Mapping[str, str]) -> None:
        """Validate executor-specific compatibility with global child environment variables.

        :param child_env: shared child-process environment overrides
        """
        pass


class HybridGpuConfig(BaseModel):
    """GPU scheduling options for the hybrid grid-search executor.

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
            if isinstance(value, str, tuple | list):
                return {"devices": value}
            if isinstance(value, int):
                return {"workers_per_device": value}
        return value
    
    def get_env(self, device_index: int) -> dict[str, str]:
        """Return child environment overrides for one GPU slot."""
        env = {"CUDA_VISIBLE_DEVICES": str(device_index)}
        if self.memory_fraction is not None:
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.memory_fraction)
        if self.preallocate is not None:
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(self.preallocate).lower()
        return env


class HybridCpuConfig(BaseModel):
    """CPU scheduling options for the hybrid grid-search executor.

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


class HybridExecutorConfig(ExecutorConfig):
    """Device-aware executor settings for running grid-search cases on GPU and CPU slots.

    :param gpu: GPU slot and JAX/XLA memory settings
    :param cpu: CPU slot settings
    """

    kind: Literal["hybrid"] = "hybrid"
    gpu: HybridGpuConfig = Field(default_factory=HybridGpuConfig)
    cpu: HybridCpuConfig = Field(default_factory=HybridCpuConfig)

    def build_slots(self) -> list[_DeviceSlot]:
        """Build concrete CPU/GPU scheduling slots from a hybrid executor config.

        :return: ordered device slots
        """
        if self.gpu.devices == "auto":
            gpu_indices = _jax_visible_gpu_indices()
        else:
            gpu_indices = tuple(self.gpu.devices)

        slots: list[_DeviceSlot] = []
        for device_index in gpu_indices:
            for _ in range(self.gpu.workers_per_device):
                slots.append(
                    _DeviceSlot(
                        kind="gpu",
                        index=int(device_index),
                        env=self.gpu.get_env(int(device_index)),
                    )
                )

        cpu_workers = self.cpu.max_workers
        if cpu_workers is None:
            cpu_workers = os.cpu_count() or 1
        for _ in range(cpu_workers):
            slots.append(_DeviceSlot(kind="cpu", index=None, env={"JAX_PLATFORMS": "cpu"}))

        if not slots:
            raise RoutineError("Hybrid GridSearch executor has no available GPU or CPU slots.")
        return slots

    def validate_child_env(self, child_env: Mapping[str, str]) -> None:
        """Validate that hybrid-owned JAX device environment is configured on the executor.

        :param child_env: shared child-process environment overrides
        :raises RoutineError: if ``child_env`` contains scheduler-owned keys
        """
        reserved = sorted(set(child_env) & _HYBRID_ENV_KEYS)
        if reserved:
            keys = ", ".join(reserved)
            raise RoutineError(f"Hybrid GridSearch controls these child_env keys through executor config: {keys}")

    def _spec_for_slot(self, spec: "_CaseSpec", slot: "_DeviceSlot") -> "_CaseSpec":
        """Return a case spec with slot-specific environment and manifest metadata."""
        env = dict(spec.env)
        env.update(slot.env)
        return _CaseSpec(
            name=spec.name,
            root=spec.root,
            config_path=spec.config_path,
            command=spec.command,
            env=env,
            quiet=spec.quiet,
            reuse_existing=spec.reuse_existing,
            device=slot.manifest(),
        )

    def run_cases(
        self,
        specs: Sequence["_CaseSpec"],
        on_result: Callable[["GridSearchCaseResult"], None] | None = None,
    ) -> list["GridSearchCaseResult"]:
        """Run cases on dynamically assigned CPU/GPU subprocess slots.

        :param specs: prepared case specs
        :param show_progress: whether to show a progress bar
        :param on_result: optional callback invoked after each completed case result is available
        :return: case results
        """
        slots = self.build_slots()
        results: list[GridSearchCaseResult] = []
        with self.progress_context(len(specs)) as bar:
            pending = iter(spec for spec in specs if not spec.reuse_existing)
            for spec in specs:
                if spec.reuse_existing:
                    results.append(_reuse_case_result(spec))
                    bar()

            active: dict[Future[GridSearchCaseResult], _DeviceSlot] = {}
            with ThreadPoolExecutor(max_workers=len(slots)) as executor:
                for slot in slots:
                    try:
                        spec = next(pending)
                    except StopIteration:
                        break
                    active[executor.submit(_run_case_subprocess, self._spec_for_slot(spec, slot))] = slot

                while active:
                    done, _ = wait(active, return_when=FIRST_COMPLETED)
                    for future in done:
                        slot = active.pop(future)
                        result = future.result()
                        results.append(result)
                        if on_result is not None:
                            on_result(result)
                        bar()
                        try:
                            spec = next(pending)
                        except StopIteration:
                            continue
                        active[executor.submit(_run_case_subprocess, self._spec_for_slot(spec, slot))] = slot
        return results


class SerialExecutorConfig(ExecutorConfig):
    """Serial grid-search executor."""

    kind: Literal["serial"] = "serial"

    def run_cases(
        self,
        specs: Sequence["_CaseSpec"],
        on_result: Callable[["GridSearchCaseResult"], None] | None = None,
    ) -> list["GridSearchCaseResult"]:
        """Run case specs one at a time in the parent process.

        :param specs: prepared case specs
        :param on_result: optional callback invoked after each completed case result is available
        :return: case results
        """
        results: list[GridSearchCaseResult] = []
        with self.progress_context(len(specs)) as bar:
            for spec in specs:
                result = _reuse_case_result(spec) if spec.reuse_existing else _run_case_subprocess(spec)
                results.append(result)
                if on_result is not None:
                    on_result(result)
                bar()
        return results


class ConcurrentExecutorConfig(ExecutorConfig):
    """Concurrent-futures grid-search executor.

    :param kind: concurrent executor backend
    :param max_workers: worker count for the underlying executor
    """

    kind: Literal["thread", "process"] = "process"
    max_workers: PositiveInt | None = None

    def context(self) -> ThreadPoolExecutor | ProcessPoolExecutor:
        """Return the configured concurrent-futures executor."""
        if self.kind == "thread":
            return ThreadPoolExecutor(max_workers=self.max_workers)
        if self.kind == "process":
            return ProcessPoolExecutor(max_workers=self.max_workers)
        raise RoutineError(f"Unsupported concurrent executor: {self.kind}")

    def run_cases(
        self,
        specs: Sequence["_CaseSpec"],
        on_result: Callable[["GridSearchCaseResult"], None] | None = None,
    ) -> list["GridSearchCaseResult"]:
        """Run case specs with a concurrent-futures executor.

        :param specs: prepared case specs
        :param on_result: optional callback invoked after each completed case result is available
        :return: case results
        """
        results: list[GridSearchCaseResult] = []
        with self.progress_context(len(specs)) as bar, self.context() as executor:
            future_to_name: dict[Future[GridSearchCaseResult], str] = {
                executor.submit(_run_case_subprocess, spec): spec.name
                for spec in specs
                if not spec.reuse_existing
            }
            for spec in specs:
                if spec.reuse_existing:
                    result = _reuse_case_result(spec)
                    results.append(result)
                    if on_result is not None:
                        on_result(result)
                    bar()
            for future in as_completed(future_to_name):
                result = future.result()
                results.append(result)
                if on_result is not None:
                    on_result(result)
                bar()
        return results


@dataclass(frozen=True)
class _DeviceSlot:
    kind: _DeviceKind
    index: int | None
    env: dict[str, str]

    def manifest(self) -> dict[str, int | str | None]:
        """Return a YAML-safe representation of this device slot."""
        return {"kind": self.kind, "index": self.index}


class GridOverride(BaseModel):
    """One hyperparameter path and its candidate values.

    :param path: path in the base YAML tree to override
    :param cases: candidate values for the override path
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: TreePath
    cases: tuple[Any, ...]

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> TreePath:
        path = coerce_tree_path(value)
        if not isinstance(path, tuple) or not all(isinstance(token, str | int) for token in path):
            raise ValueError("Grid override path must be a sequence of string or integer tokens.")
        if any(isinstance(token, int) and token < 0 for token in path):
            raise ValueError("Grid override paths do not support negative list indices.")
        return path

    @field_validator("cases", mode="before")
    @classmethod
    def _coerce_cases(cls, value: Any) -> tuple[Any, ...]:
        if isinstance(value, str | bytes) or not isinstance(value, Sequence):
            value = [value]
        if not value:
            raise ValueError("Grid override cases must contain at least one value.")
        return tuple(value)


class _SavePolicy(BaseModel):
    """Normalized case retention policy."""

    mode: Literal["all", "best", "rolling"] = "all"
    count: PositiveInt | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if value == {}:
            return {"mode": "all"}
        if value == "all":
            return {"mode": "all"}
        if value == "best":
            return {"mode": "best", "count": 1}
        if value == "rolling":
            return {"mode": "rolling", "count": 1}
        if isinstance(value, Mapping):
            if len(value) != 1:
                raise ValueError("GridSearch save_policy mappings must have exactly one key: 'best' or 'rolling'.")
            key = next(iter(value))
            if key == "best":
                return {"mode": "best", "count": value[key]}
            if key == "rolling":
                return {"mode": "rolling", "count": value[key]}
            raise ValueError("GridSearch save_policy mappings must have exactly one key: 'best' or 'rolling'.")
        return value


@dataclass(frozen=True)
class _CaseSpec:
    name: str
    root: Path
    config_path: Path
    command: tuple[str, ...]
    env: dict[str, str]
    quiet: bool
    reuse_existing: bool = False
    device: dict[str, int | str | None] | None = None


@dataclass(frozen=True)
class GridSearchCaseResult:
    """Subprocess execution result for one grid-search case.

    :param name: case name
    :param root: case root directory
    :param config_path: generated case config path
    :param exit_code: subprocess exit code
    :param start_time: human-readable start timestamp
    :param end_time: human-readable end timestamp
    :param stdout_path: captured stdout path, if captured
    :param stderr_path: captured stderr path, if captured
    :param device: device slot assigned by a device-aware executor
    """

    name: str
    root: Path
    config_path: Path
    exit_code: int
    start_time: str
    end_time: str
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    device: dict[str, int | str | None] | None = None

    @property
    def status(self) -> Literal["succeeded", "failed"]:
        """Return the coarse case status."""
        return "succeeded" if self.exit_code == 0 else "failed"


def _timestamp() -> str:
    """Return a human-readable local timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _run_case_subprocess(spec: _CaseSpec) -> GridSearchCaseResult:
    """Run one generated case config in a child ``romx run`` process.

    :param spec: prepared case execution spec
    :return: subprocess result
    """
    start = _timestamp()
    env = os.environ.copy()
    env.update(spec.env)
    stdout_path = spec.root / "stdout.log" if spec.quiet else None
    stderr_path = spec.root / "stderr.log" if spec.quiet else None

    # Run from the same working directory (to preserve all relative yaml paths)
    if spec.quiet:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.run(
                spec.command,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
    else:
        process = subprocess.run(spec.command, env=env, check=False)

    return GridSearchCaseResult(
        name=spec.name,
        root=spec.root,
        config_path=spec.config_path,
        exit_code=process.returncode,
        start_time=start,
        end_time=_timestamp(),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        device=spec.device,
    )


def _reuse_case_result(spec: _CaseSpec) -> GridSearchCaseResult:
    """Return a successful result for a reused case directory."""
    timestamp = _timestamp()
    return GridSearchCaseResult(
        name=spec.name,
        root=spec.root,
        config_path=spec.config_path,
        exit_code=0,
        start_time=timestamp,
        end_time=timestamp,
        device=spec.device,
    )


def _jax_visible_gpu_indices() -> tuple[int, ...]:
    """Return GPU indices visible to JAX without failing when no GPU backend is available."""
    try:
        import jax

        return tuple(range(len(jax.devices("gpu"))))
    except Exception:
        return ()


def _set_override_path(tree: Any, path: TreePath, value: Any) -> Any:
    """Set one sparse override path into a sparse tree.

    :param tree: existing sparse tree
    :param path: path to set
    :param value: value to write
    :return: updated sparse tree
    """
    return pytree_merge(tree, set_subtree(None, path, value))


def _manifest_value(value: Any) -> Any:
    """Return a YAML-safe manifest representation for an arbitrary override value."""
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Path):
        return _yaml_path_text(value)
    if isinstance(value, Mapping):
        return {str(key): _manifest_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_manifest_value(item) for item in value]
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    return repr(value)


def _yaml_path_text(path: str | Path | PathLike[str]) -> str:
    """Return a YAML-safe path string that uses forward slashes."""
    return str(path).replace("\\", "/")


def orbax_metric(case_root: Path) -> float:
    """Return the best loss recorded for a training case.

    The fast path reads ``loss.csv`` written by :class:`romjax.train.Train`. If no loss history is present, the
    function attempts to inspect the latest Orbax checkpoint metrics.

    :param case_root: root directory for one grid-search case
    :return: minimum observed loss
    :raises RoutineError: if no numeric loss metric can be found
    """
    loss_path = case_root / "loss.csv"
    if loss_path.exists():
        values = np.atleast_2d(np.loadtxt(loss_path, delimiter=",", skiprows=1))
        if values.size and values.shape[1] >= 2:
            return float(np.nanmin(values[:, 1]))

    try:
        from orbax.checkpoint import v1 as ocp

        with ocp.training.Checkpointer(case_root.resolve()) as ckptr:
            latest = ckptr.latest
            metrics = getattr(latest, "metrics", None)
            if isinstance(metrics, Mapping) and "loss" in metrics:
                return float(metrics["loss"])
    except Exception as exc:
        raise RoutineError(f"No readable GridSearch metric found in {case_root}.") from exc

    raise RoutineError(f"No readable GridSearch metric found in {case_root}.")


_METRIC_REGISTRY = {
    "orbax": orbax_metric
}


class MetricCallable(CallableModel):
    """Rank grid search cases by a metric."""

    callable: Annotated[Callable[[Path], float], BeforeValidator(functools.partial(from_registry, _METRIC_REGISTRY))]

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, value):
        if isinstance(value, str):
            return {"callable": value}
        return value


@dataclass
class _RollingSaveTracker:
    """Track and prune the current best case directories on a rolling basis."""

    count: int
    ranked: list[tuple[float, str, Path]] = field(default_factory=list)

    @property
    def retained(self) -> set[str]:
        """Return the currently retained case names."""
        return {name for _, name, _ in self.ranked}

    def observe(self, result: GridSearchCaseResult, metric: float) -> None:
        """Update the tracked top cases with one completed result."""
        self.ranked.append((metric, result.name, result.root))
        self.ranked.sort(key=lambda item: item[0])
        if len(self.ranked) > self.count:
            _, _, evicted_root = self.ranked.pop()
            if evicted_root.exists():
                shutil.rmtree(evicted_root)

    def reject(self, result: GridSearchCaseResult) -> None:
        """Delete a completed case that cannot participate in rolling retention."""
        if result.root.exists():
            shutil.rmtree(result.root)


class GridSearch(Routine):
    """Run a Cartesian-product grid search by generating override YAML files and dispatching ``romx run`` cases.

    :param root: directory containing generated cases, best-copy outputs, and manifest
    :param base: base routine YAML file to override
    :param override: hyperparameter override specifications
    :param write_policy: behavior when root or case artifacts already exist
    :param save_policy: which completed cases to retain after ranking
    :param metric: metric alias or callable used to rank successful cases
    :param executor: execution backend configuration
    :param case_root_path: path or paths to inject with the per-case root directory
    :param child_env: extra environment variables for child subprocesses
    :param child_quiet: whether to capture child stdout and stderr to files
    :param command: optional command prefix replacing the default current-interpreter romx entrypoint
    """

    root: Path
    base: Path
    override: list[GridOverride]
    write_policy: WritePolicy = "reuse"
    save_policy: _SavePolicy = Field(default_factory=_SavePolicy)
    metric: MetricCallable = "orbax"
    executor: ExecutorConfig = Field(default_factory=SerialExecutorConfig)
    case_root_path: tuple[TreePath, ...] = (("root",),)
    child_env: dict[str, str] = Field(default_factory=dict)
    child_quiet: bool = True
    command: tuple[str, ...] | None = None

    @field_validator("base", mode="before")
    @classmethod
    def _resolve_base(cls, value: Any) -> Any:
        if isinstance(value, str | Path):
            return romjax.YamlLoader.resolve_parent_path(value)
        return value

    @field_validator("executor", mode="before")
    @classmethod
    def _coerce_executor(cls, value: Any) -> Any:
        return ExecutorConfig.from_dict(value)

    @field_validator("save_policy", mode="before")
    @classmethod
    def _coerce_save_policy(cls, value: Any) -> Any:
        return _SavePolicy.model_validate(value)

    @field_validator("case_root_path", mode="before")
    @classmethod
    def _coerce_case_root_path(cls, value: Any) -> tuple[TreePath, ...]:
        paths = tuple(coerce_tree_paths(value))
        if any(len(path) == 0 for path in paths):
            raise ValueError("case_root_path entries cannot be empty.")
        if any(any(isinstance(token, int) and token < 0 for token in path) for path in paths):
            raise ValueError("case_root_path entries do not support negative list indices.")
        return paths

    @model_validator(mode="after")
    def _validate_paths(self) -> "GridSearch":
        self.root = self.root.resolve()
        self.base = self.base.resolve()
        if not self.base.exists():
            raise RoutineError(f"GridSearch base config does not exist: {self.base}")
        self.executor.validate_child_env(self.child_env)
        
        if self.root.exists() and any(self.root.iterdir()):
            if self.write_policy == "error":
                raise RoutineError(f"GridSearch root already contains artifacts: {self.root}")
            if self.write_policy == "overwrite":
                for artifact in (self.root / "cases", self.root / "best"):
                    if artifact.exists():
                        shutil.rmtree(artifact)
                (self.root / "grid_search_manifest.yml").unlink(missing_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)

        return self

    def _case_values(self) -> list[tuple[Any, ...]]:
        """Return the Cartesian-product grid values."""
        return list(product(*(item.cases for item in self.override)))

    def _case_metadata(self, values: Sequence[Any]) -> dict[str, Any]:
        """Return manifest metadata for a case's override values."""
        return {
            ".".join(str(token) for token in override.path): _manifest_value(value)
            for override, value in zip(self.override, values)
        }

    def _case_command(self, config_path: Path) -> tuple[str, ...]:
        """Return the command for a generated case config."""
        if self.command is not None:
            return (*self.command, str(config_path))
        return (sys.executable, "-c", "from romjax.romx_cli import main; main()", "run", str(config_path))

    def _write_case_config(self, case_root: Path, values: Sequence[Any]) -> Path:
        """Write one case override config.

        :param case_root: generated case root
        :param values: override values for this case
        :return: generated YAML path
        """
        case_root.mkdir(parents=True, exist_ok=True)
        tree: Any = {}
        for override, value in zip(self.override, values):
            tree = _set_override_path(tree, override.path, value)
        for root_path in self.case_root_path:
            tree = _set_override_path(tree, root_path, _yaml_path_text(case_root))

        try:
            base_ref = _yaml_path_text(os.path.relpath(self.base, start=case_root))
            override_tag = f"!overrides:__parent__/{base_ref}"
        except ValueError:
            override_tag = f"!overrides:{_yaml_path_text(self.base)}"
        config_path = case_root / "case.yml"
        yaml_body = romjax.YamlLoader.dump(tree, sort_keys=False)
        config_path.write_text(f"{override_tag}\n{yaml_body}", encoding="utf-8")
        return config_path

    def _prepare_cases(self) -> tuple[list[_CaseSpec], dict[str, dict[str, Any]]]:
        """Create all case directories/configs and return execution specs plus manifest entries."""
        cases_dir = self.root / "cases"
        specs: list[_CaseSpec] = []
        manifest_cases: dict[str, dict[str, Any]] = {}
        for index, values in enumerate(self._case_values()):
            name = f"case_{index:04d}"
            case_root = cases_dir / name
            reuse_existing = self.write_policy == "reuse" and case_root.exists() and any(case_root.iterdir())
            config_path = case_root / "case.yml" if reuse_existing else self._write_case_config(case_root, values)
            specs.append(
                _CaseSpec(
                    name=name,
                    root=case_root,
                    config_path=config_path,
                    command=self._case_command(config_path),
                    env=dict(self.child_env),
                    quiet=self.child_quiet,
                    reuse_existing=reuse_existing,
                )
            )
            manifest_cases[name] = {
                "path": _yaml_path_text(case_root),
                "config": _yaml_path_text(config_path),
                "overrides": self._case_metadata(values),
            }
        return specs, manifest_cases

    def _rank_cases(self, results: Sequence[GridSearchCaseResult]) -> list[tuple[str, float]]:
        """Evaluate metrics and return successful cases ordered from best (least) to worst (most)."""
        metric_fn = self.metric
        ranked: list[tuple[str, float]] = []
        for result in results:
            if result.exit_code != 0:
                continue
            try:
                ranked.append((result.name, float(metric_fn(result.root))))
            except Exception:
                continue
        return sorted(ranked, key=lambda item: item[1])

    def _build_rolling_tracker(self) -> _RollingSaveTracker:
        """Return a rolling save-policy tracker for this search."""
        if self.save_policy.count is None:
            raise RoutineError("Rolling GridSearch save_policy requires a positive case count.")
        return _RollingSaveTracker(count=self.save_policy.count)

    def _handle_rolling_result(
        self,
        tracker: _RollingSaveTracker,
        scores: dict[str, float],
        result: GridSearchCaseResult,
    ) -> None:
        """Apply rolling retention logic to one completed case result."""
        if result.exit_code != 0:
            tracker.reject(result)
            return
        try:
            metric = float(self.metric(result.root))
        except Exception:
            tracker.reject(result)
            return
        scores[result.name] = metric
        tracker.observe(result, metric)

    def _apply_save_policy(self, ranked: Sequence[tuple[str, float]]) -> list[str]:
        """Apply retention policy and copy the best case.

        :param ranked: ranked successful cases
        :return: retained case names
        """
        if self.save_policy.mode == "all":
            retained = [case_root.name for case_root in sorted((self.root / "cases").iterdir()) if case_root.is_dir()]
        else:
            retained = [name for name, _ in ranked[: self.save_policy.count]]
            for case_root in (self.root / "cases").iterdir():
                if case_root.is_dir() and case_root.name not in retained:
                    shutil.rmtree(case_root)

        return retained

    def _copy_best_case(self, best_name: str) -> None:
        """Copy the current best case directory to ``best/``."""
        best_path = self.root / "best"
        if best_path.exists():
            shutil.rmtree(best_path)
        shutil.copytree(self.root / "cases" / best_name, best_path)

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        """Write the grid-search manifest."""
        path = self.root / "grid_search_manifest.yml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(dict(manifest), fh, sort_keys=False)

    def run(self) -> int:
        """Run the grid search and write a manifest."""
        specs, manifest_cases = self._prepare_cases()
        rolling_tracker = self._build_rolling_tracker() if self.save_policy.mode == "rolling" else None
        rolling_scores: dict[str, float] = {}
        results = self.executor.run_cases(
            specs,
            on_result=(
                functools.partial(self._handle_rolling_result, rolling_tracker, rolling_scores)
                if rolling_tracker is not None
                else None
            ),
        )
        result_by_name = {result.name: result for result in results}
        ranked = (
            sorted(rolling_scores.items(), key=lambda item: item[1])
            if rolling_tracker is not None
            else self._rank_cases(results)
        )
        metric_by_name = dict(ranked)
        retained = rolling_tracker.retained if rolling_tracker is not None else self._apply_save_policy(ranked)
        if ranked:
            self._copy_best_case(ranked[0][0])

        exit_code = 0
        for spec in specs:
            result = result_by_name[spec.name]
            if result.exit_code != 0 and exit_code == 0:
                exit_code = result.exit_code
            manifest_cases[spec.name].update(
                {
                    "status": result.status,
                    "exit_code": result.exit_code,
                    "metric": metric_by_name.get(spec.name),
                    "start_time": result.start_time,
                    "end_time": result.end_time,
                    "stdout": _yaml_path_text(result.stdout_path) if result.stdout_path is not None else None,
                    "stderr": _yaml_path_text(result.stderr_path) if result.stderr_path is not None else None,
                    "device": result.device,
                    "retained": spec.name in retained,
                }
            )

        self._write_manifest(
            {
                "root": _yaml_path_text(self.root),
                "base": _yaml_path_text(self.base),
                "executor": self.executor.model_dump(),
                "save_policy": self.save_policy.model_dump(),
                "best": ranked[0][0] if ranked else None,
                "ranking": [{"case": name, "metric": metric} for name, metric in ranked],
                "cases": manifest_cases,
            }
        )
        return exit_code
