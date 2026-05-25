"""Reduced-order model training routine."""
import functools
import shutil
import time
import warnings
from collections.abc import Mapping, Sequence
from datetime import timedelta
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterator, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from alive_progress import alive_bar
from jaxtyping import ArrayLike, PyTree
from loguru import logger
from orbax.checkpoint import v1 as ocp
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    SkipValidation,
    field_validator,
    model_validator,
)

from romjax.graph import FunctionGraph
from romjax.model import ImplicitSampleable
from romjax.plotting import PlotSpec, gridplot
from romjax.routine import Routine, RoutineError
from romjax.tree import UnaryOperator, get_subtree, get_unary_operator, pytree_norm, set_subtree
from romjax.typing import CallableModel, ThirdPartyType, from_registry, from_yaml, require_type
from romjax.utils import load_h5

__all__ = ["Train", "GraphLoss", "GraphTest", "GraphDataLoader", "BatchDataLoader"]


def _prettify_timedelta(delta: float) -> str:
    total_seconds = int(delta)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    if days > 0:
        return f"{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if minutes > 0:
        return f"{minutes:02d}:{seconds:02d}"
    return f"{delta:.3f} s"


class BatchDataLoader[T: Any](BaseModel, Iterator):
    """
    Helper for basic mini-batch data loading.

    !!! Example
        ```python
        data = list(range(10))
        
        for batch in BatchDataLoader(data=data, batch_size=2):
            print(batch)  # [0, 1],  [2, 3],  [4, 5], ...
        ```

    :ivar data: the sequence of data to load batches from (i.e. a list, array, etc.), if none then will just load empty
                tuples infinitely (for use with the `Train` routine as default). Will try to access items simply by
                integer index. If this is an ndarray, then it will take along the first axis per usual. If this is a
                tuple of equal-length sequences, each item will be batched independently and the loader will yield a
                tuple of mini-batches.
    :ivar batch_size: the number of items per batch, if none then loads the entire dataset at each iteration (default)
    :ivar shuffle_seed: the random seed for shuffling data at each epoch, if none then does not shuffle (default)
    :ivar max_epochs: maximum number of iterations through full dataset, if none then continues indefinitely (default)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: Annotated[Sequence[T] | None, SkipValidation] = None  # default to empty loader
    batch_size: PositiveInt | None = None    # default to all
    shuffle_seed: int | None = None          # default no shuffle
    max_epochs: PositiveInt | None = None    # defaults to infinite

    _size: PositiveInt | None = PrivateAttr(default=None)
    _iterator: Iterator[Sequence[T]] | None = PrivateAttr(default=None)

    def _get_size(self):
        if self._size is None:
            if self.data is not None:
                if isinstance(self.data, tuple):
                    if len(self.data) == 0:
                        raise ValueError("Tuple data for batch loader must contain at least one sequence")

                    length = len(self.data[0])
                    for item in self.data[1:]:
                        if len(item) != length:
                            raise ValueError("All tuple items for batch loader must have equal length")

                    self._size = length
                else:
                    self._size = len(self.data)
        return self._size

    @model_validator(mode="after")
    def _validate_model(self):
        if self.data is not None:
            if not hasattr(self.data, "__len__"):
                raise ValueError("Data for batch loader must have a finite length")
            if isinstance(self.data, tuple) and len(self.data) > 0:
                size = len(self.data[0])
                for item in self.data[1:]:
                    if len(item) != size:
                        raise ValueError("All tuple items for batch loader must have equal length")
            self._get_size()
        
        return self

    @staticmethod
    def _batch_item(item: Sequence[T], window: np.ndarray) -> Sequence[T]:
        try:
            return item[window]
        except Exception:
            if isinstance(item, list):
                return [item[idx] for idx in window]
            if isinstance(item, tuple):
                return tuple(item[idx] for idx in window)
            return itemgetter(*window)(item)

    def _generator(self, start: int = 0) -> Generator[Sequence[T], None, None]:
        """Main mini-batch loading routine. Loads a single mini-batch from original data at a time."""
        size = self._get_size()

        if size is None or self.data is None:
            while True:
                yield ()  # empty dataloader
        
        batch_size = self.batch_size or size
        cursor = 0
        epoch = 0

        def _shuffle_indices(epoch: int):
            seed = np.random.SeedSequence([self.shuffle_seed, epoch])
            return np.random.default_rng(seed).permutation(size)
        
        indices = np.arange(size) if self.shuffle_seed is None else _shuffle_indices(epoch)
        
        def _advance_cursor(cursor, epoch, batch_size, size):
            next_cursor = cursor + batch_size
            
            if next_cursor >= size:
                cursor = 0
                epoch += 1
            else:
                cursor = next_cursor
            
            return cursor, epoch

        # Move the cursor up based on the starting index
        for _ in range(start):
            cursor, epoch = _advance_cursor(cursor, epoch, batch_size, size)

            if self.max_epochs is not None and epoch >= self.max_epochs:
                return
        
        while True:
            if self.shuffle_seed is not None:
                indices = _shuffle_indices(epoch)
            
            window = indices[cursor: cursor + batch_size]

            if isinstance(self.data, tuple):
                yield tuple(self._batch_item(item, window) for item in self.data)
            else:
                try:
                    yield self.data[window]  # try numpy fancy-indexing first
                except Exception:
                    yield itemgetter(*window)(self.data)

            cursor, epoch = _advance_cursor(cursor, epoch, batch_size, size)

            if self.max_epochs is not None and epoch >= self.max_epochs:
                return
    
    def set_iterator(self, start: int = 0) -> None:
        """Start the iterator at a given index."""
        self._iterator = self._generator(start)
    
    def __next__(self) -> Sequence[T]:
        if self._iterator is None:
            self.set_iterator()
        return next(self._iterator)
    
    def __iter__(self):
        self.set_iterator()
        return self


class DatasetConfig(BaseModel):
    """
    File-based load configuration for datasets created by :class:`romjax.data_gen.DataGeneration`.
    
    :param batch_size: number of output samples per yielded mini-batch
    :param shuffle_seed: seed for shuffling mini-batch data
    :param name: the dataset name (if None, will search the dataset root dir by default)
    :param max_samples: maximum loaded output samples  (defaults to all)
    :param max_input_samples: maximum loaded input samples  (defaults to all)
    :param max_outputs_per_input: maximum loaded output samples below each input sample  (defaults to all)
    :param max_epochs: maximum times to iterate through all available data (defaults to infinite loop)
    :param skip_input: decide whether to skip a particular input sample when loading
    :param skip_output: decide whether to skip a particular output sample when loading
    """

    batch_size: PositiveInt = 16
    shuffle_seed: int = 0
    name: str | None = None
    max_samples: PositiveInt | None = None
    max_input_samples: PositiveInt | None = None
    max_outputs_per_input: PositiveInt | None = None
    max_epochs: PositiveInt | None = None
    skip_input: Callable[[Path], bool] | None = None
    skip_output: Callable[[Path], bool] | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_plain_dataset_name(cls, value):
        if isinstance(value, str):
            return {"name": value}
        return value


class GraphDataLoader(BaseModel):
    """
    File-backed mini-batch loader for datasets created by :class:`romjax.data_gen.DataGeneration` for a FunctionGraph.

    The yielded batch payload is a mapping keyed by sampled dataset names:
    ``{dataset_name: {"inputs": batch, "outputs": batch, "residuals": batch}}``.

    :param root: root directory for loading datasets, expected structure is `root/dataset_name`
    :param datasets: configs for loading independent datasets under root
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    root: Path
    datasets: Sequence[DatasetConfig] = Field(default_factory=list)
    _iterator: Iterator[dict[str, PyTree]] | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _from_root(cls, value):
        """Validate a loader with all default options from just a plain root directory."""
        if isinstance(value, str | Path):
            return {"root": value}
        return value
    
    @field_validator("datasets", mode="before")
    @classmethod
    def _allow_single_dataset(cls, value):
        if isinstance(value, str) or not isinstance(value, Sequence):
            return [value]
        return value
    
    @model_validator(mode="after")
    def _validate_datasets(self):
        """Use all available datasets if not specified. Try to infer dataset names and raise an error if we cannot."""
        dataset_dirs = sorted(d for d in self.root.iterdir() if d.is_dir() and not d.name.startswith("."))

        if len(self.datasets) == 0:
            self.datasets = [DatasetConfig(name=d.name) for d in dataset_dirs]
        else:
            for ds_idx, ds_cfg in enumerate(self.datasets):
                if not ds_cfg.name and ds_idx > len(dataset_dirs)-1:
                    raise ValueError(f"Can't infer dataset name at index {ds_idx}. Please provide explicitly.")
                if not ds_cfg.name:
                    ds_cfg.name = dataset_dirs[ds_idx].name
        
        return self

    @staticmethod
    def walk_sample_directories(
        root: Path, 
        skip_input: Callable[[Path], bool] | None = None, 
        skip_output: Callable[[Path], bool] | None = None
    ) -> Generator[tuple[Path, Path], None, None]:
        """Walk a nested input/output directory structure generated by `gen_keys` via `DataGeneration`."""
        if skip_input is None:
            skip_input = lambda p: False
        if skip_output is None:
            skip_output = lambda p: False

        for input_seed_dir in sorted(root.iterdir()):
            if not input_seed_dir.is_dir() or not input_seed_dir.name.startswith("seed_"):
                continue

            for input_sample_dir in sorted(input_seed_dir.iterdir()):
                if (not input_sample_dir.is_dir() or 
                    not input_sample_dir.name.startswith("sample_") or
                    skip_input(input_sample_dir)):
                    continue

                for output_seed_dir in sorted(input_sample_dir.iterdir()):
                    if not output_seed_dir.is_dir() or not output_seed_dir.name.startswith("seed_"):
                        continue

                    for output_sample_dir in sorted(output_seed_dir.iterdir()):
                        if (not output_sample_dir.is_dir() or
                            not output_sample_dir.name.startswith("sample_") or
                            skip_output(output_sample_dir)):
                            continue

                        yield input_sample_dir, output_sample_dir

    @staticmethod
    def _select_dataset_batch(
        paths: Sequence[tuple[Path, Path]],
        indices: jax.Array,
        cursor: int,
        config: DatasetConfig,
    ) -> tuple[list[tuple[Path, Path]], int]:
        """Select one dataset batch from a shuffled epoch, allowing a partial batch at epoch end."""
        batch_paths: list[tuple[Path, Path]] = []
        unique_inputs: set[Path] = set()
        outputs_per_input: dict[Path, int] = {}
        limit = min(cursor + config.batch_size, len(paths))

        while cursor < limit:
            input_path, output_path = paths[int(indices[cursor])]
            cursor += 1

            next_total = len(batch_paths) + 1
            next_input_total = len(unique_inputs) + int(input_path not in unique_inputs)
            next_outputs_per_input = outputs_per_input.get(input_path, 0) + 1

            if config.max_samples is not None and next_total > config.max_samples:
                continue
            if config.max_input_samples is not None and next_input_total > config.max_input_samples:
                continue
            if config.max_outputs_per_input is not None and next_outputs_per_input > config.max_outputs_per_input:
                continue

            batch_paths.append((input_path, output_path))
            unique_inputs.add(input_path)
            outputs_per_input[input_path] = next_outputs_per_input

        return batch_paths, cursor
    
    @staticmethod
    def _stack_batch(items: Sequence[PyTree]) -> PyTree:
        """Stack matching sample pytrees along a leading batch axis."""
        return jax.tree.map(lambda *xs: jnp.stack(xs), *items)

    def _iter_datasets(self, train_step: int = 0):
        """
        Walk over all available datasets at once, using available configurations to limit sample sizes.
        Yields a mapping of dataset names to PyTree batches of data.
        
        :param train_step: initialize from this training step (default: 0)
        """
        ds_config = {}
        ds_paths = {}
        ds_totals = {}

        for ds_cfg in self.datasets:
            ds_name = ds_cfg.name

            ds_paths[ds_name] = list(
                self.walk_sample_directories(
                    self.root / ds_name, skip_input=ds_cfg.skip_input, skip_output=ds_cfg.skip_output
                )
            )
            if len(ds_paths[ds_name]) == 0:
                raise ValueError(f"No samples found for dataset '{ds_name}' in {self.root / ds_name}")

            ds_totals[ds_name] = len(ds_paths[ds_name])  # Total available data for each dataset
            ds_config[ds_name] = ds_cfg

        # Advance dataset epoch and cursor indices based on current training step
        # Epoch - number of iterations through an entire dataset
        # Cursor - starting index within a dataset for the next mini-batch
        
        ds_epochs = {name: 0 for name in ds_config}
        ds_cursors = {name: 0 for name in ds_config}
        ds_active = {name: True for name in ds_config}

        def _shuffle_indices(name: str, epoch: int) -> np.ndarray:
            """Build a deterministic per-epoch permutation without invoking JAX random ops."""
            seed = np.random.SeedSequence([ds_config[name].shuffle_seed, epoch])
            return np.random.default_rng(seed).permutation(ds_totals[name])

        ds_indices = {name: _shuffle_indices(name, ds_epochs[name]) for name in ds_config}

        def _advance_dataset(name: str) -> None:
            """Advance one dataset by a single batch and update its termination state."""
            if not ds_active[name]:
                return

            _, next_cursor = self._select_dataset_batch(
                ds_paths[name],
                ds_indices[name],
                ds_cursors[name],
                ds_config[name],
            )
            if next_cursor >= ds_totals[name]:
                ds_epochs[name] += 1
                if ds_config[name].max_epochs is not None and ds_epochs[name] >= ds_config[name].max_epochs:
                    ds_active[name] = False
                    ds_cursors[name] = ds_totals[name]
                    return

                ds_cursors[name] = 0
                ds_indices[name] = _shuffle_indices(name, ds_epochs[name])
            else:
                ds_cursors[name] = next_cursor

        for _ in range(train_step):
            for name in ds_config:
                _advance_dataset(name)

            if not any(ds_active.values()):
                return

        while True:
            active_names = [name for name in ds_paths if ds_active[name]]
            if len(active_names) == 0:
                return

            ds_batch = {}

            for name in active_names:
                batch_paths, next_cursor = self._select_dataset_batch(
                    ds_paths[name], ds_indices[name], ds_cursors[name], ds_config[name]
                )
                loaded: dict[str, list[PyTree]] = {"inputs": [], "outputs": [], "residuals": []}

                for input_path, output_path in batch_paths:
                    loaded["inputs"].append(load_h5({}, input_path / "input.h5", jax=True))

                    output_file = output_path / "output.h5"
                    if output_file.exists():
                        loaded["outputs"].append(load_h5({}, output_file, jax=True))

                    residual_file = output_path / "residual.h5"
                    if residual_file.exists():
                        loaded["residuals"].append(load_h5({}, residual_file, jax=True))

                ds_batch[name] = {
                    key: self._stack_batch(value)
                    for key, value in loaded.items()
                    if len(value) > 0
                }

                if next_cursor >= ds_totals[name]:
                    ds_epochs[name] += 1
                    if ds_config[name].max_epochs is not None and ds_epochs[name] >= ds_config[name].max_epochs:
                        ds_active[name] = False
                        ds_cursors[name] = ds_totals[name]
                    else:
                        ds_cursors[name] = 0
                        ds_indices[name] = _shuffle_indices(name, ds_epochs[name])
                else:
                    ds_cursors[name] = next_cursor

            yield ds_batch
    
    def set_iterator(self, train_step: int = 0):
        """Set the iterator based on the current train step."""
        self._iterator = self._iter_datasets(train_step)

    def __next__(self):
        """Continue existing iterator if possible, otherwise start from beginning."""
        if self._iterator is None:
            self.set_iterator()
        return next(self._iterator)
    
    def __iter__(self):
        """Restart iteration from 0."""
        self.set_iterator()
        return self


def reconstruction_loss(params: PyTree, single_data: PyTree, graph: FunctionGraph, path: list[str] = None):
    return graph.reconstruction_error(single_data, path, edge_payload_patches=params)


def tikhonov_regularization(params: PyTree, single_data: PyTree, graph: FunctionGraph):
    del single_data, graph
    return pytree_norm(params)


_LOSS_REGISTRY = {
    "reconstruction": reconstruction_loss,
    "tikhonov": tikhonov_regularization,
}


type GraphLossFunctionCallable = Annotated[
    Callable[[PyTree, PyTree, FunctionGraph], ArrayLike],
    BeforeValidator(functools.partial(from_registry, _LOSS_REGISTRY))
]

class GraphLossFunction(CallableModel):
    """Loss function for a single data sample."""

    callable: GraphLossFunctionCallable

    @model_validator(mode="before")
    @classmethod
    def _from_str(cls, value):
        if isinstance(value, str):
            return {"callable": value}
        return value


class GraphLossTerm(BaseModel):
    """
    One weighted term in a :class:`GraphLoss`. Aggregates a loss function over batch data.

    :param function: function to apply to a single sample of data
    :param edge: which edge to read data from
    :param weight: scalar term weight
    :param batch_reduce: reduce the loss over batch data; skip batch reduce if none
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    function: GraphLossFunction
    edge: str | None = None
    weight: float = 1.0
    batch_reduce: UnaryOperator | None = "mean"

    @field_validator("batch_reduce", mode="before")
    @classmethod
    def _get_unary_operator(cls, value):
        if value is not None:
            return get_unary_operator(value)
        return value

    @model_validator(mode="before")
    @classmethod
    def _from_plain_function(cls, value):
        if callable(value) or isinstance(value, str) or (isinstance(value, Mapping) and "callable" in value):
            return {"function": value}
        return value

    def __call__(
        self, 
        params: Mapping[str, PyTree], 
        batch_data: Mapping[str, PyTree], 
        graph: FunctionGraph
    ) -> jax.Array:
        if self.batch_reduce is not None:
            term_batch = batch_data[self.edge] if self.edge is not None else batch_data

            def body(carry, single_data):
                return carry, self.function(params, single_data, graph)

            _, losses = jax.lax.scan(body, None, term_batch)
            return jnp.asarray(self.weight) * self.batch_reduce(losses)
    
        else:
            return jnp.asarray(self.weight) * self.function(params, batch_data, graph)


class GraphLoss(BaseModel):
    """
    Loss function for a `FunctionGraph`.

    :param terms: loss terms combined by weighted summation
    :param graph: the FunctionGraph, leave as None to defer to `Train.graph`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    terms: Sequence[GraphLossTerm]
    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None

    __hash__ = object.__hash__

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value):
        if isinstance(value, list):
            return {"terms": value}
        return value

    @model_validator(mode="after")
    def _bind_default_edges(self):
        if self.graph is not None:
            self.terms = self._set_default_edge(self.terms, self.graph)
        return self
    
    @staticmethod
    def _set_default_edge(terms, graph):
        """Grab the first sampleable edge as the default edge, i.e. typically there is only one."""
        _default_edge = None
        for edge_name, edge in graph.edges.items():
            if isinstance(edge, ImplicitSampleable):
                _default_edge = edge_name
            break

        for term in terms:
            if term.edge is None:
                term.edge = _default_edge
        
        return terms
    
    @staticmethod
    def _resolve_references(params: PyTree, reference_type: type = str):
        """Resolve references in a pytree to other locations in the pytree. Replace the references with pointers."""
        if reference_type is not str:
            raise ValueError("Only str-type references are supported in a param PyTree.")

        def _coerce_token(token: str) -> str | int:
            token = token.strip()
            if token.lstrip("-").isdigit():
                return int(token)
            return token

        def _coerce_reference_path(reference: str) -> tuple[str | int, ...]:
            return tuple(_coerce_token(token) for token in reference.split(","))

        def _iter_reference_paths(
            tree: PyTree,
            path: tuple[str | int, ...] = (),
        ) -> Iterator[tuple[tuple[str | int, ...], str]]:
            if isinstance(tree, reference_type):
                yield path, tree
                return

            if isinstance(tree, Mapping):
                for key, value in tree.items():
                    yield from _iter_reference_paths(value, (*path, key))
                return

            if isinstance(tree, tuple | list):
                for i, value in enumerate(tree):
                    yield from _iter_reference_paths(value, (*path, i))

        resolved = params
        for ref_path, reference in _iter_reference_paths(params):
            target_path = _coerce_reference_path(reference)
            try:
                target = get_subtree(params, target_path)
            except (AttributeError, IndexError, KeyError, TypeError) as exc:
                warnings.warn(
                    f"Could not resolve parameter reference {reference!r} at path {ref_path!r}: {exc}",
                    stacklevel=2,
                )
                continue
            resolved = set_subtree(resolved, ref_path, target)

        return resolved

    def __call__(self, params: Mapping[str, PyTree], batch: Mapping[str, PyTree]) -> jax.Array:
        """Parameters are specified on a per-edge basis. Data batches will also be passed per-edge."""
        if self.graph is None:
            raise ValueError("Must specify a FunctionGraph to evaluate GraphLoss")
        
        params = self._resolve_references(params)

        total = 0.0
        for term in self.terms:
            total = total + term(params, batch, self.graph)
        return total


class GraphTest(GraphLoss):
    """
    Just compute a graph loss function over a set of validation data.
    """

    dataloader: GraphDataLoader
    reduce: UnaryOperator | None = "mean"
    
    @field_validator("reduce", mode="before")
    @classmethod
    def _get_unary_operator(cls, value):
        if value is not None:
            return get_unary_operator(value)
        return value

    def __call__(self, params: Mapping[str, PyTree]) -> jax.Array:
        values = jnp.asarray([super().__call__(params, batch) for batch in self.dataloader])
        return self.reduce(values)


type SaveDecisionPolicy = Annotated[
    ThirdPartyType(default_modules=ocp.training.save_decision_policies.__name__),
    AfterValidator(partial(require_type, ocp.training.save_decision_policies.SaveDecisionPolicy)),
]
type PreservationPolicy = Annotated[
    ThirdPartyType(default_modules=ocp.training.preservation_policies.__name__),
    AfterValidator(partial(require_type, ocp.training.preservation_policies.PreservationPolicy)),
]
type GradientTransformation = Annotated[
    ThirdPartyType(default_modules="optax"),
    AfterValidator(partial(require_type, optax.GradientTransformation)),
]


class CheckpointerConfig(BaseModel):
    """
    Orbax-policy checkpoint configuration for :class:`GraphTrain`.

    :param save_decision_policy: Orbax save policy; short names resolve from
        ``ocp.training.save_decision_policies``
    :param preservation_policy: Orbax preservation policy; short names resolve from
        ``ocp.training.preservation_policies``
    :param step_name_format: Orbax name format for saving training steps
    :param custom_metadata: see `Checkpointer`
    :param cleanup_tmp_directories: see `Checkpointer`
    :param lightweight_initialize: see `Checkpointer`
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True, extra="allow")

    save_decision_policy: SaveDecisionPolicy | None = Field(
        default_factory=lambda: ocp.training.save_decision_policies.FixedIntervalPolicy(1)
    )
    preservation_policy: PreservationPolicy | None = Field(
        default_factory=lambda: ocp.training.preservation_policies.AnyPreservationPolicy([
            ocp.training.preservation_policies.LatestN(10),
            ocp.training.preservation_policies.EveryNSteps(5),
        ])
    )
    step_name_format: Any | None = None
    custom_metadata: dict | None = None
    cleanup_tmp_directories: bool = False
    lightweight_initialize: bool = False

    @field_validator("save_decision_policy", mode="before")
    @classmethod
    def _simple_fixed_interval_save(cls, value):
        if isinstance(value, int):
            return ocp.training.save_decision_policies.FixedIntervalPolicy(value)
        return value
    
    @field_validator("preservation_policy", mode="before")
    @classmethod
    def _simple_fixed_interval_preservation(cls, value):
        if isinstance(value, int):
            return ocp.training.preservation_policies.EveryNSteps(value)
        return value

    @field_validator("step_name_format", mode="before")
    @classmethod
    def _standard_name_format(cls, value: str | int | Mapping | ocp.path.step.NameFormat | None):
        """Build a standard name format via {step_prefix=..., step_format_fixed_length=...}."""
        if isinstance(value, Mapping):
            return ocp.path.step.standard_name_format(**value)
        elif isinstance(value, str):
            return ocp.path.step.standard_name_format(step_prefix=value)
        elif isinstance(value, int):
            return ocp.path.step.standard_name_format(step_format_fixed_length=value)
        
        return value


class DiagnosticsConfig(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    log_interval: PositiveInt | None = None
    plot_interval: PositiveInt | None = None
    test_interval: PositiveInt | None = None
    callback_interval: PositiveInt | None = None
    progress_callback: Callable[[PyTree, FunctionGraph, Path], None] | None = None
    live_plot: bool = False
    save_plot: dict = Field(default_factory=lambda: dict(fname="loss.pdf", bbox_inches="tight"))
    train_plot: PlotSpec = Field(
        default_factory=lambda: dict(
            kind="line", data=([], []), name="train",
            opts={"xlabel": "Iteration", "ylabel": "Training loss", "yscale": "log", "grid": True},
            kwargs={"color": "green"}
        )
    )
    validation_plot: PlotSpec = Field(
        default_factory=lambda: dict(
            kind="line", data=([], []), name="validation",
            opts={"xlabel": "Iteration", "ylabel": "Validation loss", "yscale": "log", "grid": True},
            kwargs={"color": "orange"}
        )
    )

    @field_validator("save_plot", mode="before")
    @classmethod
    def _from_fname(cls, value):
        """Allow just specifying a filename for save plot."""
        if isinstance(value, str | Path):
            return {"fname": value}
        return value
    
    @field_validator("train_plot", "validation_plot", mode="before")
    @classmethod
    def _fill_plot_spec_data(cls, value, info):
        """Ensure we are only doing line plots, and initialize empty data param."""
        spec = PlotSpec(kind="line", data=([], []))

        if value is None:
            return spec
        
        if not isinstance(value, Mapping):
            raise ValueError(f"{info.field_name} must be a Mapping")
        
        for key in ["opts", "kwargs", "name"]:
            if (ele := value.get(key, None)) is not None and len(ele) > 0:
                spec[key] = ele
        
        return spec


class TerminationConfig(BaseModel):
    """
    Training termination criteria.

    :param max_steps: maximum optimizer steps
    :param loss_tol: rolling relative loss tolerance; disabled when non-positive
    :param test_tol: validation/test tolerance; disabled when non-positive
    :param grad_tol: gradient norm tolerance; disabled when non-positive
    :param max_runtime: runtime limit in seconds or a ``datetime.timedelta` supported string
    """

    @model_validator(mode="before")
    @classmethod
    def _from_plain_max_steps(cls, value):
        if isinstance(value, int):
            return {"max_steps": value}
        return value

    max_steps: PositiveInt = 200
    loss_tol: PositiveFloat | None = None
    test_tol: PositiveFloat | None = None
    grad_tol: PositiveFloat | None  = None
    max_runtime: timedelta = timedelta(seconds=300.0)


class _NullCheckpointer:

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @property
    def latest(self):
        return None
    
    def save_checkpointables(*args, **kwargs):
        pass


class Train(Routine):
    """
    ROM training routine.

    :param loss: loss function, callable as `loss(params, Any) -> float`
    :param init_params: PyTree of initial optimization parameters. Can specify as a callable to generate params.
    :param optimizer: Optax optimizer specification

    :param test: function to compute a validation test score, callable as `test(params) -> float`
    :param dataloader: Optionally load extra data for the loss function
    :param termination: stopping criteria
    :param diagnostics: configs for plotting and logging

    :param root: run directory for checkpoints, logs, and history (optional)
    :param write_policy: ``reuse`` restores checkpoints, ``overwrite`` replaces artifacts, ``error`` fails
    :param checkpointer: Orbax policy checkpoint options

    :param init_seed: random seed for initializing parameters (if init_params is callable)
    :param graph: graph object or YAML path (optional, for graph-related losses and dataloaders)
    """

    # Required
    loss: Callable[[PyTree, Any], float]
    init_params: PyTree
    optimizer: GradientTransformation

    # Optional
    test: Callable[[PyTree], float] | None = None
    dataloader: Iterator[Any] = Field(default_factory=BatchDataLoader)  # empty loading by default
    termination: TerminationConfig = Field(default_factory=TerminationConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    # Persistence
    root: Path | None = None
    write_policy: Literal["reuse", "overwrite", "error"] = "reuse"
    checkpointer: CheckpointerConfig = Field(default_factory=CheckpointerConfig)

    # Other
    init_seed: int = 0
    graph: Annotated[FunctionGraph | None, BeforeValidator(from_yaml)] = None

    @model_validator(mode="after")
    def _setup_extra_train_configs(self):
        """Do some assorted things after model validation."""
        if self.root is not None:
            self.root = self.root.resolve()

        # If init params implements a 'sample' function, then initialize the parameter pytree.
        if (sample_fn := getattr(self.init_params, "sample", None)) is not None and callable(sample_fn):
            self.init_params = self.init_params.sample(jax.random.key(self.init_seed))

        # Pass graph object to loss, test, and dataloader if requested
        if self.graph is not None:
            for attr in ["loss", "test", "dataloader"]:
                if hasattr(ele := getattr(self, attr), "graph"):
                    if ele.graph is None:
                        ele.graph = self.graph
                
                        if isinstance(ele, GraphLoss):
                            GraphLoss._set_default_edge(ele.terms, ele.graph)
        
        # Start dataloader from current training step if applicable
        if self.root is not None and hasattr(self.dataloader, "set_iterator"):
            with ocp.training.Checkpointer(self.root, **dict(self.checkpointer)) as ckptr:
                if ckptr.latest is not None:
                    self.dataloader.set_iterator(ckptr.latest.step + 1)
        
        return self

    @field_validator("write_policy", mode="after")
    @classmethod
    def _check_write_policy(cls, policy: bool, info):
        root = info.data["root"]
        if root is not None:
            if root.exists() and any(root.iterdir()):
                if policy == "error":
                    raise RoutineError(f"Training root already contains artifacts: {root} and policy is '{policy}'")
                if policy == "overwrite":
                    shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
        
        return policy
    
    def run(self) -> int:
        """For compatibility with Routine."""
        self.__call__()
        return 0

    def __call__(self) -> PyTree:

        test_fn = eqx.filter_jit(self.test) if self.test is not None else None
        _, static_params = eqx.partition(self.init_params, eqx.is_array)

        def _checkpoint_params(params: PyTree) -> PyTree:
            """Persist only array-valued leaves so Orbax never sees static callables/modules."""
            return eqx.filter(params, eqx.is_array)

        def _restore_params(dynamic_params: PyTree) -> PyTree:
            """Rebuild the full parameter pytree from checkpointed arrays and the init template."""
            return eqx.combine(dynamic_params, static_params)

        @eqx.filter_jit
        def _step(params: PyTree, opt_state: optax.OptState, batch: PyTree):
            loss, grads = eqx.filter_value_and_grad(self.loss)(params, batch)
            updates, opt_state = self.optimizer.update(grads, opt_state, eqx.filter(params, eqx.is_array))
            params = eqx.apply_updates(params, updates)
            return params, opt_state, loss, grads

        def _save_plot(fig):
            if fig is not None and self.root is not None:
                save_opts = dict(self.diagnostics.save_plot)
                fname = save_opts.pop("fname", "loss.pdf")
                fig.savefig(self.root / fname, **save_opts)
        
        def _load_history_csv(fname: str) -> tuple[list[int], list[float]]:
            if self.root:
                p = self.root / fname
                if not p.exists():
                    return [], []
                arr = np.atleast_2d(np.loadtxt(p, delimiter=",", skiprows=1))
                return arr[:, 0].astype(int).tolist(), arr[:, 1].tolist()
            else:
                return [], []
    
        def _save_history_csv(fname: str, iterations: list[int], values: list[float]):
            if self.root:
                arr = np.column_stack((
                    np.asarray(iterations, dtype=int),
                    np.asarray(values, dtype=float),
                ))
                np.savetxt(
                    self.root / fname, 
                    arr,
                    fmt="%d,%.6e",
                    header="Iteration,Value",
                    comments="",
                )
        
        if self.root is not None:
            checkpointer_context = ocp.training.Checkpointer(self.root, **dict(self.checkpointer))
        else:
            checkpointer_context = _NullCheckpointer()
            
        with checkpointer_context as ckptr:
            ## INITIALIZE/LOAD
            abstract_checkpointables = {
                "params": _checkpoint_params(self.init_params),
                "opt_state": self.optimizer.init(eqx.filter(self.init_params, eqx.is_array)),
            }

            if ckptr.latest is None:
                params = self.init_params
                opt_state = abstract_checkpointables["opt_state"]
                curr_step = 0
                total_steps = self.termination.max_steps
                logger.info("Initialized train")
            else:
                _loaded = ckptr.load_checkpointables(abstract_checkpointables=abstract_checkpointables)
                params = _restore_params(_loaded["params"])
                opt_state = _loaded["opt_state"]
                curr_step = ckptr.latest.step + 1  # starting on next iteration
                total_steps = self.termination.max_steps - curr_step

                if total_steps <= 0:
                    logger.info(f"Training already reached max_steps={self.termination.max_steps} from checkpoint.")
                    return 0
                
                logger.info(f"Restarting train from step {curr_step-1}")
            
            log_interval = self.diagnostics.log_interval or float('inf')
            test_interval = self.diagnostics.test_interval or float('inf')
            plot_interval = self.diagnostics.plot_interval or float('inf')
            callback_interval = self.diagnostics.callback_interval or float('inf')
            
            loss_hist = _load_history_csv("loss.csv")
            test_hist = _load_history_csv("test.csv")
            fig, axs, lines = None, None, None

            if 0 < plot_interval < float('inf'):
                if self.diagnostics.live_plot:
                    plt.ion()

                plot_specs = [self.diagnostics.train_plot]

                if test_fn is not None:
                    plot_specs.append(self.diagnostics.validation_plot)
                
                fig, axs = gridplot(plot_specs)
                lines = [ax.lines[0] for ax in axs.ravel()]
                lines[0].set_data(*loss_hist)

                if test_fn is not None:
                    lines[1].set_data(*test_hist)
            
            def _save_final(metrics=None):
                if 0 < plot_interval < float('inf') and self.diagnostics.live_plot:
                    plt.ioff()
                ckptr.save_checkpointables(
                    step=curr_step, 
                    checkpointables={"params": _checkpoint_params(params), "opt_state": opt_state}, 
                    metrics=metrics, 
                    force=True,
                    overwrite=True
                )
                _save_plot(fig)
                _save_history_csv("loss.csv", *loss_hist)
                if test_fn is not None:
                    _save_history_csv("test.csv", *test_hist)

            t_start = time.time()
                
            with alive_bar(total_steps) as bar:
                while True:
                    ## OPTIMIZER UPDATES
                    try:
                        batch = next(self.dataloader)
                    except StopIteration:
                        logger.info(f"Train dataloader has stopped at step {curr_step}. Terminating...")
                        break
                    
                    try:
                        params, opt_state, loss, grads = _step(params, opt_state, batch)
                        loss = jax.block_until_ready(loss)
                        loss_hist[0].append(curr_step)
                        loss_hist[1].append(float(loss))
                    except Exception as exc:
                        logger.exception(f"Exception encountered during train step {curr_step}. Saving checkpoint...")
                        _save_final()
                        raise RoutineError("Optimizer update failure") from exc
                    
                    ## METRICS AND CHECKPOINT
                    metrics = {"loss": float(loss)}
                    test_score, grad_norm = None, None

                    if test_fn is not None and 0 < test_interval < float('inf') and curr_step % test_interval == 0:
                        test_score = float(test_fn(params))
                        test_hist[0].append(curr_step)
                        test_hist[1].append(test_score)
                        metrics["test_score"] = test_score

                    if self.termination.grad_tol:
                        grad_norm = pytree_norm(grads)
                        metrics["grad_norm"] = float(grad_norm)
                    
                    ckptr.save_checkpointables(
                        step=curr_step, 
                        checkpointables={"params": _checkpoint_params(params), "opt_state": opt_state},
                        metrics=metrics
                    )

                    ## DIAGNOSTICS
                    stats_str = f"loss={float(loss):.2e}"
                    if test_score is not None:
                        stats_str += f" test={test_score:.2e}"
                    if grad_norm is not None:
                        stats_str += f" grad={grad_norm:.2e}"
                    
                    bar()
                    bar.text = stats_str

                    if curr_step % log_interval == 0:
                        logger.info(f"Elapsed: {_prettify_timedelta(time.time() - t_start)} "
                                    f"| step={curr_step} {stats_str}")
                    
                    if 0 < plot_interval < float('inf') and curr_step % plot_interval == 0:
                        lines[0].set_data(*loss_hist)
                        _save_history_csv("loss.csv", *loss_hist)

                        if test_fn is not None:
                            lines[1].set_data(*test_hist)
                            _save_history_csv("test.csv", *test_hist)
                        
                        for ax in axs.ravel():
                            ax.relim()
                            ax.autoscale_view()
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        
                        _save_plot(fig)
                    
                    if (self.diagnostics.progress_callback is not None
                        and 0 < callback_interval < float('inf')
                        and curr_step % callback_interval == 0):
                        self.diagnostics.progress_callback(params, self.graph, self.root)
                    
                    ## END CONDITIONS
                    if self.termination.test_tol and test_score is not None and test_score < self.termination.test_tol:
                        logger.info(f"Termination criteria reached: test score "
                                    f"{test_score:.2e} < {self.termination.test_tol:.2e}")
                        break
                        
                    if self.termination.grad_tol and grad_norm is not None:
                        if not jnp.isfinite(grad_norm):
                            logger.warning("Grad norm is not finite. Terminating...")
                            break

                        if grad_norm < self.termination.grad_tol:
                            logger.info(f"Termination criteria reached: gradient norm "
                                        f"{grad_norm:.2e} < {self.termination.grad_tol:.2e}")
                            break
                    
                    if self.termination.loss_tol and float(loss) < self.termination.loss_tol:
                        logger.info(f"Termination criteria reached: loss "
                                    f"{float(loss):.2e} < {self.termination.loss_tol:.2e}")
                        break

                    if curr_step+1 >= self.termination.max_steps:
                        logger.info(f"Termination criteria reached: "
                                    f"{curr_step+1} / {self.termination.max_steps} iterations")
                        break

                    if (t_diff := time.time() - t_start) >= self.termination.max_runtime.total_seconds():
                        logger.info(f"Termination criteria reached: max runtime "
                                    f"{_prettify_timedelta(t_diff)} / "
                                    f"{_prettify_timedelta(self.termination.max_runtime.total_seconds())}")
                        break

                    curr_step += 1
            
            logger.info(f"Train finished. Elapsed: {_prettify_timedelta(time.time()-t_start)}")
            _save_final(metrics)
        
        return params
