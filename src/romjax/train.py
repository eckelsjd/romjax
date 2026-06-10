"""Reduced-order model training routine."""
import functools
import shutil
import time
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

from romjax.data_gen import DataLoader, LoadDataConfig
from romjax.graph import FunctionGraph
from romjax.model import ImplicitSampleable, SourceSampleable
from romjax.plotting import PlotSpec, gridplot
from romjax.routine import Routine, RoutineError
from romjax.tree import (
    TreeErrorOperator,
    UnaryOperator,
    get_subtree,
    get_unary_operator,
    pytree_norm,
    pytree_path_iter,
    pytree_resolve_refs,
    pytree_square_norm,
)
from romjax.typing import CallableModel, ThirdPartyType, from_registry, from_yaml, require_type, resolve_graph_refs
from romjax.utils import _NullProgress

__all__ = ["Train", "GraphLoss", "GraphTest", "BatchLoader"]


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


class BatchLoader[T: Any](BaseModel, Iterator):
    """
    Helper for basic mini-batch data loading.

    !!! Example
        ```python
        data = list(range(10))
        
        for batch in BatchLoader(data=data, batch_size=2):
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


def reconstruction_loss(
    params: PyTree, 
    single_data: PyTree, 
    graph: FunctionGraph, 
    path: list[str] = None,
    error_op: TreeErrorOperator | None = None,
    ignore: set | None = None,
):
    return graph.reconstruction_error(single_data, path, edge_payload_patches=params, error_op=error_op, ignore=ignore)


def tikhonov_regularization(params: PyTree, single_data: PyTree, graph: FunctionGraph):
    del single_data, graph
    return pytree_square_norm(params)


def orthogonal_regularization(params: PyTree, single_data: PyTree, graph: FunctionGraph, ref: list[str] = None):
    del single_data, graph
    matrix = get_subtree(params, ref)  # expected Array[r x N] projection matrix
    if matrix is None:
        raise ValueError("Can't locate matrix for orthogonal regularization via ref: '{ref}'")
    
    gram = matrix @ matrix.T
    return pytree_square_norm(gram - jnp.eye(gram.shape[0]))


_LOSS_REGISTRY = {
    "reconstruction": reconstruction_loss,
    "tikhonov": tikhonov_regularization,
    "orthogonal": orthogonal_regularization,
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
    :param dataset: which dataset name to read data from
    :param weight: scalar term weight
    :param batch_reduce: reduce the loss over batch data; skip batch reduce if none
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    function: GraphLossFunction
    dataset: str | None = None
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
            if self.dataset is not None and self.dataset not in batch_data:
                return jnp.asarray(0.0)  # if a dataset runs out during iteration
            
            term_batch = batch_data[self.dataset] if self.dataset is not None else batch_data

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
    def _bind_default_datasets(self):
        if self.graph is not None:
            self._set_default_datasets()
        return self
    
    def _set_default_datasets(self):
        """Grab the first sampleable edge as the default dataset, i.e. typically there is only one."""
        if self.graph is not None:
            _default_edge = None
            for edge_name, edge in self.graph.edges.items():
                if isinstance(edge, ImplicitSampleable | SourceSampleable):
                    _default_edge = edge_name
                break

            for term in self.terms:
                if term.dataset is None:
                    term.dataset = _default_edge

    def __call__(self, params: Mapping[str, PyTree], batch: Mapping[str, PyTree]) -> jax.Array:
        """Parameters are specified on a per-edge basis. Data batches will also be passed per-edge."""
        if self.graph is None:
            raise ValueError("Must specify a FunctionGraph to evaluate GraphLoss")
        
        params = pytree_resolve_refs(params)

        total = 0.0
        for term in self.terms:
            total = total + term(params, batch, self.graph)
        return total


class GraphTest(GraphLoss):
    """
    Just compute a graph loss function over a set of validation data.
    """

    loader: DataLoader
    reduce: UnaryOperator | None = "mean"
    _batch_loss: Callable[[PyTree, PyTree], ArrayLike] = PrivateAttr()
    
    @field_validator("reduce", mode="before")
    @classmethod
    def _get_unary_operator(cls, value):
        if value is not None:
            return get_unary_operator(value)
        return value
    
    @model_validator(mode="after")
    def _validate_loader_and_loss(self):
        for _, ds_cfg in pytree_path_iter(self.loader.datasets, is_leaf=lambda leaf: isinstance(leaf, LoadDataConfig)):
            ds_cfg.max_epochs = 1   # Only load data once

        self._batch_loss = eqx.filter_jit(lambda batch, params: super(GraphTest, self).__call__(batch, params))
        return self

    def __call__(self, params: Mapping[str, PyTree]) -> jax.Array:
        values = jnp.asarray([self._batch_loss(params, batch) for batch in self.loader])
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
    show_progress: bool = True  # progress bar
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
    init_params: Annotated[PyTree, BeforeValidator(from_yaml)]
    optimizer: GradientTransformation

    # Optional
    test: Callable[[PyTree], float] | None = None
    dataloader: Iterator[Any] = Field(default_factory=BatchLoader)  # empty loading by default
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
        # Pass graph object to loss, test, and dataloader if requested
        if self.graph is not None:
            for attr in ["loss", "test", "dataloader"]:
                if hasattr(ele := getattr(self, attr), "graph"):
                    if ele.graph is None:
                        ele.graph = self.graph
                
                        if isinstance(ele, GraphLoss):
                            ele._set_default_datasets()

            self.init_params = resolve_graph_refs(self.init_params, self.graph)

        sample_fn = getattr(self.init_params, "sample", None)
        if callable(sample_fn):
            self.init_params = sample_fn(jax.random.key(self.init_seed))
        
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
                    # Just rm previous train artifacts
                    for path in root.iterdir():
                        if path.is_dir() and path.joinpath("_CHECKPOINT_METADATA").is_file():
                            shutil.rmtree(path)
                        elif path.is_file() and path.suffix in {".csv", ".pdf"}:
                            path.unlink()
            root.mkdir(parents=True, exist_ok=True)
        
        return policy
    
    def run(self) -> int:
        """For compatibility with Routine."""
        self.__call__()
        return 0

    def __call__(self) -> PyTree:

        test_fn = self.test if self.test is not None else None
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
                logger.debug("Initialized train")
            else:
                _loaded = ckptr.load_checkpointables(abstract_checkpointables=abstract_checkpointables)
                params = _restore_params(_loaded["params"])
                opt_state = _loaded["opt_state"]
                curr_step = ckptr.latest.step + 1  # starting on next iteration
                total_steps = self.termination.max_steps - curr_step

                if total_steps <= 0:
                    logger.debug(f"Training already reached max_steps={self.termination.max_steps} from checkpoint.")
                    return 0
                
                logger.debug(f"Restarting train from step {curr_step-1}")
            
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

            ctxt = alive_bar(total_steps) if self.diagnostics.show_progress else _NullProgress()
                
            with ctxt as bar:
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
                        logger.debug(f"Elapsed: {_prettify_timedelta(time.time() - t_start)} "
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
            
            logger.debug(f"Train finished. Elapsed: {_prettify_timedelta(time.time()-t_start)}")
            _save_final(metrics)
        
        return params
