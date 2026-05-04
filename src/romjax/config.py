"""Maintain configuration schemas for various romjax utilities, especially the rom CLI."""
import os
from pathlib import Path
from typing import Literal, Sequence, Callable, Any, Annotated

import jax
import jax.numpy as jnp
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
    BeforeValidator,
    AfterValidator,
)
from jaxtyping import PyTree, ArrayLike

from romjax.graph import FunctionGraph, EdgePyTree
from romjax.model import Sampleable
from romjax.tree import pytree_iter, pytree_size, get_tree_operator, TreeErrorOperator


type LossFunction = Callable[[PyTree, Any], float]


def romjax_from_file(value: str | Path | Any) -> Any:
    """Try to load a romjax object from config file. Useful as a pydantic validator."""
    if isinstance(value, str | Path):
        import romjax
        return romjax.load(value)
    return value


def ensure_path_exists(value: Path):
    if not value.exists():
        os.makedirs(value, exist_ok=True)


class GraphLossSpec(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: Annotated[FunctionGraph, BeforeValidator(romjax_from_file)]


@jax.jit
def reconstruction_error(
    params: EdgePyTree, 
    batch_data: PyTree, 
    graph: FunctionGraph, 
    edge: str,
    error_fn: TreeErrorOperator = "mean-relative"
) -> float:
    """
    Push data through a graph edge and back, and return the reconstruction error.
    
    :param params: pytree containing a map of edge->params. The params are patched into the payload at runtime.
    :param batch_data: batched training data compatible with the starting node of the specified edge
    :param graph: the function graph object that knows how to evaluate edges
    :param edge: the edge name to be used as the reconstruction pathway
    :param leaf_error: how to compute per-leaf errors (see `pytree_error`)
    :param leaf_reducer: how to combine all leaf errors into final result
    :return: the reconstruction error
    """
    batch_size = pytree_size(batch_data)
    error_fn = get_tree_operator(error_fn)
    gen_pytree = pytree_iter(batch_data)

    def body(i, acc):
        payload = next(gen_pytree)
        reconstructed = graph.push_path(payload, [edge, edge], edge_payload_patches={edge: params[edge]})
        return acc + error_fn(payload, reconstructed)
    
    total = jax.lax.fori_loop(0, batch_size, body, 0.0)

    return total / batch_size


class TrainConfig(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    root: Annotated[Path, AfterValidator(ensure_path_exists)]
    loss_fn: LossFunction

    @field_validator("loss_fn", mode="before")
    @classmethod
    def _from_graph_spec(cls, value: LossFunction | GraphLossSpec) -> LossFunction:
        """Generate a preset loss function for a FunctionGraph."""
        if not callable(value):
            spec = GraphLossSpec.model_validate(value)
        
        else:
            return value


class SampleConfig(BaseModel):
    """
    Sampling configuration for a nested input/output sampling strategy.

    :ivar input_samples: number of input samples
    :ivar outputs_per_input: number of output samples for each input sample
    :ivar input_seed: random seed for inputs
    :ivar output_seed: random seed for outputs
    """

    input_samples: int
    outputs_per_input: int
    input_seed: int
    output_seed: int
        

class GenDataConfig(BaseModel):
    """
    Data generation config for a FunctionGraph.
    
    :ivar root: root directory for saving data
    :ivar graph: the FunctionGraph object specifying all models and connections. 
                 May point to a yaml file with the FunctionGraph spec implemented at the top-level
    :ivar train: sampling configuration for each model (see `SampleConfig`) for training dataset
    :ivar validation: sampling configurations for validation dataset
    :ivar to_sample: the names of the models to sample. Each must implement the `Sampleable` protocol
    :ivar format: the data format to save samples. Only `h5` supported.
    :ivar dataset_policy: reuse existing data, overwrite existing data, or throw an error if existing data found
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    root: Path 
    graph: Annotated[FunctionGraph, BeforeValidator(romjax_from_file)]
    train: Sequence[SampleConfig]
    validation: Sequence[SampleConfig]
    to_sample: list[str] | None = None
    batch_size: int = 1
    format: Literal["h5"] = "h5"
    dataset_policy: Literal["reuse", "overwrite", "error"] = "reuse"

    @field_validator("root", mode="after")
    @classmethod
    def _make_root(cls, value: Path):
        if not value.exists():
            os.makedirs(value, exist_ok=True)
        os.makedirs(value / "train", exist_ok=True)
        os.makedirs(value / "validation", exist_ok=True)
        return value
    
    @field_validator("train", "validation", mode="before")
    @classmethod
    def _allow_single_config(cls, value: SampleConfig | Sequence[SampleConfig]) -> Sequence[SampleConfig]:
        if not isinstance(value, Sequence):
            return [value]
        return value
    
    @field_validator("to_sample", mode="wrap")
    @classmethod
    def _validate_sampleable(
        cls, 
        value: str | list[str] | None, 
        handler: ValidatorFunctionWrapHandler, 
        info: ValidationInfo
    ) -> list[str]:
        """If none, default to all sampleables in the graph."""
        if value is None:
            value = []
            for edge_name, edge in info.data['graph'].edges.items():
                if isinstance(edge, Sampleable):
                    value.append(edge_name)
        
        if not isinstance(value, list):
            value = [value]
        
        value = handler(value)

        # Check for the Sampleable required methods
        for name in value:
            if name not in info.data['graph'].edges:
                raise ValueError(f"Model name '{name}' not an edge in the graph, so it cannot be sampled.")
            edge = info.data['graph'].edges[name]
            if not hasattr(edge, 'sample_inputs') or not hasattr(edge, 'sample_outputs'):
                raise ValueError(f"Graph edge object {edge} does not have the required 'sample_inputs' and "
                                 f"'sample_outputs' methods, so it cannot be sampled.")
        return value
    
    @model_validator(mode="after")
    def _validate_sequences(self):
        """Make sure train, validation configs match the length of sampleables (will broadcast len=1)."""
        num_sample = len(self.to_sample)

        if num_sample == 0:
            raise ValueError("Must specify at least one model to sample.")

        if num_sample > 1:
            if len(self.train) == 1:
                for i in range(1, num_sample):
                    new_config = self.train[0].copy()
                    new_config.seed += i
                    self.train.append(new_config)
            
            if len(self.validation) == 1:
                for i in range(1, num_sample):
                    new_config = self.validation[0].copy()
                    new_config.seed += i
                    self.validation.append(new_config)
        
        if len(self.train) != num_sample:
            raise ValueError(f"Number of training configs: {len(self.train)}. Expected {num_sample}")
        if len(self.validation) != num_sample:
            raise ValueError(f"Number of validation configs: {len(self.validation)}. Expected {num_sample}")
        
        return self
