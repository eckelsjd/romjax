"""Maintain configuration schemas for various romjax utilities, especially the rom CLI."""
from typing import Literal, Sequence, Mapping
from pathlib import Path
import os

from pydantic import (
    BaseModel, 
    ConfigDict, 
    field_validator, 
    ValidationInfo, 
    model_validator, 
    ValidatorFunctionWrapHandler,
) 

from romjax.typing import RoxObject
from romjax.graph import FunctionGraph
from romjax.model import Sampleable
from romjax import YamlLoader


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
        

class GenDataConfig(BaseModel, RoxObject):
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
    graph: FunctionGraph
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

    @field_validator("graph", mode="before")
    @classmethod
    def _load_graph(cls, value: str | Path | FunctionGraph) -> FunctionGraph:
        """Load a graph from a yaml file."""
        if isinstance(value, str | Path) and value.endswith((".yml", ".yaml")):
            return YamlLoader.load(value)
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
