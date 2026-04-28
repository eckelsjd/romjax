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
    Field
) 

from romjax.typing import RoxObject
from romjax.graph import FunctionGraph
from romjax.model import Sampleable
from romjax import YamlLoader
from romjax.utils import Logger


class SampleConfig(BaseModel):

    input_samples: int
    outputs_per_input: int
    input_seed: int
    output_seed: int
        

class GenDataConfig(BaseModel, RoxObject):
    """Data generation config."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_default=True)

    root: Path 
    graph: FunctionGraph
    train: Sequence[SampleConfig]
    validation: Sequence[SampleConfig]
    batch_size: int = 1
    to_sample: list[str] | None = None
    use_solution: Sequence[bool] = True
    format: Literal["h5"] = "h5"
    dataset_policy: Literal["reuse", "overwrite", "error"] = "reuse"
    logger: Logger = Field(default_factory=Logger)

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
    
    @field_validator("use_solution", mode="before")
    @classmethod
    def _allow_single_use_solution(cls, value):
        if not isinstance(value, Sequence):
            return [value]
        return value
    
    @field_validator("logger", mode="before")
    @classmethod
    def _add_logger_name(cls, value):
        if hasattr(value, "name"):
            value.name = "Data generation"
        elif isinstance(value, Mapping):
            value['name'] = "Data generation"
        
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
            
            if len(self.use_solution) == 1:
                for i in range(1, num_sample):
                    self.use_solution.append(self.use_solution[0])
        
        if len(self.train) != num_sample:
            raise ValueError(f"Number of training configs: {len(self.train)}. Expected {num_sample}")
        if len(self.validation) != num_sample:
            raise ValueError(f"Number of validation configs: {len(self.validation)}. Expected {num_sample}")
        if len(self.use_solution) != num_sample:
            raise ValueError(f"Number of use_solution configs: {len(self.use_solution)}. Expected {num_sample}")
        
        return self
