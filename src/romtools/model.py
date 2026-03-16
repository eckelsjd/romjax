from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from romtools.typing import PyTree

# Logging
# file save/load (h5, abstract, common format across solvers)
# output pytree or array from evaluate (wrapping inputs for PDE solvers?)

class Model(BaseModel, ABC):
    """A function that maps inputs/outputs to residuals."""
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    @abstractmethod
    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate forward residual function."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, inputs: PyTree, residuals: PyTree) -> PyTree:
        """Solve inverse residual function."""
        raise NotImplementedError

    @classmethod
    def yaml_tag(cls) -> str:
        """YAML tag used by YamlLoader for this model class."""
        return f"!model:{cls.__module__}.{cls.__name__}"
