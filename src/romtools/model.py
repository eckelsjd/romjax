from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

# Logging
# file save/load (h5, abstract, common format across solvers)

class Model(BaseModel, ABC):
    """A function that maps inputs/outputs to residuals."""
    model_config = ConfigDict(validate_assignment=True, validate_default=True)

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate forward residual function."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, *args: Any, **kwargs: Any) -> Any:
        """Solve inverse residual function."""
        raise NotImplementedError

    @classmethod
    def yaml_tag(cls) -> str:
        """YAML tag used by YamlLoader for this model class."""
        return f"!model:{cls.__module__}.{cls.__name__}"
