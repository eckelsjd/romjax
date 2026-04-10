from abc import abstractmethod, ABC

import jax
import equinox as eqx
from jaxtyping import Key, PyTree

from romjax.graph import Edge


class Sampleable(ABC):
    """Mixin for a model to indicate the ability to sample input and output spaces (i.e. the 'coordinates')."""

    @abstractmethod
    def sample_inputs(self, key: Key) -> PyTree:
        """Sample a single model input for the given key."""
        raise NotImplementedError
    
    @abstractmethod
    def sample_outputs(self, key: Key, inputs: PyTree | None = None, solution: PyTree | None = None) -> PyTree:
        """
        Produce one sample of outputs for the given key.
        
        :param key: the random key
        :param inputs: optionally condition on inputs
        :param solution: for efficiency, optionally condition on the precomputed solution of solve(inputs)=0
        :return: the outputs sample
        """
        raise NotImplementedError


class ImplicitModel(Edge, ABC):
    """
    An implicit function f(b,u) that maps inputs/outputs to residuals.
    
    The forward/backward functions are the augmented (and invertible) residual given by F(b,u) = (Id(b), f(b,u)).
    Must implement residual evaluate/solve methods that map (b,u) -> r and (b,r) -> u, respectively.
    """

    def forward(self, x: PyTree) -> PyTree:
        """Pass inputs through and evaluate residuals.
        
        :param x: must be of the form {"inputs": ..., "outputs": ...}
        :return: pytree of the form   {"inputs": ..., "residuals": ...}
        """
        return {"inputs": x["inputs"], "residuals": self.evaluate(x["inputs"], x["outputs"])}
    
    def backward(self, x: PyTree) -> PyTree:
        """Pass inputs through and solve for outputs.
        
        :param x: must be of the form {"inputs": ..., "residuals": ...}
        :return: pytree of the form   {"inputs": ..., "outputs": ...}
        """
        return {"inputs": x["inputs"], "outputs": self.solve(x["inputs"], x["residuals"])}

    @abstractmethod
    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate forward residual function f(b,u)."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, inputs: PyTree, residuals: PyTree) -> PyTree:
        """Solve inverse residual function f(b,u)=r."""
        raise NotImplementedError
    

class ExplicitModel(ImplicitModel, ABC):
    """
    Compute an explicit model via the pushforward: outputs = G(inputs).
    
    Assumes the residual has the same tree structure as the outputs and the pushforward.
    """

    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate the residual as f(b,u) = u - G(b)"""
        return jax.tree.map(lambda u, uhat: u - uhat, outputs, self.pushforward(inputs))
    
    def solve(self, inputs: PyTree, residuals: PyTree) -> PyTree:
        """Solve the inverse (which just computes the pushforward)."""
        return jax.tree.map(lambda r, uhat: r + uhat, residuals, self.pushforward(inputs))
    
    @abstractmethod
    def pushforward(self, inputs: PyTree) -> PyTree:
        """Compute explict outputs from inputs."""
        raise NotImplementedError
    

class FilterModel(Edge):
    """
    Flexible PyTree->PyTree mapping using simple input/output filtering and configurable callables.
    """
    filters: list[]

    def forward(self, x: PyTree) -> PyTree:
        pass

    def backward(self, x: PyTree) -> PyTree:
        pass


class EquinoxInputs(DictModel):
    x: PyTree
    params: eqx.Module
    

def eqxforward(inputs: EquinoxInputs, )

params0 = {
    'edge0': {
        'k1': 1.0,
        'k2': 2.0
    },
    'edge1': {
        'linear': eqx.Module,
        'nonlinear': eqx.Module
    },
    'edge3': eqx.Module
}

def loss_fn(params, args):
    zi, zj, ri, rj = args
    for edge in path:
