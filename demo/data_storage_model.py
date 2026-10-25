"""Importable algebraic implicit model for the indexed-storage demonstration."""

import jax
import jax.numpy as jnp
from jaxtyping import Key, PyTree

from romjax.graph import Edge
from romjax.model import ImplicitSampleable


class ToyImplicit(Edge, ImplicitSampleable):
    """Solve y = 2x and sample condition-controlled perturbations of its solution."""

    def forward(self, x: PyTree) -> PyTree:
        """Return the input tree."""
        return x

    def backward(self, x: PyTree) -> PyTree:
        """Return the input tree."""
        return x

    def sample_inputs(self, key: Key) -> PyTree:
        """Sample one model input."""
        return {"x": jax.random.uniform(key, (1,))}

    def solve(self, inputs: PyTree) -> PyTree:
        """Solve the algebraic equation."""
        return {"y": 2 * inputs["x"]}

    def sample_conditions(self, key: Key) -> PyTree:
        """Sample the output perturbation."""
        return {"noise": 0.1 * jax.random.normal(key, (1,))}

    def sample_outputs(
        self, key: Key, inputs: PyTree | None = None, solution: PyTree | None = None, conditions: PyTree | None = None
    ) -> PyTree:
        """Perturb the base solution using the persisted conditions."""
        del key
        solution = self.solve(inputs) if solution is None else solution
        return {"y": solution["y"] + conditions["noise"]}

    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate the algebraic residual."""
        return {"r": outputs["y"] - jnp.multiply(2, inputs["x"])}
