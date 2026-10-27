"""Small deterministic regression problem for the Optuna study demo."""
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from optuna import Trial

from romjax.configuration import ConfigAssignment
from romjax.typing import DictModel


class Initial(DictModel):
    """Initialize a two-coefficient linear model."""

    def sample(self, key: jax.Array) -> PyTree:
        """Return zero coefficients; the demo has no initialization randomness."""
        return {'weights': jnp.zeros(2)}


class RegressionLoss(DictModel):
    """Training error with a configurable regularization family and weight."""

    penalty: Literal['l1', 'l2'] = 'l2'
    weight: float = 0.001

    def __call__(self, params: PyTree, batch: object) -> jax.Array:
        """Evaluate differentiable training loss for one coefficient tree."""
        x = jnp.array([-1., 0., 1., 2.])
        weights = params['weights']
        error = jnp.mean((weights[0] * x + weights[1] - (1.5 * x + 0.2)) ** 2)
        penalty = jnp.sum(jnp.abs(weights)) if self.penalty == 'l1' else jnp.sum(weights ** 2)
        return error + self.weight * penalty


def validation(params: PyTree) -> jax.Array:
    """Evaluate unregularized error on held-out inputs."""
    x = jnp.array([-0.5, 0.5, 1.5])
    weights = params['weights']
    return jnp.mean((weights[0] * x + weights[1] - (1.5 * x + 0.2)) ** 2)


def conditional_optimizer(trial: Trial) -> list[ConfigAssignment]:
    """Choose an optimizer family and conditionally suggest momentum."""
    family = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    rate = trial.suggest_float(f'{family}_learning_rate', 1e-4, 0.1, log=True)
    kwargs = {'learning_rate': rate}
    if family == 'sgd':
        kwargs['momentum'] = trial.suggest_float('momentum', 0.0, 0.95)
    return [ConfigAssignment(path=('optimizer',), value={'name': f'optax.{family}', 'kwargs': kwargs})]
