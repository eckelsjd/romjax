"""Importable tiny training functions for isolated study integration tests."""
import jax.numpy as jnp
import optuna

from romjax.typing import DictModel


class Initial(DictModel):
    def sample(self, key):
        return {'w': jnp.array(1.0)}


def loss(params, batch):
    return params['w'] ** 2


def score(params):
    return params['w'] ** 2


class AlwaysPrune(optuna.pruners.BasePruner):
    def prune(self, study, trial):
        return True


def stop_after_first(study, trial):
    study.stop()


def broken_loss(params, batch):
    raise ValueError('loss failed')


def broken_callback(study, trial):
    raise RuntimeError('callback failed')
