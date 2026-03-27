"""Module for assorted processing utilities."""
from typing import Mapping
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from pydantic import BaseModel

from romtools.typing import PyTree

__all__ = ['to_pytree', 'merge_pytrees', 'get_logger', 'tree_l2_norm']

LOG_FORMATTER = logging.Formatter(u"%(asctime)s — [%(levelname)s] — %(name)-15s — %(message)s")


@jax.jit
def tree_l2_norm(tree: PyTree):
    return jnp.sqrt(jax.tree.reduce(lambda acc, x: acc + jnp.sum(x**2), tree, 0.0))


def to_pytree(value: PyTree) -> PyTree:
    """Convert nested pydantic models and dicts to a PyTree of just 
    dicts,tuples,lists -- anything else is left as a leaf node (i.e. jax arrays)."""
    if isinstance(value, BaseModel):
        data = value.model_dump()
        return {k: to_pytree(v) for k, v in data.items()}
    if isinstance(value, Mapping):
        return {k: to_pytree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_pytree(v) for v in value)
    if isinstance(value, list):
        return [to_pytree(v) for v in value]
    
    return value


def merge_pytrees(defaults: PyTree, overrides: PyTree) -> PyTree:
    """Merge pytrees, overwriting existing paths and adding any new ones.
    
    :param defaults: the existing pytree
    :param overrides: the pytree to merge
    :return: a new merged pytree
    """
    if overrides is None:
        return defaults
    if defaults is None:
        return overrides
    
    if isinstance(defaults, Mapping) and isinstance(overrides, Mapping):
        merged: dict = dict(defaults)
        for key, value in overrides.items():
            if key in merged:
                merged[key] = merge_pytrees(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    if isinstance(defaults, tuple) and isinstance(overrides, tuple):
        merged = []
        common = min(len(defaults), len(overrides))
        for idx in range(common):
            merged.append(merge_pytrees(defaults[idx], overrides[idx]))
        if len(defaults) > common:
            merged.extend(defaults[common:])
        if len(overrides) > common:
            merged.extend(overrides[common:])
        return tuple(merged)
    
    if isinstance(defaults, list) and isinstance(overrides, list):
        merged_list: list = []
        common = min(len(defaults), len(overrides))
        for idx in range(common):
            merged_list.append(merge_pytrees(defaults[idx], overrides[idx]))
        if len(defaults) > common:
            merged_list.extend(defaults[common:])
        if len(overrides) > common:
            merged_list.extend(overrides[common:])
        return merged_list
    
    return overrides


def get_logger(name: str, stdout: bool = True, log_file: str | Path = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return a file/stdout logger with the given name.

    :param name: the name of the logger to return
    :param stdout: whether to add a stdout stream handler to the logger
    :param log_file: add file logging to this file (optional)
    :param level: the logging level to set
    :returns: the logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    if stdout:
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(std_handler)
    if log_file is not None:
        f_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        f_handler.setLevel(level)
        f_handler.setFormatter(LOG_FORMATTER)
        logger.addHandler(f_handler)

    return logger
