"""Reproducible, file-based sampling via jax.random with pydantic validation."""
import copy
import os
import time
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Literal, Mapping, Optional, Protocol, runtime_checkable

import jax
import jaxtyping
from jax.typing import ArrayLike
from pydantic import field_validator

from romjax.typing import DictModel

__all__ = ['Distribution', 'parametric_sampler', 'gen_keys', 'SamplerCallable']


type DistributionName = Literal['uniform', 'normal']
type ParamName = str


@runtime_checkable
class SamplerCallable(Protocol):
    """Take in a random key and return a single pytree sample. May take any kwargs."""
    def __call__(self, key: jaxtyping.Key, **kwargs: dict[str, Any]) -> jaxtyping.PyTree: ...


@runtime_checkable
class DistributionCallable(Protocol):
    """Take in a random key and extra options and produce an array of samples."""
    def __call__(self, key: jaxtyping.Key, **kwargs: dict[str, Any]) -> ArrayLike: ...


def normal(key: jaxtyping.Key, mean: ArrayLike = 0.0, std: ArrayLike = 1.0, **kwargs) -> ArrayLike:
    """Small wrapper of jax.random.normal to support mean/std args."""
    return jax.random.normal(key, **kwargs) * std + mean


class Distribution(DictModel):
    """Simple class that provides validation for common distributions."""
    distribution: DistributionCallable

    @field_validator('distribution', mode='before')
    @classmethod
    def _coerce_distribution(cls, value: DistributionName | DistributionCallable) -> DistributionCallable:
        if isinstance(value, str):
            mapping: Mapping[DistributionName, DistributionCallable] = {
                'uniform': jax.random.uniform,
                'normal': normal
            }
            if value not in mapping:
                raise ValueError(f"Unknown distribution function: {value!r}")
            return mapping[value]
        if callable(value):
            return value
        raise TypeError("Distribution must be a supported name or a callable")

    @property
    def opts(self):
        return self.model_extra
    
    def sample(self, key: jaxtyping.Key):
        return self.distribution(key, **self.opts)


def parametric_sampler(key: jaxtyping.Key, **params: dict[ParamName, Distribution]) -> jaxtyping.PyTree:
    """
    Independently sample all parameter distributions passed in as kwargs and return as a pytree.
    
    :param key: the jax random key
    :param **params: a map from param names to their distributions (see `Distribution`)
    :return: a PyTree representing a single sample of all parameters
    """
    params = copy.copy(params)
    for k in list(params.keys()):
        if not isinstance(params[k], Mapping):
            raise TypeError("All params for parametric sampling must be a Distribution-like mapping.")
        params[k] = Distribution(**params[k])

    num_rvs = len(params)
    subkeys = jax.random.split(key, num_rvs)
    sample = {rv: dist.sample(subkey) for (rv, dist), subkey in zip(params.items(), subkeys)}

    return sample


def gen_keys(
    indices: int | Iterable[int],
    seed: int | None = None,
    path: str | os.PathLike | None = None,
    skip: Callable[[str | os.PathLike], bool] | Literal['existing'] | None = None
) -> Generator[tuple[jaxtyping.Key, Optional[str | os.PathLike]], None, None]:
    """
    Generator of jax random keys from a given seed. Optionally generate a matching path for saving samples.

    :param indices: the number of keys to generate, or the specific indices (defaults to range(indices))
    :param seed: the jax random seed for the base key
    :param path: optional, if provided also creates and yields a path/seed_i/sample_j folder structure
    :param skip: optional, if provided along with path, this decides whether to skip a certain directory
    :yield: (key_i, Optional[path_i]), the jax random key for each index and optionally the associated path
    """
    if seed is None:
        seed = int(time.time())
    base_key = jax.random.key(seed)

    if isinstance(indices, int):
        indices = range(indices)

    if skip == 'existing':
        skip = lambda p: Path(p).exists()
    if skip is None:
        skip = lambda p: False

    if path is not None:
        seed_path = Path(path) / f"seed_{seed}"
        os.makedirs(seed_path, exist_ok=True)
        info_path = Path(path) / "romjax.txt"
        if not info_path.exists():
            with open(info_path, "w") as fd:
                fd.write(f"Date: {time.asctime()}\n\n"
                        f"This directory has been used as a `romjax` sampling location.\n"
                        f"The structure is seed_i/sample_j for the jth sample of random seed i.\n")
    
    for i in indices:
        if path is not None:
            p = seed_path / f"sample_{i}"

            if skip(p):
                continue

            os.makedirs(p, exist_ok=True)
            yield jax.random.fold_in(base_key, i), p
        
        else:
            yield jax.random.fold_in(base_key, i)
