"""Reproducible, file-based sampling via jax.random with pydantic validation."""
import time
from pathlib import Path
from typing import Protocol, Iterable, Any, Literal, Mapping, runtime_checkable
import os
import copy

import jax
import jaxtyping
import jax.numpy as jnp
from jax.typing import ArrayLike
from pydantic import field_validator

from romtools.utils import save_h5
from romtools.typing import DictModel

__all__ = ['Distribution', 'parametric_sampler', 'sampling_keys', 'BatchSampler']


type DistributionName = Literal['uniform', 'normal']
type ParamName = str

@runtime_checkable
class BatchSampler(Protocol):
    """Take in random keys and corresponding paths, write samples to paths however you want. May take any kwargs."""
    def __call__(self, keys: Iterable[jaxtyping.Key], paths: Iterable[str | Path], 
                 **kwargs: dict[str, Any]) -> None: ...


@runtime_checkable
class DistributionCallable(Protocol):
    """Take in a random key and extra options and produce an array of samples."""
    def __call__(self, key: jaxtyping.Key, **kwargs: dict[str, Any]) -> ArrayLike: ...


def normal(key: jaxtyping.Key, mean: ArrayLike = 0.0, std: ArrayLike = 1.0, **kwargs) -> ArrayLike:
    """Small wrapper of jax.random.normal to support mean/std args."""
    return jax.random.normal(key, **kwargs) * std + mean


class Distribution(DictModel):
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
        raise TypeError(f"Distribution must be a supported name or a callable")

    @property
    def opts(self):
        return self.model_extra
    
    def sample(self, key: jaxtyping.Key):
        return self.distribution(key, **self.opts)


def parametric_sampler(
    keys: Iterable[jaxtyping.Key], 
    paths: Iterable[str | Path],
    format: Literal['h5', 'txt'] = 'h5',
    prefix: str = 'sample',
    write_summary: bool = True,
    **params: dict[ParamName, Distribution]
) -> None:
    """Independently sample all parameter distributions passed in as kwargs and save to file.
    
    :param keys: the jax random keys per sample
    :param paths: the base paths to save samples associated with keys
    :param format: data format for the generated arrays (defaults to h5). For txt, only scalars are supported.
    :param write_summary: write summary of saved data array names and shapes (default true)
    :param **params: a map from param names to their distributions (see `Distribution`)
    """
    _supported = ['h5', 'txt']
    if format not in _supported:
        raise ValueError(f"Format '{format}' not supported. Only {_supported}.")

    params = copy.copy(params)
    for k in list(params.keys()):
        if not isinstance(params[k], Mapping):
            raise TypeError(f"All params for parametric sampling must be a Distribution-like mapping.")
        params[k] = Distribution(**params[k])
    num_rvs = len(params)

    for key, path in zip(keys, paths):
        subkeys = jax.random.split(key, num_rvs)
        sample = {rv: dist.sample(subkey) for (rv, dist), subkey in zip(params.items(), subkeys)}

        if format == 'h5':
            save_h5(sample, Path(path) / f"{prefix}.h5")
        elif format == 'txt':
            if any([len(arr.squeeze().shape) > 0 for arr in sample.values()]):
                raise ValueError("Can't write non-scalar array to txt format")
            with open(Path(path) / f"{prefix}.txt", "w") as fd:
                header = " ".join(list(sample.keys()))
                data = " ".join([f"{arr.squeeze():.6E}" for arr in sample.values()])
                fd.write(f"{header}\n{data}\n")

        if write_summary:
            with open(Path(path) / f"{prefix}_summary.txt", 'w') as fd:
                param_lines = [f"{param}: shape={arr.shape} dtype={arr.dtype}" for param, arr in sample.items()]
                fd.writelines([f"Date: {time.asctime()}\n\n"] + param_lines)


def sampling_keys(
    size: int,
    path: str | Path, 
    seed: int | None = None, 
    reuse: bool = True,
) -> tuple[list[jaxtyping.Key], list[Path]]:
    """Setup and return keys/paths for file-based reproducible sampling using jax."""
    if seed is None:
        seed = int(time.time())
    base_key = jax.random.key(seed)

    seed_path = Path(path) / f"seed_{seed}"
    os.makedirs(seed_path, exist_ok=True)

    info_path = Path(path) / "ROMTOOLS.txt"
    if not info_path.exists():
        with open(info_path, "w") as fd:
            fd.write(f"Date: {time.asctime()}\n\n"
                     f"This directory has been used as a `romtools` sampling location.\n"
                     f"The structure is seed_i/sample_j for the jth sample of random seed i.\n"
                     f"Sample directories with 'ignore' in the name will be ignored.\n")

    existing = [f for f in os.listdir(seed_path) if Path(f).is_dir() and str(f).startswith("sample_") and 
                "ignore" not in str(f)]
    existing_int = {int(f.split("_")[1]) for f in existing}
    new_int = [i for i in range(size) if not reuse or (i not in existing_int)]

    keys, paths = [], []
    for i in new_int:
        p = seed_path / f"sample_{i}"
        os.makedirs(p, exist_ok=True)
        paths.append(p)
        keys.append(jax.random.fold_in(base_key, i))
    
    return keys, paths
