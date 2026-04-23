"""Reproducible, file-based sampling via jax.random with pydantic validation."""
import copy
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Literal, Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jaxtyping
from jax.typing import ArrayLike
from pydantic import field_validator

from romjax.typing import DictModel

__all__ = ['Distribution', 'parametric_sampler', 'validate_distribution_pytree', 'pytree_sampler',
           'near_solution_sampler', 'gen_keys', 'SamplerCallable']


type DistributionName = Literal['uniform', 'normal']
type ParamName = str
type ScaleReducerName = Literal['mean', 'rms', 'max_abs']
type ScaleReducer = Callable[[ArrayLike], ArrayLike]
type RelativeScale = tuple[ScaleReducer | ScaleReducerName, float]


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


def validate_distribution_pytree(template: jaxtyping.PyTree) -> jaxtyping.PyTree:
    """Validate every leaf in a pytree-like template as a :class:`Distribution`."""
    if isinstance(template, Distribution):
        return template
    if isinstance(template, Mapping):
        if "distribution" in template:
            return Distribution(**template)
        return {key: validate_distribution_pytree(value) for key, value in template.items()}
    if isinstance(template, tuple):
        return tuple(validate_distribution_pytree(value) for value in template)
    if isinstance(template, list):
        return [validate_distribution_pytree(value) for value in template]
    raise TypeError("All leaves for pytree sampling must be Distribution-like mappings.")


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


def pytree_sampler(key: jaxtyping.Key, template: jaxtyping.PyTree) -> jaxtyping.PyTree:
    """
    Sample an arbitrary pytree template whose leaves are ``Distribution``-like specs.

    :param key: the JAX random key
    :param template: pytree whose leaves are Distribution instances or Distribution-like mappings
    :return: pytree with the same structure as ``template`` and sampled array leaves
    """
    template = validate_distribution_pytree(template)
    leaves, treedef = jax.tree.flatten(template, is_leaf=lambda value: isinstance(value, Distribution))
    subkeys = jax.random.split(key, len(leaves))
    samples = [dist.sample(subkey) for dist, subkey in zip(leaves, subkeys)]
    return jax.tree.unflatten(treedef, samples)


def _is_relative_scale_spec(value: Any) -> bool:
    """Return True when ``value`` is a ``(reducer, factor)`` relative-scale specification."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and (callable(value[0]) or value[0] in {"mean", "rms", "max_abs"})
    )


def _coerce_scale_reducer(reducer: ScaleReducer | ScaleReducerName) -> ScaleReducer:
    """Resolve supported relative-scale reducers from callables or string names."""
    if callable(reducer):
        return reducer
    mapping: Mapping[ScaleReducerName, ScaleReducer] = {
        "mean": jnp.mean,
        "rms": lambda x: jnp.sqrt(jnp.mean(jnp.square(x))),
        "max_abs": lambda x: jnp.max(jnp.abs(x)),
    }
    if reducer not in mapping:
        raise ValueError(f"Unknown relative scale reducer: {reducer!r}")
    return mapping[reducer]


def _resolve_scale_leaf(reference: ArrayLike, scale_spec: ArrayLike | RelativeScale) -> ArrayLike:
    """Resolve one scale leaf, supporting both absolute and relative scale specifications."""
    if _is_relative_scale_spec(scale_spec):
        reducer, factor = scale_spec
        reducer_fn = _coerce_scale_reducer(reducer)
        return jnp.asarray(reducer_fn(jnp.asarray(reference))) * jnp.asarray(factor)
    return jnp.asarray(scale_spec)


def _broadcast_like(template: jaxtyping.PyTree, value: ArrayLike | jaxtyping.PyTree) -> jaxtyping.PyTree:
    """Broadcast a scalar or pytree ``value`` across the structure of ``template`` when needed."""
    if _is_relative_scale_spec(value):
        return jax.tree.map(lambda _: value, template)
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return jax.tree.map(lambda _: value, template)


def near_solution_sampler(
    key: jaxtyping.Key,
    *,
    inputs: jaxtyping.PyTree | None = None,
    solution: jaxtyping.PyTree | None = None,
    noise: jaxtyping.PyTree | None = None,
    scale: ArrayLike | jaxtyping.PyTree = 1.0,
    **noise_kwargs: dict[str, Any],
) -> jaxtyping.PyTree:
    """
    Sample an output pytree by adding sampled noise to a reference solution pytree.

    ``noise`` may be supplied explicitly, or inferred from ``noise_kwargs`` for convenient
    YAML configuration where output keys map directly to ``Distribution`` specs.

    :param key: the JAX random key
    :param inputs: optional conditioning inputs passed for API consistency
    :param solution: reference output pytree to perturb
    :param noise: optional pytree template of Distribution-like leaves
    :param scale: optional scalar or pytree multiplier applied to sampled noise before addition
    :param **noise_kwargs: fallback noise template when ``noise`` is omitted
    :return: sampled output pytree with the same structure as ``solution``
    """
    del inputs
    if solution is None:
        raise ValueError("near_solution sampler requires a reference solution.")
    if noise is None:
        if not noise_kwargs:
            raise ValueError("near_solution sampler requires a noise pytree or Distribution-like kwargs.")
        noise = noise_kwargs
    elif noise_kwargs:
        raise ValueError("Pass noise specs either via `noise=` or keyword template entries, not both.")

    sampled_noise = pytree_sampler(key, noise)
    scale_tree = _broadcast_like(solution, scale)
    resolved_scale = jax.tree.map(
        _resolve_scale_leaf,
        solution,
        scale_tree,
    )
    scaled_noise = jax.tree.map(
        lambda sample, sample_scale: jnp.asarray(sample) * jnp.asarray(sample_scale),
        sampled_noise,
        resolved_scale,
    )
    return jax.tree.map(lambda ref, delta: jnp.asarray(ref) + delta, solution, scaled_noise)


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
