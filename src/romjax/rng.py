"""Reproducible, file-based sampling via jax.random with pydantic validation."""
import functools
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jaxtyping
from jax.typing import ArrayLike
from pydantic import AfterValidator, BeforeValidator, Field, TypeAdapter, model_validator

from romjax.operators import UnaryOp
from romjax.random_field import darcy, kle
from romjax.tree import pytree_merge
from romjax.typing import CallableModel, ThirdPartyType, from_registry, require_type

__all__ = ['Distribution', 'SamplerCallable', 'DistributionCallable', 'DistributionPyTree', 'PyTreeSampler',
           'NearSolutionSampler', 'gen_keys']


type RelativeScale = tuple[UnaryOp, float]


class SamplerCallable(CallableModel):
    """Take in a random key and return a single pytree sample. May take any kwargs."""

    def __call__(self, key: jaxtyping.Key, **kwargs: dict[str, Any]) -> jaxtyping.PyTree:
        return super().__call__(key, **kwargs)


def uniform(key: jaxtyping.Key, shape=(), dtype=None, minval=0.0, maxval=1.0, *, out_sharding=None):
    """Small wrapper of jax.random.uniform to convert list bounds into arrays."""
    return jax.random.uniform(key, shape=shape, dtype=dtype, minval=jnp.asarray(minval), maxval=jnp.asarray(maxval),
                              out_sharding=out_sharding)


def log_uniform(key: jaxtyping.Key, shape=(), dtype=None, minval=-6.0, maxval=0.0, *, out_sharding=None):
    """Sample values uniformly in log10 space between ``10**minval`` and ``10**maxval``."""
    log10_sample = jax.random.uniform(
        key,
        shape=shape,
        dtype=dtype,
        minval=jnp.asarray(minval),
        maxval=jnp.asarray(maxval),
        out_sharding=out_sharding,
    )
    return jnp.power(jnp.asarray(10, dtype=log10_sample.dtype), log10_sample)


def normal(key: jaxtyping.Key, mean: ArrayLike = 0.0, std: ArrayLike = 1.0, **kwargs) -> ArrayLike:
    """Small wrapper of jax.random.normal to support mean/std args."""
    return jax.random.normal(key, **kwargs) * jnp.asarray(std) + jnp.asarray(mean)


def dirac(key: jaxtyping.Key, value: ArrayLike = 0.0) -> ArrayLike:
    """Just return a constant value as a dirac distribution."""
    return jnp.asarray(value)


def random_eqx_module(
    key: jaxtyping.Key, 
    name: str = None, 
    args: tuple | list = (), 
    kwargs: dict = None,
    default_modules: str | Sequence[str] = "romjax.nn"
) -> eqx.Module:
    """
    Construct a random eqx.Module object using the provided key and class constructor info. The class constructor
    must take `key` as a kwarg in the init method. For compatibility with PyTreeSampler, the returned object should 
    subclass eqx.Module.

    See `ThirdPartyType` for details.
    """
    if name is None:
        raise ValueError("Cannot construct third-party object with empty 'name'.")
    if kwargs is None:
        kwargs = {}
    
    type EquinoxModule = Annotated[
        ThirdPartyType(default_modules=default_modules),
        AfterValidator(functools.partial(require_type, eqx.Module)),
    ]

    kwargs["key"] = key
    ta = TypeAdapter(EquinoxModule)

    return ta.validate_python({"name": name, "args": args, "kwargs": kwargs})


_distribution_registry = {
    "uniform": uniform,
    "log_uniform": log_uniform,
    "normal": normal,
    "darcy": darcy,
    "kle": kle,
    "dirac": dirac,
    "eqx_module": random_eqx_module
}

type DistributionCallable = Annotated[Callable[[jaxtyping.Key], ArrayLike | eqx.Module], 
                                      BeforeValidator(functools.partial(from_registry, _distribution_registry))]
"""Take in a random key and extra options and produce an array of samples."""


class Distribution(CallableModel):
    """Simple class that provides validation for common distributions."""

    callable: DistributionCallable

    @model_validator(mode="before")
    @classmethod
    def _dirac_distribution_from_number(cls, value):
        if isinstance(value, float | int):
            return {"callable": dirac, "value": value}
        return value
    
    def sample(self, key: jaxtyping.Key):  # Just an alias
        return super().__call__(key)


def validate_distribution_pytree(template: jaxtyping.PyTree) -> jaxtyping.PyTree:
    """Validate every leaf in a pytree-like template as a :class:`Distribution`. Leave anything else untouched."""
    if callable(template) or isinstance(template, float | int):
        return Distribution.model_validate(template)
    if isinstance(template, Mapping):
        if "callable" in template:
            return Distribution(**template)
        if "name" in template:
            return Distribution(callable="eqx_module", **template)
        return {key: validate_distribution_pytree(value) for key, value in template.items()}
    if isinstance(template, tuple):
        return tuple(validate_distribution_pytree(value) for value in template)
    if isinstance(template, list):
        return [validate_distribution_pytree(value) for value in template]
    
    return template


type DistributionPyTree = Annotated[jaxtyping.PyTree, BeforeValidator(validate_distribution_pytree)]


class PyTreeSampler(SamplerCallable):

    template: DistributionPyTree = Field(default_factory=dict)
    __pydantic_extra__: dict[str, Any] = Field(init=False, default_factory=dict)

    @model_validator(mode="after")
    def _merge_distributions(self):
        validated_extra = validate_distribution_pytree(self.model_extra)
        object.__setattr__(self, "template", pytree_merge(validated_extra, self.template))
        object.__setattr__(self, "__pydantic_extra__", {})
        return self

    def __call__(self, key: jaxtyping.Key, **kwargs: dict[str, Any]) -> jaxtyping.PyTree:
        runtime_template = kwargs.pop("template", {})
        if runtime_template:
            runtime_template = validate_distribution_pytree(runtime_template)
        template = pytree_merge(runtime_template, self.template)
        callable_fn = type(self).model_fields["callable"].default
        return callable_fn(self, key, **kwargs, **template)
    
    def callable(self, key: jaxtyping.Key, **template) -> jaxtyping.PyTree:
        """
        Sample an arbitrary pytree template whose leaves are ``Distribution``-like specs.

        Assumes third-party type samplers return eqx.Module, which will be treated as leaves.

        :param key: the JAX random key
        :param template: pytree whose leaves are Distribution instances or Distribution-like mappings
        :return: pytree with the same structure as ``template`` and sampled array leaves
        """
        dist_tree, other_tree = eqx.partition(template, lambda leaf: isinstance(leaf, Distribution))
        leaves, treedef = jax.tree.flatten(dist_tree, is_leaf=lambda leaf: isinstance(leaf, Distribution))
        subkeys = jax.random.split(key, len(leaves))
        samples = [dist.sample(subkey) for dist, subkey in zip(leaves, subkeys)]
        ret = eqx.combine(
            jax.tree.unflatten(treedef, samples), 
            other_tree,
            is_leaf = lambda val: eqx.is_array(val) | isinstance(val, eqx.Module)
        )
        return ret

    def sample(self, key: jaxtyping.Key):  # alias
        return self(key)


class NearSolutionSampler(PyTreeSampler):

    scale: jaxtyping.PyTree[ArrayLike | RelativeScale] = 1.0

    def _scale_noise(self, noise: jaxtyping.PyTree, solution: jaxtyping.PyTree) -> jaxtyping.PyTree:
        """Scale noise using a reference solution."""
        def _is_relative_scale_spec(value: Any) -> bool:
            """Return True when ``value`` is a ``(reducer, factor)`` relative-scale specification."""
            return isinstance(value, tuple | list) and len(value) == 2 and (
                isinstance(value[0], str | UnaryOp) or callable(value[0])
            )

        def _resolve_scale_leaf(reference: ArrayLike, scale_spec: ArrayLike | RelativeScale) -> ArrayLike:
            """Resolve one scale leaf, supporting both absolute and relative scale specifications."""
            if _is_relative_scale_spec(scale_spec):
                reducer, factor = scale_spec
                reducer_fn = UnaryOp(reducer)
                return jnp.asarray(reducer_fn(jnp.asarray(reference))) * jnp.asarray(factor)
            return jnp.asarray(scale_spec)

        def _broadcast_like(template: jaxtyping.PyTree, value: ArrayLike | jaxtyping.PyTree) -> jaxtyping.PyTree:
            """Broadcast a scalar or pytree ``value`` across the structure of ``template`` when needed."""
            if _is_relative_scale_spec(value):
                return jax.tree.map(lambda _: value, template)
            if isinstance(value, Mapping) or (isinstance(value, Sequence) and not isinstance(value, str | bytes)):
                return value
            return jax.tree.map(lambda _: value, template)
        
        scale_tree = _broadcast_like(solution, self.scale)
        resolved_scale = jax.tree.map(_resolve_scale_leaf, solution, scale_tree)
        scaled_noise = jax.tree.map(
            lambda sample, sample_scale: jnp.asarray(sample) * jnp.asarray(sample_scale),
            noise,
            resolved_scale,
        )
        return scaled_noise

    def callable(
        self, 
        key: jaxtyping.Key, 
        inputs: jaxtyping.PyTree | None = None, 
        solution: jaxtyping.PyTree | None = None,
        **template
    ) -> jaxtyping.PyTree:
        """
        Sample an output pytree by adding sampled noise to a reference solution pytree.

        :param key: the JAX random key
        :param inputs: optional conditioning inputs passed for API consistency
        :param solution: reference output pytree to perturb
        :param **template: contains a Pytree of Distribution-like objects for sampling noise
        :return: sampled output pytree with the same structure as ``solution``
        """
        del inputs
        if solution is None:
            raise ValueError("NearSolutionSampler requires a reference solution.")
        
        noise = PyTreeSampler.model_fields["callable"].default(self, key, **template)
        scaled_noise = self._scale_noise(noise, solution)
        return jax.tree.map(lambda ref, delta: jnp.asarray(ref) + delta, solution, scaled_noise)
    
    def sample(
        self, 
        key: jaxtyping.Key, 
        inputs: jaxtyping.PyTree | None = None, 
        solution: jaxtyping.PyTree | None = None
    ) -> jaxtyping.PyTree:
        """Just a convenience alias."""
        return self(key, inputs=inputs, solution=solution)


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
