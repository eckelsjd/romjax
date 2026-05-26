"""PyTree utilities.

This module provides small operator models for array and pytree reductions that are
convenient to configure from YAML-friendly string specs while remaining usable inside
``jax.jit`` and ``equinox.filter_jit`` computations.

Usage
-----
Unary operators accept either a string spec or a callable:

- ``UnaryOperator("mean")``
- ``UnaryOperator("max-abs")``
- ``UnaryOperator(jnp.mean)``

Error operators compare two arrays:

- ``ErrorOperator("abs")`` computes ``abs(x - xhat)``
- ``ErrorOperator(("abs", "norm"))`` computes ``abs(x - xhat) / norm(x)``
- ``get_error_operator("rmse")`` resolves a predefined alias

Tree error operators compare two pytrees:

- ``TreeErrorOperator("mean")`` reduces leafwise differences with ``mean``
- ``TreeErrorOperator("mse")`` resolves a predefined tree-error alias
- ``TreeErrorOperator({"reduce_op": "norm", "leaf_op": ("abs", "norm")})``

Caching
-------
String-defined operators are cached so repeated validation of the same spec reuses the
same Python callable and JIT wrapper instead of rebuilding them each time.

- Canonical string normalization is cached.
- Composite unary callables are cached by canonical unary spec.
- Array error callables are cached by ``(op_spec, norm_spec)``.
- Tree reducer callables are cached by reducer spec.
- Tree error callables are cached by ``(reduce_spec, leaf_spec, norm_spec)``.
- Convenience constructors like ``get_unary_operator("mean")`` also cache the
  corresponding operator model objects.

Only string/canonical-spec paths are cached. Arbitrary user-provided Python callables are
still accepted, but those instances are not shared through the module-level caches.
"""
from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from functools import lru_cache
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_serializer, model_validator

__all__ = [
    "at",
    "merge",
    "size",
    "norm",
    "reduce",
    "mean",
    "map_error",
    "pytree_iter",
    "pytree_path_iter",
    "to_pytree",
    "get_subtree",
    "set_subtree",
    "UnaryOperator",
    "ErrorOperator",
    "TreeErrorOperator",
    "get_unary_operator",
    "get_error_operator",
    "get_tree_operator",
]

type UnaryCallable = Callable[[ArrayLike], ArrayLike]
type ErrorCallable = Callable[[ArrayLike, ArrayLike], ArrayLike]
type TreeErrorCallable = Callable[[PyTree, PyTree], ArrayLike]
type ErrorSpecKey = tuple[str, str | None]
type PathToken = str | int
type TreePath = tuple[PathToken, ...]

_OPERATOR_CACHE_SIZE = 256


def _noop(x: PyTree) -> PyTree:
    return x


def _serialize_unary_value(value: "UnaryOperator | UnaryCallable | None") -> str | UnaryCallable | None:
    if value is None:
        return None
    if isinstance(value, UnaryOperator):
        return value.op if isinstance(value.op, str) else value.function
    return value


_unary_aliases: dict[str, str] = {
    "max_abs": "max-abs",
    "rms": "sqrt-mean-square",
}

_error_aliases: dict[str, ErrorSpecKey] = {
    "mse": ("mean-square", None),
    "mae": ("mean-abs", None),
    "max-abs": ("max-abs", None),
    "rmse": ("sqrt-mean-square", None),
    "relative": ("norm", "norm"),
    "pointwise-relative": ("abs", "abs"),
}

_tree_error_aliases: dict[str, dict[str, str | ErrorSpecKey]] = {
    "mse": {"reduce_op": "mean", "leaf_op": "square"},
    "mae": {"reduce_op": "mean", "leaf_op": "abs"},
    "max-abs": {"reduce_op": "max", "leaf_op": "abs"},
    "rmse": {"reduce_op": "sqrt-mean", "leaf_op": "square"},
    "relative": {"reduce_op": "norm", "norm": "norm"},
    "mean-relative": {"reduce_op": "mean", "leaf_op": ("norm", "norm")},
    "mean-norm": {"reduce_op": "mean", "leaf_op": "norm"}
}


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _canonical_unary_spec(spec: str) -> str:
    return _unary_aliases.get(spec, spec)


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _resolve_unary_part(name: str) -> UnaryCallable:
    canonical = _canonical_unary_spec(name)
    override = {"norm": jnp.linalg.norm, "noop": _noop}
    if canonical in override:
        return override[canonical]
    if not hasattr(jnp, canonical):
        raise ValueError(f"Unknown jax unary operator part: {name!r}")
    return getattr(jnp, canonical)


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _build_unary_callable(spec: str) -> UnaryCallable:
    canonical = _canonical_unary_spec(spec)
    funcs = tuple(_resolve_unary_part(part) for part in reversed(canonical.split("-")))

    @jax.jit
    def composite(x: ArrayLike) -> ArrayLike:
        value = x
        for func in funcs:
            value = func(value)
        return value

    return composite


def _build_uncached_error_callable(op_fn: UnaryCallable, norm_fn: UnaryCallable | None) -> ErrorCallable:
    if norm_fn is None:
        @jax.jit
        def error_fn(x: ArrayLike, xhat: ArrayLike) -> ArrayLike:
            return op_fn(x - xhat)
    else:
        @jax.jit
        def error_fn(x: ArrayLike, xhat: ArrayLike) -> ArrayLike:
            return op_fn(x - xhat) / norm_fn(x)
    return error_fn


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _build_error_callable(op_spec: str, norm_spec: str | None) -> ErrorCallable:
    op_fn = _build_unary_callable(op_spec)
    norm_fn = None if norm_spec is None else _build_unary_callable(norm_spec)
    return _build_uncached_error_callable(op_fn, norm_fn)


def _array_tree(tree: PyTree) -> PyTree:
    return eqx.filter(tree, eqx.is_array_like)


def _flatten_tree_arrays(tree: PyTree) -> jax.Array:
    leaves = [jnp.ravel(jnp.asarray(leaf)) for leaf in jax.tree.leaves(_array_tree(tree))]
    if not leaves:
        return jnp.asarray([], dtype=jnp.float32)
    return jnp.concatenate(leaves)


@eqx.filter_jit
def pytree_norm(tree: PyTree) -> jax.Array:
    """Compute L2 norm over the pytree. Equivalent to pytree_reduce(tree, 'norm') but quicker."""
    total = jax.tree.reduce(
        lambda acc, leaf: acc + jnp.sum(jnp.square(jnp.asarray(leaf))),
        _array_tree(tree),
        jnp.asarray(0.0),
    )
    return jnp.sqrt(total)


@eqx.filter_jit
def pytree_mean(tree: PyTree) -> jax.Array:
    """Compute the mean over the pytree. Equivalent to pytree_reduce(tree, 'mean') but quicker."""
    count, total = jax.tree.reduce(
        lambda acc, leaf: (
            acc[0] + jnp.asarray(jnp.size(leaf)),
            acc[1] + jnp.sum(jnp.asarray(leaf)),
        ),
        _array_tree(tree),
        (jnp.asarray(0), jnp.asarray(0.0)),
    )
    return total / count


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _build_tree_reducer(spec: str) -> Callable[[PyTree], jax.Array]:
    canonical = _canonical_unary_spec(spec)
    if canonical == "mean":
        return pytree_mean
    if canonical == "norm":
        return pytree_norm

    unary_fn = _build_unary_callable(canonical)

    @eqx.filter_jit
    def reducer(tree: PyTree) -> jax.Array:
        return unary_fn(_flatten_tree_arrays(tree))

    return reducer


def _build_uncached_tree_error_callable(
    reduce_fn: Callable[[PyTree], ArrayLike],
    leaf_fn: ErrorCallable,
    norm_fn: Callable[[PyTree], ArrayLike] | None,
) -> TreeErrorCallable:
    if norm_fn is None:
        @jax.jit
        def error_fn(tree: PyTree, tree_hat: PyTree) -> ArrayLike:
            return reduce_fn(jax.tree.map(leaf_fn, _array_tree(tree), _array_tree(tree_hat)))
    else:
        @jax.jit
        def error_fn(tree: PyTree, tree_hat: PyTree) -> ArrayLike:
            return reduce_fn(jax.tree.map(leaf_fn, _array_tree(tree), _array_tree(tree_hat))) / norm_fn(tree)
    return error_fn


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _build_tree_error_callable(
    reduce_spec: str,
    leaf_spec: ErrorSpecKey,
    norm_spec: str | None,
) -> TreeErrorCallable:
    reduce_fn = _build_tree_reducer(reduce_spec)
    leaf_fn = _build_error_callable(*leaf_spec)
    norm_fn = None if norm_spec is None else _build_tree_reducer(norm_spec)
    return _build_uncached_tree_error_callable(reduce_fn, leaf_fn, norm_fn)


class UnaryOperator(BaseModel):
    """
    Jax unary operation ``f(array) -> array``.

    Supports hyphen-separated composites such as ``"sqrt-mean-square"`` for RMS.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    op: str | UnaryCallable
    _callable: UnaryCallable = PrivateAttr()

    def __init__(
        self,
        value: str | UnaryCallable | Mapping | UnaryOperator | None = None,
        /,
        **data,
    ) -> None:
        if value is not None:
            if data:
                raise TypeError("Pass either a positional operator spec or keyword fields, not both.")
            if isinstance(value, Mapping):
                data = dict(value)
            elif isinstance(value, UnaryOperator):
                data = {"op": value.op}
            else:
                data = {"op": value}
        super().__init__(**data)

    @field_validator("op", mode="before")
    @classmethod
    def _from_spec(cls, value: str | UnaryCallable | UnaryOperator) -> str | UnaryCallable:
        if isinstance(value, UnaryOperator):
            return value.op
        if isinstance(value, str):
            return _canonical_unary_spec(value)
        if callable(value):
            return value
        raise TypeError("UnaryOperator requires a string spec or callable.")

    def model_post_init(self, __context) -> None:
        fn = _build_unary_callable(self.op) if isinstance(self.op, str) else self.op
        object.__setattr__(self, "_callable", fn)

    @property
    def op_str(self) -> str | None:
        """Return the canonical string specification when available."""
        return self.op if isinstance(self.op, str) else None

    @property
    def function(self) -> UnaryCallable:
        """Return the callable used at runtime."""
        return self._callable

    @model_serializer(mode="plain")
    def _serialize(self) -> str | UnaryCallable:
        return self.op if isinstance(self.op, str) else self.function

    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self._callable(x)


class ErrorOperator(BaseModel):
    """
    Compute the error between two arrays ``f(array, array) -> array``.

    When ``error_fn`` is omitted, this computes ``op(x - xhat) / norm(x)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    op: UnaryOperator | None = None
    norm: UnaryOperator | None = None
    error_fn: ErrorCallable | None = None
    _callable: ErrorCallable = PrivateAttr()

    def __init__(
        self,
        value: str | tuple | list | ErrorCallable | Mapping | ErrorOperator | None = None,
        /,
        **data,
    ) -> None:
        if value is not None:
            if data:
                raise TypeError("Pass either a positional operator spec or keyword fields, not both.")
            if isinstance(value, Mapping):
                data = dict(value)
            elif isinstance(value, ErrorOperator):
                data = value.model_dump()
            elif isinstance(value, tuple | list):
                data = {"op": value[0], "norm": value[1]}
            elif callable(value):
                data = {"error_fn": value}
            else:
                data = {"op": value}
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, value):
        if isinstance(value, ErrorOperator):
            return value.model_dump()
        if isinstance(value, str | UnaryOperator):
            return {"op": value}
        if isinstance(value, tuple | list):
            return {"op": value[0], "norm": value[1]}
        if callable(value):
            return {"error_fn": value}
        return value

    @field_validator("op", "norm", mode="before")
    @classmethod
    def _coerce_unary(cls, value):
        if value is None or isinstance(value, UnaryOperator):
            return value
        return get_unary_operator(value)

    @model_validator(mode="after")
    def _validate_after(self) -> ErrorOperator:
        if self.error_fn is None and self.op is None:
            raise ValueError("Must either specify error_fn or op")
        return self

    def model_post_init(self, __context) -> None:
        spec_key = self.spec_key
        if self.error_fn is not None:
            fn = jax.jit(self.error_fn)
        elif spec_key is not None:
            fn = _build_error_callable(*spec_key)
        else:
            assert self.op is not None
            norm_fn = None if self.norm is None else self.norm.function
            fn = _build_uncached_error_callable(self.op.function, norm_fn)
        object.__setattr__(self, "_callable", fn)

    @property
    def spec_key(self) -> ErrorSpecKey | None:
        """Return a cache key when the operator is fully described by string specs."""
        if self.error_fn is not None or self.op is None or self.op.op_str is None:
            return None
        norm_spec = None if self.norm is None else self.norm.op_str
        if self.norm is not None and norm_spec is None:
            return None
        return (self.op.op_str, norm_spec)

    @property
    def function(self) -> ErrorCallable:
        """Return the callable used at runtime."""
        return self._callable

    @model_serializer(mode="plain")
    def _serialize(self) -> str | UnaryCallable | dict[str, str | UnaryCallable | None] | ErrorCallable:
        if self.error_fn is not None:
            return self.error_fn
        assert self.op is not None
        op_value = _serialize_unary_value(self.op)
        norm_value = _serialize_unary_value(self.norm)
        if norm_value is None:
            return op_value
        return {"op": op_value, "norm": norm_value}

    def __call__(self, x: ArrayLike, xhat: ArrayLike) -> ArrayLike:
        return self._callable(x, xhat)


class TreeErrorOperator(BaseModel):
    """
    Compute the error between two pytrees ``f(tree, tree) -> scalar``.

    When ``error_fn`` is omitted, this computes ``reduce_op(leaf_op(x, xhat)) / norm(tree)``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    reduce_op: UnaryOperator | None = None
    leaf_op: ErrorOperator = Field(default_factory=lambda: ErrorOperator("noop"))
    norm: UnaryOperator | None = None
    error_fn: TreeErrorCallable | None = None
    _callable: TreeErrorCallable = PrivateAttr()

    def __init__(
        self,
        value: str | tuple | list | TreeErrorCallable | Mapping | TreeErrorOperator | None = None,
        /,
        **data,
    ) -> None:
        if value is not None:
            if data:
                raise TypeError("Pass either a positional operator spec or keyword fields, not both.")
            if isinstance(value, Mapping):
                data = dict(value)
            elif isinstance(value, TreeErrorOperator):
                data = value.model_dump()
            elif isinstance(value, tuple | list):
                data = {"reduce_op": value[0], "leaf_op": value[1]}
            elif callable(value):
                data = {"error_fn": value}
            else:
                data = dict(_tree_error_aliases.get(value, {"reduce_op": value}))
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, value):
        if isinstance(value, TreeErrorOperator):
            return value.model_dump()
        if isinstance(value, str | UnaryOperator):
            if isinstance(value, str) and value in _tree_error_aliases:
                return dict(_tree_error_aliases[value])
            return {"reduce_op": value}
        if isinstance(value, tuple | list):
            return {"reduce_op": value[0], "leaf_op": value[1]}
        if callable(value):
            return {"error_fn": value}
        return value

    @field_validator("reduce_op", "norm", mode="before")
    @classmethod
    def _coerce_reduce_unary(cls, value):
        if value is None or isinstance(value, UnaryOperator):
            return value
        return get_unary_operator(value)

    @field_validator("leaf_op", mode="before")
    @classmethod
    def _coerce_leaf_op(cls, value):
        if isinstance(value, ErrorOperator):
            return value
        return get_error_operator(value)

    @model_validator(mode="after")
    def _validate_after(self) -> TreeErrorOperator:
        if self.error_fn is None and self.reduce_op is None:
            raise ValueError("Must either specify error_fn or reduce_op")
        return self

    def model_post_init(self, __context) -> None:
        spec_key = self.spec_key
        if self.error_fn is not None:
            fn = jax.jit(self.error_fn)
        elif spec_key is not None:
            fn = _build_tree_error_callable(*spec_key)
        else:
            assert self.reduce_op is not None
            reduce_fn = _build_tree_reducer(self.reduce_op.op_str) if self.reduce_op.op_str is not None else None
            if reduce_fn is None:
                reduce_op = self.reduce_op

                @eqx.filter_jit
                def reduce_fn(tree: PyTree) -> ArrayLike:
                    return pytree_reduce(reduce_op, tree)

            norm_fn = None
            if self.norm is not None:
                if self.norm.op_str is not None:
                    norm_fn = _build_tree_reducer(self.norm.op_str)
                else:
                    norm_op = self.norm

                    @eqx.filter_jit
                    def norm_fn(tree: PyTree) -> ArrayLike:
                        return pytree_reduce(norm_op, tree)

            fn = _build_uncached_tree_error_callable(reduce_fn, self.leaf_op.function, norm_fn)
        object.__setattr__(self, "_callable", fn)

    @property
    def spec_key(self) -> tuple[str, ErrorSpecKey, str | None] | None:
        """Return a cache key when the operator is fully described by string specs."""
        if self.error_fn is not None or self.reduce_op is None or self.reduce_op.op_str is None:
            return None
        leaf_key = self.leaf_op.spec_key
        if leaf_key is None:
            return None
        norm_spec = None if self.norm is None else self.norm.op_str
        if self.norm is not None and norm_spec is None:
            return None
        return (self.reduce_op.op_str, leaf_key, norm_spec)

    @property
    def function(self) -> TreeErrorCallable:
        """Return the callable used at runtime."""
        return self._callable

    @model_serializer(mode="plain")
    def _serialize(
        self,
    ) -> (
        str
        | UnaryCallable
        | dict[str, str | UnaryCallable | ErrorCallable | dict[str, str | UnaryCallable | None] | None]
        | TreeErrorCallable
    ):
        if self.error_fn is not None:
            return self.error_fn
        assert self.reduce_op is not None
        reduce_value = _serialize_unary_value(self.reduce_op)
        norm_value = _serialize_unary_value(self.norm)
        leaf_default = self.leaf_op.spec_key == ("noop", None)
        leaf_value = self.leaf_op.model_dump()

        if leaf_default and norm_value is None:
            return reduce_value

        payload: dict[str, str | UnaryCallable | ErrorCallable | dict[str, str | UnaryCallable | None] | None] = {
            "reduce_op": reduce_value
        }
        if not leaf_default:
            payload["leaf_op"] = leaf_value
        if norm_value is not None:
            payload["norm"] = norm_value
        return payload

    def __call__(self, tree: PyTree, tree_hat: PyTree) -> ArrayLike:
        return self._callable(tree, tree_hat)


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _get_cached_unary_operator(spec: str) -> UnaryOperator:
    return UnaryOperator(spec)


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _get_cached_error_operator(op_spec: str, norm_spec: str | None) -> ErrorOperator:
    return ErrorOperator(op=op_spec, norm=norm_spec)


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _get_cached_error_alias(alias: str) -> ErrorOperator:
    op_spec, norm_spec = _error_aliases[alias]
    return _get_cached_error_operator(op_spec, norm_spec)


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _get_cached_tree_operator(reduce_spec: str, leaf_spec: ErrorSpecKey, norm_spec: str | None) -> TreeErrorOperator:
    payload: dict[str, str | ErrorSpecKey] = {"reduce_op": reduce_spec, "leaf_op": leaf_spec}
    if norm_spec is not None:
        payload["norm"] = norm_spec
    return TreeErrorOperator(**payload)


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _get_cached_tree_alias(alias: str) -> TreeErrorOperator:
    alias_spec = _tree_error_aliases[alias]
    reduce_spec = alias_spec["reduce_op"]
    assert isinstance(reduce_spec, str)
    norm_spec = alias_spec.get("norm")
    assert norm_spec is None or isinstance(norm_spec, str)
    leaf_value = alias_spec.get("leaf_op", "noop")
    if isinstance(leaf_value, tuple):
        leaf_spec = leaf_value
    else:
        assert isinstance(leaf_value, str)
        leaf_spec = (leaf_value, None)
    return _get_cached_tree_operator(reduce_spec, leaf_spec, norm_spec)


def get_unary_operator(value: str | UnaryCallable | UnaryOperator) -> UnaryOperator:
    """Convenience method to either get a predefined operator or make a new one."""
    if isinstance(value, UnaryOperator):
        return value
    if isinstance(value, str):
        return _get_cached_unary_operator(_canonical_unary_spec(value))
    return UnaryOperator(value)


def get_error_operator(value: str | tuple | list | ErrorCallable | ErrorOperator) -> ErrorOperator:
    """Convenience method to either get a predefined operator or make a new one."""
    if isinstance(value, ErrorOperator):
        return value
    if isinstance(value, str):
        if value in _error_aliases:
            return _get_cached_error_alias(value)
        return _get_cached_error_operator(_canonical_unary_spec(value), None)
    return ErrorOperator(value)


def get_tree_operator(value: str | tuple | list | TreeErrorCallable | TreeErrorOperator) -> TreeErrorOperator:
    """Convenience method to either get a predefined operator or make a new one."""
    if isinstance(value, TreeErrorOperator):
        return value
    if isinstance(value, str):
        if value in _tree_error_aliases:
            return _get_cached_tree_alias(value)
        return _get_cached_tree_operator(_canonical_unary_spec(value), ("noop", None), None)
    return TreeErrorOperator(value)


@eqx.filter_jit
def pytree_map_error(error_op: ErrorOperator, tree: PyTree, tree_hat: PyTree) -> PyTree:
    """Apply an error function per-leaf between two trees."""
    operator = get_error_operator(error_op)
    return jax.tree.map(operator, _array_tree(tree), _array_tree(tree_hat))


@eqx.filter_jit
def pytree_reduce(reducer: UnaryOperator, tree: PyTree) -> jax.Array:
    """Reduce an array pytree to a scalar JAX array."""
    operator = get_unary_operator(reducer)
    return operator(_flatten_tree_arrays(tree))


def to_pytree(value: PyTree) -> PyTree:
    """Convert nested pydantic models and dicts to a pytree of dicts, tuples, and lists."""
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


def pytree_merge(defaults: PyTree, overrides: PyTree) -> PyTree:
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
                merged[key] = pytree_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    if isinstance(defaults, tuple) and isinstance(overrides, tuple):
        merged = []
        common = min(len(defaults), len(overrides))
        for idx in range(common):
            merged.append(pytree_merge(defaults[idx], overrides[idx]))
        if len(defaults) > common:
            merged.extend(defaults[common:])
        if len(overrides) > common:
            merged.extend(overrides[common:])
        return tuple(merged)

    if isinstance(defaults, list) and isinstance(overrides, list):
        merged_list: list = []
        common = min(len(defaults), len(overrides))
        for idx in range(common):
            merged_list.append(pytree_merge(defaults[idx], overrides[idx]))
        if len(defaults) > common:
            merged_list.extend(defaults[common:])
        if len(overrides) > common:
            merged_list.extend(overrides[common:])
        return merged_list

    return overrides


def pytree_iter(tree: PyTree) -> Generator[PyTree, None, None]:
    """Yield per-sample pytrees from a batched pytree with a leading batch axis."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    if not leaves:
        return
    batch_size = leaves[0].shape[0]
    for i in range(batch_size):
        yield jax.tree_util.tree_unflatten(treedef, [leaf[i] for leaf in leaves])


def pytree_path_iter(tree: PyTree, is_leaf: Callable[[Any], bool] = eqx.is_array) -> Generator[Any, None, None]:
    """Yield tuples of (path, leaf) for all leaves in the tree."""
    leaves_with_path, _ = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)

    for key_path, leaf in leaves_with_path:
        path: list[PathToken] = []
        for key in key_path:
            if isinstance(key, jax.tree_util.DictKey):
                path.append(key.key)
            elif isinstance(key, jax.tree_util.SequenceKey):
                path.append(key.idx)
            elif isinstance(key, jax.tree_util.GetAttrKey):
                path.append(key.name)
            elif isinstance(key, jax.tree_util.FlattenedIndexKey):
                path.append(key.key)
            else:
                raise TypeError(f"Unsupported pytree key type: {type(key)!r}")
        yield tuple(path), leaf


def pytree_at(tree: PyTree, index: int) -> PyTree:
    """Return a pytree with each leaf at the provided index."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return jax.tree_util.tree_unflatten(treedef, [leaf[index] for leaf in leaves])


def pytree_size(tree: PyTree) -> int:
    """Get the leading dimension of the arrays in the pytree."""
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return len(leaves[0])


def get_subtree(tree: PyTree, path: TreePath) -> PyTree:
    """Return subtree located at ``path``."""
    node = tree
    for token in path:
        if isinstance(node, Mapping):
            node = node[token]
        elif isinstance(node, (list, tuple)):
            node = node[token]
        elif isinstance(token, int):
            node = node[token]
        else:
            node = getattr(node, token)
    return node


def set_subtree(tree: PyTree | None, path: TreePath, value: PyTree) -> PyTree:
    """Set one subtree in a nested dict/list/tuple tree and return the updated tree."""
    if len(path) == 0:
        return value

    head, tail = path[0], path[1:]

    if isinstance(head, str):
        out = {} if tree is None else dict(tree)
        out[head] = set_subtree(out.get(head), tail, value)
        return out

    if tree is None:
        out_list: list[Any] = []
    elif isinstance(tree, tuple):
        out_list = list(tree)
    elif isinstance(tree, list):
        out_list = list(tree)
    else:
        raise TypeError(f"Cannot index non-sequence node with integer token {head!r}.")

    while len(out_list) <= head:
        out_list.append(None)
    out_list[head] = set_subtree(out_list[head], tail, value)

    if isinstance(tree, tuple):
        return tuple(out_list)
    return out_list


at = pytree_at
merge = pytree_merge
size = pytree_size
norm = pytree_norm
reduce = pytree_reduce
mean = pytree_mean
map_error = pytree_map_error
