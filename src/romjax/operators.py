"""PyTree-native configurable operators.

This module defines the public operator API for ``romjax``. Operators are ordinary
callables over PyTrees, with single arrays handled as the limiting one-leaf case.
String specifications resolve through JAX functions and are cached so repeated
configuration does not rebuild common jitted callables.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_serializer, model_validator

from romjax.tree import TreePath, coerce_tree_paths, pytree_path_iter

__all__ = ["UnaryOp", "BinaryOp"]

type UnaryCallable = Callable[[PyTree], PyTree]
type BinaryCallable = Callable[[PyTree, PyTree], PyTree]
type PathToken = str | int

_OPERATOR_CACHE_SIZE = 256


def _noop(x: PyTree) -> PyTree:
    return x


_unary_aliases: dict[str, str] = {
    "max_abs": "max-abs",
    "rms": "sqrt-mean-square",
}

_binary_aliases: dict[str, dict[str, Any]] = {
    "mse": {"op": "mean-square"},
    "mae": {"op": "mean-abs"},
    "max-abs": {"op": "max-abs"},
    "rmse": {"op": "sqrt-mean-square"},
    "relative": {"op": "norm", "norm": "norm"},
    "square-relative": {"op": "sum-square", "norm": "sum-square"},
    "pointwise-relative": {"op": "abs", "norm": "abs"},
    "mean-relative": {"reduce_op": "mean", "leaf_op": ("norm", "norm")},
    "mean-square-relative": {"reduce_op": "mean", "leaf_op": ("sum-square", "sum-square")},
    "mean-norm": {"reduce_op": "mean", "leaf_op": "norm"},
    "mean-max-square": {"reduce_op": "mean", "leaf_op": ("sum-square", "max-square")}  # norm each leaf by max square
}


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _canonical_unary_spec(spec: str) -> str:
    return _unary_aliases.get(spec, spec)


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _resolve_unary_part(name: str) -> Callable[[ArrayLike], ArrayLike]:
    canonical = _canonical_unary_spec(name)
    override = {"norm": jnp.linalg.norm, "noop": _noop}
    if canonical in override:
        return override[canonical]
    if canonical.startswith("p") and canonical[1:].isdigit():
        q = int(canonical[1:])

        def percentile(x: ArrayLike) -> ArrayLike:
            return jnp.percentile(x, q)

        return percentile
    if not hasattr(jnp, canonical):
        raise ValueError(f"Unknown JAX unary operator part: {name!r}")
    return getattr(jnp, canonical)


@lru_cache(maxsize=_OPERATOR_CACHE_SIZE)
def _build_unary_array_callable(spec: str) -> Callable[[ArrayLike], ArrayLike]:
    canonical = _canonical_unary_spec(spec)
    funcs = tuple(_resolve_unary_part(part) for part in reversed(canonical.split("-")))

    @jax.jit
    def composite(x: ArrayLike) -> ArrayLike:
        value = x
        for func in funcs:
            value = func(value)
        return value

    return composite


def _is_array_leaf(value: Any) -> bool:
    return eqx.is_array_like(value) and not isinstance(value, str | bytes)


def _is_ignored(path: TreePath, ignore: tuple[TreePath, ...]) -> bool:
    return any(path[:idx] in ignore for idx in range(1, len(path) + 1))


def _combined_ignore(configured: tuple[TreePath, ...], runtime: Any) -> tuple[TreePath, ...]:
    return (*configured, *coerce_tree_paths(runtime))


def _selected_array_leaves(tree: PyTree, ignore: tuple[TreePath, ...] = ()) -> list[ArrayLike]:
    if _is_ignored((), ignore):
        return []
    if _is_array_leaf(tree):
        return [tree]
    return [
        leaf
        for path, leaf in pytree_path_iter(tree, is_leaf=lambda value: _is_array_leaf(value))
        if _is_array_leaf(leaf) and not _is_ignored(path, ignore)
    ]


def _flatten_arrays(leaves: list[ArrayLike]) -> jax.Array:
    if not leaves:
        raise ValueError("Cannot apply operator to a PyTree with no selected array leaves.")
    return jnp.concatenate([jnp.ravel(jnp.asarray(leaf)) for leaf in leaves])


def _reduce_leaves(leaves: list[ArrayLike], reducer: str | UnaryCallable) -> PyTree:
    if callable(reducer):
        if len(leaves) == 1:
            return reducer(leaves[0])
        return reducer(_flatten_arrays(leaves))

    canonical = _canonical_unary_spec(reducer)
    if len(leaves) == 1:
        return _build_unary_array_callable(canonical)(leaves[0])

    flat = _flatten_arrays(leaves)

    return _build_unary_array_callable(canonical)(flat)


def _leaf_pair_items(
    x: PyTree,
    y: PyTree,
    ignore: tuple[TreePath, ...],
    overlap: Literal["common", "strict"],
) -> list[tuple[TreePath, ArrayLike, ArrayLike]]:
    items: list[tuple[TreePath, ArrayLike, ArrayLike]] = []

    def visit(lhs: Any, rhs: Any, path: TreePath) -> None:
        if _is_ignored(path, ignore):
            return

        lhs_is_array = _is_array_leaf(lhs)
        rhs_is_array = _is_array_leaf(rhs)
        if lhs_is_array and rhs_is_array:
            items.append((path, lhs, rhs))
            return
        if lhs_is_array or rhs_is_array:
            if overlap == "strict":
                raise ValueError(f"Tree leaves differ at path {path!r}.")
            return

        if isinstance(lhs, Mapping) and isinstance(rhs, Mapping):
            lhs_keys = set(lhs)
            rhs_keys = set(rhs)
            if overlap == "strict" and lhs_keys != rhs_keys:
                missing_lhs = sorted(rhs_keys - lhs_keys, key=str)
                missing_rhs = sorted(lhs_keys - rhs_keys, key=str)
                raise ValueError(
                    f"Tree mapping keys differ at path {path!r}: "
                    f"missing from lhs={missing_lhs}, missing from rhs={missing_rhs}."
                )
            for key in lhs:
                if key not in rhs:
                    continue
                visit(lhs[key], rhs[key], (*path, key))
            return

        if isinstance(lhs, tuple | list) and isinstance(rhs, tuple | list):
            if overlap == "strict" and len(lhs) != len(rhs):
                raise ValueError(f"Tree sequence lengths differ at path {path!r}: {len(lhs)} != {len(rhs)}.")
            for index in range(min(len(lhs), len(rhs))):
                visit(lhs[index], rhs[index], (*path, index))
            return

        if overlap == "strict" and type(lhs) is not type(rhs):
            raise ValueError(f"Tree node types differ at path {path!r}: {type(lhs).__name__} != {type(rhs).__name__}.")

    visit(x, y, ())
    if not items:
        raise ValueError("Cannot apply binary operator to pytrees with no selected overlapping array leaves.")
    return items


def _serialize_unary_spec(value: str | UnaryCallable | None) -> str | UnaryCallable | None:
    if isinstance(value, str):
        return _canonical_unary_spec(value)
    return value


class UnaryOp(BaseModel):
    """Unary PyTree operator.

    :param op: string spec or callable applied to one array/PyTree
    :param leaf_op: optional per-leaf unary operator used before ``reduce_op``
    :param reduce_op: optional whole-tree reduction operator
    :param ignore: tree paths to exclude when selecting leaves
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    op: str | UnaryCallable | None = None
    leaf_op: str | UnaryCallable | None = None
    reduce_op: str | UnaryCallable | None = None
    ignore: tuple[TreePath, ...] = Field(default_factory=tuple)
    _function: UnaryCallable = PrivateAttr()

    def __init__(
        self,
        value: str | UnaryCallable | Mapping[str, Any] | "UnaryOp" | None = None,
        /,
        **data: Any,
    ) -> None:
        if value is not None:
            if isinstance(value, Mapping):
                data = {**dict(value), **data}
            elif isinstance(value, UnaryOp):
                data = {**{
                    "op": value.op,
                    "leaf_op": value.leaf_op,
                    "reduce_op": value.reduce_op,
                    "ignore": value.ignore,
                }, **data}
            else:
                data = {"op": value, **data}
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def _from_plain_value(cls, value: Any) -> Any:
        if isinstance(value, UnaryOp):
            return {
                "op": value.op,
                "leaf_op": value.leaf_op,
                "reduce_op": value.reduce_op,
                "ignore": value.ignore,
            }
        if isinstance(value, str) or callable(value):
            return {"op": value}
        return value

    @field_validator("op", "leaf_op", "reduce_op", mode="before")
    @classmethod
    def _coerce_unary_spec(cls, value: Any) -> Any:
        if value is None or callable(value):
            return value
        if isinstance(value, str):
            return _canonical_unary_spec(value)
        raise TypeError("UnaryOp fields require a string spec or callable.")

    @field_validator("ignore", mode="before")
    @classmethod
    def _coerce_ignore(cls, value: Any) -> tuple[TreePath, ...]:
        return tuple(coerce_tree_paths(value))

    @model_validator(mode="after")
    def _validate_shape(self) -> "UnaryOp":
        if self.op is None and self.reduce_op is None:
            raise ValueError("UnaryOp requires either 'op' or 'reduce_op'.")
        if self.op is not None and (self.leaf_op is not None or self.reduce_op is not None):
            raise ValueError("Use either 'op' or the 'leaf_op'/'reduce_op' structured form, not both.")
        return self

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_function", self._build_function())

    @model_serializer(mode="plain")
    def _serialize(self) -> str | UnaryCallable | dict[str, Any]:
        if self.op is not None and not self.ignore:
            return _serialize_unary_spec(self.op)
        payload: dict[str, Any] = {}
        if self.op is not None:
            payload["op"] = _serialize_unary_spec(self.op)
        if self.leaf_op is not None:
            payload["leaf_op"] = _serialize_unary_spec(self.leaf_op)
        if self.reduce_op is not None:
            payload["reduce_op"] = _serialize_unary_spec(self.reduce_op)
        if self.ignore:
            payload["ignore"] = self.ignore
        return payload

    def _build_function(self) -> UnaryCallable:
        if callable(self.op):
            return self.op

        def function(x: PyTree) -> PyTree:
            return self._evaluate(x, ignore=None)

        return function

    def _evaluate(self, x: PyTree, ignore: Any = None) -> PyTree:
        combined_ignore = _combined_ignore(self.ignore, ignore)
        if callable(self.op):
            return self.op(x)
        if isinstance(self.op, str):
            leaves = _selected_array_leaves(x, combined_ignore)
            return _reduce_leaves(leaves, self.op)

        assert self.reduce_op is not None
        leaf_op = self.leaf_op or "noop"
        leaves = [_reduce_leaves([leaf], leaf_op) for leaf in _selected_array_leaves(x, combined_ignore)]
        return _reduce_leaves(leaves, self.reduce_op)

    def __call__(self, x: PyTree, *, ignore: Any = None) -> PyTree:
        return self._evaluate(x, ignore=ignore)


class BinaryOp(BaseModel):
    """Binary PyTree operator.

    :param op: unary operation applied to ``x - y``
    :param norm: optional unary normalization applied to ``x``
    :param leaf_op: optional per-shared-leaf binary operation used before ``reduce_op``
    :param reduce_op: optional whole-tree reduction operator
    :param ignore: tree paths to exclude when selecting shared leaves
    :param overlap: whether to compare common leaves or require exact shared structure
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    op: str | UnaryCallable | None = None
    norm: str | UnaryCallable | None = None
    leaf_op: Any = None
    reduce_op: str | UnaryCallable | None = None
    callable: BinaryCallable | None = None  # directly compute binary op from a plain callable
    ignore: tuple[TreePath, ...] = Field(default_factory=tuple)
    overlap: Literal["common", "strict"] = "common"
    _function: BinaryCallable = PrivateAttr()

    def __init__(
        self,
        value: str | tuple[Any, Any] | list[Any] | BinaryCallable | Mapping[str, Any] | "BinaryOp" | None = None,
        /,
        **data: Any,
    ) -> None:
        if value is not None:
            if isinstance(value, Mapping):
                data = {**dict(value), **data}
            elif isinstance(value, BinaryOp):
                data = {**{
                    "op": value.op,
                    "norm": value.norm,
                    "leaf_op": value.leaf_op,
                    "reduce_op": value.reduce_op,
                    "callable": value.callable,
                    "ignore": value.ignore,
                    "overlap": value.overlap,
                }, **data}
            elif isinstance(value, tuple | list):
                data = {"op": value[0], "norm": value[1], **data}
            elif callable(value):
                data = {"callable": value, **data}
            elif isinstance(value, str):
                data = {**dict(_binary_aliases.get(value, {"op": value})), **data}
            else:
                data = {"op": value, **data}
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def _from_plain_value(cls, value: Any) -> Any:
        if isinstance(value, BinaryOp):
            return {
                "op": value.op,
                "norm": value.norm,
                "leaf_op": value.leaf_op,
                "reduce_op": value.reduce_op,
                "callable": value.callable,
                "ignore": value.ignore,
                "overlap": value.overlap,
            }
        if isinstance(value, str):
            return dict(_binary_aliases.get(value, {"op": value}))
        if isinstance(value, tuple | list):
            return {"op": value[0], "norm": value[1]}
        if callable(value):
            return {"callable": value}
        return value

    @field_validator("op", "norm", "reduce_op", mode="before")
    @classmethod
    def _coerce_unary_spec(cls, value: Any) -> Any:
        if value is None or callable(value):
            return value
        if isinstance(value, str):
            return _canonical_unary_spec(value)
        raise TypeError("BinaryOp unary fields require a string spec or callable.")

    @field_validator("ignore", mode="before")
    @classmethod
    def _coerce_ignore(cls, value: Any) -> tuple[TreePath, ...]:
        return tuple(coerce_tree_paths(value))

    @field_validator("leaf_op", mode="before")
    @classmethod
    def _coerce_leaf_op(cls, value: Any) -> Any:
        if value is None or isinstance(value, BinaryOp):
            return value
        return BinaryOp(value)

    @model_validator(mode="after")
    def _validate_shape(self) -> "BinaryOp":
        if self.callable is not None:
            if self.op is not None or self.norm is not None or self.leaf_op is not None or self.reduce_op is not None:
                raise ValueError("Callable BinaryOp cannot also define op/norm/leaf_op/reduce_op.")
            return self
        if self.op is None and self.reduce_op is None:
            raise ValueError("BinaryOp requires 'op', 'reduce_op', or 'callable'.")
        if self.op is not None and (self.leaf_op is not None or self.reduce_op is not None):
            raise ValueError("Use either 'op' or the 'leaf_op'/'reduce_op' structured form, not both.")
        return self

    def model_post_init(self, __context: Any) -> None:
        object.__setattr__(self, "_function", self._build_function())

    @model_serializer(mode="plain")
    def _serialize(self) -> BinaryCallable | str | dict[str, Any]:
        if self.callable is not None:
            return self.callable
        if self.op is not None and self.norm is None and not self.ignore and self.overlap == "common":
            return _serialize_unary_spec(self.op)
        payload: dict[str, Any] = {}
        if self.op is not None:
            payload["op"] = _serialize_unary_spec(self.op)
        if self.norm is not None:
            payload["norm"] = _serialize_unary_spec(self.norm)
        if self.leaf_op is not None:
            payload["leaf_op"] = self.leaf_op.model_dump() if isinstance(self.leaf_op, BinaryOp) else self.leaf_op
        if self.reduce_op is not None:
            payload["reduce_op"] = _serialize_unary_spec(self.reduce_op)
        if self.ignore:
            payload["ignore"] = self.ignore
        if self.overlap != "common":
            payload["overlap"] = self.overlap
        return payload

    def _build_function(self) -> BinaryCallable:
        if self.callable is not None:
            return self.callable

        def function(x: PyTree, y: PyTree) -> PyTree:
            return self._evaluate(x, y, ignore=None)

        return function

    def _evaluate(self, x: PyTree, y: PyTree, ignore: Any = None) -> PyTree:
        if self.callable is not None:
            return self.callable(x, y)

        combined_ignore = _combined_ignore(self.ignore, ignore)
        pairs = _leaf_pair_items(x, y, combined_ignore, self.overlap)

        if self.op is not None:
            diff_leaves = [lhs - rhs for _, lhs, rhs in pairs]
            value = _reduce_leaves(diff_leaves, self.op)
        else:
            assert self.reduce_op is not None
            leaf_op = self.leaf_op if isinstance(self.leaf_op, BinaryOp) else BinaryOp("noop")
            value = _reduce_leaves([leaf_op(lhs, rhs) for _, lhs, rhs in pairs], self.reduce_op)

        if self.norm is None:
            return value
        return value / _reduce_leaves([lhs for _, lhs, _ in pairs], self.norm)

    def __call__(self, x: PyTree, y: PyTree, *, ignore: Any = None) -> PyTree:
        return self._evaluate(x, y, ignore=ignore)
