"""PyTree utilities."""
from __future__ import annotations

import warnings
from collections.abc import Callable, Generator, Mapping
from typing import Annotated, Any, Iterator, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from pydantic import BaseModel, BeforeValidator

__all__ = [
    "at",
    "merge",
    "size",
    "norm",
    "reduce",
    "mean",
    "stack",
    "pytree_iter",
    "pytree_path_iter",
    "pytree_resolve_refs",
    "to_pytree",
    "get_subtree",
    "set_subtree",
    "shape_dtype_like",
    "is_shape_dtype",
    "as_shape_dtype_pytree",
    "ShapeDtypePyTree",
    "TreePath",
    "coerce_tree_path",
    "coerce_tree_paths",
]

type PathToken = str | int
type TreePath = tuple[PathToken, ...]


def coerce_tree_path(value: Any) -> TreePath:
    """Coerce a single tree path."""
    if isinstance(value, tuple):
        return tuple(int(token) if isinstance(token, str) and token.lstrip("-").isdigit() else token for token in value)
    if isinstance(value, list):
        return tuple(int(token) if isinstance(token, str) and token.lstrip("-").isdigit() else token for token in value)
    if isinstance(value, str):
        return (value,)
    return value


def coerce_tree_paths(value: Any) -> list[TreePath]:
    """Coerce multiple tree paths."""
    if value is None:
        return []
    if isinstance(value, str | int):
        return [(value,)]
    if isinstance(value, list | tuple):
        if len(value) == 0:
            return []
        if all(isinstance(token, str | int) for token in value):
            return [tuple(coerce_tree_path(value))]
        return [tuple(coerce_tree_path(path)) for path in value]
    return value


def _array_tree(tree: PyTree) -> PyTree:
    return eqx.filter(tree, eqx.is_array_like)


def is_shape_dtype(value: Any) -> bool:
    """Return whether ``value`` is JAX shape/dtype metadata used as a static template leaf."""
    return isinstance(value, jax.ShapeDtypeStruct)


def as_shape_dtype_pytree(template: PyTree) -> PyTree:
    """
    Normalize array-like leaves and YAML-friendly shape/dtype mappings into JAX template leaves.

    A supported mapping contains ``"shape"`` and optionally ``"dtype"`` as its only keys. When ``"dtype"`` is
    omitted, the active JAX default floating-point dtype is used. Other leaves and container structure are preserved.

    :param template: pytree containing arrays, shape/dtype mappings, and static leaves
    :return: matching pytree with template leaves represented by :class:`jax.ShapeDtypeStruct`
    """
    if is_shape_dtype(template):
        return template
    if eqx.is_array(template):
        return jax.ShapeDtypeStruct(jnp.shape(template), jnp.asarray(template).dtype)
    if isinstance(template, Mapping):
        if "shape" in template and set(template) <= {"shape", "dtype"}:
            dtype = template.get("dtype", jnp.asarray(0.0).dtype)
            return jax.ShapeDtypeStruct(tuple(template["shape"]), dtype)
        return {key: as_shape_dtype_pytree(value) for key, value in template.items()}
    if isinstance(template, tuple):
        return tuple(as_shape_dtype_pytree(value) for value in template)
    if isinstance(template, list):
        return [as_shape_dtype_pytree(value) for value in template]
    return template


type ShapeDtypePyTree = Annotated[Any, BeforeValidator(as_shape_dtype_pytree)]
"""Pydantic-compatible pytree type that normalizes array template leaves."""


def shape_dtype_like(
    tree: PyTree,
    leaf_filter: Callable[[Any], bool] = eqx.is_array,
) -> PyTree:
    """
    Create a lightweight template preserving selected array leaf shapes and dtypes.

    :param tree: source pytree
    :param leaf_filter: predicate selecting leaves to replace with shape/dtype metadata
    :return: pytree with selected leaves replaced by :class:`jax.ShapeDtypeStruct`
    """
    return jax.tree_util.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(jnp.shape(leaf), jnp.asarray(leaf).dtype) if leaf_filter(leaf) else leaf,
        tree,
    )


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
def pytree_square_norm(tree: PyTree) -> jax.Array:
    """Compute ||x||^2. Same as sum-square."""
    total = jax.tree.reduce(
        lambda acc, leaf: acc + jnp.sum(jnp.square(jnp.asarray(leaf))),
        _array_tree(tree),
        jnp.asarray(0.0),
    )
    return total


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


@eqx.filter_jit
def pytree_reduce(reducer: Callable[[PyTree], PyTree] | str, tree: PyTree) -> jax.Array:
    """Reduce an array pytree to a scalar JAX array."""
    from romjax.operators import UnaryOp

    return UnaryOp(reducer)(tree)


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


def pytree_path_iter(
    tree: PyTree,
    is_leaf: Callable[[Any], bool] = eqx.is_array,
) -> Generator[tuple[TreePath, PyTree], None, None]:
    """Yield tuples of ``(path, leaf)`` for all leaves in the tree.

    This walks standard ``dict``/``list``/``tuple`` containers directly so the
    yielded order matches Python's native container order.
    """

    def _iter(node: PyTree, path: list[PathToken]) -> Generator[tuple[TreePath, PyTree], None, None]:
        if is_leaf(node) or not isinstance(node, (Mapping, list, tuple)):
            yield tuple(path), node
            return

        if isinstance(node, Mapping):
            for key, value in node.items():
                path.append(key)
                yield from _iter(value, path)
                path.pop()
            return

        for index, value in enumerate(node):
            path.append(index)
            yield from _iter(value, path)
            path.pop()

    yield from _iter(tree, [])


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
        if node is None:
            return None
        if isinstance(node, Mapping):
            node = node.get(token, None)
        elif isinstance(node, (list, tuple)) and isinstance(token, int):
            node = node[token] if token < len(node) else None
        elif isinstance(token, int):
            try:
                node = node[token]
            except (IndexError, TypeError):
                return None
        else:
            node = getattr(node, token, None)
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


def pytree_stack(items: Sequence[PyTree]) -> PyTree:
    """Stack matching sample pytrees along a leading batch axis."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *items)


def pytree_resolve_refs(tree: PyTree, reference_type: type = str):
    """Resolve references in a pytree to other locations in the pytree. Replace the references with pointers."""
    if reference_type is not str:
        raise ValueError("Only str-type references are supported in a param PyTree.")

    def _coerce_token(token: str) -> str | int:
        token = token.strip()
        if token.lstrip("-").isdigit():
            return int(token)
        return token

    def _coerce_reference_path(reference: str) -> tuple[str | int, ...]:
        return tuple(_coerce_token(token) for token in reference.split(","))

    def _iter_reference_paths(
        tree: PyTree,
        path: tuple[str | int, ...] = (),
    ) -> Iterator[tuple[tuple[str | int, ...], str]]:
        if isinstance(tree, reference_type):
            yield path, tree
            return

        if isinstance(tree, Mapping):
            for key, value in tree.items():
                yield from _iter_reference_paths(value, (*path, key))
            return

        if isinstance(tree, tuple | list):
            for i, value in enumerate(tree):
                yield from _iter_reference_paths(value, (*path, i))

    resolved = tree
    for ref_path, reference in _iter_reference_paths(tree):
        target_path = _coerce_reference_path(reference)
        try:
            target = get_subtree(tree, target_path)
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            warnings.warn(
                f"Could not resolve parameter reference {reference!r} at path {ref_path!r}: {exc}",
                stacklevel=2,
            )
            continue
        resolved = set_subtree(resolved, ref_path, target)

    return resolved


at = pytree_at
merge = pytree_merge
size = pytree_size
norm = pytree_norm
reduce = pytree_reduce
mean = pytree_mean
stack = pytree_stack
