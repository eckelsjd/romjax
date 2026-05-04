"""PyTree utilities."""
from typing import Callable, Mapping, Generator
import functools

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree, ArrayLike
from pydantic import BaseModel, model_validator, field_validator


__all__ = ["at", "merge", "size", "norm", "reduce", "mean", "map_error", "pytree_iter", "to_pytree",
           "UnaryOperator", "ErrorOperator", "TreeErrorOperator",
           "get_unary_operator", "get_error_operator", "get_tree_operator"]


def _noop(x: PyTree) -> PyTree:
    return x


class UnaryOperator(BaseModel):
    """
    Jax unary operation f(array)->array. Supports convenient validation for composite operators.
    Can specify with hypen-separated string, e.g. 'sqrt-mean-square' for rms.
    """
    op: Callable[[ArrayLike], ArrayLike]
    _op_str: str | None = None

    @model_validator(mode="wrap")
    @classmethod
    def _validate(cls, value: str | Callable | Mapping | UnaryOperator, handler):
        # Try to save the string op name if that is how we are constructing
        _op_str = None
        if isinstance(value, str):
            _op_str = value
        elif isinstance(value, Mapping):
            if "op" in value and isinstance(value["op"], str):
                _op_str = value["op"]
        elif isinstance(value, UnaryOperator):
            _op_str = value._op_str

        # Validate from a string or callable directly
        if isinstance(value, str) or callable(value):
            value = {"op": value}
        
        value = handler(value)  # pydantic validation here
        value._op_str = _op_str
        
        return value
    
    @field_validator("op", mode="before")
    @classmethod
    def _from_spec(cls, value: str | Callable) -> Callable:
        if callable(value):
            return value
        return cls.get_jax_operator(value)
    
    @property
    def op_str(self):
        """Useful for tracking the original string spec (optional), may be none."""
        return self._op_str
    
    @staticmethod
    def get_jax_operator(spec_str: str):
        """Return a composite jax operator, e.g. 'max-abs' -> jnp.max(jnp.abs(*)). Split operators by hyphen '-'."""
        override = {"norm": jnp.linalg.norm, "noop": _noop}
        funcs = [override[part] if part in override else getattr(jnp, part) for part in reversed(spec_str.split("-"))]

        @jax.jit
        def composite(x):
            for f in funcs:
                x = f(x)
            return x
        
        return composite
        
    def __call__(self, x: ArrayLike) -> ArrayLike:
        return self.op(x) 


class ErrorOperator(BaseModel):
    """
    Compute the error between two arrays f(array, array) -> array.
    Can specify with op and norm, which will compute op(x - xhat) / norm(x).
    """
    op: UnaryOperator | None = None
    norm: UnaryOperator | None = None
    error_fn: Callable[[ArrayLike, ArrayLike], ArrayLike] | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, value):
        if isinstance(value, str | UnaryOperator):
            return {"op": value}
        if isinstance(value, tuple | list):
            return {"op": value[0], "norm": value[1]}
        if callable(value):
            return {"error_fn": value}

        return value
    
    @model_validator(mode="after")
    def _compile_error_fn(self):
        """Compute error as op(x-xhat)/norm(x) by default if error_fn is not provided. Compile with jit."""
        if self.error_fn is None:
            if self.op is None:
                raise ValueError(f"Must either specify error_fn or op")
            if self.norm is None:
                self.error_fn = lambda x, xhat: self.op(x - xhat)
            else:
                self.error_fn = lambda x, xhat: self.op(x - xhat) / self.norm(x)
        
        self.error_fn = jax.jit(self.error_fn)
        return self

    def __call__(self, x: ArrayLike, xhat: ArrayLike) -> ArrayLike:
        return self.error_fn(x, xhat)


class TreeErrorOperator(BaseModel):
    """
    Compute the error between two pytrees f(tree, tree) -> float.
    Can specify error per-leaf and over full tree via leaf_op and reduce_op. Can also specify a norm. 
    This will compute error_fn(tree, tree_hat) as:
        `reduce_op(leaf_op(x, xhat)) / norm(tree)`
    where x are the leaves of tree, and xhat are the leaves of tree_hat.
    
    By default, leaf_op="noop" will just compute pointwise difference x-xhat for each leaf.
    """
    reduce_op: UnaryOperator | None = None
    leaf_op: ErrorOperator = ErrorOperator("noop")  # just plain (x - xhat)
    norm: UnaryOperator | None = None
    error_fn: Callable[[PyTree, PyTree], float] | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, value):
        if isinstance(value, str | UnaryOperator):
            return {"reduce_op": value}
        if isinstance(value, tuple | list):
            return {"reduce_op": value[0], "leaf_op": value[1]}
        if callable(value):
            return {"error_fn": value}

        return value
    
    @model_validator(mode="after")
    def _compile_error_fn(self):
        """Compute error as reduce(leaf(x,xhat))/norm(x) by default if error_fn is not provided. Compile with jit."""
        if self.error_fn is None:
            if self.reduce_op is None:
                raise ValueError(f"Must either specify error_fn or reduce_op")
            
            override = {"mean": pytree_mean, "norm": pytree_norm}  # more efficient implementations
            reduce_op = override.get(self.reduce_op.op_str, functools.partial(pytree_reduce, self.reduce_op))
            
            if self.norm is None:
                self.error_fn = lambda t, t_hat: reduce_op(pytree_map_error(self.leaf_op, t, t_hat))
            else:
                norm = override.get(self.norm.op_str, functools.partial(pytree_reduce, self.norm))
                self.error_fn = lambda t, t_hat: reduce_op(pytree_map_error(self.leaf_op, t, t_hat)) / norm(t)
        
        self.error_fn = jax.jit(self.error_fn)
        return self

    def __call__(self, tree: PyTree, tree_hat: PyTree) -> float:
        return self.error_fn(tree, tree_hat)


_error_aliases = {
    "mse": ("mean-square", None),
    "mae": ("mean-abs", None),
    "max-abs": ("max-abs", None),
    "rmse": ("sqrt-mean-square", None),
    "relative": ("norm", "norm"),
    "pointwise-relative": ("abs", "abs")
}

_tree_error_aliases = {
    "mse": dict(reduce_op="mean", leaf_op="square"),
    "mae": dict(reduce_op="mean", leaf_op="abs"),
    "max-abs": dict(reduce_op="max", leaf_op="abs"),
    "rmse": dict(reduce_op="sqrt-mean", leaf_op="square"),
    "relative": dict(reduce_op="norm", norm="norm"),                   # Relative over full tree
    "mean-relative": dict(reduce_op="mean", leaf_op=("norm", "norm"))  # Per-leaf relative
}

# Predefine 
_unary_operators = {key: UnaryOperator.model_validate(key) for key in ["mean", "norm", "abs", "max-abs", "noop"]}
_error_operators = {key: ErrorOperator.model_validate(value) for key, value in _error_aliases.items()}
_tree_operators = {key: TreeErrorOperator.model_validate(value) for key, value in _tree_error_aliases.items()}

def get_unary_operator(value: str | Callable) -> UnaryOperator:
    """Convenience method to either get a predefined operator or make a new one."""
    return _unary_operators.get(value, UnaryOperator.model_validate(value))

def get_error_operator(value: str | tuple | Callable) -> ErrorOperator:
    """Convenience method to either get a predefined operator or make a new one."""
    return _error_operators.get(value, ErrorOperator.model_validate(value))

def get_tree_operator(value: str | tuple | Callable) -> ErrorOperator:
    """Convenience method to either get a predefined operator or make a new one."""
    return _tree_operators.get(value, TreeErrorOperator.model_validate(value))


@jax.jit
def pytree_map_error(error_op: ErrorOperator, tree: PyTree, tree_hat: PyTree) -> PyTree:
    """Apply an error function per-leaf between two trees."""
    return jax.tree.map(
        get_error_operator(error_op),
        eqx.filter(tree, eqx.is_array_like),
        eqx.filter(tree_hat, eqx.is_array_like)
    )


@jax.jit
def pytree_norm(tree: PyTree) -> float:
    """Compute L2 norm over the pytree. Equivalent to pytree_reduce(tree, 'norm') but quicker."""
    return float(jnp.sqrt(jax.tree.reduce(lambda acc,x: acc + jnp.sum(x**2), eqx.filter(tree, eqx.is_array_like), 0.0)))


@jax.jit
def pytree_mean(tree: PyTree) -> float:
    """Compute the mean over the pytree. Equivalent to pytree_reduce(tree, 'mean') but quicker."""
    cnt, sum = jax.tree.reduce(
        lambda acc, x: (acc[0] + jnp.size(x), acc[1] + jnp.sum(x)), 
        eqx.filter(tree, eqx.is_array_like), 
        (0, 0.0)
    )
    return float(sum / cnt)


@jax.jit
def pytree_reduce(reducer: UnaryOperator, tree: PyTree) -> float:
    """Reduce array pytree to a float."""
    arr = jnp.concatenate([leaf.ravel() for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array_like))])
    return float(get_unary_operator(reducer)(arr))


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


def pytree_at(tree: PyTree, index: int) -> PyTree:
    """Return a pytree with each leaf at the provided index."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return jax.tree_util.tree_unflatten(treedef, [leaf[index] for leaf in leaves])


def pytree_size(tree: PyTree) -> int:
    """Get the leading dimension of the arrays in the pytree."""
    leaves, _ = jax.tree_util.tree_flatten(tree)
    return len(leaves[0])


at = pytree_at
merge = pytree_merge
size = pytree_size
norm = pytree_norm
reduce = pytree_reduce
mean = pytree_mean
map_error = pytree_map_error
