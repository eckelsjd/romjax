"""
Configurable normalization utilities for graph edges.

General call structure is:
  - EdgeNormConfig.forward/backward.pre/post -> NormTree
  - NormTree() -> _apply_norm_spec -> _apply_operator_to_tree -> NormOperator
  - NormOperator() -> NormOperator._callable -> _call_norm_part

Kind of convoluted, but here is the justification (I guess):
- EdgeNormConfig  -- pydantic validation of forward/backward pre/post with inference of inverses
- NormTree        -- pydantic validation for nested NormOperators
- NormOperator    -- validation from artifacts, registered strings, and composites
- _call_norm_part -- make sure we only pass supported kwargs to each norm function in a composite

Primary supported normalizations:
- zscore
- minmax
- log / log1p
- sqrt
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, PyTree
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_serializer, model_validator

from romjax.typing import CallableModel
from romjax.utils import load_h5

__all__ = [
    "EdgeNormConfig",
    "EdgeNormDirection",
    "NormOperator",
    "NormTree",
    "minmax",
    "unminmax",
    "unzscore",
    "zscore",
]

_NORM_ARTIFACT_SUFFIXES = {".h5", ".hdf5"}


def zscore(x: ArrayLike, mean: ArrayLike = 0.0, std: ArrayLike = 1.0) -> ArrayLike:
    """
    Normalize an array by subtracting a mean and dividing by a standard deviation.

    :param x: input array
    :param mean: broadcast-compatible mean
    :param std: broadcast-compatible standard deviation
    :return: normalized array
    """
    return (jnp.asarray(x) - jnp.asarray(mean)) / jnp.asarray(std)


def unzscore(x: ArrayLike, mean: ArrayLike = 0.0, std: ArrayLike = 1.0) -> ArrayLike:
    """
    Invert :func:`zscore`.

    :param x: normalized array
    :param mean: broadcast-compatible mean
    :param std: broadcast-compatible standard deviation
    :return: denormalized array
    """
    return jnp.asarray(x) * jnp.asarray(std) + jnp.asarray(mean)


def minmax(
    x: ArrayLike,
    xmin: ArrayLike = 0.0,
    xmax: ArrayLike = 1.0,
    ymin: ArrayLike = 0.0,
    ymax: ArrayLike = 1.0,
) -> ArrayLike:
    """
    Affinely map an array from ``[xmin, xmax]`` into ``[ymin, ymax]``.

    :param x: input array
    :param xmin: broadcast-compatible lower input bound
    :param xmax: broadcast-compatible upper input bound
    :param ymin: broadcast-compatible lower output bound
    :param ymax: broadcast-compatible upper output bound
    :return: normalized array
    """
    x = jnp.asarray(x)
    return (x - jnp.asarray(xmin)) / (jnp.asarray(xmax) - jnp.asarray(xmin)) * (
        jnp.asarray(ymax) - jnp.asarray(ymin)
    ) + jnp.asarray(ymin)


def unminmax(
    x: ArrayLike,
    xmin: ArrayLike = 0.0,
    xmax: ArrayLike = 1.0,
    ymin: ArrayLike = 0.0,
    ymax: ArrayLike = 1.0,
) -> ArrayLike:
    """
    Invert :func:`minmax`.

    :param x: normalized array
    :param xmin: broadcast-compatible lower input bound
    :param xmax: broadcast-compatible upper input bound
    :param ymin: broadcast-compatible lower output bound
    :param ymax: broadcast-compatible upper output bound
    :return: denormalized array
    """
    x = jnp.asarray(x)
    return (x - jnp.asarray(ymin)) / (jnp.asarray(ymax) - jnp.asarray(ymin)) * (
        jnp.asarray(xmax) - jnp.asarray(xmin)
    ) + jnp.asarray(xmin)


def _pow10(x: ArrayLike) -> ArrayLike:
    return jnp.power(jnp.asarray(10, dtype=jnp.asarray(x).dtype), x)


_NORM_REGISTRY: dict[str, Callable[..., ArrayLike]] = {
    "identity": lambda x: x,
    "noop": lambda x: x,
    "zscore": zscore,
    "unzscore": unzscore,
    "minmax": minmax,
    "unminmax": unminmax,
    "log": jnp.log,
    "exp": jnp.exp,
    "log10": jnp.log10,
    "pow10": _pow10,
    "sqrt": jnp.sqrt,
    "square": jnp.square,
    "log1p": jnp.log1p,
    "expm1": jnp.expm1,
}

_INVERSE_REGISTRY: dict[str, str] = {
    "identity": "identity",
    "noop": "noop",
    "zscore": "unzscore",
    "unzscore": "zscore",
    "minmax": "unminmax",
    "unminmax": "minmax",
    "log": "exp",
    "exp": "log",
    "log10": "pow10",
    "pow10": "log10",
    "sqrt": "square",
    "square": "sqrt",
    "log1p": "expm1",
    "expm1": "log1p",
}


def _is_norm_artifact_path(value: str | Path) -> bool:
    path = Path(value)
    return path.suffix in _NORM_ARTIFACT_SUFFIXES


def _decode_h5_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "shape") and value.shape == ():
        return _decode_h5_scalar(value.item())
    return value


def _resolve_norm_artifact_path(value: str | Path) -> Path:
    import romjax

    return romjax.YamlLoader.resolve_parent_path(value)


def _read_h5_leaf_group(group: h5py.Group) -> dict[str, Any]:
    """Read one HDF5 group as a normalization leaf spec."""
    callable_name = _decode_h5_scalar(group.attrs.get("callable"))
    if callable_name is None:
        raise ValueError(f"Normalization leaf group {group.name!r} is missing a callable attribute.")

    payload: dict[str, Any] = {"callable": callable_name}
    for key, value in group.attrs.items():
        if key in {"callable", "axes", "container", "input_shape"}:
            continue
        payload[key] = _decode_h5_scalar(value)
    for key, value in group.items():
        if isinstance(value, h5py.Dataset):
            payload[key] = jnp.asarray(value[()])
    return payload


def _read_h5_tree_group(group: h5py.Group) -> Any:
    """Read an HDF5 group that mirrors a runtime NormTree."""
    if "callable" in group.attrs:
        return _read_h5_leaf_group(group)

    container = _decode_h5_scalar(group.attrs.get("container"))
    if container in {"list", "tuple"}:
        values = [_read_h5_tree_group(group[key]) for key in sorted(group.keys(), key=int)]
        return tuple(values) if container == "tuple" else values
    return {key: _read_h5_tree_group(group[key]) for key in group}


def _load_norm_tree_artifact(value: str | Path) -> Any:
    """Load a full pytree of normalization callable configs from an HDF5 artifact."""
    path = _resolve_norm_artifact_path(value)
    with h5py.File(path, "r") as h5:
        if _decode_h5_scalar(h5.attrs.get("romjax_type")) != "norm_tree":
            return _load_norm_artifact(path)
        if "tree" not in h5:
            raise ValueError(f"Normalization tree artifact {path} is missing a 'tree' group.")
        return _read_h5_tree_group(h5["tree"])


def _load_norm_artifact(value: str | Path) -> dict[str, Any]:
    """Load a single normalization callable config from an HDF5 artifact."""
    path = _resolve_norm_artifact_path(value)
    with h5py.File(path, "r") as h5:
        if _decode_h5_scalar(h5.attrs.get("romjax_type")) == "norm_tree":
            raise ValueError("Whole-tree normalization artifacts must be loaded through NormTree.")

    data = load_h5({}, path, jax=True)
    payload = dict(data.get("opts", data))
    callable_name = payload.pop("callable", None)

    with h5py.File(path, "r") as h5:
        if callable_name is None and "callable" in h5.attrs:
            callable_name = h5.attrs["callable"]

    callable_name = _decode_h5_scalar(callable_name)
    if callable_name is None:
        raise ValueError(f"Normalization artifact {path} is missing a callable name.")
    return {"callable": callable_name, **payload}


def _is_leaf_norm_spec(value: Any) -> bool:
    return isinstance(value, Mapping) and "callable" in value


def _merge_norm_override(base: Any, override: Any, path: tuple[str | int, ...] = ()) -> Any:
    """Deep-merge a YAML override into a loaded norm tree, creating missing mapping paths."""
    if override is None:
        return base
    if isinstance(base, Mapping) and isinstance(override, Mapping):
        if _is_leaf_norm_spec(base):
            merged = dict(base)
            merged.update(override)
            return merged

        merged = dict(base)
        for key, value in override.items():
            merged[key] = _merge_norm_override(merged.get(key), value, (*path, key))
        return merged
    if isinstance(base, list) and isinstance(override, Sequence) and not isinstance(override, str | bytes):
        merged = list(base)
        for index, value in enumerate(override):
            if index >= len(merged):
                raise IndexError(f"Normalization override index {index} is out of range at path {path!r}.")
            merged[index] = _merge_norm_override(merged[index], value, (*path, index))
        return merged
    if isinstance(base, tuple) and isinstance(override, Sequence) and not isinstance(override, str | bytes):
        merged = list(base)
        for index, value in enumerate(override):
            if index >= len(merged):
                raise IndexError(f"Normalization override index {index} is out of range at path {path!r}.")
            merged[index] = _merge_norm_override(merged[index], value, (*path, index))
        return tuple(merged)
    return override


def _resolve_norm_part(name: str) -> Callable[..., ArrayLike]:
    if name in _NORM_REGISTRY:
        return _NORM_REGISTRY[name]
    if hasattr(jnp, name):
        return getattr(jnp, name)
    if "." in name:
        parts = name.split(".")
        for index in range(len(parts) - 1, 0, -1):
            try:
                module = import_module(".".join(parts[:index]))
            except ModuleNotFoundError:
                continue
            resolved: Any = module
            for attr in parts[index:]:
                resolved = getattr(resolved, attr)
            if callable(resolved):
                return resolved
    raise ValueError(f"Unknown normalization operator part {name!r}.")


def _call_norm_part(fn: Callable[..., ArrayLike], x: ArrayLike, opts: Mapping[str, Any]) -> ArrayLike:
    """Call one composite norm part with only supported keyword options."""
    try:
        fn_signature = signature(fn)
    except (TypeError, ValueError):
        return fn(x)

    params = fn_signature.parameters
    if any(param.kind is Parameter.VAR_KEYWORD for param in params.values()):
        return fn(x, **opts)
    supported_opts = {
        key: value
        for key, value in opts.items()
        if key in params and params[key].kind in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
    }
    return fn(x, **supported_opts)


def _coerce_callable_spec(value: Any) -> Any:
    if isinstance(value, NormOperator):
        return value.model_dump()
    if isinstance(value, str | Path) and _is_norm_artifact_path(value):
        return _load_norm_artifact(value)
    if isinstance(value, str | Callable):
        return {"callable": value}
    return value


class NormOperator(CallableModel):
    """
    A configured leaf-wise normalization callable.

    String callables may be composite specs separated by ``"-"``. Composite order matches
    :class:`romjax.tree.UnaryOperator`: ``"log-minmax"`` applies ``minmax`` first, then ``log``.

    :param callable: callable object or registered string spec
    """

    callable: str | Callable[..., ArrayLike]
    _callable: Callable[..., ArrayLike] = PrivateAttr()

    @model_validator(mode="before")
    @classmethod
    def _from_spec(cls, value: Any) -> Any:
        return _coerce_callable_spec(value)

    @field_validator("callable", mode="before")
    @classmethod
    def _coerce_callable(cls, value: Any) -> Any:
        if isinstance(value, str | Path) and _is_norm_artifact_path(value):
            return _load_norm_artifact(value)["callable"]
        return value

    def model_post_init(self, __context: Any) -> None:
        callable_spec = self.callable
        if isinstance(callable_spec, str):
            parts = tuple(reversed(callable_spec.split("-")))
            functions = tuple(_resolve_norm_part(part) for part in parts)

            def composite(x: ArrayLike, **kwargs: Any) -> ArrayLike:
                value = x
                for fn in functions:
                    value = _call_norm_part(fn, value, kwargs)
                return value

            object.__setattr__(self, "_callable", composite)
        else:
            object.__setattr__(self, "_callable", callable_spec)

    @property
    def spec(self) -> str | None:
        """Return the string spec when the operator was configured from a registered string."""
        return self.callable if isinstance(self.callable, str) else None

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any] | Callable[..., ArrayLike]:
        if callable(self.callable) and not isinstance(self.callable, str):
            return self.callable
        return {"callable": self.callable, **(self.model_extra or {})}

    def inverse(self) -> "NormOperator":
        """
        Return a configured inverse normalization operator.

        :return: inverse operator with the same configured constants
        :raises ValueError: if this operator does not have a registered inverse
        """
        if self.spec is None:
            raise ValueError("Cannot infer inverse for an arbitrary normalization callable.")
        inverse_parts: list[str] = []
        for part in reversed(self.spec.split("-")):
            if part not in _INVERSE_REGISTRY:
                raise ValueError(f"Cannot infer inverse for normalization operator part {part!r}.")
            inverse_parts.append(_INVERSE_REGISTRY[part])
        return NormOperator(callable="-".join(inverse_parts), **(deepcopy(self.model_extra) or {}))

    def __call__(self, x: ArrayLike, **kwargs: Any) -> ArrayLike:
        opts = dict(self.model_extra or {})
        opts.update(kwargs)
        return self._callable(x, **opts)


def _is_operator_spec(value: Any) -> bool:
    return (
        isinstance(value, NormOperator)
        or callable(value)
        or isinstance(value, str | Path)
        or (isinstance(value, Mapping) and "callable" in value)
    )


def _validate_norm_spec(value: Any) -> Any:
    if value is None or isinstance(value, NormTree):
        return value
    if _is_operator_spec(value):
        return NormOperator.model_validate(value)
    if isinstance(value, Mapping):
        return {key: _validate_norm_spec(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_validate_norm_spec(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_validate_norm_spec(item) for item in value)
    return value


def _inverse_norm_spec(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, NormTree):
        return value.inverse().root
    if isinstance(value, NormOperator):
        return value.inverse()
    if isinstance(value, Mapping):
        return {key: _inverse_norm_spec(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_inverse_norm_spec(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_inverse_norm_spec(item) for item in value)
    raise ValueError(f"Cannot infer inverse for normalization spec {value!r}.")


def _apply_operator_to_tree(operator: NormOperator, x: PyTree, aux: Any = None) -> PyTree:
    opts = aux if isinstance(aux, Mapping) else {}

    def apply_leaf(leaf: Any) -> Any:
        if not eqx.is_array_like(leaf):
            return leaf
        return operator(leaf, **opts)

    return jax.tree.map(apply_leaf, x)


def _apply_norm_spec(spec: Any, x: PyTree, aux: Any = None) -> PyTree:
    if spec is None:
        return x
    if isinstance(spec, NormTree):
        return spec(x, aux=aux)
    if isinstance(spec, NormOperator):
        return _apply_operator_to_tree(spec, x, aux=aux)
    if isinstance(spec, Mapping):
        if not isinstance(x, Mapping):
            raise TypeError(f"Normalization spec expects a mapping payload, got {type(x).__name__}.")
        out = dict(x)
        aux_map = aux if isinstance(aux, Mapping) else {}
        for key, child_spec in spec.items():
            if key not in x:
                continue
            out[key] = _apply_norm_spec(child_spec, x[key], aux=aux_map.get(key))
        return out
    if isinstance(spec, list | tuple):
        if not isinstance(x, Sequence) or isinstance(x, str | bytes):
            raise TypeError(f"Normalization spec expects a sequence payload, got {type(x).__name__}.")
        aux_seq = aux if isinstance(aux, Sequence) and not isinstance(aux, str | bytes) else [None] * len(spec)
        out = list(x)
        for idx, child_spec in enumerate(spec):
            if idx >= len(x):
                continue
            child_aux = aux_seq[idx] if idx < len(aux_seq) else None
            out[idx] = _apply_norm_spec(child_spec, x[idx], aux=child_aux)
        return type(x)(out) if isinstance(x, tuple) else out
    return x


class NormTree(BaseModel):
    """
    A broadcast or pytree-structured normalization specification.

    :param root: either one :class:`NormOperator` broadcast over all array leaves or a pytree of operators
    :param artifact: optional HDF5 artifact path resolved lazily at runtime
    :param overrides: optional deep-merge overrides applied after loading ``artifact``
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Any = None
    artifact: Path | None = None
    overrides: Any = None
    inverse_artifact: bool = False
    _resolved_root: Any = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _from_spec(cls, value: Any) -> Any:
        if isinstance(value, NormTree):
            return value
        if isinstance(value, str | Path) and _is_norm_artifact_path(value):
            return {"artifact": value}
        if isinstance(value, Mapping) and "artifact" in value:
            return {
                "artifact": value["artifact"],
                "overrides": value.get("overrides"),
                "inverse_artifact": value.get("inverse_artifact", False),
            }
        if isinstance(value, Mapping) and "root" in value:
            root = value["root"]
            if isinstance(root, str | Path) and _is_norm_artifact_path(root):
                return {
                    "artifact": root,
                    "overrides": value.get("overrides"),
                    "inverse_artifact": value.get("inverse_artifact", False),
                }
            if isinstance(root, Mapping) and "artifact" in root:
                return {
                    "artifact": root["artifact"],
                    "overrides": root.get("overrides", value.get("overrides")),
                    "inverse_artifact": root.get("inverse_artifact", value.get("inverse_artifact", False)),
                }
        if isinstance(value, Mapping) and set(value.keys()) == {"root"}:
            return value
        return {"root": value}

    @field_validator("root", mode="before")
    @classmethod
    def _coerce_root(cls, value: Any) -> Any:
        return _validate_norm_spec(value)

    @field_validator("artifact", mode="before")
    @classmethod
    def _coerce_artifact(cls, value: Any) -> Path | None:
        if value is None:
            return None
        return _resolve_norm_artifact_path(value)

    @model_serializer(mode="plain")
    def _serialize(self) -> Any:
        if self.artifact is not None:
            payload: dict[str, Any] = {"artifact": str(self.artifact)}
            if self.overrides is not None:
                payload["overrides"] = self.overrides
            if self.inverse_artifact:
                payload["inverse_artifact"] = True
            return payload
        return self.root

    def resolve_root(self) -> Any:
        """
        Resolve the runtime normalization tree, loading artifact-backed specs at most once.

        :return: validated runtime normalization spec
        """
        if self._resolved_root is not None:
            return self._resolved_root
        if self.artifact is None:
            self._resolved_root = self.root
            return self._resolved_root

        loaded = _load_norm_tree_artifact(self.artifact)
        merged = _merge_norm_override(loaded, self.overrides)
        resolved = _validate_norm_spec(merged)
        if self.inverse_artifact:
            resolved = _inverse_norm_spec(resolved)
        self._resolved_root = resolved
        return self._resolved_root

    def inverse(self) -> "NormTree":
        """
        Return a norm tree with every configured operator replaced by its inverse.

        :return: inverse norm tree
        """
        if self.artifact is not None and self._resolved_root is None:
            return NormTree(
                artifact=self.artifact,
                overrides=deepcopy(self.overrides),
                inverse_artifact=not self.inverse_artifact,
            )
        return NormTree(root=_inverse_norm_spec(self.resolve_root()))

    def __call__(self, x: PyTree, aux: Any = None) -> PyTree:
        return _apply_norm_spec(self.resolve_root(), x, aux=aux)


def _coerce_norm_tree(value: Any) -> NormTree | None:
    if value is None or isinstance(value, NormTree):
        return value
    return NormTree.model_validate(value)


class EdgeNormDirection(BaseModel):
    """
    Normalization stages for one edge direction.

    :param pre: normalization applied before the underlying edge map
    :param post: normalization applied after the underlying edge map
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    pre: NormTree | None = None
    post: NormTree | None = None

    @field_validator("pre", "post", mode="before")
    @classmethod
    def _coerce_stage(cls, value: Any) -> NormTree | None:
        return _coerce_norm_tree(value)


class EdgeNormConfig(BaseModel):
    """
    Bidirectional normalization configuration for an edge.

    Missing inverse stages are inferred when all involved operators have registered inverses:
    ``backward.post = inverse(forward.pre)`` and ``backward.pre = inverse(forward.post)``.

    :param forward: forward-direction pre/post normalization
    :param backward: backward-direction pre/post normalization
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    forward: EdgeNormDirection = Field(default_factory=EdgeNormDirection)
    backward: EdgeNormDirection = Field(default_factory=EdgeNormDirection)

    @model_validator(mode="before")
    @classmethod
    def _none_as_empty(cls, value: Any) -> Any:
        if value is None:
            return {}
        return value

    @model_validator(mode="after")
    def _infer_missing_inverses(self) -> "EdgeNormConfig":
        if self.backward.post is None and self.forward.pre is not None:
            self.backward.post = self.forward.pre.inverse()
        if self.backward.pre is None and self.forward.post is not None:
            self.backward.pre = self.forward.post.inverse()
        if self.forward.post is None and self.backward.pre is not None:
            self.forward.post = self.backward.pre.inverse()
        if self.forward.pre is None and self.backward.post is not None:
            self.forward.pre = self.backward.post.inverse()
        return self
