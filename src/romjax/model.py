from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key, PyTree
from pydantic import Field, field_validator

from romjax.graph import Edge
from romjax.typing import DictModel
from romjax.utils import merge_pytrees

type FilterSpec = bool | Callable[[Any], bool]
type PathToken = str | int
type TreePath = tuple[PathToken, ...]


class Sampleable(ABC):
    """Mixin for models that can sample input/output spaces."""

    @abstractmethod
    def sample_inputs(self, key: Key) -> PyTree:
        """Sample a single model input for the given key."""
        raise NotImplementedError

    @abstractmethod
    def sample_outputs(self, key: Key, inputs: PyTree | None = None, solution: PyTree | None = None) -> PyTree:
        """
        Produce one sample of outputs for the given key.

        :param key: random key
        :param inputs: optional conditioning inputs
        :param solution: optional precomputed solution of ``solve(inputs)=0``
        :return: sampled outputs
        """
        raise NotImplementedError


class ImplicitModel(Edge, ABC):
    """
    Implicit residual model with invertible augmented forward/backward maps.

    For ``f(b, u)=r``, the edge forward/backward maps are:

    - ``forward: (b, u) -> (b, r)``
    - ``backward: (b, r) -> (b, u)``
    """

    def forward(self, x: PyTree) -> PyTree:
        """
        Pass inputs through and evaluate residuals.

        :param x: tree with ``{"inputs": ..., "outputs": ...}``
        :return: tree with ``{"inputs": ..., "residuals": ...}``
        """
        return {"inputs": x["inputs"], "residuals": self.evaluate(x["inputs"], x["outputs"])}

    def backward(self, x: PyTree) -> PyTree:
        """
        Pass inputs through and solve for outputs.

        :param x: tree with ``{"inputs": ..., "residuals": ...}``
        :return: tree with ``{"inputs": ..., "outputs": ...}``
        """
        return {"inputs": x["inputs"], "outputs": self.solve(x["inputs"], x["residuals"])}

    @abstractmethod
    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate forward residual function ``f(b, u)``."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, inputs: PyTree, residuals: PyTree) -> PyTree:
        """Solve inverse residual function ``f(b, u)=r`` for ``u``."""
        raise NotImplementedError


class ExplicitModel(ImplicitModel, ABC):
    """
    Explicit model as an implicit model with pushforward ``outputs = G(inputs)``.

    Residual and inverse are handled as:

    - ``evaluate: f(b, u) = u - G(b)``
    - ``solve: u = r + G(b)``
    """

    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate residual as ``u - G(b)``."""
        return jax.tree.map(lambda u, uhat: u - uhat, outputs, self.pushforward(inputs))

    def solve(self, inputs: PyTree, residuals: PyTree) -> PyTree:
        """Compute inverse residual solve using the pushforward."""
        return jax.tree.map(lambda r, uhat: r + uhat, residuals, self.pushforward(inputs))

    @abstractmethod
    def pushforward(self, inputs: PyTree) -> PyTree:
        """Compute explicit outputs from inputs."""
        raise NotImplementedError


class PathSpec(DictModel):
    """Path-based override for one subtree in an Equinox filter-spec tree."""

    path: TreePath
    spec: FilterSpec = True

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> TreePath:
        return _coerce_path(value)


class TreeRoute(DictModel):
    """
    Route one subtree from callable output into a destination path in the assembled output tree.

    :param source: source path in callable output, defaults to the callable output root
    :param target: destination path in assembled edge output
    """

    source: TreePath = ()
    target: TreePath

    @field_validator("source", "target", mode="before")
    @classmethod
    def _coerce_paths(cls, value: Any) -> TreePath:
        return _coerce_path(value)


class FilterModelSpec(DictModel):
    """
    One configurable filter component used by :class:`FilterModel`.

    The input side for each direction is controlled via ``in_spec``/``in_paths`` and
    ``out_spec``/``out_paths``. Callable results are merged into the assembled output by either:

    - explicit routes (`forward_routes`/`backward_routes`) for precise patch placement, or
    - legacy output filtering (`out_spec` for forward, `in_spec` for backward) if no routes are set.

    :param forward: callable for forward mapping
    :param backward: callable for backward mapping
    :param in_spec: base filter spec for forward input and backward output
    :param out_spec: base filter spec for forward output and backward input
    :param in_paths: path overrides merged into ``in_spec``
    :param out_paths: path overrides merged into ``out_spec``
    :param forward_routes: explicit routing from forward callable output to final output tree
    :param backward_routes: explicit routing from backward callable output to final output tree
    :param forward_opts: keyword args passed to ``forward``
    :param backward_opts: keyword args passed to ``backward``
    """

    forward: Callable[[PyTree, Any], PyTree]
    backward: Callable[[PyTree, Any], PyTree]

    in_spec: PyTree[FilterSpec] | None = None
    out_spec: PyTree[FilterSpec] | None = None
    in_paths: list[PathSpec] = Field(default_factory=list)
    out_paths: list[PathSpec] = Field(default_factory=list)

    forward_routes: list[TreeRoute] = Field(default_factory=list)
    backward_routes: list[TreeRoute] = Field(default_factory=list)

    forward_opts: dict[str, Any] = Field(default_factory=dict)
    backward_opts: dict[str, Any] = Field(default_factory=dict)

    @field_validator("in_paths", "out_paths", mode="before")
    @classmethod
    def _coerce_filter_paths(cls, value: Any) -> list[PathSpec]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        out: list[PathSpec] = []
        for item in value:
            if isinstance(item, PathSpec):
                out.append(item)
            elif isinstance(item, dict):
                out.append(PathSpec(**item))
            else:
                out.append(PathSpec(path=item))
        return out

    @field_validator("forward_routes", "backward_routes", mode="before")
    @classmethod
    def _coerce_routes(cls, value: Any) -> list[TreeRoute]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        out: list[TreeRoute] = []
        for item in value:
            if isinstance(item, TreeRoute):
                out.append(item)
            elif isinstance(item, dict):
                out.append(TreeRoute(**item))
            else:
                out.append(TreeRoute(target=item))
        return out

    def _resolve_spec(
        self,
        tree: PyTree,
        spec: PyTree[FilterSpec] | None,
        path_specs: list[PathSpec],
    ) -> PyTree[FilterSpec]:
        """
        Resolve a final Equinox filter spec by combining a base spec with path overrides.

        Resolution behavior:

        - no base spec + no paths -> keep whole tree
        - callable base spec + no paths -> use callable directly
        - bool base spec -> broadcast bool over the full tree
        - no base spec + paths -> start with all ``False`` then apply overrides
        """
        if spec is None and not path_specs:
            return True
        if callable(spec):
            if path_specs:
                raise ValueError("Path-based overrides cannot be applied to a callable filter spec.")
            return spec

        if spec is None:
            spec_tree = jax.tree_util.tree_map(lambda _: False, tree)
        elif isinstance(spec, bool):
            spec_tree = jax.tree_util.tree_map(lambda _: spec, tree)
        else:
            spec_tree = spec

        for path_spec in path_specs:
            subtree = _get_subtree(tree, path_spec.path)
            replacement = _constant_spec_like(subtree, path_spec.spec)
            spec_tree = eqx.tree_at(
                lambda t, p=path_spec.path: _get_subtree(t, p),
                spec_tree,
                replacement,
            )
        return spec_tree

    def resolve_in_spec(self, tree: PyTree) -> PyTree[FilterSpec]:
        """Resolve effective input-side filter specification."""
        return self._resolve_spec(tree, self.in_spec, self.in_paths)

    def resolve_out_spec(self, tree: PyTree) -> PyTree[FilterSpec]:
        """Resolve effective output-side filter specification."""
        return self._resolve_spec(tree, self.out_spec, self.out_paths)

    def route_forward(self, candidate: PyTree) -> PyTree:
        """Route forward callable output into a patch tree."""
        if self.forward_routes:
            return _build_routed_patch(candidate, self.forward_routes)
        return eqx.filter(candidate, self.resolve_out_spec(candidate))

    def route_backward(self, candidate: PyTree) -> PyTree:
        """Route backward callable output into a patch tree."""
        if self.backward_routes:
            return _build_routed_patch(candidate, self.backward_routes)
        return eqx.filter(candidate, self.resolve_in_spec(candidate))


class FilterModel(Edge):
    """
    Flexible pytree-to-pytree edge map with configurable filtering, callable transforms, and patch routing.

    The runtime input may include an optional ``"filters"`` entry that provides one callable argument per
    :class:`FilterModelSpec` element. This is useful for passing learnable objects (for example Equinox modules)
    through JAX transforms.

    :param filters: list of directional filter specs
    :param forward_base: base output tree for forward (`"empty"` or `"input"`)
    :param backward_base: base output tree for backward (`"empty"` or `"input"`)
    """

    filters: list[FilterModelSpec]
    forward_base: Literal["empty", "input"] = "empty"
    backward_base: Literal["empty", "input"] = "empty"

    def _split_payload_and_args(self, x: PyTree) -> tuple[PyTree, list[Any]]:
        """Split runtime payload and per-filter callable args."""
        payload = x
        runtime_args: Any = None

        if isinstance(x, Mapping) and "filters" in x:
            payload = dict(x)
            runtime_args = payload.pop("filters")

        if runtime_args is None:
            args = [None] * len(self.filters)
        else:
            if not isinstance(runtime_args, (list, tuple)):
                runtime_args = [runtime_args]
            args = list(runtime_args)
            if len(args) < len(self.filters):
                args.extend([None] * (len(self.filters) - len(args)))
            elif len(args) > len(self.filters):
                raise ValueError(
                    f"Received {len(args)} runtime filter args but model has only {len(self.filters)} filter specs."
                )

        return payload, args

    def _run(
        self,
        x: PyTree,
        direction: Literal["forward", "backward"],
        base_mode: Literal["empty", "input"],
    ) -> PyTree:
        payload, filter_args = self._split_payload_and_args(x)
        assembled: PyTree | None = payload if base_mode == "input" else None

        for spec, args in zip(self.filters, filter_args):
            if direction == "forward":
                view = eqx.filter(payload, spec.resolve_in_spec(payload))
                candidate = spec.forward(view, args, **spec.forward_opts)
                patch = spec.route_forward(candidate)
            else:
                view = eqx.filter(payload, spec.resolve_out_spec(payload))
                candidate = spec.backward(view, args, **spec.backward_opts)
                patch = spec.route_backward(candidate)

            assembled = patch if assembled is None else _merge_filtered_trees(assembled, patch)

        return payload if assembled is None else assembled

    def forward(self, x: PyTree) -> PyTree:
        """
        Evaluate the forward-direction filter map.

        :param x: input pytree, optionally including ``"filters"`` runtime args
        :return: assembled forward output pytree
        """
        return self._run(x, direction="forward", base_mode=self.forward_base)

    def backward(self, x: PyTree) -> PyTree:
        """
        Evaluate the backward-direction filter map.

        :param x: input pytree, optionally including ``"filters"`` runtime args
        :return: assembled backward output pytree
        """
        return self._run(x, direction="backward", base_mode=self.backward_base)


class LinearProjection(eqx.Module):
    """Linear projection module with tied transpose reconstruction."""

    matrix: ArrayLike

    def reduce(self, x: ArrayLike) -> ArrayLike:
        """
        Project from full to reduced coordinates using ``z = x W^T``.

        :param x: full-space vector/tensor with last axis ``n_full``
        :return: reduced coordinates with last axis ``n_latent``
        """
        matrix = jnp.asarray(self.matrix)
        return jnp.matmul(jnp.asarray(x), jnp.swapaxes(matrix, -1, -2))

    def reconstruct(self, z: ArrayLike) -> ArrayLike:
        """
        Reconstruct from reduced to full coordinates using ``x_hat = z W``.

        :param z: reduced coordinates with last axis ``n_latent``
        :return: reconstructed full coordinates with last axis ``n_full``
        """
        return jnp.matmul(jnp.asarray(z), jnp.asarray(self.matrix))

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Alias for :meth:`reduce`."""
        return self.reduce(x)


class ConvAutoencoder2D(eqx.Module):
    """
    Small convolutional autoencoder for 2D fields.

    Single-sample tensor convention is ``(channels, height, width)``.
    """

    encoder_conv: eqx.nn.Conv2d
    encoder_linear: eqx.nn.Linear
    decoder_linear: eqx.nn.Linear
    decoder_conv: eqx.nn.ConvTranspose2d

    input_shape: tuple[int, int] = eqx.field(static=True)
    in_channels: int = eqx.field(static=True)
    hidden_channels: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)

    def __init__(
        self,
        input_shape: tuple[int, int],
        latent_dim: int,
        key: Key,
        in_channels: int = 1,
        hidden_channels: int = 4,
    ):
        """
        Build a convolutional autoencoder with stride-2 encoder and transpose-conv decoder.

        :param input_shape: spatial input shape ``(height, width)`` (both must be even)
        :param latent_dim: latent code dimension
        :param key: random key for initialization
        :param in_channels: input channels
        :param hidden_channels: hidden channel count
        """
        h, w = input_shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError("ConvAutoencoder2D expects even input_shape in both dimensions.")

        h2, w2 = h // 2, w // 2
        flat_dim = hidden_channels * h2 * w2
        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.encoder_conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            key=k1,
        )
        self.encoder_linear = eqx.nn.Linear(flat_dim, latent_dim, key=k2)
        self.decoder_linear = eqx.nn.Linear(latent_dim, flat_dim, key=k3)
        self.decoder_conv = eqx.nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            key=k4,
        )

        self.input_shape = input_shape
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

    def encode(self, x: ArrayLike) -> ArrayLike:
        """
        Encode one sample to latent coordinates.

        :param x: input sample with shape ``(channels, height, width)``
        :return: latent vector with shape ``(latent_dim,)``
        """
        y = self.encoder_conv(jnp.asarray(x))
        y = jax.nn.tanh(y)
        return self.encoder_linear(jnp.ravel(y))

    def decode(self, z: ArrayLike) -> ArrayLike:
        """
        Decode one latent vector to a reconstructed sample.

        :param z: latent vector with shape ``(latent_dim,)``
        :return: reconstructed sample with shape ``(channels, height, width)``
        """
        h2, w2 = self.input_shape[0] // 2, self.input_shape[1] // 2
        y = self.decoder_linear(jnp.asarray(z))
        y = jax.nn.tanh(y)
        y = jnp.reshape(y, (self.hidden_channels, h2, w2))
        return self.decoder_conv(y)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Autoencode one sample."""
        return self.decode(self.encode(x))


def _coerce_path(value: Any) -> TreePath:
    """Coerce a path-like object into a tuple path."""
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _get_subtree(tree: PyTree, path: TreePath) -> PyTree:
    """Return subtree located at ``path``."""
    node = tree
    for token in path:
        if isinstance(node, Mapping):
            node = node[token]
        elif isinstance(node, (list, tuple)):
            node = node[token]
        else:
            node = getattr(node, token)
    return node


def _constant_spec_like(tree: PyTree, value: Any) -> PyTree:
    """Fill a spec tree with one constant value."""
    return jax.tree_util.tree_map(lambda _: value, tree)


def _set_subtree(tree: PyTree | None, path: TreePath, value: PyTree) -> PyTree:
    """Set one subtree in a nested dict/list/tuple tree and return the updated tree."""
    if len(path) == 0:
        return value

    head, tail = path[0], path[1:]

    if isinstance(head, str):
        out = {} if tree is None else dict(tree)
        out[head] = _set_subtree(out.get(head), tail, value)
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
    out_list[head] = _set_subtree(out_list[head], tail, value)

    if isinstance(tree, tuple):
        return tuple(out_list)
    return out_list


def _build_routed_patch(candidate: PyTree, routes: list[TreeRoute]) -> PyTree:
    """Construct one patch tree by routing selected candidate subtrees."""
    patch: PyTree | None = None
    for route in routes:
        source_value = candidate if len(route.source) == 0 else _get_subtree(candidate, route.source)
        patch = _set_subtree(patch, route.target, source_value)
    return {} if patch is None else patch


def _merge_filtered_trees(base: PyTree, patch: PyTree) -> PyTree:
    """Merge patch into base, recursively overriding only paths present in patch."""
    return merge_pytrees(base, patch)


def eqx_evaluate(
    x: PyTree,
    module: eqx.Module | Callable[[PyTree], PyTree],
    reshape: Literal["flat", "stack"] | Callable[[PyTree], ArrayLike] | None = None,
    input_path: TreePath | PathToken | None = None,
    method: str | None = None,
    method_kwargs: dict[str, Any] | None = None,
    batched: bool = False,
    in_axes: int = 0,
    out_axes: int = 0,
    output_path: TreePath | PathToken | None = None,
) -> PyTree:
    """
    Evaluate array-like inputs with an Equinox-like module or callable.

    :param x: numeric pytree input
    :param module: Equinox module or callable
    :param reshape: optional pre-processing of ``x`` before ``module(x)``
        - ``None``: no reshape
        - ``"flat"``: flatten and concatenate all leaves into one 1D array
        - ``"stack"``: stack leaves along a leading axis (requires compatible leaf shapes)
        - callable: custom reshape function
    :param input_path: optional subtree path selected before reshape/evaluation
    :param method: optional attribute name on ``module`` to call (for example ``\"encode\"``)
    :param method_kwargs: optional kwargs passed to the selected callable
    :param batched: if ``True``, apply ``jax.vmap`` over the selected/reshaped input
    :param in_axes: ``vmap`` input axis when ``batched`` is ``True``
    :param out_axes: ``vmap`` output axis when ``batched`` is ``True``
    :param output_path: optional path used to wrap output into a pytree
    :return: callable output (optionally wrapped at ``output_path``)
    """
    x_selected = _get_subtree(x, _coerce_path(input_path)) if input_path is not None else x

    if reshape is None:
        x_eval = x_selected
    elif callable(reshape):
        x_eval = reshape(x_selected)
    elif reshape == "flat":
        leaves = jax.tree_util.tree_leaves(x_selected)
        if len(leaves) == 0:
            x_eval = jnp.zeros((0,))
        else:
            flat_leaves = [jnp.ravel(jnp.asarray(leaf)) for leaf in leaves]
            x_eval = jnp.concatenate(flat_leaves, axis=0)
    elif reshape == "stack":
        leaves = jax.tree_util.tree_leaves(x_selected)
        if len(leaves) == 0:
            x_eval = jnp.zeros((0,))
        else:
            x_eval = jnp.stack([jnp.asarray(leaf) for leaf in leaves], axis=0)
    else:
        raise ValueError(f"Unknown reshape specification {reshape!r}")

    call_kwargs = {} if method_kwargs is None else method_kwargs
    eval_fn = getattr(module, method) if method is not None else module
    if not callable(eval_fn):
        raise TypeError("Resolved module evaluation target is not callable.")

    if batched:
        y = jax.vmap(lambda xx: eval_fn(xx, **call_kwargs), in_axes=in_axes, out_axes=out_axes)(x_eval)
    else:
        y = eval_fn(x_eval, **call_kwargs)

    if output_path is not None:
        return _set_subtree(None, _coerce_path(output_path), y)
    return y
