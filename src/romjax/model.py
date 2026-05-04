from abc import ABC, abstractmethod
from collections.abc import Mapping
from inspect import Signature, signature
from typing import Any, Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key, PyTree
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from romjax.graph import Edge, Node
from romjax.tree import pytree_merge

type PathToken = str | int
type TreePath = tuple[PathToken, ...]


__all__ = ['Sampleable', 'eqx_evaluate', 'identity_filter', 'ImplicitModel', 'ExplicitModel', 'FilterModel']


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

        Implementations may delegate to configurable samplers with the conditioning contract
        ``sampler(key, inputs=..., solution=..., **opts)``.

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

    source: Node = Node(name="inputs")
    target: Node = Node(name="outputs")

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


class OuterToInnerRoute(BaseModel):
    """
    Route one subtree from the outer edge payload into the inner callable-facing input tree.

    If ``inner`` is omitted, the selected subtree keeps the same structure in the inner tree.

    :param outer: source path in the outer edge payload
    :param inner: destination path in the inner callable input tree
    """

    outer: TreePath
    inner: TreePath | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_path(cls, value: Any) -> Any:
        if isinstance(value, tuple | list):
            return {"outer": value}
        return value

    @model_validator(mode="after")
    def _set_default_inner(self):
        if self.inner is None:
            self.inner = self.outer
        return self

    @field_validator("outer", "inner", mode="before")
    @classmethod
    def _coerce_paths(cls, value: Any) -> TreePath | None:
        if value is None:
            return None
        return _coerce_path(value)


class InnerToOuterRoute(BaseModel):
    """
    Route one subtree from the inner callable output tree into the assembled outer edge payload.

    ``inner`` defaults to the callable output root when omitted.

    :param outer: destination path in the assembled outer edge payload
    :param inner: source path in the inner callable output tree
    """

    outer: TreePath
    inner: TreePath = ()

    @model_validator(mode="before")
    @classmethod
    def _from_path(cls, value: Any) -> Any:
        if isinstance(value, tuple | list):
            return {"outer": value}
        return value

    @field_validator("outer", "inner", mode="before")
    @classmethod
    def _coerce_paths(cls, value: Any) -> TreePath:
        return _coerce_path(value)


def identity_filter(x: PyTree, args: Any, **kwargs):
    return x


def eqx_evaluate(
    x: PyTree,
    module: eqx.Module | Callable[[PyTree], PyTree],
    gather: Literal["flat", "stack"] | Callable[[PyTree], ArrayLike] | None = None,
    scatter: Literal["flat", "stack"] | Callable[[ArrayLike, PyTree], PyTree] | None = None,
    leaf_filter: Callable[[Any], bool] = eqx.is_array,
    method: str | None = None,
    method_kwargs: dict[str, Any] | None = None,
    aux: Mapping[str, Any] | None = None,
    return_aux: bool = False,
    template: PyTree | None = None,
    capture_template: bool = True,
) -> PyTree | tuple[PyTree, dict[str, Any]]:
    """
    Evaluate a filtered pytree with an Equinox-like module or callable.

    ``FilterModel`` is expected to handle input/output path selection and routing. This helper only handles:
    input collection/gathering, callable evaluation, and optional output reconstruction/scattering.

    :param x: filtered pytree input
    :param module: Equinox module or callable
    :param gather: optional pre-processing of ``x`` before ``module(x)``
        - ``None``: pass tree through unchanged
        - ``"flat"``: flatten and concatenate all leaves into one 1D array
        - ``"stack"``: stack leaves along a leading axis (requires compatible leaf shapes)
        - callable: custom gather function
    :param scatter: optional post-processing of callable output
        - ``None``: return callable output unchanged
        - ``"flat"``: split one flat array into template leaf shapes
        - ``"stack"``: split leading axis into template leaves
        - callable: custom scatter function ``f(y, template) -> pytree``
    :param leaf_filter: predicate selecting leaves used in ``"flat"``/``"stack"`` gather and scatter/reconstruction
    :param method: optional attribute name on ``module`` to call (for example ``\"encode\"``)
    :param method_kwargs: optional kwargs passed to the selected callable
    :param aux: optional auxiliary payload. If ``template`` is not provided, this checks ``aux["template"]``
    :param return_aux: if True, return ``(value, aux_out)``
    :param template: optional pytree template for scatter/reconstruction
    :param capture_template: if True, include inferred shape template in aux output
    :return: callable output (optionally reconstructed to a pytree)
    """
    aux_out: dict[str, Any] = {}
    if capture_template:
        aux_out["template"] = _shape_template_like(x, leaf_filter=leaf_filter)

    if gather is None:
        x_eval = x
    elif callable(gather):
        x_eval = gather(x)
    elif gather == "flat":
        x_eval = _gather_tree_array(x, mode="flat", leaf_filter=leaf_filter)
    elif gather == "stack":
        x_eval = _gather_tree_array(x, mode="stack", leaf_filter=leaf_filter)
    else:
        raise ValueError(f"Unknown gather specification {gather!r}")

    call_kwargs = {} if method_kwargs is None else method_kwargs
    eval_fn = getattr(module, method) if method is not None else module
    if not callable(eval_fn):
        raise TypeError("Resolved module evaluation target is not callable.")
    y = eval_fn(x_eval, **call_kwargs)

    if scatter is None:
        if return_aux:
            return y, aux_out
        return y

    template_tree = template
    if template_tree is None and aux is not None:
        template_tree = aux.get("template")
    if template_tree is None:
        raise ValueError("Reconstruction requires a template supplied explicitly or through auxiliary data.")

    if callable(scatter):
        reconstructed = scatter(y, template_tree)
    elif scatter == "flat":
        reconstructed = _scatter_tree_array(y, template_tree, mode="flat", leaf_filter=leaf_filter)
    elif scatter == "stack":
        reconstructed = _scatter_tree_array(y, template_tree, mode="stack", leaf_filter=leaf_filter)
    else:
        raise ValueError(f"Unknown scatter specification {scatter!r}")

    if return_aux:
        return reconstructed, aux_out
    return reconstructed


class FilterDirectionSpec(BaseModel):
    """
    One directional callable configuration for :class:`FilterModelSpec`.

    Each call direction follows one routing model:

    1. assemble an inner callable input tree from the outer payload using ``input_routes``,
    2. call ``callable(inner_input, call_arg, **opts)``,
    3. merge the callable result back into the outer output using ``output_routes``.

    If ``input_routes`` is omitted, the full outer payload is passed through as the inner input.
    If ``output_routes`` is omitted, the callable output is used directly as the outer patch.

    :param callable: directional callable
    :param input_routes: routes from outer payload to inner callable input
    :param output_routes: routes from inner callable output to outer payload
    :param opts: keyword args passed to the directional callable
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    callable: Callable[[PyTree, Any], PyTree] = identity_filter
    input_routes: list[OuterToInnerRoute] = Field(default_factory=list)
    output_routes: list[InnerToOuterRoute] = Field(default_factory=list)
    opts: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input_routes", mode="before")
    @classmethod
    def _coerce_input_routes(cls, value: Any) -> list[OuterToInnerRoute]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        out: list[OuterToInnerRoute] = []
        for item in value:
            if isinstance(item, OuterToInnerRoute):
                out.append(item)
            elif isinstance(item, dict):
                out.append(OuterToInnerRoute(**item))
            else:
                out.append(OuterToInnerRoute(outer=item))
        return out

    @field_validator("output_routes", mode="before")
    @classmethod
    def _coerce_output_routes(cls, value: Any) -> list[InnerToOuterRoute]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        out: list[InnerToOuterRoute] = []
        for item in value:
            if isinstance(item, InnerToOuterRoute):
                out.append(item)
            elif isinstance(item, dict):
                out.append(InnerToOuterRoute(**item))
            else:
                out.append(InnerToOuterRoute(outer=item))
        return out

    def assemble_inner_input(self, outer_payload: PyTree) -> PyTree:
        """Assemble the callable-facing inner input tree for one direction."""
        if not self.input_routes:
            return outer_payload
        return _assemble_inner_input(outer_payload, self.input_routes)

    def assemble_outer_output(self, inner_output: PyTree) -> PyTree:
        """Assemble the outer patch returned by one directional callable."""
        if not self.output_routes:
            return inner_output
        return _assemble_outer_patch(inner_output, self.output_routes)


class FilterModelSpec(BaseModel):
    """
    One self-contained bidirectional route-driven transform used by :class:`FilterModel`.

    Each spec defines one forward direction and one backward direction. In both cases the spec:

    1. routes data from the outer edge payload into a callable-facing inner tree,
    2. evaluates the configured callable with optional runtime call input,
    3. routes the callable result back into the assembled outer payload.

    Runtime call inputs are supplied by :class:`FilterModel` via top-level ``call_args``.
    Cached inverse-pass state is transported separately through ``forward_aux`` / ``backward_aux``.

    :param forward: forward-direction routing and callable config
    :param backward: backward-direction routing and callable config
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    forward: FilterDirectionSpec = Field(default_factory=FilterDirectionSpec)
    backward: FilterDirectionSpec = Field(default_factory=FilterDirectionSpec)


class FilterModel(Edge):
    """
    Flexible route-driven pytree-to-pytree edge map.

    The runtime payload may include an optional top-level ``"call_args"`` entry. ``call_args`` supplies
    runtime call inputs for the spec callables and is kept distinct from graph-transported auxiliary state.
    When :meth:`romjax.graph.FunctionGraph.push_path` is used with ``edge_payload_patches``, those graph-level
    patches choose which ``call_args`` payload reaches this edge. This class still only handles the per-edge
    normalization and distribution of ``call_args`` across its own filter specs.

    ``call_args`` accepts either:

    - a single shared value, broadcast to every spec,
    - ``{"shared": value}`` for an explicit shared value, or
    - ``{"per_spec": [value0, value1, ...]}`` for one value per spec.

    When there is exactly one filter spec, a bare value of any type is treated as that spec's runtime input.

    :param filters: list of directional filter specs
    """

    filters: list[FilterModelSpec]

    _CALL_ARGS_KEY = "call_args"
    _SHARED_CALL_ARGS_KEY = "shared"
    _PER_SPEC_CALL_ARGS_KEY = "per_spec"
    _CACHED_STATES_KEY = "cached_states"

    def _split_outer_payload_and_runtime_inputs(self, x: PyTree) -> tuple[PyTree, Any, list[Any]]:
        """Split the outer payload from normalized per-spec runtime call inputs."""
        outer_payload = x
        runtime_args: Any = None

        if isinstance(x, Mapping) and self._CALL_ARGS_KEY in x:
            outer_payload = dict(x)
            runtime_args = outer_payload.pop(self._CALL_ARGS_KEY)

        if runtime_args is None:
            return outer_payload, None, [None] * len(self.filters)

        if (
            isinstance(runtime_args, Mapping)
            and set(runtime_args.keys()).issubset({self._SHARED_CALL_ARGS_KEY, self._PER_SPEC_CALL_ARGS_KEY})
        ):
            if self._SHARED_CALL_ARGS_KEY in runtime_args and self._PER_SPEC_CALL_ARGS_KEY in runtime_args:
                raise ValueError("call_args must provide either 'shared' or 'per_spec', not both.")
            if self._SHARED_CALL_ARGS_KEY in runtime_args:
                return (
                    outer_payload,
                    runtime_args,
                    [runtime_args[self._SHARED_CALL_ARGS_KEY]] * len(self.filters),
                )
            runtime_args = runtime_args.get(self._PER_SPEC_CALL_ARGS_KEY)
            if not isinstance(runtime_args, (list, tuple)):
                raise ValueError("call_args['per_spec'] must be a list or tuple aligned with the filter specs.")

        if len(self.filters) == 1:
            return outer_payload, runtime_args, [runtime_args]

        if isinstance(runtime_args, (list, tuple)):
            args = list(runtime_args)
            if len(args) != len(self.filters):
                raise ValueError(
                    f"Received {len(args)} per-spec runtime inputs but model has {len(self.filters)} filter specs."
                )
            return outer_payload, runtime_args, args

        return outer_payload, runtime_args, [runtime_args] * len(self.filters)

    def _extract_cached_states(self, aux: Any) -> list[Any]:
        """Normalize graph-transported cached call state into one entry per spec."""
        if aux is None:
            return [None] * len(self.filters)
        if not isinstance(aux, Mapping) or self._CACHED_STATES_KEY not in aux:
            raise ValueError("FilterModel auxiliary state must be a mapping with a 'cached_states' entry.")
        cached_states = aux[self._CACHED_STATES_KEY]
        if not isinstance(cached_states, (list, tuple)):
            raise ValueError("FilterModel auxiliary 'cached_states' must be a list or tuple aligned with the filters.")
        cached_state_list = list(cached_states)
        if len(cached_state_list) != len(self.filters):
            raise ValueError(
                f"Received {len(cached_state_list)} cached states but model has {len(self.filters)} filter specs."
            )
        return cached_state_list

    def _package_cached_states(self, cached_states: list[Any] | None) -> dict[str, list[Any]] | None:
        """Package per-spec cached call state for graph transport."""
        if cached_states is None or all(item is None for item in cached_states):
            return None
        return {self._CACHED_STATES_KEY: cached_states}

    def _run(
        self,
        x: PyTree,
        direction: Literal["forward", "backward"],
        aux: Any = None,
        return_aux: bool = False,
    ) -> tuple[PyTree, list[Any] | None]:
        outer_payload, runtime_payload, runtime_inputs = self._split_outer_payload_and_runtime_inputs(x)
        cached_states = self._extract_cached_states(aux)
        assembled_outer: PyTree | None = None
        produced_cached_states: list[Any] | None = [] if return_aux else None

        for spec, runtime_input, cached_state_in in zip(self.filters, runtime_inputs, cached_states):
            direction_spec = spec.forward if direction == "forward" else spec.backward
            inner_input = direction_spec.assemble_inner_input(outer_payload)
            inner_output, cached_state_out = _call_direction_callable(
                direction_spec.callable,
                inner_input,
                runtime_input,
                opts=direction_spec.opts,
                cached_state=cached_state_in,
                return_aux=return_aux,
            )
            outer_patch = direction_spec.assemble_outer_output(inner_output)
            assembled_outer = (
                outer_patch
                if assembled_outer is None
                else pytree_merge(assembled_outer, outer_patch)
            )
            if produced_cached_states is not None:
                produced_cached_states.append(cached_state_out)

        output = outer_payload if assembled_outer is None else assembled_outer
        if runtime_payload is not None and isinstance(output, Mapping):
            output = dict(output)
            output[self._CALL_ARGS_KEY] = runtime_payload
        return output, produced_cached_states

    def forward(self, x: PyTree) -> PyTree:
        """
        Evaluate the forward-direction filter map.

        :param x: input pytree, optionally including top-level ``call_args``
        :return: assembled forward output pytree
        """
        output, _ = self._run(x, direction="forward")
        return output

    def backward(self, x: PyTree) -> PyTree:
        """
        Evaluate the backward-direction filter map.

        :param x: input pytree, optionally including top-level ``call_args``
        :return: assembled backward output pytree
        """
        output, _ = self._run(x, direction="backward")
        return output

    def forward_aux(self, x: PyTree, aux: PyTree | None = None) -> tuple[PyTree, PyTree | None]:
        """Evaluate forward map and return auxiliary payload for a later backward map."""
        output, produced_aux = self._run(x, direction="forward", aux=aux, return_aux=True)
        return output, self._package_cached_states(produced_aux)

    def backward_aux(self, x: PyTree, aux: PyTree | None = None) -> tuple[PyTree, PyTree | None]:
        """Evaluate backward map and return auxiliary payload for a later forward map."""
        output, produced_aux = self._run(
            x,
            direction="backward",
            aux=aux,
            return_aux=True,
        )
        return output, self._package_cached_states(produced_aux)


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
        elif isinstance(token, int):
            node = node[token]
        else:
            node = getattr(node, token)
    return node


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


def _assemble_inner_input(outer_payload: PyTree, routes: list[OuterToInnerRoute]) -> PyTree:
    """Construct one callable-facing inner tree from routed outer payload subtrees."""
    patch: PyTree | None = None
    for route in routes:
        source_value = _get_subtree(outer_payload, route.outer)
        patch = _set_subtree(patch, route.inner or (), source_value)
    return {} if patch is None else patch


def _assemble_outer_patch(inner_output: PyTree, routes: list[InnerToOuterRoute]) -> PyTree:
    """Construct one outer patch tree from routed inner callable output subtrees."""
    patch: PyTree | None = None
    for route in routes:
        source_value = inner_output if len(route.inner) == 0 else _get_subtree(inner_output, route.inner)
        patch = _set_subtree(patch, route.outer, source_value)
    return {} if patch is None else patch


def _get_callable_signature(fn: Callable[..., Any]) -> Signature | None:
    """Best-effort callable signature extraction."""
    try:
        return signature(fn)
    except (TypeError, ValueError):
        return None


def _accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    """Return True if callable explicitly lists a named kwarg."""
    sig = _get_callable_signature(fn)
    if sig is None:
        return False
    param = sig.parameters.get(name)
    return param is not None and param.kind in {
        param.POSITIONAL_OR_KEYWORD,
        param.KEYWORD_ONLY,
    }


def _call_direction_callable(
    fn: Callable[..., Any],
    inner_input: PyTree,
    runtime_input: Any,
    opts: dict[str, Any],
    cached_state: Any,
    return_aux: bool,
) -> tuple[PyTree, Any]:
    """
    Call one directional filter-model callable.

    The callable contract is:

    - positional args: ``fn(inner_input, runtime_input, ...)``
    - optional kwarg: ``aux=...`` receives cached state from the inverse pass
    - optional kwarg: ``return_aux=True`` requests ``(value, cached_state)`` output

    If ``return_aux`` is requested and supported by the callable, the second tuple element is cached
    and passed to the inverse-direction call through :class:`FilterModel` auxiliary transport.
    """
    kwargs = dict(opts)
    supports_aux = _accepts_kwarg(fn, "aux")
    supports_return_aux = _accepts_kwarg(fn, "return_aux")

    if supports_aux and "aux" not in kwargs:
        kwargs["aux"] = cached_state
    if return_aux and supports_return_aux and "return_aux" not in kwargs:
        kwargs["return_aux"] = True

    result = fn(inner_input, runtime_input, **kwargs)
    if return_aux and supports_return_aux:
        if not (isinstance(result, tuple) and len(result) == 2):
            raise TypeError("Directional callable returned invalid auxiliary output; expected `(value, cached_state)`.")
        return result[0], result[1]
    return result, None


def _shape_template_like(tree: PyTree, leaf_filter: Callable[[Any], bool]) -> PyTree:
    """Create a lightweight template preserving array leaf shapes and dtypes."""
    return jax.tree_util.tree_map(
        lambda leaf: jnp.zeros_like(jnp.asarray(leaf)) if leaf_filter(leaf) else leaf,
        tree,
    )


def _gather_tree_array(
    tree: PyTree,
    mode: Literal["flat", "stack"],
    leaf_filter: Callable[[Any], bool],
) -> ArrayLike:
    """Collect/gather selected tree leaves into one array."""
    selected_leaves = [jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree) if leaf_filter(leaf)]
    if not selected_leaves:
        return jnp.zeros((0,))

    if mode == "flat":
        if len(selected_leaves) == 1:
            return jnp.ravel(selected_leaves[0])
        return jnp.concatenate([jnp.ravel(leaf) for leaf in selected_leaves], axis=0)

    ref_shape = selected_leaves[0].shape
    for leaf in selected_leaves[1:]:
        if leaf.shape != ref_shape:
            raise ValueError(f"'stack' gather requires identical leaf shapes, got {leaf.shape} and {ref_shape}.")
    return jnp.stack(selected_leaves, axis=0)


def _scatter_tree_array(
    value: ArrayLike,
    template: PyTree,
    mode: Literal["flat", "stack"],
    leaf_filter: Callable[[Any], bool],
) -> PyTree:
    """Reconstruct/scatter selected template leaves from one array output."""
    leaves, treedef = jax.tree_util.tree_flatten(template)
    selected = [(idx, jnp.asarray(leaf)) for idx, leaf in enumerate(leaves) if leaf_filter(leaf)]
    if not selected:
        return template

    output_leaves = list(leaves)
    if mode == "flat":
        flat_value = jnp.ravel(jnp.asarray(value))
        expected = sum(leaf.size for _, leaf in selected)
        if flat_value.shape[0] != expected:
            raise ValueError(f"'flat' scatter expected {expected} values but received {flat_value.shape[0]}.")

        offset = 0
        for idx, leaf in selected:
            size = leaf.size
            output_leaves[idx] = jnp.reshape(flat_value[offset : offset + size], leaf.shape)
            offset += size
    else:
        stacked_value = jnp.asarray(value)
        if stacked_value.shape[0] != len(selected):
            raise ValueError(
                f"'stack' scatter expected leading axis {len(selected)} but received {stacked_value.shape[0]}."
            )
        for i, (idx, leaf) in enumerate(selected):
            candidate = stacked_value[i]
            if candidate.shape != leaf.shape:
                if candidate.size != leaf.size:
                    raise ValueError(
                        f"'stack' scatter cannot reshape leaf {i} from {candidate.shape} to {leaf.shape}."
                    )
                candidate = jnp.reshape(candidate, leaf.shape)
            output_leaves[idx] = candidate

    return jax.tree_util.tree_unflatten(treedef, output_leaves)
