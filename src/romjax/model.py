from abc import ABC, abstractmethod
from collections.abc import Mapping
from inspect import Signature, signature
from typing import Any, Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Key, PyTree
from pydantic import Field, field_validator, model_validator

from romjax.graph import Edge, Node
from romjax.typing import DictModel
from romjax.utils import merge_pytrees

type FilterSpec = bool | Callable[[Any], bool]
type PathToken = str | int
type TreePath = tuple[PathToken, ...]

__all__ = ['Sampleable', 'eqx_evaluate', 'ImplicitModel', 'ExplicitModel', 'FilterModel']


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


class PathSpec(DictModel):
    """Path-based override for one subtree in an Equinox filter-spec tree."""

    path: TreePath
    spec: FilterSpec = True

    @model_validator(mode="before")
    @classmethod
    def _from_path(cls, value):
        if isinstance(value, tuple | list):
            return {"path": value}
        return value

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


class FilterModelSpec(DictModel):
    """
    One configurable filter component used by :class:`FilterModel`.

    The input side for each direction is controlled via ``in_spec``/``in_paths`` and
    ``out_spec``/``out_paths``. Callable results are merged into the assembled output by either:

    - explicit routes (`forward_routes`/`backward_routes`) for precise patch placement, or
    - legacy output filtering (`out_spec` for forward, `in_spec` for backward) if no routes are set.

    :param forward: callable for forward mapping (defaults to eqx_evaluate)
    :param backward: callable for backward mapping (defaults to eqx_evaluate)
    :param in_spec: base filter spec for forward input and backward output
    :param out_spec: base filter spec for forward output and backward input
    :param in_paths: path overrides merged into ``in_spec``
    :param out_paths: path overrides merged into ``out_spec``
    :param forward_routes: explicit routing from forward callable output to final output tree
    :param backward_routes: explicit routing from backward callable output to final output tree
    :param forward_opts: keyword args passed to ``forward``
    :param backward_opts: keyword args passed to ``backward``
    """

    forward: Callable[[PyTree, Any], PyTree] = eqx_evaluate
    backward: Callable[[PyTree, Any], PyTree] = eqx_evaluate

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

    _AUX_RUNTIME_ARG_KEY = "__romjax_filter_runtime_arg__"  # for passing equinox modules through "filters" input arg
    _AUX_CALL_DATA_KEY = "__romjax_filter_aux__"            # for other aux data returned by forward/backward functions

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

    def _split_filter_aux(self, aux: Any) -> list[Any]:
        """Split runtime per-filter auxiliary values."""
        if aux is None:
            return [None] * len(self.filters)
        if not isinstance(aux, (list, tuple)):
            aux = [aux]
        aux_list = list(aux)
        if len(aux_list) < len(self.filters):
            aux_list.extend([None] * (len(self.filters) - len(aux_list)))
        elif len(aux_list) > len(self.filters):
            raise ValueError(f"Received {len(aux_list)} aux entries but model has only {len(self.filters)} filters.")
        return aux_list

    def _unpack_filter_aux(self, aux: Any) -> tuple[Any, Any]:
        """Unpack internal aux envelope into runtime arg and callable aux."""
        if isinstance(aux, Mapping) and self._AUX_RUNTIME_ARG_KEY in aux:
            return aux.get(self._AUX_RUNTIME_ARG_KEY), aux.get(self._AUX_CALL_DATA_KEY)
        return None, aux

    def _pack_filter_aux(self, runtime_arg: Any, call_aux: Any) -> Any:
        """Pack runtime arg and callable aux into one graph-transportable payload."""
        if runtime_arg is None:
            return call_aux
        return {
            self._AUX_RUNTIME_ARG_KEY: runtime_arg,
            self._AUX_CALL_DATA_KEY: call_aux,
        }

    def _run(
        self,
        x: PyTree,
        direction: Literal["forward", "backward"],
        base_mode: Literal["empty", "input"],
        aux: Any = None,
        return_aux: bool = False,
    ) -> tuple[PyTree, list[Any] | None]:
        payload, filter_args = self._split_payload_and_args(x)
        filter_aux = self._split_filter_aux(aux)
        assembled: PyTree | None = payload if base_mode == "input" else None
        produced_aux: list[Any] | None = [] if return_aux else None

        for spec, runtime_arg, aux_in in zip(self.filters, filter_args, filter_aux):
            aux_runtime_arg, call_aux_in = self._unpack_filter_aux(aux_in)
            args = runtime_arg if runtime_arg is not None else aux_runtime_arg

            if direction == "forward":
                view = eqx.filter(payload, spec.resolve_in_spec(payload))
                candidate, aux_out = _call_filter_spec(
                    spec.forward,
                    view,
                    args,
                    opts=spec.forward_opts,
                    aux=call_aux_in,
                    return_aux=return_aux,
                )
                patch = spec.route_forward(candidate)
            else:
                view = eqx.filter(payload, spec.resolve_out_spec(payload))
                candidate, aux_out = _call_filter_spec(
                    spec.backward,
                    view,
                    args,
                    opts=spec.backward_opts,
                    aux=call_aux_in,
                    return_aux=return_aux,
                )
                patch = spec.route_backward(candidate)

            assembled = patch if assembled is None else _merge_filtered_trees(assembled, patch)
            if produced_aux is not None:
                produced_aux.append(self._pack_filter_aux(args, aux_out))

        output = payload if assembled is None else assembled
        if produced_aux is not None and all(item is None for item in produced_aux):
            produced_aux = None
        return output, produced_aux

    def forward(self, x: PyTree) -> PyTree:
        """
        Evaluate the forward-direction filter map.

        :param x: input pytree, optionally including ``"filters"`` runtime args
        :return: assembled forward output pytree
        """
        output, _ = self._run(x, direction="forward", base_mode=self.forward_base)
        return output

    def backward(self, x: PyTree) -> PyTree:
        """
        Evaluate the backward-direction filter map.

        :param x: input pytree, optionally including ``"filters"`` runtime args
        :return: assembled backward output pytree
        """
        output, _ = self._run(x, direction="backward", base_mode=self.backward_base)
        return output

    def forward_aux(self, x: PyTree, aux: PyTree | None = None) -> tuple[PyTree, PyTree | None]:
        """Evaluate forward map and return auxiliary payload for a later backward map."""
        output, produced_aux = self._run(x, direction="forward", base_mode=self.forward_base, aux=aux, return_aux=True)
        return output, produced_aux

    def backward_aux(self, x: PyTree, aux: PyTree | None = None) -> tuple[PyTree, PyTree | None]:
        """Evaluate backward map and return auxiliary payload for a later forward map."""
        output, produced_aux = self._run(
            x,
            direction="backward",
            base_mode=self.backward_base,
            aux=aux,
            return_aux=True,
        )
        return output, produced_aux


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


def _get_callable_signature(fn: Callable[..., Any]) -> Signature | None:
    """Best-effort callable signature extraction."""
    try:
        return signature(fn)
    except (TypeError, ValueError):
        return None


def _accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    """Return True if callable supports a named kwarg or ``**kwargs``."""
    sig = _get_callable_signature(fn)
    if sig is None:
        return False
    for param in sig.parameters.values():
        if param.kind == param.VAR_KEYWORD or param.name == name:
            return True
    return False


def _call_filter_spec(
    fn: Callable[..., Any],
    view: PyTree,
    filter_arg: Any,
    opts: dict[str, Any],
    aux: Any,
    return_aux: bool,
) -> tuple[PyTree, Any]:
    """Call one filter function with optional auxiliary-data protocol."""
    kwargs = dict(opts)
    supports_aux = _accepts_kwarg(fn, "aux")
    supports_return_aux = _accepts_kwarg(fn, "return_aux")

    if supports_aux and "aux" not in kwargs:
        kwargs["aux"] = aux
    if return_aux and supports_return_aux and "return_aux" not in kwargs:
        kwargs["return_aux"] = True

    result = fn(view, filter_arg, **kwargs)
    if return_aux and supports_return_aux:
        if not (isinstance(result, tuple) and len(result) == 2):
            raise TypeError("Filter callable returned invalid auxiliary output; expected `(value, aux)` tuple.")
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
            return selected_leaves[0]
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
