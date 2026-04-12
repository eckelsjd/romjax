from abc import abstractmethod, ABC
from typing import Any, Callable, Literal

import jax
import equinox as eqx
from pydantic import Field, field_validator
from jaxtyping import Key, PyTree, ArrayLike

from romjax.graph import Edge
from romjax.typing import DictModel


class Sampleable(ABC):
    """Mixin for a model to indicate the ability to sample input and output spaces (i.e. the 'coordinates')."""

    @abstractmethod
    def sample_inputs(self, key: Key) -> PyTree:
        """Sample a single model input for the given key."""
        raise NotImplementedError
    
    @abstractmethod
    def sample_outputs(self, key: Key, inputs: PyTree | None = None, solution: PyTree | None = None) -> PyTree:
        """
        Produce one sample of outputs for the given key.
        
        :param key: the random key
        :param inputs: optionally condition on inputs
        :param solution: for efficiency, optionally condition on the precomputed solution of solve(inputs)=0
        :return: the outputs sample
        """
        raise NotImplementedError


class ImplicitModel(Edge, ABC):
    """
    An implicit function f(b,u) that maps inputs/outputs to residuals.
    
    The forward/backward functions are the augmented (and invertible) residual given by F(b,u) = (Id(b), f(b,u)).
    Must implement residual evaluate/solve methods that map (b,u) -> r and (b,r) -> u, respectively.
    """

    def forward(self, x: PyTree) -> PyTree:
        """Pass inputs through and evaluate residuals.
        
        :param x: must be of the form {"inputs": ..., "outputs": ...}
        :return: pytree of the form   {"inputs": ..., "residuals": ...}
        """
        return {"inputs": x["inputs"], "residuals": self.evaluate(x["inputs"], x["outputs"])}
    
    def backward(self, x: PyTree) -> PyTree:
        """Pass inputs through and solve for outputs.
        
        :param x: must be of the form {"inputs": ..., "residuals": ...}
        :return: pytree of the form   {"inputs": ..., "outputs": ...}
        """
        return {"inputs": x["inputs"], "outputs": self.solve(x["inputs"], x["residuals"])}

    @abstractmethod
    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate forward residual function f(b,u)."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, inputs: PyTree, residuals: PyTree) -> PyTree:
        """Solve inverse residual function f(b,u)=r."""
        raise NotImplementedError
    

class ExplicitModel(ImplicitModel, ABC):
    """
    Compute an explicit model via the pushforward: outputs = G(inputs).
    
    Assumes the residual has the same tree structure as the outputs and the pushforward.
    """

    def evaluate(self, inputs: PyTree, outputs: PyTree) -> PyTree:
        """Evaluate the residual as f(b,u) = u - G(b)"""
        return jax.tree.map(lambda u, uhat: u - uhat, outputs, self.pushforward(inputs))
    
    def solve(self, inputs: PyTree, residuals: PyTree) -> PyTree:
        """Solve the inverse (which just computes the pushforward)."""
        return jax.tree.map(lambda r, uhat: r + uhat, residuals, self.pushforward(inputs))
    
    @abstractmethod
    def pushforward(self, inputs: PyTree) -> PyTree:
        """Compute explicit outputs from inputs."""
        raise NotImplementedError


type FilterSpec = bool | Callable[[Any], bool]  # determine if a pytree leaf should be kept or filtered


class PathSpec(DictModel):
    """Path-based override for one subtree in an Equinox filter spec tree."""
    path: tuple[str | int, ...]
    spec: FilterSpec = True

    @field_validator("path", mode="before")
    @classmethod
    def _coerce_path(cls, value):
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value,)


class FilterModelSpec(DictModel):
    """
    Configuration for one filtered forward/backward component of a FilterModel.

    Supports either direct Equinox filter specs (`in_spec`/`out_spec`) or path-based overrides (`in_paths`/`out_paths`).
    Path overrides are merged into the full spec tree that is passed to `eqx.filter`.

    The pairs (in_spec, in_paths) and (out_spec, out_paths) are resolved according to the following:

    If spec and paths are both empty, then no filtering is done (whole tree is returned).
    If spec is callable, this is applied to the whole tree and path overrides are not allowed.
    If spec is a boolean or a tree, this is the default filter before applying overrides.
    If spec is none but paths are provided, then the default filter is False on all leaves.
    """
    forward: Callable[[PyTree, Any], PyTree]
    backward: Callable[[PyTree, Any], PyTree]

    in_spec: PyTree[FilterSpec] | None = None
    out_spec: PyTree[FilterSpec] | None = None
    in_paths: list[PathSpec] = Field(default_factory=list)
    out_paths: list[PathSpec] = Field(default_factory=list)
    forward_opts: dict[str, Any] = Field(default_factory=dict)
    backward_opts: dict[str, Any] = Field(default_factory=dict)

    @field_validator("in_paths", "out_paths", mode="before")
    @classmethod
    def _coerce_paths(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        out: list[Any] = []
        for item in value:
            if isinstance(item, PathSpec):
                out.append(item)
            elif isinstance(item, dict):
                out.append(PathSpec(**item))
            elif isinstance(item, (list, tuple)):
                out.append(PathSpec(path=item))
            else:
                out.append(PathSpec(path=(item,)))
        return out

    def _resolve_spec(self, tree: PyTree, spec: PyTree[FilterSpec], path_specs: list[PathSpec]) -> PyTree[FilterSpec]:
        """Return a filter spec tree compatible with `eqx.filter` by merging path overrides with default spec."""
        if spec is None and not path_specs:
            return True  # Return whole tree by default

        if callable(spec) and path_specs:
            raise ValueError("Path-based overrides cannot be applied to a callable filter spec.")

        # Setup the full default spec tree
        if spec is None:
            spec_tree = jax.tree_util.tree_map(lambda _: False, tree)
        elif isinstance(spec, bool):
            spec_tree = jax.tree_util.tree_map(lambda _: spec, tree)
        else:
            spec_tree = spec

        # Apply path overrides
        for path_spec in path_specs:
            subtree = self._get_subtree(tree, path_spec.path)
            replacement = self._constant_spec_like(subtree, path_spec.spec)
            spec_tree = eqx.tree_at(
                lambda t, p=path_spec.path: self._get_subtree(t, p),
                spec_tree,
                replacement,
            )
        return spec_tree
    
    @staticmethod
    def _get_subtree(tree: PyTree, path: tuple[str | int, ...]) -> PyTree:
        """Return the subtree located at `path`."""
        node = tree
        for token in path:
            if isinstance(node, dict):
                node = node[token]
            elif isinstance(token, int):
                node = node[token]
            else:
                node = getattr(node, token)
        return node
    
    @staticmethod
    def _constant_spec_like(tree: PyTree, value: Any) -> PyTree:
        """Fill a spec tree with `value`."""
        return jax.tree_util.tree_map(lambda _: value, tree)

    def resolve_in_spec(self, tree: PyTree) -> PyTree[FilterSpec]:
        return self._resolve_spec(tree, self.in_spec, self.in_paths)

    def resolve_out_spec(self, tree: PyTree) -> PyTree[FilterSpec]:
        return self._resolve_spec(tree, self.out_spec, self.out_paths)


class FilterModel(Edge):
    """
    Flexible PyTree->PyTree mapping using simple input/output filtering and configurable callables.

    The inputs/outputs get pushed through a set of prescribed filters, allowing reuse of common "black-box" functions
    regardless of the input/output pytree structures.

    Pass extra 'filters' key to the forward/backward functions to give extra args to the individual filters.
    """
    filters: list[FilterModelSpec]

    def forward(self, x: PyTree) -> PyTree:
        """Evaluate forward model.
        
        :param x: must be of the form {..., 'filters': []}, where filters contain extra args specific to each filter
                  forward function, and everything else gets filtered before being passed to the forward function
        :return: the merged pytree of the results from all forward filter functions
        """
        assembled: PyTree | None = None
        filter_args = x.pop('filters', [None for _ in range(len(self.filters))])
        for spec, args in zip(self.filters, filter_args):
            in_view = eqx.filter(x, spec.resolve_in_spec(x))
            candidate = spec.forward(in_view, args, **spec.forward_opts)

            # TODO: merging is not quite right
            # Each filter should specify an out_path for where the result should be saved in the final pytree
            # And vice versa for the backward direction
            patch = eqx.filter(candidate, spec.resolve_out_spec(candidate))
            assembled = patch if assembled is None else _merge_filtered_trees(assembled, patch)
        if assembled is None:
            return x
        return assembled

    def backward(self, x: PyTree) -> PyTree:
        """Evaluate backward model.
        
        :param x: must be of the form {..., 'filters': []}, where filters contain extra args specific to each
                  filter backward function
        :return: the merged pytree of the results from all backward filter functions
        """
        assembled: PyTree | None = None
        filter_args = x.pop('filters', [None for _ in range(len(self.filters))])
        for spec, args in zip(self.filters, filter_args):
            out_view = eqx.filter(x, spec.resolve_out_spec(x))
            candidate = spec.backward(out_view, args, **spec.backward_opts)

            # TODO: merging is not quite right
            patch = eqx.filter(candidate, spec.resolve_in_spec(candidate))
            assembled = patch if assembled is None else _merge_filtered_trees(assembled, patch)
        if assembled is None:
            return x
        return assembled


def _merge_filtered_trees(base: PyTree, patch: PyTree) -> PyTree:
    return jax.tree_util.tree_map(
        lambda a, b: b if b is not None else a,
        base,
        patch,
        is_leaf=lambda x: x is None,
    )
    

def eqx_evaluate(
    x: PyTree,
    module: eqx.Module,
    reshape: Literal["flat", "stack"] | Callable[[PyTree], ArrayLike] | None = None
    ) -> ArrayLike:
    """
    Evaluate array-like inputs using an equinox nn module (assumed callable). 
    Optionally provide reshaping to array from pytree.

    This is for consistency with jax grad on eqx modules, where the network params are stored in the module itself.
    
    :param x: the numeric inputs
    :param module: the equinox module
    :param reshape: how to convert the numeric input PyTree to an array. "flat" will stack all into a 1d array.
                    "stack" will try to concatenate all arrays along a new leading axis. "None" will do nothing.
    :return: the eqx Module evaluation on the inputs
    """
    if reshape is None:
        reshape = lambda x: x
    if reshape == "flat":
        # TODO: flatten the input tree and stack all arrays in one big flat 1d array
        pass
    if reshape == "stack":
        # TODO:  flatten the input tree and stack all arrays along a new leading axis (assume all are the same shape)
        pass

    x = reshape(x)
    y = module(x)
    return y
