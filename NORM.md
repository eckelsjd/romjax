## Overview

The goal is to implement a clean and configurable way to add normalization to graph Edge forward() and backward() methods.
The normalization will act as a pre-processing step and should take the form of a PyTree of functions, or alternatively a function that takes a PyTree input, and ultimately produces a PyTree output that matches the PyTree input. In total, there should be 4 such normalization steps, one for either side of both the forward and backward functions. Ideally, the pre- and post-processing norm calls are integrated as wrappers around the existing forward() and backward() methods in the parent abstract Edge class, so that all children Edge classes can get normalization for "free" by simple configuration. It should also be possible for child classes to implement instance methods that override the parent's default way of handling normalization.

The default method for normalization should be a configurable pytree of callables (with args), where each callable is applied to each corresponding leaf of the payload `x`, and produces a corresponding normalized leaf in the output `xhat`. This normalized pytree then proceeds to the `forward` or `backward` method per usual. Likewise, before the `forward` or `backward` method returns its output, that output passes through the post-processing norm callable before being returned to the caller. By default, no normalization is applied, i.e. all pytrees pass through pre- and post- norms unchanged.

## Key software design points

Your design should cleanly integrate at the top-level Edge class, where the new normalization functionality should act as a wrapper around existing forward/backward infrastructure.

You will need to come up with a good public API config structure for adding normalization to graph Edges. The main use case is to apply unary-like operators at the leaf-level, very likely with the need to include additional arguments (like norm constants) from the config itself or by loading from file. Norms will likely always be applied on a per-leaf basis, but a single norm should broadcast. Consider similar designs used for specifying callables in a Pytree, such as the PyTreeSampler.

To allow complete control, it should also optionally be possible for child classes to implement instance methods that completely override the default PyTree normalization scheme. This would allow the user to insert any additional transformations to the PyTree payloads that aren't already handled by the default infrastructure.

When specifying norms in the forward direction, the corresponding "inverse" norms should be applied when data comes back through the backward direction. In this way, the normalizations are just pre/post process steps, but do not actually change the underlying data irreversibly. The primary use case is for training neural networks, where we will likely precompute norm constants, and then normalize all incoming data online during graph evaluation.

## Requirements

- All existing graph Edge APIs should be unaffected. Normalization should be integrated as wrappers around forward/backward methods.
- It should be possible to configure all Edge classes via pydantic how to handle pre- and post- normalization for both forward and backward.
- The pre-norm for forward should be "undone/inversed" by the post-norm of backward. Likewise, the pre-norm of backward should be undone by the post-norm of forward.
- If only a single callable is specified for a norm, then this should validate as broadcasting over the full input pytree.
- Several common normalizations should be registered and easily validated from named strings. Specifically, `zscore` and `minmax`. The zscore should normalize via (x-mean)/std, and the minmax should normalize via (x-xmin)/(xmax-xmin) * (ymax - ymin) + ymin. The extra params (mean, std) or (xmax, xmin, ymin, ymax) will need to be configured for these.
- It should be possible to also specify a UnaryOperator (see tree.py), such as the composite function `sqrt-log` that will first take the log, then sqrt. Your implementation will need to handle norm methods that also take in additional arguments. For example, `log-minmax` should first apply `minmax` using its config, then apply log. Essentially, this means supported callables like minmax and zscore should integrate with the existing UnaryOperator machinery.
- Norm constants can be any ArrayLike compatible with the leaf arrays. For example, they may be scalars, or they may be arrays that broadcast with the leaf arrays. The norm methods should work as long as usual broadcasting rules work.
- Likely, norm constants will need to be computed in an offline stage and saved to file. Just like the ImplicitIterativeGalerkin class allows validating/resolving compression data from file, the normalizations should support loading/validating normalization data from file
- It should additionally be possible to specify overrides for norm constants/args online by passing in auxiliary data. This should be handled by the existing aux workflow, rather than by passing args through the `x` payloads.

## Tips for implementation

If the norm callables are specified using `CallableModel`, then all norm constants can be configured up front by passing in extra configs. For example: `{callable: zscore, mean: 1.1, std: 0.34 }`, where mean and std are precomputed normalization constants. This function can then be integrated with existing UnaryOperator machinery to support composite norms like `log-minmax`.

## Example
The example configuration below demonstrates a skeleton of how your norm implementation is roughly to operate:
```yaml
edge: !pd:path.to.MyCustomEdge
  source: input_node
  target: output_node
  name: my_model
  norm:
    forward:
      pre:
        tree_root:
          a_branch: 
            a_leaf: { callable: minmax, xmin: 0.1, xmax: 10.1, ymin: 0, ymax: 1 }
          b_branch:
            b_leaf: [ { another: leaf }, { callable: log10 } ] # handle arbitrary nested leaf nodes within mappings/lists
          c_leaf: { callable: log10-zscore, mean: 0.0, std: 0.3 } # handle composite norms
          d_leaf: saved_norm.h5  # load/validate a callable norm from a saved file
          e_leaf: { callable: !!python/name:custom_norm, custom_args: [...] }  # can use any norm we want by just changing the callable
      post:
        output_leaf: log10 # simple string validation to log10 norm
    backward:
      pre:  ...  # (optional) by default, this should undo forward-post
      post: ...  # (optional) by default, this should undo forward-pre
```

Note how norms are just callables (possibly with args via CallableModel), and specified on a per-leaf basis. We only want to provide special treatment for registred string callables (minmax and zscore), as well as composite strings for supported unary operators. Finally, we also want to support loading/validating from file, when we need to save for example large arrays as norm constants. The save file can be h5 with metadata that indicates what norm function to use.

Also note that specifying both pre- and post- in the forward direction, should carry default inverses for the backward pre- and post- methods (for supported/registered) functions. However, if we cannot infer the inverses, then the user must supply the backward pre- and post- methods manually. Vice versa if the user only supplies backward but not forward. Try to infer the inverses where possible, but let the user override/configure all norms.

Below is an example of how the user may be able to completely override the default pytree norm implementation:

```python

class CustomEdge(Edge):

    def _forward_pre_norm(self, x: PyTree):
        ... # This overrides the parent completely
    
    def _backward_post_norm(self, x: PyTree):
        ...
```

Then when `CustomEdge.forward` is called, the order of execution will be the overriden `_forward_pre_norm` method, then `forward`, then the default forward post norm.

## Testing and done-ness criteria

Please additionally implement new unit tests for any new functionality. At a minimum, you should ensure:

- existing Edge APIs are unaffected
- pydantic validation works for configuring normalization
- norms can be loaded from yaml config
- norm constants can be validated/loaded from file
- registed/supported zscore and minmax work as expected (and are invertible)
- composite unary operators work (and their inverses)
- inferring the inverse works when possible, otherwise a validation error is raised if the user does not supply one
- complete child class norm override instance methods work as expected

Iterate until all new unit tests pass and no regressions are introduced.