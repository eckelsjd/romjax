## Your overall goal
Design a `FilterModel` class API to enable flexible pytree-pytree mapping for use as an `Edge` in a `FunctionGraph`, similar to how `ImplicitModel` is used for PDEs. The `FilterModel` will act as a graph edge to map between two PDE `ImplicitModel` edges, and will generally work using the equinox library for neural networks to form black-box, data-driven functions between vector spaces.

## High-level idea for FilterModel
The FilterModel class should implement a flexible pytree->pytree map that is fully configurable from yaml with pydantic. It should be useful across a wide-variety of use cases without needing to subclass or edit the source code. The intended usage is to map between two spaces that are defined by more concrete models, such as PDEs like Poisson2D that have very rigid/meaningful input/output specs. While the concrete models are highly-specific in form and structure, the FilterModel generally only cares about mapping array-like data to array-like data. The primary target is to use the equinox neural network library to actually perform the array->array computations (i.e. linear maps, convolutions, etc.). The main challenge with the FilterModel then, is to flexibly collect input arrays from one concrete model, do some computations, and flexibly reformat/rearrange the output arrays into the specific form required by another concrete model.

Since the FilterModel is an instance of an Edge in a FunctionGraph, it needs to map in both the forward and backward directions, i.e. map arrays from one concrete space to another, and back. For example, this could be as simple as transposing a matrix, or it may be more complicated like an encoder/decoder structure using convolutional neural networks in the equinox library. In all cases, this behavior should be configurable and transparent.

Since equinox neural networks are the primary target for computation, it may be desirable to have special structure in the FilterModel input/output pytrees to support passing in eqx.Module objects, which serve as the primary interface to equinox neural networks. Note that these Modules must be passed through the forward/backward functions to support auto-differentiation of their parameters: eqx.Modules *are* pytreees themselves, so they are just another part of the inputs to the FilterModel.

## End-to-end example of desired behavior
Let's say the input+output space to PDE #1 can be specified as a pytree:
```
{
    "inputs": {
        "conductivity": {
            "const": jnp.array([...]),  # 2d input field
            "alpha": 1.0                # scalar param
        },
        "boundary": [
            ({"type": "dirichlet", "value": 0.0}, {"type": "dirichlet", "value": 0.0}),
            ({"type": "neumann", "value": 0.0}, {"type": "neumann", "value": 0.0}),
        ]
    },
    "outputs": {
        "phi": jnp.array([...])         # 2d field potential
    }
}
```
Let's then say the corresponding input+output space of PDE #2 can be specifed as:
```
{
    "inputs" {
        "x_inputs": jnp.array(...),     # just a function of inputs
        "x_outputs": jnp.array(...),    # just a function of outputs
        "x_shared": jnp.array(...),     # function of both inputs/outputs
    },
    "outputs" {
        "phi_red": jnp.array(...)       # a lower dimensional scalar potential field
    }
}
```
In this case, PDE #2 can be thought of as a "reduced" model of PDE #1, and we are trying to learn how to map information from PDE #1 to PDE #2 (and vice versa). When mapping from #1 to #2, we might want to support a few cases:
```
pytree["inputs"]["x_inputs"] = f(pytree["inputs"])
pytree["inputs"]["x_outputs"] = f(pytree["outputs"])
pytree["inputs"]["x_shared"] = f(pytree["inputs"] + pytree["outputs"])
pytree["outputs"]["phi_red"] = f(pytree["outputs"]["phi"])
```
In this example, the functions "f" above may be computed by equinox Modules, but they may also be user-specified in case the user has some specific structure they want to implement. The benefit with equinox modules, is that we can take filtered input pytrees, and perform common operations such as flattening/reshaping before passing to a neural network.

The FilterModel should transparently and flexibly handle all these cases (and others that are similar in spirit), so that we can map some "filtered" version of pytree #1 in several ways to "patches" of the output for pytree #2. The sum total result of this filter->compute->merge pattern is that we can map between arbitrary concrete pytree structures in a highly configurable way.

In this example, I only show mapping from PDE #1 -> PDE #2, but keep in mind that we also need to do the inverse, i.e. map from PDE #2 -> PDE #1 in a similar way. You have some flexibility in how to implement this, so long as you maintain a clean and interpretable interface with yaml/pydantic configuration, and as long as the FilterModel maintains consistency with being an Edge in a FunctionGraph.

## Some likely challenges to be solved
Once some array data gets computed in the forward direction by a neural network, it needs to get placed in the output pytree in a place that the consumer of this new pytree understands (i.e. PDE #2) For example, you may take `pytree["inputs"] -> pytree["inputs"]["x_inputs"]` in the example above. This is likely very doable using the eqx.filter pipeline we have already built. However, when you go in the backward direction, the information in `pytree["inputs"]["x_inputs"]` may need to get split back into the various sources from the original pytree (i.e. the various subtrees in `pytree["inputs"]` in PDE #1). So it seems clear to me how to map "many subtrees" -> "one subtree", but it is less clear how to go from "one subtree" -> "many subtrees". You will likely need to come up with a new, clean API that enables this functionality to be specified from yaml in the same way that we have enabled "input filter specs".

Relatedly, eqx.Module objects may be shared in both the forward and backward directions, for example a linear map and its transpose. But it is still possible that the forward/backward directions may require different parameters or eqx.Modules (i.e. an encoder/decoder -- although this may be shared too depending on implementation).

Your API design should account for these challenges, and you are not necessarily confined to keep the current structure. For example, maybe it would be better to completely split the configurations for the forward and backward directions, specifying a list of "input filters" for each separately. But keep in mind the challenge that some parameters may be shared in both directions of the FilterModel, in much the same way that a PDE model uses one set of parameters for both directions.

## An example use case in optimization
Let's say we have a FunctionGraph with two PDE edges (using the ImplicitModel class for both) and two FilterModel edges representing the connections between the two PDE models. Each edge has a set of parameters that we want to optimize:
```
params = {
    'pde_edge_1': {
        'inputs': { ... }
    },
    'pde_edge_2': { ... }
    'filter_edge_1': {
        'filters': [ eqx.Module, eqx.Module, ...]
    },
    'filter_edge_2': {
        'filters': [...]
    }
}
```
In total, the "params" object is a PyTree that we can pass to `optax` to optimize, and the FunctionGraph itself is used in the loss function to take the parameters and some inputs and compute a mean-squared error through the graph, such as:
```
def loss_fn(params, inputs, targets):
    loss = 0.0
    for edge, pytree in params.items():
        pred = graph[edge](pytree, inputs)
        loss += jnp.sum((pred - targets)**2)
```
Note how we use eqx.Module objects as the maps for the FilterModel. 

## Current progress on implementation
You can see the current progress on this API in `model.py`. Right now, the idea is for the `FilterModel` to have a list of "filter specs" that tell it how to filter incoming inputs from a pytree and how to compute the forward direction for each filtered input. The input filtering works okay, but as mentioned before, there are some difficulties with how to properly do the merging, and how to go in the backward direction. Your job is to finish this implementation according to the descriptions in this document, including a complete rewrite of FilterModel if you believe it be necessary.

## Your tasks and instructions
You are free to complete these tasks however you wish, including rewriting the existing API for the FilterModel in `model.py`. But please provide thorough documentation on your thinking and all public APIs, and please describe in detail how you are solving the problems that come up and how they align with the overall goals explained in this document. You can create and modify as many new files as you need, including for testing and documentation purposes.

- [ ] Design and implement a `FilterModel` API according to the specifications and use-cases in this document.
- [ ] Come up with and implement a suite of tests to make sure the basic functionality works. This should include simple unit tests demonstrating the basic usage, as well as integration tests for how well this works with toy PDE models inside a FunctionGraph. Your tests should also include loading and using configs from yaml with pydantic validation, and demonstrate round-trip loading/serializing consistency.
- [ ] Implement a simple filter model that performs linear projection using a matrix for the forward reduction step, and its transpose for the backward reconstruction step. Verify that optimizing this matrix on a simple dataset gives comparable results to the proper orthogonal decomposition
- [ ] Implement a simple convolutional autoencoder in equinox and demonstrate how it can be integrated directly into a FunctionGraph and optimized alongside parameters for other edges in the graph

