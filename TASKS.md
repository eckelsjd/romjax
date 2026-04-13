## Objective
Clean up the eqx_evaluate and FilterModel pipeline to support handling of auxiliary data gathered by the forward function, and passed to the backward function later on. This may require a rework of the larger Edge API as well to support auxiliary data in the forward and backward directions.

Your goal is also to plan and enable smooth caching and passing of information around a FunctionGraph, including proper handling of this new auxiliary data API.

## Tasks
- Rework the FilterModel and eqx_evaluate API to support automatic handling of auxiliary data needed by the backward function from the forward function and vice versa. Specifically for eqx_evaluate, the backward function needs the pytree "template" argument, which can be gathered during the forward evaluation by evaluating the passed inputs and inferring their array shapes.
- Think about and generalize this idea of auxiliary data passing so that graph Edges besides just FilterModel can make use of and use auxiliary data.
- Adjust the larger Edge API if needed to support the passing of auxiliary data alongside the main forward/backward pytrees. You may need to adjust the ImplicitModel and ExplicitModel classes to support this as well.
- Adjust the Poisson2D class to be compatible with this new auxiliary data passing API. While the Poisson2D class does not currently use auxiliary data, it may be desirable to pass information regarding solver statistics for example.
- Rewrite the tests in test_model.py to support the new API.
- Specifically for the test_simple_eqx_filter_model test in test_model.py, rewrite the FilterModel to use as many default options as possible and no extra callables. For example, use "stack" collection and reconstruction, and make sure the "template" auxiliary data is automatically collected and passed by the new API.
- Make sure that the automatic templates gathered and used by FilterModel/eqx_evaluate support scalars and 1d arrays as well, as well as nested pytrees.
- Rework the LinearProjection class so that it is initialized by a jax key and the size of the matrix (much like the autoencoder is initialized)
- Plan and document how auxiliary data will be handled and cached when composing multiple edges around a FunctionGraph.
- Implement a method in FunctionGraph to cleanly and transparently push inputs starting at one node along a specified path of edges to a target destination node, i.e. function composition. 

## Constraints
- Everything must remain jax/jit friendly and compatible, i.e. only pure functions with no side-effects
- You are free to adjust the FunctionGraph and Edge APIs to support auxiliary data passing as long as everything is documented and kept clean and interpretable.
- It should be possible to easily pass inputs from one node to another along the edges of the FunctionGraph without ever thinking about or manually handling the auxiliary data. That is, the auxiliary data should be transparently handled by your API design (and optional if the user does not need it).
- The configuration of the FunctionGraph should remain simple, concise, and easy to use, without excessive new options and configurations to worry about for every edge case. The defaults should be sufficient for most use cases.

## Definition of Done
You are done when your unit tests pass and accurately capture all of the new behavior. Please also run all other tests with `uv run pytest` and fix all side issues that come up from your new implementation, to ensure consistency with the existing API.

## Beware of the sharp bits
The general idea with this new auxiliary data API is that the forward and backward functions of a graph Edge may "learn" something at runtime about the data being passed through it that the opposite function might need to know in order to do its job correctly. For example, the FilterModel.backward pass with eqx_evaluate needs to know the pytree template of the inputs gathered during the forward pass. We could configure this template statically up front if we know what the data looks like ahead of time, but our design philosophy is that we want to be as flexible as possible with the incoming data, and instead infer this auxiliary information at runtime. Another use case example might be if a PDE solver fails at runtime, we can capture this information and handle it downstream rather than crashing. Since we are working in a pure-function jax environment, we need to compute and return this extra information at runtime and somehow make it available to the user in a transparent and clean manner.

Here is where an issue arises that your design needs to handle. The FunctionGraph will compose and transport PyTrees around the graph by passing them through the Edge forward/backward functions. However, if we are attaching auxiliary data somehow to the inputs/outputs of the edge functions, the information from one edge may accidentally get propagated to an Edge that does not know how to handle it. Instead, the FunctionGraph and Edges should work with a clean and consistent API/structure so that everyone knows where to put their auxiliary information and how to retrieve the information relevant to their forward/backward functions.

When coming up with and implementing your design, be aware of and make sure you handle a couple cases listed here:
- We may inititiate the traversal of a path, compute auxiliary data as we go, and then use that auxiliary data where needed during runtime (e.g. going forward and then backward with eqx_evaluate to a latent space)
- We may traverse a path, compute auxiliary data as we go, and not end up using any of it in that specific traversal, instead returning the aux data to the caller
- We may start traversing a path, but then try to take a route (e.g. backward direction) that requires aux data that we haven't computed or passed in. This should raise an error. 
- We may traverse a path that we know ahead of time requires auxiliary data. The user can pass this in to the inputs before evaluating the path (e.g. the aux data may be known ahead of time or precomputed from a previous traversal). This should work fine just as if the aux data had been computed by the traversal itself.

In all these cases, it should not be the Edge's job to filter or handle extra auxiliary data. It should be the FunctionGraph's job when traversing a path to make sure the right auxiliary data is collected and passed to the right Edge and nothing more.