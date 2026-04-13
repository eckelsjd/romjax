## Objective
Refine the eqx_evaluate function to be more modular and reusable within the FilterModel as a forward or backward function.

## Tasks
- [ ] Refine the eqx_evaluate function so that is reusable for all equinox-like evaluations. At a minimum, it should work as both the forward and backward function for FilterModel, with both LinearProjection and the CNN autoencoder.
- [ ] Update the tests in test_model.py so that the eqx_evaluate function is used for the LinearProjection and CNN autoencoder maps (both forward and backward)

## Constraints
- Keep the general structure of FilterModel (but you are allowed to make small changes if needed to support the new tasks)
- You may add as many changes or additional options as needed to make eqx_evaluate reusable as described in this document.

## Definition of Done
When all of your unit tests in `test_model.py` pass with `uv run pytest`, and when all new features are covered by the tests -- specifically the reuse of the eqx_evaluate function.

## Key files
- `model.py` - where FilterModel and eqx_evaluate live
- `test_model.py` - where the tests should go
- `graph.py` - where the FunctionGraph and Edge classes live

## Some challenges to overcome
FilterModel has the ability to route arbitrary pytree results in both the forward and backward directions. For eqx_evaluate to be reusable, it will likely need to take advantage of this. However, the output of eqx_evaluate in the forward direction will generally just be an array representing the latent space. It is not clear how the backward direction should take this latent array, reconstruct the full array, and then split and reroute the full array into all its original consituent pytree paths. This is a problem you will need to solve, and it may need to take a similar strategy as the forward_routes and backward_routes in FilterModel.

Note this also brings up the difficulty of different calling methods for different directions. For example, you need to call "encode" for the CNN in the forward direction, but "decode" in the backward direction. Maybe your solution allows this behavior to be configurable up front so that at runtime, the same eqx_evaluate function can be used regardless of direction.

## The ideal outcome
A lot of data-driven, black-box type modeling follows the same pattern for evaluation -- you collect all arrays into some sort of structure (flat, stacked, etc) and pass it through a neural network forward evaluation. Our FilterModel allows the configuration of the input filtering / array collection and the output routing, and it also allows specifying the forward/backward callables. The main reason for doing this is that we don't want to rewrite the callables every time -- we just want to specify how to collect/filter/route the arrays, and then we reuse a few common callables (i.e. eqx_evaluate). If we can't reuse the callables, then we might as well just implement the full Edge class for every individual use case, which we want to avoid.