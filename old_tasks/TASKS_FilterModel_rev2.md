## Objective
Rewrite eqx_evaluate along with FilterModel so that we can cleanly map from arbitrary pytree -> array -> pytree using a FilterModel, where the array is typically a shared latent space computed by an equinox module. For example, we might use LinearProjection, or a ConvAutoencoder to map to/from this latent space. 

## Tasks
- Make eqx_evaluate and FilterModel work so that we can do pytree->array->pytree mappings
- Clean up eqx_evaluate to remove redundant input/output path filtering (assume this is handled by the external FilterModel)
- Clean up eqx_evaluate so it does not try to do batch/vmap operations
- Finish/rewrite test_simple_eqx_filter in test_model.py to learn a shared latent space for two fields using linear projection
- Reconfigure clean all additional tests in test_model.py to conform to the new API
- Make sure all tests in test_model.py pass with `uv run pytest test_model.py`

## Constraints
- The eqx_evaluate function should be reusable and configurable across various equinox modules
- It should be configurable how to collect and reshape pytree inputs before computing the latent space
- It should be configurable how to reconstruct and reshape pytree outputs from a latent space array

## Definition of Done
When all pytests pass and the tests accurately capture the pytree->array->pytree design.

## Design philosophy notes
In general, we may specify several of these latent-space mappings via FilterModel.filters. Each filter in FilterModel will first collect all inputs and send the filtered input pytree to the equinox modules. The equinox module will then handle either reducing via the forward map, or reconstructing via the backward map. What is currently missing is this backward map needs to optionally recreate the original pytree structure that was used when compressing, so that we can go in both directions seamlessly. Finally, after all latent spaces have been computed, the FilterModel evaluation will patch all the results together into the final output pytree using the forward/backward routes that we have already configured.
