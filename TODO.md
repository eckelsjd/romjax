Keeping track of ideas, bugs, thoughts, etc.

## Critical
- [ ] Normalization for eqx transforms

## Needs testing
- [ ] Training on param references

## Blind spots
- [ ] Support raw arrays and lists/tuples in HDF5 loader
- [ ] Resolving references in GraphLoss may be flaky -- what if some eqx.Module has string params that aren't meant to be references?
- [ ] Might be necessary (and nice) to allow jax dtype structs for filter model templates during reconstruction

## Backburner
- [ ] Keep an eye on orbax checkpoint reloading with more complicated param trees
- [ ] Go back and make sure graph loss functions actually give the same result as normal regresion (linear, mlp)
- [ ] Need to be more consistent with suppressing or showing logger/progress bars in library code via logger.disable("romjax") for example
- [ ] Other data save formats outside of H5

## Ergonomics
- [ ] !include pydantic tag for loading from another .yml file

## Serialization
- [ ] from_registry items back to string
- [ ] !include back to string paths
- [ ] Ignoring large arrays and default items
- [ ] Better abstract param templates for saving/loading with orbax. Would be nice for these templates to serialize/yaml with jax Dtype structs, and to be automatically handled

## Routines
- [ ] Automated batch routines (for hyperparameter optimization)
- [ ] Routine dependencies and snakemake/makefile

## Task CLI
- [ ] [Issues with codex sandbox shell commands](https://github.com/openai/codex/issues/17525)

## Major design changes
- [ ] Move heavy pydantic workflow to separate library (custom models, yaml loading, routines, cli, etc.)
- [ ] Move agents workflow to separate libary or copier-numpy template

## Documentation
- [ ] Pretty much everything
- [ ] Some simple train and gen data examples would be good
- [ ] How to use filter model with eqx_evaluate for projection, neural nets, etc.

## Profiling
- [ ] Make sure the expensive part of data generation is the model evaluation
- [ ] Make sure the expensive part of train is the optimizer update (be on the lookout for big recursive pytree operations) -- within this, make sure we are spending most time computing rather than filtering/flattening/assembling/caching/allocating/etc.
- [ ] Check performance of poisson and make sure it reasonable for the problem size (lookout for optx)