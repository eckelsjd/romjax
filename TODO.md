Keeping track of ideas, bugs, thoughts, etc.

**Short term**
- Optimize single run (multi-threaded/jit/caching/overhead, poisson, etc.)
  - gpu seems slower
  - cpu is only going up to 100%, sometimes more -- not much multi-threading going on in the jit
- normalization before/after training


## Critical
- [ ] Normalization for eqx transforms

## Needs testing
- [ ] Training on param references

## Blind spots
- [ ] Support raw arrays and lists/tuples in HDF5 loader
- [ ] Resolving references in GraphLoss may be flaky -- what if some eqx.Module has string params that aren't meant to be references?
- [ ] Might be necessary (and nice) to allow jax dtype structs for filter model templates during reconstruction

## Backburner
- [ ] Go back and make sure graph loss functions actually give the same result as normal regresion (linear, mlp)
- [ ] Other data save formats outside of H5

## Ergonomics
- [ ] !include pydantic tag for loading from another .yml file
- [ ] Might be neat for the !include mechanism to work like python import, so we don't have to keep worrying about relative paths
- [ ] More consistent handling of exceptions in the cli. For example, when to show messages, expected vs unexpected failures, etc.
- [ ] Would be good to copy the entire resolved yaml config to routine root dirs, (after resolving all the overrides)

## Serialization
- [ ] from_registry items back to string
- [ ] !include back to string paths
- [ ] Ignoring large arrays and default items
- [ ] Better abstract param templates for saving/loading with orbax. Would be nice for these templates to serialize/yaml with jax Dtype structs, and to be automatically handled

## Routines
- [ ] ~~Routine dependencies and snakemake/makefile~~

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

## Optimizations
- [ ] Prefetching for dataloader
- [ ] Caching data globally across training runs (train data and graph/models to avoid repeated pydantic validation)
- [ ] Stacked h5 datasets (many samples per file)
- [ ] Something with pre-compiling (globally) jax loss functions to avoid jit cache misses
- [ ] Seems to be some recompiling when using cached data versus disk data -- keep an eye on this

## Training
- [ ] EvolutionSearch or similar instead of grid search (still dispatch in parallel)
- [ ] Multi-node dispatch (MPI executor) -- likely we will never have a single train run that is bigger than a single machine, so we can always do embarrassingly parallel dispatch, and we don't have to worry about sharding
- [ ] Get this all to work on Mac OS

## Future projects
- Building surrogate for graph comm as a model error indicator (scalar or field)
- Online adaptive time-stepping for time-based simulations