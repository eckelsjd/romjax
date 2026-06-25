Keeping track of ideas, bugs, thoughts, etc.


## Critical
- [ ] Need a better way to evaluate ODE residual, especially in light of small time scales.
- [ ] Normalization for eqx transforms

## Needs testing
- [ ] Training on param references

## Blind spots
- [ ] Support raw arrays and lists/tuples in HDF5 loader
- [ ] Resolving references in GraphLoss may be flaky -- what if some eqx.Module has string params that aren't meant to be references?
- [ ] Might be necessary (and nice) to allow jax dtype structs for filter model templates during reconstruction
- [ ] How to handle config for controller/solver states for diffrax solver restarts -- probably downstream user will have some sort of loop and their own save format
- [ ] Limited to uniform time-grids for ODEs -- can't handle multiple time-scales. The best fix is likely to allow fields to carry their coordinates with them (e.g. the non-uniform time grid), then somehow encode this everywhere in the FunctionGraph, e.g. via neural operators. Hm. But the main issue is just when calling evaluate() -- fd gradients of a numerical solution are not good for sharp changes.

## Backburner
- [ ] Go back and make sure graph loss functions actually give the same result as normal regresion (linear, mlp)
- [ ] Other data save formats outside of H5
- [ ] Getting Galerkin to work for diffrax ODE models -- solving ODE in latent space, building basis over snapshots

## Ergonomics
- [ ] !include pydantic tag for loading from another .yml file
- [ ] Might be neat for the !include mechanism to work like python import, so we don't have to keep worrying about relative paths
- [ ] More consistent handling of exceptions in the cli. For example, when to show messages, expected vs unexpected failures, etc.
- [ ] Would be good to copy the entire resolved yaml config to routine root dirs, (after resolving all the overrides)
- [ ] Would be nice for !overrides to work inline, so you can just override a few values, rather than create a whole new file

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
- [ ] Move PDE/models to separate library (poisson, vlasov, euler)
- [ ] Reorganize module layout and optimize public API (with good defaults, nice validation) for documentation tutorials/demos
- [ ] Unit tests are getting pretty heavy -- breaking into multiple libraries and condensing may help

## Documentation
- [ ] Pretty much everything
- [ ] Some simple train and gen data examples would be good
- [ ] How to use filter model with eqx_evaluate for projection, neural nets, etc.

## Profiling
- [ ] Make sure the expensive part of data generation is the model evaluation
- [ ] Check performance of poisson and make sure it reasonable for the problem size (lookout for optx)
- [ ] Check performance of vlasov

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