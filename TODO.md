Keeping track of ideas, bugs, thoughts, etc.

## Critical
- [ ] Normalization for eqx transforms
- [ ] Latent space sampling (routine to save galerkin_sampler.yml from SVD bounds)

## Needs testing
- [ ] Reconstruction matching SVD
- [ ] Training on param references

## Blind spots
- [ ] Including solution data in mini-batches (with residual=0)
- [ ] Support raw arrays and lists/tuples in HDF5 loader
- [ ] Resolving references in GraphLoss may be flaky -- what if some eqx.Module has string params that aren't meant to be references?

## Backburner
- [ ] Keep an eye on orbax checkpoint reloading with more complicated param trees
- [ ] Go back and make sure graph loss functions actually give the same result as normal regresion (linear, mlp)
- [ ] Need to be more consistent with suppressing or showing logger/progress bars in library code via logger.disable("romjax") for example
- [ ] Other data save formats outside of H5

## Ergonomics
- [ ] Auto-fill linear projection full dof size
- [ ] !include pydantic tag for loading from another .yml file

## Serialization
- [ ] from_registry items back to string
- [ ] !include back to string paths
- [ ] Ignoring large arrays and default items

## Routines
- [ ] Postprocess routine for plots, tables, metrics etc. for journal
- [ ] Composite routines
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

## Profiling
- [ ] Make sure the expensive part of data generation is the model evaluation
- [ ] Make sure the expensive part of train is the optimizer update (be on the lookout for big recursive pytree operations)
- [ ] Check performance of poisson and make sure it reasonable for the problem size (lookout for optx)