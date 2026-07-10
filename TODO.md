Keeping track of ideas, bugs, thoughts, etc.

- finish objectives and poisson
- get Galerkin ODE working (small problem, then Vlasov)
- repeat galerkin training on Vlasov
- implement Euler (conservative values, handle heat flux and stress terms)
- Figure out coord/res transforms for Vlasov->Euler
- possibly CNN/FNO autoencoder 
- Train Vlasov/Euler graph
- Routine for sampling knudsen, training surrogate for E(v)

## Critical
- [ ] Need a better way to evaluate ODE residual, especially in light of small time scales.
- [ ] Getting Galerkin to work for diffrax ODE models -- solving ODE in latent space, building basis over snapshots

## Blind spots
- [ ] Resolving references in GraphLoss may be flaky -- what if some eqx.Module has string params that aren't meant to be references?
- [ ] How to handle config for controller/solver states for diffrax solver restarts -- probably downstream user will have some sort of loop and their own save format
- [ ] Limited to uniform time-grids for ODEs -- can't handle multiple time-scales. The best fix is likely to allow fields to carry their coordinates with them (e.g. the non-uniform time grid), then somehow encode this everywhere in the FunctionGraph, e.g. via neural operators. Hm. But the main issue is just when calling evaluate() -- fd gradients of a numerical solution are not good for sharp changes.

## Backburner
- [ ] Refactor all these loose private methods into a more structured OO design (especially GenNorm and GraphLoss)

## Ergonomics
- [ ] !include pydantic tag for loading from another .yml file
- [ ] Might be neat for the !include mechanism to work like python import, so we don't have to keep worrying about relative paths
- [ ] More consistent handling of exceptions in the cli. For example, when to show messages, expected vs unexpected failures, etc.
- [ ] Would be good to copy the entire resolved yaml config to routine root dirs, (after resolving all the overrides)
- [ ] All the annoying `resolve` methods you need to call on a graph after construction (norms, refs, compression, etc.)

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
- [ ] Composite functions for unary operators, normalizations, error/tree operators, etc.
- [ ] Pydantic workflow, !romx, !pd, !overrides, __parent__, etc..

## Profiling
- [ ] Make sure the expensive part of data generation is the model evaluation
- [ ] Check performance of vlasov

## Optimizations
- [ ] Caching data globally across training runs (train data and graph/models to avoid repeated pydantic validation)
- [ ] Stacked h5 datasets (many samples per file)
- [ ] Something with pre-compiling (globally) jax loss functions to avoid jit cache misses

## Training
- [ ] EvolutionSearch or similar instead of grid search (still dispatch in parallel)
- [ ] Multi-node dispatch (MPI executor) -- likely we will never have a single train run that is bigger than a single machine, so we can always do embarrassingly parallel dispatch, and we don't have to worry about sharding
- [ ] Get this all to work on Mac OS
- [ ] Early validation-based stopping criteria (and keeping best performing checkpoint)

## Assumptions / hidden behavior
- [ ] Graph loss terms get their dataset from the first implicit model in graph
- [ ] The name of the dataset edge should be the name of the generated data
- [ ] Commutativity error does not evaluate the dataset edge -- all data for both nodes should be present

## Future projects
- Building surrogate for graph comm as a model error indicator (scalar or field)
- Online adaptive time-stepping for time-based simulations
- Can reformulate loss scaling as a PID controller
- Public release:
    - Move pydantic/agents/models out
    - Clean tests, CI/CD, demos
    - Refactor to nice internal/public layout (modules, exports, defaults, param/var names)
    - Module/class/function docstrings throughout
    - Readme, quickstart, citations
    - Mkdocs website with tutorial, examples
    - Version, pip, github, bump