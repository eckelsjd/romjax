# feat-train-cli

- Type: `feat`
- Branch: `feat-train-cli`
- Worktree: `/home/eckelsjd/Documents/project-romjax/feat-train-cli`
- Status: `open`
- Created: `2026-05-04T19:12:57Z`

## Objective
Add a minimal `rom_cli train` workflow that mirrors `rom_cli generate`: load one YAML-backed `TrainConfig`, copy that config into the run root, construct a reconstruction loss from `FunctionGraph` paths, build a simple dataset iterator from generated H5 samples, and call `optim.train` with validated optimizer and runtime options.

## Tasks
- [ ] Complete `TrainConfig` in `src/romjax/config.py` with YAML-friendly Pydantic submodels for: graph loading, dataset root/split selection, reconstruction-loss terms, optimizer selection, dataloader options, and `optim.train` runtime/save/log settings.
- [ ] Replace the current raw `loss_fn: Callable` idea with a minimal loss-spec registry. Start with one built-in reconstruction loss that accepts user-specified graph paths plus optional term weights and resolves to a callable using `FunctionGraph.reconstruction_error` or an equivalent batched helper.
- [ ] Extend the loss_fn configuration so that reducing over training mini-batches is reusable outside of the reconstruction loss. The reduction over mini-batches should also be configurable, but default to a simple mean over samples. Reduction over mini-batches should be efficient and jax-compatible. Prefer a jax loop over individual samples rather than vmap for evaluating on mini-batches.
- [ ] Add minimal optax configuration through validated presets instead of arbitrary callables. Support a small initial set such as `adam`, `adamw`, and `sgd`, with explicit scalar fields for common arguments and a narrow `extra_kwargs` escape hatch only if required. 
- [ ] Extend optax configuration to support custom gradient transforms via the `optax.chain` interface. For example, it should be possible to configure an optax optimizer from yaml that includes a learning rate scheduler. Consider a similar shape as how `Poisson2D` flexibly loads `optimistix` solvers with help from the `IterativeSolver` type in `typing.py`.
- [ ] Implement a train-side data loader helper in `src/romjax/rom_cli.py` that reads generated `input.h5` samples from the existing `train/` and `validation/` directory layout, stacks batches into pytrees, and yields batch payloads for the configured reconstruction-loss terms. The dataloader should be configurable to support heterogeneous data, for example also loading `output.h5` and `residual.h5` along with `input.h5`.
- [ ] Define the minimal mapping between generated datasets and reconstruction paths. Prefer one config term per loss path with an explicit dataset edge directory, so training can reuse the existing `generate` output structure without inventing a new storage format.
- [ ] Add a `train` subcommand to `src/romjax/rom_cli.py` that matches the `generate` flow: validate the config path, load via `YamlLoader`, copy the config into the run root, build optimizer/loss/dataloaders, and invoke `optim.train`.
- [ ] Migrate the train logging functionality to focus on updating a progress bar, similar to `rom_cli generate`. However, keep existing logging support for file-based logging alongside the progress bar.
- [ ] Export any new public config models from `src/romjax/__init__.py` if they are intended to be instantiated from YAML tags.
- [ ] Add focused tests covering config validation, loss-spec resolution, dataset loading from generated sample directories, and CLI dispatch for `rom_cli train`.

## Constraints
- [ ] Keep the implementation minimal and aligned with the current filesystem workflow; do not introduce a second training data format or a broad experiment framework.
- [ ] Prefer Pydantic models and small registries over arbitrary YAML-loaded third-party callables; validation should catch unsupported optimizer or loss settings early.
- [ ] Preserve JAX-friendly execution: resolved loss callables should operate on pytrees and remain compatible with `jit`/`grad` through `optim.train`.
- [ ] Limit the first pass to reconstruction loss over user-specified graph paths; defer more general graph objectives, metric composition, or callback systems.
- [ ] Reuse `optim.train` where possible, but you may consider any changes or optimizations to support the train CLI workflow, which takes precedence over the existing `optim.train` function.

## Definition of Done
`rom_cli train <config>` is planned around a concrete `TrainConfig` that can be loaded from YAML, validated with Pydantic, pointed at an existing generated dataset root, and resolved into a reconstruction-loss callable, an optax optimizer, and dataloaders suitable for `optim.train`. The plan is complete when the required modules, tests, and boundaries are explicit enough for a later implementation pass to proceed without re-deciding the config shape.

## Relevant Files
- [ ] `src/romjax/config.py` for `TrainConfig` and loss/optimizer config models.
- [ ] `src/romjax/rom_cli.py` for the new `train` subcommand and generated-data loader helpers.
- [ ] `src/romjax/optim.py` for the runtime arguments that `TrainConfig` should expose cleanly.
- [ ] `src/romjax/graph.py` for path-based reconstruction loss behavior.
- [ ] `src/romjax/__init__.py` for any YAML-tag-visible public config exports.
- [ ] `tests/test_rom_cli.py` for CLI and generated-data-loading coverage.
- [ ] `tests/test_optim.py` or a new `tests/test_config.py` for config and loss-resolution tests.

## Key challenges and sharp points
The main sharp point is configuration of third-party and callable-heavy concepts without making YAML unsafe or brittle. `optax` and loss functions should be represented as small validated specs that resolve internally, not as arbitrary user-provided callables. But it should be flexible enough to allow user customization of gradient transforms. The second sharp point is dataset selection: generated data is organized per sampled edge, while reconstruction loss is organized per graph path, so the config should make that mapping explicit instead of guessing from nodes. In general, it should be easy to configure and adjust how dataloading works.

## More context (optional)
`TrainConfig` already exists as a stub in `src/romjax/config.py`, and the current `generate` command already establishes the desired workflow pattern: a YAML-loaded config object, a root directory for artifacts, and a fixed on-disk sample layout. The cleanest next step is to extend that same pattern for training rather than introducing a separate experiment runner. A representative minimal YAML shape should include: a run root, a graph, a generated dataset root, one or more reconstruction terms with `dataset_edge` + `path`, a simple optimizer preset, dataloader batch/shuffle settings, and the subset of `optim.train` controls needed for logging, saving, and termination.
