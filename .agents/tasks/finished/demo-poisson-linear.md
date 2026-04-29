# demo-poisson-linear

- Type: `demo`
- Branch: `demo-poisson-linear`
- Worktree: `/home/eckelsjd/Documents/project-romjax/demo-poisson-linear`
- Status: `open`
- Created: `2026-04-27T19:30:18Z`

## Objective
Plan and implement a reproducible Poisson demo that loads `demo/poisson_graph.yml`, learns two linear projection
operators at the same time, and minimizes reconstruction error across the reduced graph. The two trained projections
should correspond to the existing `FilterModel` edges named `coordinate transform` and `residual transform`, with a
two-stage workflow that first generates and persists stratified training data on disk, then trains with mini-batches
loaded from those saved files. The training loop should be able to pass different runtime modules to each edge while
staying compatible with `jax`, mini-batching, and the current YAML-driven graph workflow.

## Tasks
- [ ] Inspect `demo/poisson_graph.yml` and confirm the exact training path through the graph:
      `hf_coord -> lf_coord -> lf_res -> hf_res`, where the learned modules are consumed by
      `coordinate transform` and `residual transform` and the fixed PDE edge is `poisson`.
- [ ] Design a small demo-facing configuration model, preferably a `DictModel`, that keeps the script reproducible and
      ergonomic. Include at least:
      PRNG seed, latent dimensions for both projections, train/validation sample counts, batch size, number of steps or
      epochs, optimizer settings, logging cadence, output directory layout, reuse/overwrite policy, and
      dataset-generation options for stratified Poisson input/output sampling.
- [ ] Decide whether the demo should keep configuration inline in Python, load a small YAML companion file, or support
      both via a config object plus a thin CLI entry point. Favor a shape that can later become a reusable example for
      `romjax` training scripts.
- [ ] Split the workflow into two explicit stages:
      1. data generation and persistence, and
      2. training from saved samples.
      Keep the boundary clean enough that rerunning the script can skip stage 1 when compatible saved data already
      exists.
- [ ] Add a disk-backed dataset-generation pipeline for the Poisson graph using `romjax.rng.gen_keys` to create
      reproducible sample keys and paths. At the top level, use `gen_keys` to generate the input-sample directories and
      associated random keys for high-fidelity input sampling.
- [ ] For each generated input sample, solve the Poisson problem once, persist the input and solution pytrees with
      `save_h5`, then run `gen_keys` again inside that input sample directory to create multiple output-sample seeds and
      paths tied to that fixed solution. Use those nested keys to generate sampled outputs and any derived residual data
      needed for training, saving each pytree with `save_h5`.
- [ ] Define the on-disk layout clearly so the sampling hierarchy is visible and reusable, for example by split and
      sample id:
      split directory -> input sample directory -> output sample files / metadata.
      Make sure the saved structure is sufficient to reconstruct deterministic training/validation datasets without
      retaining everything in memory.
- [ ] Reuse existing seeds and saved samples when the same script is run again with compatible configuration. The plan
      should include a simple policy for detecting when data generation can be skipped versus when it should fail fast
      or regenerate data.
- [ ] Keep data generation memory-aware. Do not accumulate all full-resolution solutions, sampled outputs, or residuals
      in RAM before saving; generate, save, and release sample data incrementally.
- [ ] Define the actual training objective clearly before implementation. The minimum loss should measure reconstruction
      error for both learned projections, for example:
      solution reconstruction through `coordinate transform` forward/backward, and
      residual reconstruction through `residual transform` forward/backward.
      If helpful, include optional weighted terms for latent consistency or graph-path consistency, but keep the first
      demo centered on reconstruction.
- [ ] Plan the parameterization of the two learned modules explicitly. The simplest target is two independent
      `LinearProjection` instances, one for sampled outputs and one for sampled residuals, with dimensions inferred from
      the flattened Poisson state size and configurable latent widths.
- [ ] Extend `FunctionGraph.push_path(...)` so callers can provide edge-specific runtime payload additions instead of one
      shared `call_args` value for the entire path. The likely requirement is a new optional argument that maps edge
      names to runtime payload patches or runtime call inputs, applied only when that edge is evaluated.
- [ ] Keep the new graph API narrow and readable. Prefer a contract such as edge-scoped runtime inputs keyed by edge
      name over a general-purpose mutation hook. Document how this interacts with existing payload propagation and aux
      caching.
- [ ] Verify how the new path API should behave with `CompositeEdge`. The `galerkin rom` edge currently expands to
      `[coordinate transform, poisson, residual transform]`, so edge-specific runtime inputs must survive recursive
      composite traversal without leaking to unrelated edges.
- [ ] Review whether `FilterModel` itself needs a small API clarification now that this is the first case with more
      than one learned `FilterModel` in the same graph path. In particular, document the distinction between:
      per-spec runtime inputs inside one `FilterModel`, and
      per-edge runtime inputs across multiple `FilterModel` edges in `FunctionGraph.push_path(...)`.
- [ ] Implement the demo entry point under `demo/` so it can run data generation first and training second, reusing the
      existing `demo/poisson_graph.py` exploration as a starting point but turning it into a full workflow with:
      graph loading,
      data-generation commands or flags,
      module initialization,
      batch loading from disk,
      batched loss evaluation,
      optimizer setup,
      train/validation metrics, and
      a concise final summary of reconstruction scores.
- [ ] Build the training mini-batch loader around `load_h5` so batches are assembled from saved sample files rather
      than from one in-memory dataset object. The loader should iterate over persisted output samples, gather the small
      set of pytrees needed for one batch, and release them after each step.
- [ ] Make batching explicit and memory-aware. The script should avoid repeatedly materializing large flattened arrays
      when a batch-local flattening step is sufficient, and it should not require holding all transformed latent states,
      aux caches, or saved dataset contents in memory at once.
- [ ] Decide whether to use `romjax.optim.train(...)` directly or add a tiny helper around it for epoch-based batching
      and validation reporting. If the existing helper is reused, keep the demo loader interface simple and JIT-safe.
- [ ] If the script supports both generation and training in one command, make sure a repeated run can reuse the same
      persisted seeds/samples and proceed directly to training unless the user explicitly requests regeneration.
- [ ] Add targeted regression tests for any reusable API changes:
      graph tests covering edge-specific runtime inputs through `push_path`, including `CompositeEdge`,
      model tests covering the interaction between graph-level edge routing and `FilterModel` per-spec `call_args`, and
      any focused helper tests for the new disk-backed sampling layout or seed-reuse logic if that logic is factored out
      into reusable functions.
- [ ] Run targeted verification with `uv run pytest tests/test_graph.py tests/test_model.py` and any new focused demo
      helper tests. Run `uv run rr lint` once the implementation is in place.

## Constraints
- [ ] Keep the scope centered on a developer-facing demo and the smallest reusable graph API needed to support it; do
      not turn this task into a full training framework refactor.
- [ ] Preserve the existing YAML-driven graph definition in `demo/poisson_graph.yml`; prefer runtime-injected modules
      over hard-coding learned projections into the YAML itself.
- [ ] Maintain `jax` compatibility throughout the numerical path. The loss should be expressible as a pure function of
      parameters and batch data, suitable for `jit`, `grad`, and mini-batch evaluation.
- [ ] Favor deterministic, configurable data generation and optimization settings so runs are easy to repeat and later
      compare in the paper workflow.
- [ ] Use `romjax.rng.gen_keys` for both the top-level input sampling and the nested per-input output sampling so the
      persisted dataset structure is deterministic and easy to resume.
- [ ] Use `save_h5` and `load_h5` for persisted pytrees rather than introducing a parallel serialization path.
- [ ] Keep memory overhead in mind: generate and save samples incrementally, batch the training loop from disk, avoid
      unnecessary duplication of full-resolution solution and residual trees, and only cache auxiliary data at the
      batch granularity needed for inverse reconstruction.
- [ ] Treat the graph API change conservatively. The new `push_path` feature should solve edge-specific runtime inputs
      cleanly without weakening existing semantics for plain payload propagation or aux reuse.
- [ ] Preserve readability for paper-oriented users. The final demo should make it obvious what is fixed graph
      structure, what is learned runtime module state, what data is generated offline and persisted on disk, and what is
      recomputed or loaded per batch.

## Definition of Done
The task is complete when a plan has been implemented into a demo that:

- generates a deterministic, stratified dataset on disk by sampling inputs first, then sampling multiple outputs per
  input using nested `gen_keys` calls and `save_h5` persistence
- loads `demo/poisson_graph.yml` and trains two independent linear projection modules for the `coordinate transform`
  and `residual transform` edges
- loads training mini-batches from saved files with `load_h5` instead of requiring full-dataset transformed states or
  graph caches in memory
- can rerun against an existing compatible dataset and reuse the same seeds/samples without regenerating data by
  default
- exposes reproducible configuration for data generation and optimization with a clear path toward future reusable
  training scripts
- includes the minimum necessary `FunctionGraph.push_path(...)` API extension for edge-specific runtime inputs, with
  tests covering both direct paths and composite-path behavior
- documents or tests the first multi-`FilterModel` use case clearly enough that later demos can reuse the same pattern
- passes targeted pytest coverage for the changed graph/model behavior and any added helpers

## Relevant Files
- [ ] [demo/poisson_graph.yml](/home/eckelsjd/Documents/project-romjax/romjax/demo/poisson_graph.yml):
      source graph definition whose `coordinate transform`, `residual transform`, and `galerkin rom` edges define the
      training topology
- [ ] [demo/poisson_graph.py](/home/eckelsjd/Documents/project-romjax/romjax/demo/poisson_graph.py):
      current exploratory script to replace or evolve into the new reproducible data-generation + training demo
- [ ] [src/romjax/rng.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/rng.py):
      use `gen_keys` to define reproducible top-level input sampling and nested per-input output sampling paths
- [ ] [src/romjax/graph.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/graph.py):
      add the `push_path(...)` runtime-input extension and any composite-edge plumbing/docstring updates
- [ ] [src/romjax/model.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/model.py):
      review whether `FilterModel` docs or small helper behavior need clarification for graph-level multi-edge runtime
      inputs versus per-spec `call_args`
- [ ] [src/romjax/optim.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/optim.py):
      existing mini-batch training utility to reuse if it fits the demo loop cleanly
- [ ] [src/romjax/nn.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/nn.py):
      expected home of `LinearProjection`, which should parameterize both learned transforms
- [ ] [src/romjax/poisson.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/poisson.py):
      Poisson sampling, solve, and residual evaluation behavior that drives dataset generation
- [ ] Search for the existing `load_h5` / `save_h5` utilities in `src/romjax/` and reuse them for persisted pytree
      storage rather than adding a new serialization helper
- [ ] [tests/test_graph.py](/home/eckelsjd/Documents/project-romjax/romjax/tests/test_graph.py):
      primary regression coverage for new `push_path(...)` runtime-input semantics and composite-edge propagation
- [ ] [tests/test_model.py](/home/eckelsjd/Documents/project-romjax/romjax/tests/test_model.py):
      relevant if the implementation needs explicit coverage for the interaction between graph-level edge routing and
      `FilterModel` runtime `call_args`

## More context
The key implementation question is not how to train one `FilterModel`; that path already exists. The new problem is
that one graph traversal now needs two different learned runtime modules, each consumed by a different `FilterModel`
edge in the same path. `FilterModel` already knows how to normalize per-spec `call_args` inside a single edge, but
`FunctionGraph.push_path(...)` currently forwards one payload through the whole path and has no first-class way to add
different runtime inputs at different edges.

Because of that, the cleanest likely API change is at the graph level rather than by overloading `FilterModel` again.
The later implementation should keep the distinction sharp:

- graph-level edge runtime inputs choose which module or runtime payload each edge receives
- `FilterModel.call_args` still controls how one edge distributes runtime inputs across its internal filter specs
- graph aux caches remain responsible only for inverse-pass state, not for smuggling runtime modules across edges

For the demo itself, prioritize an ergonomic and repeatable workflow over maximal sophistication. The requested
sampling pattern is explicitly stratified: first sample and persist inputs, then for each fixed input/solution pair
sample and persist multiple output perturbations. That layout should make the dataset inspectable on disk, make reruns
cheap when seeds already exist, and let the training stage stream batches from files instead of rebuilding or retaining
the full dataset in memory.

If this demo works well, it can become the template for later ROM training scripts that combine YAML graph
definitions, nested RNG-managed sample directories, runtime-injected learnable modules, and memory-conscious mini-batch
optimization over persisted datasets.
