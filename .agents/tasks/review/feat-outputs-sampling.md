# feat-outputs-sampling

- Type: `feat`
- Branch: `feat-outputs-sampling`
- Worktree: `/home/eckelsjd/Documents/project-romjax/feat-outputs-sampling`
- Status: `open`
- Created: `2026-04-23T18:12:19Z`

## Objective
Add a configurable output-sampling feature for `Sampleable` models, with the first concrete implementation centered on
`Poisson2D.sample_outputs`. The API should let users provide a custom sampler, but `Poisson2D` should also ship with an
easy default sampler that conditions on `inputs`, optionally reuses a precomputed `solution`, and generates plausible
`phi` samples near the solved state by adding configurable perturbations such as Gaussian noise, uniform noise, or a
smooth random field.

## Tasks
- [ ] Design the public sampling interface for outputs so it remains consistent with the existing
      `outputs_sampler`/`outputs_sampler_opts` pattern and clearly documents the conditioning contract
      `sampler(key, inputs=..., solution=..., **opts) -> outputs`.
- [ ] Implement one built-in `Poisson2D` output sampler entry point that can be selected from YAML by name, for example
      `"near_solution"`, instead of requiring every user to pass a custom callable.
- [ ] Define a YAML-friendly configuration model for the default sampler options, including the perturbation mode,
      amplitude/scale parameters, and any shape or truncation settings needed for field-based noise.
- [ ] Update `Poisson2D` validation so built-in output sampler names are resolved to callables and built-in sampler opts
      are validated with `pydantic` rather than passed through as untyped dictionaries.
- [ ] Implement default `Poisson2D.sample_outputs` behavior that:
      1. uses the provided `solution` when available,
      2. otherwise computes `solve(inputs)` when the default sampler needs a reference solution,
      3. passes `inputs` through to the sampler so custom samplers can condition on them,
      4. returns a `{"phi": ...}` pytree matching the existing output structure.
- [ ] Implement a near-solution sampler that supports at least:
      1. additive Gaussian white noise,
      2. additive uniform white noise,
      3. additive smooth random-field noise for spatially correlated perturbations (keep in mind the existing kle sampler)
- [ ] Keep the near-solution definition flexible by allowing either direct absolute scales or relative scales derived
      from the solution magnitude, while keeping defaults simple enough for demo use.
- [ ] Add targeted unit tests covering sampler-name resolution, option validation, solution reuse vs internal solve,
      output shape preservation, deterministic sampling for a fixed key, and custom callable support.
- [ ] Add a Poisson-focused demo script that samples several `phi` fields for one input, plots the forcing/input,
      reference solution, and sampled outputs side by side, and serves as the canonical example for the new feature.
- [ ] Run targeted checks with `uv run pytest tests/test_poisson.py tests/test_loader.py` and `uv run rr lint`.

## Constraints
- [ ] Preserve the current `Sampleable.sample_outputs(key, inputs=None, solution=None)` contract; do not introduce a
      separate incompatible sampling path for `Poisson2D`.
- [ ] Keep the feature `jax`-friendly: sampling utilities should operate on pytrees/arrays, avoid hidden global state,
      and remain compatible with `jit`-safe numerical code where practical.
- [ ] Follow the existing YAML -> Pydantic -> runtime callable workflow for user-facing configuration. New built-in
      sampler options should be serializable and validated.
- [ ] Do not break existing behavior where users may already provide `outputs_sampler` as a direct callable.
- [ ] Avoid over-generalizing into a framework-wide sampler hierarchy unless the implementation clearly pays for itself.
      A focused `Poisson2D` implementation with reusable helper functions is preferable.
- [ ] Reuse an externally provided `solution` whenever possible so repeated output sampling for the same inputs does not
      trigger repeated nonlinear solves.
- [ ] Keep the default sampler conservative: perturbations should preserve output shape and should not silently change
      the meaning of other Poisson inputs or residual targets.
- [ ] Add tests and demo code only within task scope; do not refactor unrelated PDE, graph, or training code.

## Definition of Done
`Poisson2D` supports output sampling through either a user-provided callable or a built-in default sampler selectable
from YAML/config. Calling `sample_outputs(key, inputs=..., solution=...)` produces a valid `{"phi": ...}` sample with
documented near-solution behavior, reuses `solution` when supplied, and falls back to solving only when necessary.
Sampler options are validated, the behavior is covered by unit tests, and a demo script plots the forcing/input,
reference solution, and multiple sampled outputs for one Poisson problem. Targeted pytest coverage and `ruff` lint pass.

## Relevant Files
- [ ] `src/romjax/poisson.py` for the `Poisson2D` API, built-in sampler implementation, and config validation.
- [ ] `src/romjax/rng.py` if shared sampling helpers or distribution utilities are worth reusing for output noise.
- [ ] `src/romjax/model.py` for the `Sampleable` contract and any docstring/API clarifications.
- [ ] `src/romjax/__init__.py` if new public sampler helpers or config models should be exported.
- [ ] `tests/test_poisson.py` for behavior and regression tests around `sample_outputs`.
- [ ] `tests/test_loader.py` for YAML loader coverage if the built-in sampler is selectable by string name.
- [ ] `demo/` for a new Poisson output-sampling visualization script.

## Key challenges and sharp points
The main design tension is between flexibility and keeping the user-facing API small. `Poisson2D` already accepts
generic sampler callables, so the new default should extend that pattern rather than replace it. The built-in sampler
also needs a precise conditioning story: it should be explicit about when `inputs` are required, when `solution` can be
reused, and when `sample_outputs` is allowed to call `solve` internally.

Noise model choice matters. IID pixel noise is easy to implement but may look physically uninformative, while random
field perturbations are more useful for PDE experiments but require extra option validation and shape handling. The plan
should therefore treat Gaussian and uniform noise as the minimum supported perturbations, with smooth field noise added
through a small reusable helper rather than ad hoc demo-only code.

Another sharp edge is scale selection. Absolute noise magnitudes are simple but brittle across problem setups. Relative
scaling against the reference solution norm or amplitude is more ergonomic, but must be documented carefully so samples
remain interpretable and reproducible.

## More context (optional)
Today, `Poisson2D.sample_outputs` is only a thin wrapper around `self.outputs_sampler(...)` and returns an empty sample
when no sampler is configured. That means there is currently no out-of-the-box way to draw output perturbations near a
known solution, even though the method signature already anticipates conditioning on both `inputs` and a precomputed
`solution`.

This feature should make output-space sampling practical for demos and later learning experiments where near-manifold
data is useful. A good default would let users write concise YAML such as selecting `outputs_sampler: near_solution`
with small options for `mode: gaussian`, `scale: 0.05`, or a correlated random-field variant, while still allowing
expert users to inject a fully custom callable.

The demo should stay simple and visual: fix one Poisson input sample, solve once, draw several output perturbations,
and plot the forcing/input field, the solution, and the sampled `phi` fields together. That will validate both the API
and the usefulness of the default sampler before broader integration into ROM training workflows.
