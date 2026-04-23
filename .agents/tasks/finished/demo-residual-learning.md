# demo-residual-learning

- Type: `demo`
- Branch: `demo-residual-learning`
- Worktree: `/home/eckelsjd/Documents/project-romjax/demo-residual-learning`
- Status: `open`
- Created: `2026-04-23T14:00:01Z`

## Objective
Plan and implement a standalone demo script that compares two ways of learning a scalar residual surrogate
``f_hat(b, u) ~ f(b, u)`` for an implicit relation ``f(b, u) = w`` under a known prior ``b ~ N(0, 1)``:

1. solution-manifold learning, where training data only comes from points satisfying ``f(b, u) = 0``
2. near-manifold residual learning, where training data comes from samples ``u | b ~ N(u^*(b), sigma^2)``

The demo should make the difference visually obvious by plotting the true residual field and comparing contour/error
plots for both learned surrogates over the ``(b, u)`` plane.

## Tasks
- [ ] Choose a simple analytic scalar residual function with a closed-form solution manifold ``u^*(b)`` and
      nontrivial off-manifold behavior, for example ``f(b, u) = u - g(b)`` or a mildly nonlinear alternative such as
      ``f(b, u) = u - g(b) + alpha * u^3`` where ``u^*(b)`` remains easy to evaluate.
- [ ] Define the demo configuration in a small, reproducible way:
      fixed PRNG seed, prior ``b ~ N(0, 1)``, conditional width ``sigma``, train/validation grid extents, sample
      counts, model class/capacity, optimizer settings, and output figure path(s).
- [ ] Implement two dataset generators:
      one that samples only manifold points ``(b, u^*(b), w=0)``, and one that samples ``b`` from the prior then
      ``u`` from ``N(u^*(b), sigma^2)`` with targets ``w = f(b, u)``.
- [ ] Fit the same surrogate family for both regimes so the comparison is controlled.
      Keep the model lightweight and JAX-friendly, such as a small MLP from [src/romjax/nn.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/nn.py)
      or an Equinox module defined locally in the demo.
- [ ] Evaluate the true residual and both learned surrogates on a dense ``(b, u)`` mesh.
      Compute at least:
      global residual error over the full plotting window,
      near-manifold error inside ``|u - u^*(b)| <= c * sigma``, and
      solution-manifold consistency ``|f_hat(b, u^*(b))|``.
- [ ] Produce comparison figures using `matplotlib` and, where convenient, [src/romjax/plotting.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/plotting.py):
      one panel for the true ``f(b, u)``,
      one for the manifold-only surrogate,
      one for the near-manifold surrogate,
      and one or more error panels such as ``|f_hat - f|`` or signed residual mismatch.
- [ ] Include a clear visualization of the sampled training data on top of the contour plots so the user can see the
      coverage difference between the two regimes.
- [ ] Add a short textual summary printed by the script that reports the chosen parameters and the comparison metrics,
      so the demo is useful even without manually inspecting every figure.
- [ ] If the implementation introduces reusable helper logic beyond the script body, place that logic in a small,
      focused module under `src/romjax/` and cover it with targeted unit tests in `tests/`.

## Constraints
- [ ] Keep the scope limited to a developer-facing demo in `demo/`; do not expand this task into a new public ROM API
      unless a helper is clearly reusable.
- [ ] Use a deterministic setup with fixed seeds and explicit configuration so the same figures can be regenerated.
- [ ] Keep the numerical example scalar and analytic; avoid PDE solves, large training loops, or dependencies beyond
      the current stack.
- [ ] Preserve JAX-friendly code paths for model evaluation and training, even if plotting remains ordinary matplotlib.
- [ ] Use the same architecture, optimizer, and training budget for both learning strategies so the comparison isolates
      the effect of data support rather than model capacity.
- [ ] Choose plotting ranges wide enough to reveal off-manifold extrapolation failure, but not so wide that contours are
      dominated by irrelevant tails of the prior.

## Definition of Done
The task is complete when a reproducible demo script exists under `demo/` that:

- trains one surrogate on manifold-only data and one surrogate on near-manifold residual data
- evaluates both surrogates against a known analytic residual ``f(b, u)``
- saves or displays contour-style comparisons over the ``(b, u)`` plane with training samples overlaid
- reports quantitative error metrics showing how performance differs on and away from the solution manifold
- includes any minimal supporting tests for reusable helpers and passes targeted `uv run pytest ...` coverage for those
  additions

## Relevant Files
- [ ] [demo/kle_sampling.py](/home/eckelsjd/Documents/project-romjax/romjax/demo/kle_sampling.py):
      reference style for a lightweight developer demo script; create the new demo alongside this file, e.g.
      `demo/residual_learning.py`
- [ ] [src/romjax/plotting.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/plotting.py):
      existing plotting helper that may be reused for subplot layout and contour/pcolor figures
- [ ] [src/romjax/optim.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/optim.py):
      check whether existing training utilities are lightweight enough for the demo; otherwise keep training local to
      the script
- [ ] [src/romjax/nn.py](/home/eckelsjd/Documents/project-romjax/romjax/src/romjax/nn.py):
      inspect for an existing small neural-network component before introducing a custom demo model
- [ ] [tests/test_plotting.py](/home/eckelsjd/Documents/project-romjax/romjax/tests/test_plotting.py):
      relevant if plotting helpers need small extensions for contour overlays or layout support
- [ ] [tests/test_optim.py](/home/eckelsjd/Documents/project-romjax/romjax/tests/test_optim.py):
      relevant if reusable optimization helpers are added or adapted

## More context
This demo should illustrate the core intuition behind residual learning for implicit models: data restricted to
``f(b, u) = 0`` is enough to identify the solution manifold, but it does not constrain how a learned surrogate behaves
away from that manifold. By contrast, sampling from ``u | b ~ N(u^*(b), sigma^2)`` gives the model local information
about the residual geometry near the solution set, which should improve contour fidelity and off-manifold error.

To keep the message crisp, prefer an example where the contour plot tells the story immediately:
the manifold-only model should look accurate along ``u = u^*(b)`` but poorly identified away from it, while the
near-manifold model should recover a visibly better approximation in a tube around that curve.

If the demo is successful, it can later be extended into a more general `romjax` example by:

- wrapping the scalar experiment in a YAML-friendly config model for reproducible sweeps over ``sigma`` and model size
- reusing the same training/evaluation pattern for higher-dimensional implicit residuals or ROM edges
- turning the script outputs into paper-quality figures that motivate why commutativity objectives benefit from
  residual information beyond the exact solution manifold
