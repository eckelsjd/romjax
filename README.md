# romjax
[![Python version](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/eckelsjd/copier-numpy)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
![Code Coverage](https://img.shields.io/badge/coverage-82%25-yellowgreen?logo=codecov)

Reduced-order modeling in jax.

## ⚙️ Installation
```shell
git clone https://github.com/eckelsjd/romjax.git && cd romjax
uv sync --all-groups # use --extra cu13 for gpu-support
```

## 🔬 Profiling
The built-in harness uses `jax.profiler.trace(...)` and `StepTraceAnnotation` so you can inspect both Python
orchestration and JAX-compiled execution in TensorBoard.

Enable tracing with CLI flags:

```shell
uv run python -m romjax.romx_cli run \
  --profile \
  --profile-dir /tmp/romjax-traces \
  --profile-label train-debug \
  path/to/train.yml
```

The CLI sets `ROMJAX_PROFILE`, `ROMJAX_PROFILE_DIR`, and `ROMJAX_PROFILE_LABEL` for the launched routine so the
same flags also enable `GridSearch` child traces.

For a direct `Train` run, the trace is written under `--profile-dir` if provided, otherwise under
`<train_root>/profiles/`.

For `GridSearch`, the parent search is traced and each spawned training case gets its own trace directory under
`<case_root>/profiles/` automatically. If `--profile-dir` is set on the parent CLI, child traces are written under a
case-named subdirectory inside that root.

Open the results with TensorBoard:

```shell
tensorboard --logdir /tmp/romjax-traces
```

If you want a host-side flame graph for Python overhead, run the parent process under `py-spy`. For allocation and
copy-volume analysis, `Scalene` is the better follow-up tool.

## 🏗️ Contributing
See the [contribution](https://github.com/eckelsjd/romjax/blob/main/CONTRIBUTING.md) guidelines.

<sup><sub>Made with the [copier-numpy](https://github.com/eckelsjd/copier-numpy.git) template.</sub></sup>
