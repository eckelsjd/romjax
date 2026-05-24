# romjax
[![Python version](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/eckelsjd/copier-numpy)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
![Code Coverage](https://img.shields.io/badge/coverage-14%25-red?logo=codecov)

Reduced-order modeling in jax.

## ⚙️ Installation
```shell
git clone https://github.com/eckelsjd/romjax.git && cd romjax
uv sync --all-groups # use --extra cu13 for gpu-support
```

## Project roadmap

Right now, we have the basic `FunctionGraph` API up and running with the `Poisson2D` model as a good PDE test case. The following list summarizes where we are heading next.


- Testing dataloading and optimization of simple linear projection on some Poisson data.
- Implementing a Galerkin projection ROM ImplicitModel.
- Configuring a full Poisson<->Galerkin graph from yaml.
- Implementing various graph-theoretic objective functions (state reconstruction, solution error, etc.)
- Automate running multiple cases with different configurations (maybe scaling up hardware too)
- Automate hyperparameter optimization
- Post-processing scripts for comparing methods, plots, tables, etc.
- Automate everything from a repeatable snakemake workflow
- Then start looking at different modeling options (Vlasov, Conv2D, etc.)

When we are done, we will have all of these features implemented with a thorough regression test suite in place and passing. You will generally only be working on one of these tasks at a time with a more specific set of instructions and "doneness" criteria provided when ready.

## 🏗️ Contributing
See the [contribution](https://github.com/eckelsjd/romjax/blob/main/CONTRIBUTING.md) guidelines.

<sup><sub>Made with the [copier-numpy](https://github.com/eckelsjd/copier-numpy.git) template.</sub></sup>
