# romtools
[![Python version](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/eckelsjd/copier-numpy)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
![Code Coverage](https://img.shields.io/badge/coverage-25%25-red?logo=codecov)

[//]: # (You will need to register your Github repo with trusted publishing on PyPI if desired: https://docs.pypi.org/trusted-publishers/)

Tools for reduced-order modeling.

Currently just really basic data loading/processing and plotting with support for:

- 2d finite-volume quadrilateral mesh
- Tecplot ASCII data files
- 1d and 2d plots with slicing and animation
- Surface integrals on boundary

Some ideas for the future:

- More general abstract data loading/parsing/saving, can support Tecplot but also other common formats like vtk, pmd, yt, etc.
- Support for 1d and 3d mesh data
- Support for numerical results other than finite-volume (or even other than PDEs)
- Volume and line integrals
- Actual ROM implementations (lol)

## ⚙️ Installation
```shell
git clone https://github.com/eckelsjd/romtools.git
pip install -e romtools
```

## 🏗️ Contributing
See the [contribution](https://github.com/eckelsjd/romtools/blob/main/CONTRIBUTING.md) guidelines.

<sup><sub>Made with the [copier-numpy](https://github.com/eckelsjd/copier-numpy.git) template.</sub></sup>
