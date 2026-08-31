# romjax
[![Python version](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=cccccc)](https://www.python.org/downloads/)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/eckelsjd/copier-numpy)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
![Code Coverage](https://img.shields.io/badge/coverage-82%25-yellowgreen?logo=codecov)

Reduced-order modeling in jax. Compatible with the [jax ecosystem](https://docs.kidger.site/equinox/) for ML (equinox, optax, lineax, etc.)

## ⚙️ Installation
```shell
git clone https://github.com/eckelsjd/romjax.git && cd romjax
uv sync --all-groups  # use --extra cuda13 for gpu support
```

## 📍 Quickstart
Based on an [approach]() that abstracts the ROM training process as one of minimizing the path error in a graph.
```python
import jax.numpy as jnp
import optax

import romjax as romx

class FOM(romx.ExplicitModel):
    theta: float = 1.0  # true parameter
    def pushforward(self, inputs):
        y = self.theta * inputs["x"]**2
        return dict(y=y)
  
class ROM(romx.ExplicitModel):
    def pushforward(self, inputs):
      y = inputs["theta"] * inputs["x"]**2
      return dict(y=y)

# Graph-theoretic approach
graph = romx.FunctionGraph(
    edges=[
        FOM(source="A", target="B"),                 #  A-->B
        romx.IdentityEdge(source="A", target="C"),   #  |   |
        romx.IdentityEdge(source="B", target="D"),   #  ⌄   ⌄
        ROM(source="C", target="D"),                 #  C-->D
    ]
)

# Residual minimization (i.e. least-squares || y - theta x^2 ||^2)
loss = romx.GraphLoss([dict(
    callable="path_error", path_a=["A->B", "B->D"], path_b=["A->C", "C->D"]
)])

# Training data
def dataloader():
    xtrain = jnp.linspace(0, 1, 15)
    ytrain = 1.0 * xtrain ** 2
    data = [
        dict(inputs={"x": x}, outputs={"y": y}) for x, y in zip(xtrain, ytrain)
    ]
    while True:
        yield data

params = romx.Train(
    graph=graph,
    loss=loss,
    dataloader=dataloader(),
    optimizer=optax.adam(0.001),
    init_params={"C->D": {"theta": jnp.asarray([0.0])}}
)()

print(params["C->D"]["theta"])  # 1.0
```

Obviously, this is a lot of extra work for plain least-squares, but the same setup can be used for a variety of more complicated problems with minimal additional configuration:

- **Solution error** - optimize through a differentiable PDE solver
- **Galerkin projection** - learn reduced-basis ODEs (or data-driven alternatives such as DMD)
- **Deep learning** - train autoencoders, MLPs, and everything in between (with [equinox](https://docs.kidger.site/equinox/))
- **Closure modeling** - tune physics-based closure model coefficients
- **Residual learning** - exploit residual geometry with [non-converged solutions]()

All of these diverse learning problems share the same fundamental graph structure as the simple least-squares problem.

## 🏗️ Contributing
See the [contribution](https://github.com/eckelsjd/romjax/blob/main/CONTRIBUTING.md) guidelines.

<sup><sub>Made with the [copier-numpy](https://github.com/eckelsjd/copier-numpy.git) template.</sub></sup>

## 📎 Citation
TBA