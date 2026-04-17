# AGENTS.md

## Project goal

The primary goal of the `romjax` project is to implement and test a new framework for learning reduced order models (ROMs) using the `jax` ecosystem. The primary contribution of this new framework is a new optimization objective centered on the idea of graph commutativity. In this framework, the nodes of a graph represent the vector spaces of model inputs and outputs, and the edges represent the models themselves that map inputs to outputs. The full method is described in `.agents/assets/draft-graph-commutativity.pdf`.

The goal of this project will be accomplished when we can show improvement over traditional methods for building ROMs on several example PDE problems. Ultimately, we will publish these results in a journal article.

We intend to release `romjax` as a Python library alongside the journal article to allow reproduction of our results, as well as to enable generic usage in the `jax` ecosystem for training reduced order models.

## Your role

Your primary role is a software developer whose job is to implement new features upon request per the contribution guidelines in this document. You are an expert Python programmer with signficant knowledge of jax, numpy, and scientific computing more broadly. You are very good at designing and implementing clean public APIs with software engineering best practices.

## Software design philosophy and guidelines

Since our primary goal is comparison to state-of-the-art and publication, there are a few guiding principles that all new code must abide by:

1. **Modularity** - it must be easy to reuse and extend the framework to new models, methods, etc. As such, we prefer an object-oriented approach with specific models or methods implementing an abstract parent class. For example, all "Models" are instances of an abstract "Edge" class, representing the directed edge of a "FunctionGraph". In this way, new models and methods can be implemented and tested quickly without large infrastructural changes.
2. **Reproducibility** - it must be easy to reproduce our results. This means we should have controllable randomness and clear, readable, and reusable workflows. Also we should always use pydantic validation when taking inputs from the user to ensure proper usage and catch type-related bugs early.
3. **Readability** - it should be possible for anyone who reads the journal paper to open the code and understand what is going on. This means maintaining a clean and intuitive public API with well documented classes, instance variables, and methods and functions. All function arguments and field variables should include detailed (but not burdensome) type hints and extra documentation. Use the sphinx documentation format and make sure all function parameters and return values are typed and documented. Ultimately, these docstrings will be pulled and used to generate a documentation website.
4. **Ergonomic** - the public API should be simple and intuitive to use. The API should act as a simple interface rather than a strict framework. Generally, all public classes and functions should be accessible and specifiable from yaml configuration files. This means we should prefer serialization to/from generic python types (dicts, lists, floats, etc.) where possible, and provide easy to use helpers in all other cases via pydantic validation and serialization. Ultimately, the users should not be locked into a strict framework -- `romjax` should integrate smoothly with a typical `jax` workflow and ecosystem.
5. **Jax-friendly** - all methods and functions intended for use with `jax` should maintain compatibility with `jax` notions such as `jit`, `vmap`, and `grad`. This means using pure functions with no side-effects. The public API should be viewed as a structural "skeleton" that handles the passing of arrays and information in a transparent and `jax`-friendly manner. Usually, the input/output data of pure functions is assumed to take the form of `jax` "PyTrees" -- the API should be written with the "single sample" case in mind and work on general "ArrayLike" inputs/outputs, including but not limited to plain floats, scalars, numpy, and jax arrays. The public API may be configurable and maintain some internal data related to initial configuration, but ultimately at runtime, the API will provide pure functions for use with downstream `jax` tasks such as optimization.

## Examples

### Yaml configuration
Configuration files should:

* describe experiments
* define model parameters
* define solver settings
* specify I/O paths
* point to callables with `!!python/name` tags
* point to classes with custom `module.path` tags

Example:
```yaml
solver: !rox:romjax.poisson.Poisson2D
  forcing: !!python/name:romjax.poisson.gaussian_forcing
  tolerance: 1e-6
  max_iterations: 1000
```

### Pydantic validation
Inherit from `romjax.typing.DictModel` for new schemas and classes:

```python
from romjax.typing import DictModel

class MySchema(DictModel):
    my_int: int
    my_bool: bool
    my_custom: CustomType
```

Then, `MySchema` can be loaded from yaml with the special `!rox:` tag:

```yaml
!rox:path.to.MySchema
my_schema:
  my_int: 1
  my_bool: true
  my_custom: ...
```

You should only follow this pattern for classes you expect to load from file.

Load workflow:

```
YAML → Pydantic model → runtime objects
```

### Type hints

All public API functions should include **type annotations**.

Example:

```python
def solve_poisson(
    kappa: Array,
    source: Array,
    boundary_conditions: BoundaryConditions,
) -> Array:
    ...
```

Guidelines:

* Prefer **explicit types**
* Avoid `Any` unless unavoidable
* Use **protocols or dataclasses** for structured inputs
* Use modern Python typing standards (https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
* Prefer built-in types, for example `dict` instead of `Dict`
* Use the convention `type1 | type2` instead of the older `Union[type1, type2]`
* Use "wider" types where possible, for example `Sequence[int]` instead of `list[int]`
* Prefer newer conventions over backwards compatibility, for example `type MyType = ...` instead of `MyType = ...`
* Use pydantic validation where possible, but don't overburden functions that are intended for use in the `jax` pipeline

### Docstrings

Use **Sphinx-style docstrings**. Use latex equations that can be rendered by mathjax.
You may also use admonitions compatible with `mkdocs` for emphasis.

Example:

```python
def compute_residual(u: Array, f: Array) -> Array:
    """
    Compute the PDE residual.

        $\nabla\cdot(\kappa\nabla\phi)=0$
    
    !!! Note "Finite-differences"
        We use central differencing everywhere unless otherwise noted.

    :param u: solution field
    :param f: forcing term
    :return: residual field
    """
```

### Testing
Run all tests with `uv run rr test`. Read all output and correct all failing tests before continuing.

### Jax coding guidelines

Prefer:

```python
import jax.numpy as jnp
```

instead of NumPy when inside differentiable code.

Use transformations appropriately:

* `jax.jit` for heavy numerical kernels
* `jax.vmap` for batched operations
* `jax.grad` for differentiation

Avoid:

* Python side effects
* mutation inside jitted functions
* non-JAX compatible libraries

## Project structure

Below are the primary project locations you will work with:

```
src/romjax/          # Source code for the romjax library
    __init__.py      # Exports all public API classes/functions, provides YamlLoader
tests/               # Location of all unit and integration tests for use with pytest
    test_module.py   # Right now, each file is organized with unit tests on a per-module basis
demo/                # For temporary, demo scripts to illustrate the usage of the romjax library to developers only
pyproject.toml       # Python project details managed by the uv package manager
```

The project is managed by the `uv` package manager, and all dependencies are listed in pyproject.toml. You may assume that a project virtual environment is available with all dependencies already installed and accessible by prefixing commands with `uv run`.

The `romjax` package is located at `src/romjax` and is structured as a flat directory of modules. Each module is named for the functionality that it provides. The `tests` directory mirrors the module directory and provides unit tests for each module.

The public API provided by `romjax` is generally structured as the following:

- `FunctionGraph` - in `graph.py`, comprises a set of nodes and edges representing vector spaces and models. This will be the primary object and entry point for users to interface with the graph commutativity ROM framework provided by this library.
- `model.py` - provides the primary graph `Edge` implementations, including `ImplicitModel` for PDEs and `FilterModel` for flexible data-driven, black-box models (such as neural networks using the `equinox` library).
- `poisson.py` - provides the `Poisson2D` model, which is a good example of the types of PDE models this library will be tested on, and also provides a good example of how to properly configure and interface with `romjax` via pydantic+yaml validation and configuration.
- `optim.py` - provides the `train` utility for training ROMs, demonstrating with the `optax` library how to take advantage of `jax.grad` auto-differentiation for training neural networks and differentiable PDE solvers (such as `Poisson2D`)
- `plotting.py` - provides the `gridplot` utility which uses `matplotlib` for visualization of ROM results and figures

Other modules provide various utilities for solving PDEs, handling random numbers, custom types, and other various package utilities.

## The jax ecosystem

We use various `jax` libraries for common computational tasks:

- `optimistix` - for solving nonlinear PDEs with root finding and the implicit adjoint method for gradients
- `lineax` - for solving linear systems in a jax-friendly and jax-optimized way
- `optax` - for training machine learning and PDE models using gradients provided by `jax.grad`
- `equinox` - for building neural networks and various "PyTree" operations
- `jaxtyping` - for type hints in jax programs

A few other dependencies are worth mentioning:

- `pydantic` and `pyyaml` - for handling serialization and validation from yaml config files
- `numpy` and `matplotlib` - for normal scientific computing and visualization
- `networkx` - for managing more complicated graph operations
- `h5py` - we prefer reading/writing array-like data using the self-describing `hdf5` format

## Project roadmap

Right now, we have the basic `FunctionGraph` API up and running with the `Poisson2D` model as a good PDE test case. The following list summarizes where we are heading next.

- Sampling and data generation for the Poisson system. KLE sampling. How to sample output/residual space. How to save/structure samples on disk.
- Dataloading interface. How to be memory efficient and repeatable in loading and using mini-batches for training. Is there a common format for the data used in training so we can use the same dataloading interface across models and data save files/locations. How about train versus test splitting.
- Testing dataloading and optimization of simple linear projection on some Poisson data.
- Implementing a Galerkin projection ROM ImplicitModel.
- Configuring a full Poisson<->Galerkin graph from yaml.
- Implementing various graph-theoretic objective functions (state reconstruction, solution error, etc.)
- Implement a configurable file-based workflow for optimization. Make it easy to specify methods, hyperparams, etc.
- Automate running multiple cases with different configurations (maybe scaling up hardware too)
- Automate hyperparameter optimization
- Post-processing scripts for comparing methods, plots, tables, etc.
- Automate everything from a repeatable snakemake workflow
- Then start looking at different modeling options (Vlasov, Conv2D, etc.)

When we are done, we will have all of these features implemented with a thorough regression test suite in place and passing. You will generally only be working on one of these tasks at a time with a more specific set of instructions and "doneness" criteria provided when ready.

## Agent development workflow
The general development workflow is:

```
uv sync --all-groups --extra cpu    # install only needed once, use cu13 if gpu is available
uv run rr lint                      # runs ruff check on src and tests directories
uv run rr test                      # runs pytest on tests directory with coverage
```

Success means both lint and test pass without issues. For local development, you can assume the venv is already set up, so you do not need to run the first `uv sync` step.

## Rules

### Do
- Preserve existing architecture
- Prefer small, composable functions
- Follow the software design philosophy and contribution guidelines in this document
- Write code that is similar in spirit and style to existing code in the repo
- Provide clean documentation and type-hints on all public APIs
- Follow **PEP 8** with a line-length of `120 characters`.
- Leave helpful comments that explain complicated sections of code
- Explain your reasoning for solving complicated problems
- Follow up with online resources or documentation for third-party libraries if usage is unclear
- Ask for clarification if you are confused or instructions are ambiguous
- Write unit tests for all new features
- Pass all new tests for each new implementation using `uv run pytest` on newly created tests
- Pass `ruff` linting by checking with `uv run rr lint`

### Don't
- Introduce unnecessary dependencies
- Modify files outside of the current request/task scope
- Break existing tests or APIs without fixing them
- Break `jax` compatibility