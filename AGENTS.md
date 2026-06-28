# AGENTS.md

## Project goal

`romjax` implements a `jax`-based framework for learning reduced-order models (ROMs) using graph commutativity objectives. Nodes represent input/output vector spaces and edges represent models between them.

Success means demonstrating improvement over traditional ROM methods on representative PDE problems and releasing `romjax` as a reproducible Python library alongside the journal article.

## Your role

Implement requested features while preserving clean public APIs, reproducibility, and `jax` compatibility.

## Software design philosophy and guidelines

Since our primary goal is comparison to state-of-the-art and publication, there are a few guiding principles that all new code must abide by:

1. **Modularity** - prefer reusable abstractions and small composable pieces. Extend the framework by implementing focused subclasses such as `Edge` variants rather than introducing one-off infrastructure.
2. **Reproducibility** - keep workflows deterministic, configurable, and easy to rerun. Use `pydantic` validation for user-facing configuration to catch misuse early.
3. **Readability** - optimize for readers coming from the paper. Public APIs should be clearly typed, documented, and organized so the intent of each class or function is easy to follow.
4. **Ergonomic** - keep the library easy to use from ordinary `jax` workflows. Prefer YAML-friendly, serializable configuration and avoid turning `romjax` into a rigid framework.
5. **Jax-friendly** - numerical paths should remain pure-function oriented and compatible with `jit`, `vmap`, and `grad`. Design APIs around single-sample PyTree inputs and transparent data flow.

## Coding rules

Use YAML-friendly configuration for public, file-loaded objects. Prefer `romjax.typing.DictModel` for user-facing config schemas and preserve the `YAML -> Pydantic model -> runtime object` workflow. YAML configs may reference callables with `!!python/name` and classes with `!romx:module.path`.

* Use modern Python typing (prefer `dict` over `Dict`, `type1 | type2` over `Union[type1, type2]`, `type MyType = ...`, etc.)
* Annotate all public APIs
* Prefer "wider" interfaces where possible, for example `Sequence[int]` instead of `list[int]`
* Use Pydantic for user-facing config models
* Use **Sphinx-style docstrings** with typed parameters and returns; include math or implementation notes only when they materially help.
* Prefer `import jax.numpy as jnp`.
* Avoid Python side effects, mutation inside jitted functions, and non-JAX-compatible libraries in differentiable code.
* Run targeted unit tests with `uv run pytest ...` for specific small file changes. Run all tests with `uv run rr test` only for significant repo changes. Read the failures and fix them before continuing.
* Prefer an object-oriented approach and keep private helper methods consolidated within the classes that use them. Only move private methods outside of classes when they are reused and common to multiple classes.

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

The project is managed by `uv`. Assume the environment is already set up and use `uv run` for project commands.

The `romjax` package is located at `src/romjax` and is structured as a flat directory of modules. Each module is named for the functionality that it provides. The `tests` directory mirrors the module directory and provides unit tests for each module.

The public API provided by `romjax` is generally structured as the following:

- `graph.py` - defines `FunctionGraph`, the main graph-based ROM interface.
- `model.py` - defines core `Edge` implementations such as `ImplicitModel` and `FilterModel`.
- `poisson.py` - provides `Poisson2D`, the main PDE example and configuration reference.
- `train.py` - provides training utilities built around `jax.grad` and `optax`.
- `data_gen.py` - provides data generation utilities
- `plotting.py` - provides plotting helpers for ROM results.

Other modules provide various utilities for solving PDEs, handling random numbers, custom types, and other various package utilities.

## Agent development workflow

The general development workflow is:

```
uv sync --all-groups                # install only needed once, use --extra cu13 if gpu is available
uv run rr lint                      # runs ruff check on src and tests directories
uv run rr test                      # runs pytest on tests directory with coverage
```

Success means both lint and test pass without issues. You usually do not need to run the initial `uv sync` step during local development. You only need to run full test coverage `uv run rr test` for significant repo changes. For small targeted changes, run lightweight unit tests via `uv run pytest ...` instead.

## Rules

### Do
- Preserve existing architecture
- Prefer small, composable functions
- Prefer pydantic validation once and for all upon object init rather than at runtime when methods are called
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
- Use token-efficient options to avoid overly-verbose command outputs

### Don't
- Introduce unnecessary dependencies
- Modify files outside of the current request/task scope
- Break existing tests or APIs without fixing them
- Break `jax` compatibility
- Show large code diffs in prompt outputs or CLI responses
- Create an excessive number of standalone private functions if they are only used once
- Run the full test suite unless it is absolutely necessary. Prefer small focused tests.
