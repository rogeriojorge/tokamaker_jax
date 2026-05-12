# Contributing

This repository is a staged JAX-native port of TokaMaker. Contributions should keep the implementation differentiable, testable, and easy to reproduce.

## Development Loop

1. Install the package with development extras:

   ```bash
   pip install -e ".[dev,docs]"
   ```

2. Run formatting, linting, tests, and docs locally:

   ```bash
   ruff format .
   ruff check .
   pytest --cov=tokamaker_jax --cov-fail-under=95
   sphinx-build -W -b html docs docs/_build/html
   ```

## Porting Rules

- Preserve TokaMaker physics behavior before adding new features.
- Keep solver kernels JAX-transformable with `jit`, `grad`, `vmap`, and `scan` where practical.
- Add regression tests for every feature ported from OpenFUSIONToolkit.
- Prefer explicit typed configuration over hidden global state.
- Document user-facing APIs as they land.

