# tokamaker-jax

[![CI](https://github.com/rogeriojorge/tokamaker_jax/actions/workflows/ci.yml/badge.svg)](https://github.com/rogeriojorge/tokamaker_jax/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/tokamaker-jax/badge/?version=latest)](https://tokamaker-jax.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/rogeriojorge/tokamaker_jax/branch/main/graph/badge.svg)](https://codecov.io/gh/rogeriojorge/tokamaker_jax)
[![Python](https://img.shields.io/pypi/pyversions/tokamaker-jax.svg)](https://pypi.org/project/tokamaker-jax/)
[![License: LGPL-3.0](https://img.shields.io/github/license/rogeriojorge/tokamaker_jax.svg)](LICENSE)

`tokamaker-jax` is a JAX-native porting project for TokaMaker, the OpenFUSIONToolkit Grad-Shafranov equilibrium tool. The target is an end-to-end differentiable, accelerator-ready solver with a friendlier Python API, TOML-driven CLI, GUI-first workflow, high-quality examples, plotting, docs, and at least 95% test coverage.

This repository currently contains the project scaffold, CI/docs infrastructure, source audit, porting plan/log, and a small differentiable fixed-boundary Grad-Shafranov seed solver. The complete feature parity port is tracked in [plan.md](plan.md).

![Fixed-boundary seed equilibrium](docs/_static/fixed_boundary_seed.png)

![Axisymmetric manufactured Grad-Shafranov convergence](docs/_static/manufactured_grad_shafranov_convergence.png)

![Pressure sweep animation](docs/_static/pressure_sweep.gif)

## Install

```bash
git clone https://github.com/rogeriojorge/tokamaker_jax.git
cd tokamaker_jax
pip install -e ".[dev,gui,docs]"
```

Python 3.10 and newer are supported. TOML files use `tomllib` on Python 3.11+ and `tomli` on Python 3.10.

## Use

Launch the GUI:

```bash
tokamaker-jax
```

Run from a TOML file:

```bash
tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png
```

Use from Python:

```python
from tokamaker_jax import load_config, solve_from_config

config = load_config("examples/fixed_boundary.toml")
solution = solve_from_config(config)
print(solution.stats())
```

Run manufactured validation gates:

```bash
tokamaker-jax verify --gate grad-shafranov --subdivisions 4 8 16
```

## Current Scope

- JAX differentiable fixed-boundary seed solver for the Grad-Shafranov operator on a rectangular grid.
- p=1 triangular FEM reference kernels, dense/sparse/matrix-free assembly, weighted axisymmetric Grad-Shafranov weak-form assembly, profile source loads, and manufactured convergence gates.
- TOML configuration loader with Python 3.10 compatibility.
- CLI that launches the GUI by default and runs TOML files when supplied.
- Matplotlib plotting utilities.
- Sphinx and Read the Docs setup.
- GitHub Actions for linting, testing with coverage, docs, and release publishing.

## Porting Target

The full port will cover TokaMaker's unstructured finite-element mesh workflow, fixed/free-boundary equilibria, coil and passive conductor modeling, profile functions, reconstruction constraints, wall modes, time-dependent stepping, EQDSK/i-file IO, and publication-quality plotting. See [plan.md](plan.md) for the full breakdown, validation matrix, and implementation log.

## Attribution

This project is derived from planning and source review of [OpenFUSIONToolkit/OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit), especially its TokaMaker component. Cite Hansen et al., *Computer Physics Communications* 298, 109111 (2024), DOI: [10.1016/j.cpc.2024.109111](https://doi.org/10.1016/j.cpc.2024.109111), when using TokaMaker-derived work.
