# tokamaker-jax

[![CI](https://github.com/rogeriojorge/tokamaker_jax/actions/workflows/ci.yml/badge.svg)](https://github.com/rogeriojorge/tokamaker_jax/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/tokamaker-jax/badge/?version=latest)](https://tokamaker-jax.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/rogeriojorge/tokamaker_jax/branch/main/graph/badge.svg)](https://codecov.io/gh/rogeriojorge/tokamaker_jax)
[![Python](https://img.shields.io/pypi/pyversions/tokamaker-jax.svg)](https://pypi.org/project/tokamaker-jax/)
[![License: LGPL-3.0](https://img.shields.io/github/license/rogeriojorge/tokamaker_jax.svg)](LICENSE)

`tokamaker-jax` is a JAX-native porting project for TokaMaker, the OpenFUSIONToolkit Grad-Shafranov equilibrium tool. The target is an end-to-end differentiable, accelerator-ready solver with a friendlier Python API, TOML-driven CLI, GUI-first workflow, high-quality examples, plotting, docs, and at least 95% test coverage.

This repository currently contains the staged JAX port infrastructure, p=1 triangular FEM kernels, validation gates, GUI workflow, generated documentation assets, source audit, and porting plan/log. The complete feature parity port is tracked in [plan.md](plan.md).

## Quick Start

```bash
git clone https://github.com/rogeriojorge/tokamaker_jax.git
cd tokamaker_jax
pip install -e .
```

The default install includes GUI dependencies. Python 3.10 and newer are supported.
After the package is published on PyPI, the installed-wheel path is:

```bash
pip install tokamaker-jax
tokamaker-jax init-example fixed-boundary --output fixed_boundary.toml
tokamaker-jax fixed_boundary.toml --plot outputs/fixed_boundary.png
```

Launch the GUI:

```bash
tokamaker-jax
```

Use an explicit host/port when needed:

```bash
tokamaker-jax gui --host 127.0.0.1 --port 8081 --no-browser
```

Run a TOML case and write a plot:

```bash
tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png
```

The `examples/...` paths are available in a repository checkout. For a PyPI-only
install, create the packaged example first with `tokamaker-jax init-example`.

Run the main validation suite:

```bash
tokamaker-jax verify --gate all --subdivisions 4 8 16
```

List runnable examples and planned upstream parity fixtures:

```bash
tokamaker-jax cases --runnable-only
tokamaker-jax cases --json --output outputs/case_manifest.json
```

Summarize exact upstream OpenFUSIONToolkit/TokaMaker mesh and geometry files:

```bash
tokamaker-jax upstream-fixtures
tokamaker-jax fixed-boundary-evidence
```

Use from Python:

```python
from tokamaker_jax import load_config, solve_from_config, write_example

case_path = write_example("fixed-boundary", "fixed_boundary.toml", force=True)
config = load_config(case_path)
solution = solve_from_config(config)
print(solution.stats())
```

For development and docs work:

```bash
pip install -e ".[dev,docs]"
```

TOML files use `tomllib` on Python 3.11+ and `tomli` on Python 3.10.

Release and PyPI publishing steps are documented in
[docs/release.md](docs/release.md).

Static MHD solver explorer for GitHub Pages:
[tokamaker_jax_explorer.html](https://rogeriojorge.github.io/tokamaker_jax/_static/tokamaker_jax_explorer.html)

## Examples

Generate benchmark and literature-reproduction artifacts:

```bash
python examples/benchmark_report.py --output outputs/benchmark_report.json
python examples/reproduce_cpc_seed_family.py outputs/literature/cpc_seed_family
```

Run individual physics gates:

```bash
tokamaker-jax verify --gate grad-shafranov --subdivisions 4 8 16
tokamaker-jax verify --gate circular-loop
tokamaker-jax verify --gate oft-parity
tokamaker-jax verify --gate free-boundary-profile
```

## Visual Overview

![tokamaker-jax MHD browser explorer](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/tokamaker_jax_explorer_screenshot.png)

![GUI workflow dashboard](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/gui_workflow_dashboard.png)

![Publication validation panel](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/publication_validation_panel.png)

![Fixed-boundary seed equilibrium](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/fixed_boundary_seed.png)

![Validation dashboard](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/validation_dashboard.png)

![Case manifest status](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/case_manifest_status.png)

![Upstream mesh fixture inventory](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/upstream_fixture_mesh_sizes.png)

![Upstream fixed-boundary gEQDSK flux](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/fixed_boundary_upstream_geqdsk.png)

![Axisymmetric manufactured Grad-Shafranov convergence](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/manufactured_grad_shafranov_convergence.png)

![Closed-form circular-loop elliptic response](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/circular_loop_elliptic_response.png)

![Free-boundary profile coupling](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/free_boundary_profile_coupling.png)

![Coil current sweep](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/coil_current_sweep.gif)

![Benchmark summary](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/benchmark_summary.png)

![CPC seed-family reproduction surrogate](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/cpc_seed_family.png)

![Pressure sweep animation](https://raw.githubusercontent.com/rogeriojorge/tokamaker_jax/main/docs/_static/pressure_sweep.gif)

## Current Scope

- JAX differentiable fixed-boundary seed solver for the Grad-Shafranov operator on a rectangular grid.
- p=1 triangular FEM reference kernels, dense/sparse/matrix-free assembly, weighted axisymmetric Grad-Shafranov weak-form assembly, profile source loads, and manufactured convergence gates.
- Nonlinear p=1 profile iteration with pressure and FF' source terms, residual checks, and differentiability tests.
- Reduced large-aspect-ratio coil Green's-function fixture plus a closed-form circular-loop elliptic Green's-function kernel checked against high-resolution quadrature.
- Free-boundary/profile coupling gate that drives the nonlinear FEM iteration from circular-loop coil boundary flux and checks coil-current/profile-scale differentiability.
- OpenFUSIONToolkit/TokaMaker comparison probe that records local upstream availability and runs numeric `eval_green` parity when the original compiled library is available.
- TOML configuration loader with Python 3.10 compatibility.
- CLI that launches the GUI by default and runs TOML files when supplied.
- Packaged fixed-boundary example export through `tokamaker-jax init-example`.
- Case manifest browser exposed through the CLI, GUI, docs, and committed JSON artifacts.
- Availability-gated upstream fixture inventory for exact TokaMaker mesh/geometry files.
- Source-evidence artifact for upstream fixed-boundary notebooks and the `gNT_example` gEQDSK case.
- Reusable EQDSK/gEQDSK importer and committed fixed-boundary diagnostic gate.
- Matplotlib plotting utilities, generated validation figures, CPC seed-family reproduction surrogate, and JSON figure recipes.
- NiceGUI workflow dashboard summaries, editable TOML validation, saved-case execution, and stored-report tables for solver, validation, plotting, benchmark, and reproduction lanes.
- Benchmark-history JSON/JSONL helpers for recording hardware/context metadata with timing reports.
- Expanded documentation with equations, derivations, design decisions, input/output artifact contracts, upstream/literature comparison levels, release publishing steps, and publication-ready generated figures.
- Sphinx and Read the Docs setup.
- GitHub Actions for linting, testing with coverage, benchmark artifact upload, docs, and release publishing.

## Porting Target

The full port will cover TokaMaker's unstructured finite-element mesh workflow, fixed/free-boundary equilibria, coil and passive conductor modeling, profile functions, reconstruction constraints, wall modes, time-dependent stepping, EQDSK/i-file IO, and publication-quality plotting. See [plan.md](plan.md) for the full breakdown, validation matrix, and implementation log.

## Attribution

This project is derived from planning and source review of [OpenFUSIONToolkit/OpenFUSIONToolkit](https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit), especially its TokaMaker component. Cite Hansen et al., *Computer Physics Communications* 298, 109111 (2024), DOI: [10.1016/j.cpc.2024.109111](https://doi.org/10.1016/j.cpc.2024.109111), when using TokaMaker-derived work.
