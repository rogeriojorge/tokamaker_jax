# Design Decisions

This page records the main engineering decisions behind `tokamaker-jax`. The
goal is to keep the port maintainable as it grows from seed validation to full
TokaMaker parity.

## Scope Boundary

`tokamaker-jax` is a staged native port, not a thin wrapper around
OpenFUSIONToolkit. The comparison harness may import upstream TokaMaker for
parity tests, but production solver paths should remain pure Python/JAX unless
a compiled dependency is explicitly isolated behind an optional benchmark or
fixture.

Reasoning:

- Differentiability needs transparent residuals, fixed iteration policies, and
  explicit custom VJP/JVP boundaries.
- GPU/TPU execution needs array programs that XLA can see.
- Research users need reproducible Python, CLI, GUI, and docs workflows
  without requiring a full Fortran/MPI build.

## Data Model

The code is organized around immutable or near-immutable data passed through
pure functions:

| Layer | Current files | Design rule |
| --- | --- | --- |
| Configuration | `config.py` | TOML/Python inputs validate before solving. Python 3.10 uses `tomli`; Python 3.11+ uses `tomllib`. |
| Geometry | `geometry.py`, `mesh.py` | Region and mesh metadata are explicit. Mesh topology is treated as static for AD. |
| FEM kernels | `fem.py`, `assembly.py` | Local kernels are small, tested, JIT-compatible functions. |
| Physics | `fem_equilibrium.py`, `free_boundary.py`, `profiles.py` | Physics functions are JAX array programs with documented signs and units. |
| Solvers | `solver.py`, future solver modules | Iteration counts and relaxation are visible inputs. |
| Verification | `verification.py`, `comparison.py` | Gates return JSON-ready dataclasses or dictionaries. |
| Interfaces | `cli.py`, `gui.py`, `plotting.py` | User interfaces consume the same validated config and report objects. |

## FEM Strategy

The initial production FEM target is p=1 triangular Lagrange elements. Higher
orders remain a future extension point.

Why p=1 first:

- It matches the minimum useful unstructured-mesh workflow.
- Exact local mass/stiffness identities are simple enough for rigorous tests.
- Assembly, Dirichlet reduction, sparse storage, matrix-free apply, and source
  integration can be validated independently.
- It provides a stable base for code-to-code TokaMaker parity before adding
  high-order elements.

The code keeps local element kernels, global scatter, sparse conversion, and
matrix-free apply separate so each can be tested and optimized independently.

## Differentiability Strategy

The project distinguishes three gradient classes:

| Class | Status | Examples |
| --- | --- | --- |
| Direct AD | Implemented for fixed-shape array programs | Green's functions, local FEM kernels, response matrices, seed solver objectives. |
| Unrolled solver AD | Implemented for seed/profile gates | Fixed iteration nonlinear profile solve, source-scale sensitivities. |
| Implicit AD | Planned | Newton solves, reconstruction objectives, fixed-point equilibria, time-dependent steps. |

Nonsmooth geometry and topology events are not hidden. The plan is to report
them as events and require smooth surrogate objectives when users request
gradient-based optimization.

## GUI Strategy

The command `tokamaker-jax` launches the GUI by default; passing a TOML file
runs the reproducible CLI path. The GUI is meant to be research-grade, not a
marketing landing page:

- surface actual solver settings, residuals, convergence rates, and report
  provenance;
- show generated figures and validation artifacts;
- allow examples and TOML cases to become the same object;
- keep every GUI action reproducible as a command.

The current GUI provides seed controls, region previews, validation plots,
coil-response plots, workflow summaries, a manifest-backed case browser, editable
TOML validation, saved-file run execution with artifact capture, and stored
report tables. The explicit `tokamaker-jax gui` command exposes host, port,
reload, and no-browser controls for Windows, Linux, macOS, WSL, containers, and
remote sessions.

## Documentation Strategy

Documentation is treated as executable infrastructure:

- generated figures live under `docs/_static`;
- JSON reports are committed when they document a validation state;
- every publication-style plot should have a command and input manifest;
- every physics claim should point to an equation, source file, test, and
  reference.

This is why `examples/generate_assets.py` regenerates validation dashboards,
benchmark summaries, comparison reports, and figure panels.

## Performance Strategy

Performance gates are intentionally small but explicit:

- benchmark lanes report medians, best/worst times, repeats, warmups, and
  metadata;
- CI uploads benchmark artifacts;
- thresholds are versioned in `docs/validation/benchmark_thresholds.json`;
- hardware-normalized historical baselines are planned after the solver stack
  stops changing quickly.

The current code favors correctness and testability over premature compiler
specialization. Hot paths are kept pure enough to be moved behind `jit`, `vmap`,
or custom sparse kernels as the FEM/free-boundary implementation matures.

## Compatibility Strategy

The port follows upstream TokaMaker concepts while exposing a friendlier JAX
API. Compatibility is tracked in three levels:

| Level | Meaning |
| --- | --- |
| Concept parity | Same model concept exists, such as region, coil, profile, or diagnostic. |
| Numeric parity | Same inputs produce matching scalar/vector outputs within tolerances. |
| Workflow parity | A full upstream example has a reproducible `tokamaker-jax` TOML/Python/GUI workflow. |

Current numeric parity exists for the circular-loop Green's-function sign
comparison when local OpenFUSIONToolkit is built. Full fixed/free-boundary
equilibrium workflow parity remains planned.
