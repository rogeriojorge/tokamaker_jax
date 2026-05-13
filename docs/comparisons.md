# Upstream and Literature Comparisons

This page compares `tokamaker-jax` with upstream TokaMaker, adjacent
free-boundary equilibrium tools, differentiable JAX science codes, and
literature validation targets. It is a status map, not a marketing claim.

```{image} _static/upstream_comparison_matrix.png
:alt: Upstream and literature comparison matrix
```

## OpenFUSIONToolkit and TokaMaker

TokaMaker is the primary porting target. The published description and upstream
docs identify the key features that `tokamaker-jax` must eventually match:

- static and time-dependent Grad-Shafranov equilibrium workflows;
- unstructured triangular FEM meshes;
- engineering-like geometry definitions;
- free-boundary boundary-condition formulation;
- analytic and cross-code validation;
- Python usability backed by compiled solver components.

Current `tokamaker-jax` status:

| Area | Current status | Next parity gate |
| --- | --- | --- |
| Mesh/regions | Region primitives, `TriMesh`, OFT-style HDF5/JSON helpers | Import exact upstream example meshes and compare region/area summaries. |
| FEM weak form | p=1 triangular assembly and manufactured GS gates | Solov'ev and spheromak analytic parity on triangular meshes. |
| Free-boundary coils | Reduced Green's fixture plus circular-loop elliptic kernel | Full coil mutual/response fixtures from upstream tests. |
| OFT numeric comparison | `eval_green` parity when local OFT build is available | Fixed-boundary equilibrium vector parity. |
| Profiles | Power-profile source and nonlinear Picard iteration | Polynomial, spline, Wesson, bootstrap profile fixtures. |
| Reconstruction/time/walls | Planned | Synthetic diagnostic and passive-conductor examples. |

The committed comparison report is
`docs/_static/openfusiontoolkit_comparison_report.json`.
The case-by-case upstream fixture inventory is in [](case_manifest.md) and
`docs/_static/case_manifest.json`. Exact mesh/geometry summaries for upstream
example files are in [](upstream_fixtures.md) and
`docs/_static/upstream_fixture_summary.json`.

## FreeGS and FreeGSNKE

FreeGS provides a Python free-boundary Grad-Shafranov workflow with coils and
constraints. FreeGSNKE extends that ecosystem toward static forward, static
inverse, and evolutive free-boundary problems with passive structures and
Newton-Krylov methods.

Design lessons adopted here:

- examples must start from machine definitions users can read;
- static forward and inverse workflows should be separate, reproducible
  commands;
- passive structures and diagnostics need first-class input objects;
- convergence and residual diagnostics should be visible in plots and reports.

Current `tokamaker-jax` status is limited to early coil-response and profile
coupling gates. Passive structures, inverse reconstruction, and evolutive
solves remain future work.

## JAX-FEM and TORAX

JAX-FEM is the closest design analogue for differentiable FEM kernels. TORAX is
the closest JAX tokamak-code analogue for differentiable, compiled, modular
physics workflows.

Design lessons adopted here:

- keep kernels pure and testable;
- separate model state from solver configuration;
- report gradients as part of validation, not as an afterthought;
- design for `jit`, `vmap`, and custom derivative rules once correctness is
  established;
- keep ML/surrogate coupling optional and downstream of validated physics.

`tokamaker-jax` is not a TORAX replacement. The intended future coupling is
equilibrium geometry, flux surfaces, and current-profile information exchanged
with transport/scenario tools.

## EFIT, COCOS, and Reconstruction Literature

EFIT-style reconstruction motivates future diagnostic constraints: flux loops,
pickup coils, plasma current, pressure, q, saddles, and boundary shape targets.
COCOS is the required convention reference for EQDSK and cross-code exchange.

Current status:

- sign conventions are now explicitly documented for the circular-loop/OFT
  comparison;
- EQDSK/i-file output remains planned;
- reconstruction constraints remain planned;
- future docs must state COCOS assumptions for every imported/exported file.

## Analytic Equilibrium Literature

Analytic and semi-analytic equilibria are the strongest validation targets
because they provide field values, derivatives, separatrix topology, and
convergence expectations without relying on another code.

Planned gates:

| Reference family | Purpose | Required artifact |
| --- | --- | --- |
| Solov'ev/Cerfon-Freidberg | Fixed-boundary convergence, O/X point checks, separatrix plots | TOML case, exact-field evaluator, convergence table, figure recipe. |
| Spheromak/FRC analytic families | Topology and non-tokamak geometry stress tests | Mesh fixture, field diagnostics, contour figure. |
| Sauter/Redl bootstrap formulas | Bootstrap current and conductivity profile validation | Coefficient tables, branch tests, ITER-like profile plot. |
| EFIT/MAST-U style reconstructions | Diagnostic residual and boundary-error validation | Synthetic diagnostics, residual dashboard, comparison report. |

The current CPC seed-family figure is explicitly marked as a surrogate fixture.
It is useful for workflow and artifact validation, not yet a data-level
reproduction of a published TokaMaker figure.

## Comparison Acceptance Levels

Every comparison must declare one of these levels:

| Level | Meaning |
| --- | --- |
| `source_audit` | Source/docs were inspected and requirements were extracted. |
| `surrogate_fixture` | A generated artifact exercises the workflow but does not claim numeric parity. |
| `kernel_parity` | A scalar/vector kernel matches a source or analytic oracle within tolerance. |
| `equilibrium_parity` | A full equilibrium state matches upstream or literature diagnostics within tolerance. |
| `workflow_parity` | An upstream/literature example has reproducible Python, TOML, GUI, docs, plots, and reports. |

Current highest implemented level:

- OpenFUSIONToolkit circular-loop `eval_green`: `kernel_parity`.
- CPC seed-family example: `surrogate_fixture`.
- Manufactured FEM gates: analytic `kernel_parity` and convergence parity
  against p=1 FEM theory.

Full TokaMaker replacement status remains future work.

## Publication-Ready Figure Panel

The current docs include a four-panel summary generated from committed
validation artifacts:

```{image} _static/publication_validation_panel.png
:alt: Publication validation panel with convergence, coupling, and benchmarks
```

The panel is intentionally composed from reproducible generated figures rather
than manually edited images.
