# tokamaker-jax Porting Plan and Log

This file is both the engineering plan and the running implementation log for a JAX-native port of TokaMaker from OpenFUSIONToolkit.

## Log

### 2026-05-12 15:45 WEST

- Found an existing `/Users/rogeriojorge/local/OpenFUSIONToolkit` checkout and updated it to upstream `main` commit `729a5f9`.
- Confirmed `gh` is authenticated as `rogeriojorge` with repository and workflow scopes.
- Audited TokaMaker source, Python API, docs, examples, and tests.
- Created `/Users/rogeriojorge/local/tokamaker_jax` with package scaffold, TOML CLI, optional GUI entry point, docs, Read the Docs config, GitHub workflows, tests, README badges, citation metadata, generated PNG/GIF assets, and a differentiable JAX fixed-boundary seed solver.
- The seed solver is not the full port. It is the executable base used to validate packaging, differentiability, CLI/GUI shape, plotting, and coverage infrastructure while the full triangular FEM port proceeds.
- Published public repository: <https://github.com/rogeriojorge/tokamaker_jax>.
- Validation performed locally: Ruff formatting/linting passed, `pytest --cov=tokamaker_jax --cov-fail-under=95` passed with 98.66% coverage, and `sphinx -W -b html docs docs/_build/html` passed.
- Initial GitHub Actions CI and Docs runs succeeded. Added Node 24 opt-in to workflows to avoid the upcoming GitHub Actions Node 20 deprecation warning.

### 2026-05-12 16:19 WEST

- Expanded this planning file after a deeper scan of upstream TokaMaker source, Python API, examples, advanced workflows, and regression tests.
- Added a capability parity matrix, package architecture, data model, TOML schema, solver plan, differentiability plan, GUI plan, validation matrix, milestone acceptance criteria, issue backlog, risk register, and immediate next implementation queue.
- Refreshed external reference context using OFT docs, the TokaMaker arXiv/CPC paper, FreeGSNKE docs, JAX-FEM, and TORAX.

## Current State

Repository: <https://github.com/rogeriojorge/tokamaker_jax>

Local path: `/Users/rogeriojorge/local/tokamaker_jax`

Current branch state when this plan was expanded:

- `main` tracks `origin/main`.
- Last pushed commit before this plan expansion: `4f82889`.
- GitHub Actions CI and Docs were green.
- Package contains an executable differentiable rectangular-grid seed solver, not yet the full TokaMaker FEM/free-boundary implementation.

## Product Goal

Build `tokamaker-jax` into a public, documented, tested, JAX-native TokaMaker port with:

- Feature parity with TokaMaker's static, free-boundary, reconstruction, passive conductor, wall-mode, bootstrap, and time-dependent Grad-Shafranov workflows.
- End-to-end differentiability for the parts that are mathematically differentiable: profile coefficients, coil currents, coil geometry, target weights, diagnostic weights, current source parameters, and smooth shape objectives.
- Clear behavior around nonsmooth events: limiter transitions, X-point creation/destruction, topology changes, mesh changes, and discrete constraint activation.
- A friendlier Python API than the original OFT wrapper, while retaining a compatibility layer for users migrating existing TokaMaker notebooks.
- GUI-first usability: `tokamaker-jax` opens a GUI; `tokamaker-jax case.toml` runs reproducibly from configuration.
- Python 3.10+ support. TOML parsing uses `tomli` on Python 3.10 and `tomllib` on Python 3.11+.
- At least 95% code coverage enforced in CI, with broader validation through parity tests, benchmark tests, and notebook/GUI smoke tests.
- Documentation that can teach a new user to build, solve, inspect, optimize, and export equilibria without reading source code.

## Non-Goals

- Do not wrap OFT Fortran as the primary solver. A temporary comparison harness may call OFT, but production `tokamaker-jax` should be JAX-native.
- Do not require a GUI for batch or HPC workflows.
- Do not promise differentiability through topological discontinuities. Expose smooth objectives and warn when events are nonsmooth.
- Do not reimplement a general CAD/meshing suite. Use importers and small domain builders, and keep mesh generation outside JAX.
- Do not optimize for every backend before physics parity. First target correctness, then compile-time and runtime performance.

## Definition of Done

The full port is done when all of the following hold:

- All core TokaMaker examples have equivalent `tokamaker-jax` TOML/Python workflows.
- OFT parity tests pass within documented tolerances for fixed-boundary, free-boundary, reconstruction, wall-mode, time-dependent, bootstrap, and IO cases.
- Core solver, profile, geometry, IO, and plotting modules maintain at least 95% coverage.
- Public API and TOML schema are documented and versioned.
- GUI can load/edit/save TOML, run a solve, show convergence, plot flux/geometry/profiles/constraints, and export artifacts.
- README and docs assets are regenerated from tested scripts.
- CI passes on Python 3.10, 3.11, 3.12, and 3.13.
- A benchmark report documents CPU/GPU runtime, memory, compile time, and gradient cost against representative cases.

## Source Audit

Upstream source: `/Users/rogeriojorge/local/OpenFUSIONToolkit`

Upstream commit used for this plan: `729a5f9`

Latest release observed during initial setup: `v1.0.0-beta7`

### Fortran Physics Core

| Source | Approx. lines | Role | Porting notes |
| --- | ---: | --- | --- |
| `src/physics/grad_shaf.F90` | 6215 | Main equilibrium factory, nonlinear solve, free-boundary conditions, coil and wall sources, O/X point logic, q profiles, interpolation, export support | Central parity target. Port in slices: data model, FEM operators, source assembly, static solve, free-boundary, diagnostics, export. |
| `src/physics/grad_shaf_profiles.F90` | 1198 | Flux-function hierarchy: zero, flat, polynomial, spline, linterp, Wesson; function values and derivatives; save/load | Good early target because it is mostly pure functions and differentiability is clear. |
| `src/physics/grad_shaf_td.F90` | 1410 | Time-dependent stepping, passive conductors, wall/current operators, linearized stability | Later target after static/free-boundary parity. Needs careful implicit differentiation and `lax.scan`. |
| `src/physics/grad_shaf_fit.F90` | 1714 | Reconstruction and fitting machinery | Port after diagnostic evaluators and static solve are stable. |
| `src/physics/grad_shaf_util.F90` | 1420 | Global quantities, loop voltage, save/load, EQDSK/i-file export, Sauter helpers | Split into `diagnostics`, `io`, and `bootstrap` modules. |
| `src/physics/axi_green.F90` | 286 | Axisymmetric Green's functions and derivatives for current filaments | Early free-boundary dependency. Verify against OFT and brute-force quadrature. |
| `src/physics/gs_eq.F90` | 174 | Equilibrium interpolation support | Port as differentiable field evaluators and synthetic diagnostics. |
| `src/python/wrappers/tokamaker_f.F90` | large wrapper | Fortran/Python interface | Use as an API inventory, not implementation source. |

### Python API Surface

Primary upstream class: `OpenFUSIONToolkit.TokaMaker.TokaMaker`.

Important public methods to preserve or map:

- Setup and state: `reset`, `setup_mesh`, `setup_regions`, `setup`, `copy_eq`, `replace_eq`, `update_settings`.
- Profiles: `load_profiles`, `set_profiles`, `set_resistivity`, `ffp_scale`, `p_scale`, `alam`, `pnorm`.
- Solves: `init_psi`, `solve`, `vac_solve`, `set_psi`, `get_psi`, `set_psi_dt`.
- Targets and constraints: `set_targets`, `get_targets`, `set_isoflux`, `set_flux`, `set_psi_constraints`, `set_saddles`, `set_coil_reg`, `set_coil_bounds`, `set_coil_vsc`, `set_vcoils`.
- Coils and conductors: `set_coil_currents`, `get_coil_currents`, `get_coil_Lmat`, `set_coil_current_dist`, `get_conductor_currents`, `get_conductor_source`, `get_vfixed`.
- Diagnostics: `get_stats`, `get_globals`, `get_profiles`, `get_q`, `trace_surf`, `get_xpoints`, `get_jtor_plasma`, `get_delstar_curr`, `calc_loopvoltage`, `sauter_fc`, `area_integral`, `flux_integral`.
- Plotting: `plot_machine`, `plot_mesh`, `plot_psi`, `plot_constraints`, `plot_eddy`.
- IO: `save_eqdsk`, `save_ifile`, `save_mug`, `save_TokaMaker`.
- Stability/time: `eig_wall`, `compute_wall_modes`, `eig_td`, `compute_linear_stability`, `setup_td`, `step_td`.

Other Python modules:

- `meshing.py`: `gs_Domain`, `Mesh`, `Region`, `save_gs_mesh`, `load_gs_mesh`, Rectangle/Polygon/Annulus/Enclosed builders, Triangle/Cubit integration, plotting.
- `reconstruction.py`: constraint containers for Mirnov, Ip, flux loop, dFlux, pressure, q, saddle; `reconstruction.reconstruct`.
- `util.py`: isoflux creation, spline/power flux functions, EQDSK/i-file readers, Green's function evaluation, MHDIN/K-file helpers, force components.

### Upstream Examples Inventory

Examples to reproduce or supersede:

- Fixed boundary: `fixed_boundary_ex1.ipynb`, `fixed_boundary_ex2.ipynb`, `gNT_example`.
- Machine mesh/equilibrium examples: ITER, HBT, DIII-D, LTX, CUTE, MANTA.
- ITER extras: baseline, H-mode/bootstrap, reconstruction, disruption forces.
- HBT extras: vacuum coil workflows.
- CUTE extras: VDE.
- Dipole: mesh and equilibrium examples.
- Advanced workflows:
  - NSTX-U isoflux controller mesh generator, shape generator, and shape-control simulator.
  - Pulse design examples including CUTE and DIII-D pulse scripts.

### Upstream Regression Inventory

`src/tests/physics/test_TokaMaker.py` covers:

- Analytic Solov'ev convergence for orders 2, 3, 4 and multiple mesh resolutions.
- Spheromak analytic/eigenfunction style tests.
- Coil Green's function and mutual inductance tests, including distributed coil current cases.
- ITER equilibrium, wall eigenmodes, linear stability, reconstruction, concurrency, EQDSK export, i-file export, and equilibrium IO.
- LTX equilibrium, wall eigenmodes, and linear stability.
- ITER H-mode/bootstrap solve.
- Redl bootstrap current calculation.

These tests define the minimum physics parity suite.

## External Reference Context

This plan uses the following external sources as design constraints and comparison points:

- OFT docs describe OFT as a common high-order FEM framework on unstructured grids and TokaMaker as a time-dependent free-boundary Grad-Shafranov equilibrium code: <https://openfusiontoolkit.github.io/OpenFUSIONToolkit/>.
- Hansen et al. describe TokaMaker as a static and time-dependent Grad-Shafranov code using finite elements on unstructured triangular grids, with Python/Fortran/C/C++ components, a novel free-boundary boundary condition formulation, and analytic/cross-code validation: <https://arxiv.org/abs/2311.07719> and <https://doi.org/10.1016/j.cpc.2024.109111>.
- FreeGS provides a Python free-boundary Grad-Shafranov reference with easy examples and coil/constraint concepts: <https://github.com/freegs-plasma/freegs> and <https://freegs.readthedocs.io/en/stable/creating_equilibria.html>.
- FreeGSNKE provides a useful product benchmark: static forward, static inverse, evolutive forward, passive structures, probes, and Newton-Krylov/evolutive workflows. Its docs also list JAX-ification of core Newton-Krylov solvers as a roadmap direction: <https://docs.freegsnke.com/>.
- JAX-FEM demonstrates differentiable FEM in JAX, including triangle elements, automatic differentiation, inverse/design problems, and PETSc integration patterns: <https://github.com/deepmodeling/jax-fem> and <https://arxiv.org/abs/2212.00964>.
- TORAX is a useful JAX tokamak-code reference for differentiable PDE workflows, compilation, nonlinear PDE sensitivities, trajectory optimization, and data-driven parameter identification: <https://github.com/google-deepmind/torax>.

## Target Architecture

### Layering

1. User-facing interfaces: Python API, TOML CLI, GUI, notebooks.
2. Case model: typed dataclasses/pytrees for machine, mesh, profiles, coils, constraints, settings.
3. Mesh/FEM core: static connectivity and quadrature metadata plus JAX arrays for coordinates and differentiable geometry parameters where possible.
4. Physics kernels: Grad-Shafranov operator, sources, profiles, Green's functions, diagnostics, reconstruction residuals.
5. Solvers: linear, nonlinear, free-boundary, reconstruction, time-dependent, stability.
6. IO and plotting: EQDSK, i-file, HDF5, MUG-compatible export, images, animations.
7. Validation and benchmarks: OFT parity harness, manufactured solutions, performance profiles.

### Planned Package Map

| Module | Responsibility | Current status |
| --- | --- | --- |
| `config` | TOML parsing, schema versioning, dataclass conversion | Seed implementation exists |
| `domain` | Rectangular seed domain now; future machine/region domain objects | Seed implementation exists |
| `geometry` | Polygons, annuli, rectangles, limiters, wall geometry, geometry validation | Planned |
| `mesh` | Triangular mesh container, HDF5 import/export, region labels, mesh quality | Planned |
| `fem` | Basis functions, quadrature, element assembly, sparse and matrix-free apply | Planned |
| `operator` | Grad-Shafranov operator and source operators | Planned |
| `profiles` | Flux functions, derivatives, profile coefficient pytrees | Seed functions exist; full hierarchy planned |
| `greens` | Axisymmetric Green's functions and gradients | Planned |
| `coils` | Coil sets, distributed coils, bounds, current vectors, inductance matrices | Planned |
| `conductors` | Passive structures, resistivity, eddy current sources, wall matrices | Planned |
| `constraints` | Isoflux, flux, saddle, target, reconstruction constraints | Planned |
| `solver` | Static fixed/free-boundary, nonlinear, vacuum, time stepping | Seed implementation exists |
| `diagnostics` | Global quantities, q profiles, loop voltage, beta/li, O/X points | Planned |
| `bootstrap` | Sauter/Redl bootstrap and H-mode profile helpers | Planned |
| `io` | EQDSK, i-file, HDF5, MUG/TokaMaker compatibility | Planned |
| `plotting` | Matplotlib/Plotly plotting and animations | Seed implementation exists |
| `gui` | GUI workflow | Seed entry point exists |
| `cli` | GUI default, TOML execution, validate, benchmark, asset generation | Seed implementation exists |
| `compat` | Migration-compatible API names and helpers | Planned |

### Data Model

Core objects should be immutable where practical and valid as JAX pytrees.

| Object | Differentiable fields | Static fields | Notes |
| --- | --- | --- | --- |
| `Machine` | optional coil positions/sizes, profile params, target params | name, units, COCOS convention | Top-level container |
| `Region` | optional geometry control points | region id, type, labels | `plasma`, `vacuum`, `boundary`, `conductor`, `coil` |
| `TriMesh` | node coordinates if geometry optimization is active | connectivity, region ids, boundary edges, polynomial order | Connectivity is static under JIT |
| `Basis` | none | order, quadrature rule, reference basis values | Cache by order |
| `Profiles` | coefficients/control points | type names, boundary conditions | Must expose value, first derivative, second derivative |
| `CoilSet` | currents, optional geometry params | grouping, polarity, turns, bounds | Vector/dict conversion must be stable |
| `Constraints` | target values, weights, locations when optimized | active flags and types | Reconstruction and inverse design reuse same residual form |
| `EquilibriumState` | psi, coil currents, conductor currents, profile scales | mesh signature, settings | The main solver output pytree |
| `SolveResult` | state and diagnostics | convergence metadata | Stores residual history and warnings |

### Dependency Policy

Required runtime:

- `jax`, `jaxlib`, `numpy`, `scipy`, `matplotlib`, `tomli` only for Python 3.10.

Optional extras:

- `gui`: `nicegui`, `plotly`.
- `docs`: Sphinx stack.
- `mesh`: optional `triangle`, `meshio`, `h5py`, maybe `gmsh` importer.
- `dev`: `pytest`, `pytest-cov`, `ruff`.
- `bench`: `pytest-benchmark`, profiling helpers.
- Future solver extras should remain optional if they bring compiled dependencies.

Avoid hard dependency on GPL packages in core if license compatibility is unclear. Use JAX-FEM as a design reference; do not vendor GPL code.

## Physics and Numerical Plan

### Equation and Conventions

Port the TokaMaker Grad-Shafranov formulation:

```text
Delta* psi =
  -0.5 d(F^2)/dpsi - mu0 R^2 dP/dpsi     in plasma
  -R mu0 J_phi                            in coils/conductors
   0                                      in vacuum
```

Required convention work:

- Explicitly document sign conventions for `psi`, normalized flux, `F0`, `p_scale`, `ffp_scale`, `alam`, and `pnorm`.
- Track COCOS conventions in EQDSK export/import.
- Provide conversion tests for absolute psi to normalized psi and back.
- Preserve TokaMaker's tokamak and spheromak normalized flux conventions.

### Mesh and FEM

Milestone sequence:

1. Import existing TokaMaker HDF5 meshes, preserving `R`, `LC`, `REG`, coil dictionaries, and conductor dictionaries.
2. Implement pure-Python/JAX mesh container with region ids, boundary edges, node markers, and element adjacency.
3. Implement p=1 triangular FEM first to de-risk assembly and manufactured-solution tests.
4. Add p=2, p=3, p=4 parity to match TokaMaker's tested orders.
5. Add mesh generation wrappers for `gs_Domain` style rectangle, polygon, annulus, and enclosed regions. Mesh generation can use external libraries outside JAX.
6. Add mesh quality checks and GUI mesh preview.

Acceptance criteria:

- Mesh importer round-trips ITER, HBT, DIII-D, LTX, CUTE, MANTA, Dipole, NSTX-U HDF5 meshes without changing region ids.
- Basis function partition-of-unity and derivative tests pass for every supported order.
- Mass/stiffness matrix manufactured tests converge at expected order.
- FEM assembly is deterministic across runs.

### Profiles

Port order:

1. `zero` and `flat`.
2. `linterp` and `jphi-linterp`.
3. `polynomial`.
4. `spline`.
5. `Wesson`.
6. non-inductive profile terms.
7. Sauter/Redl bootstrap current helpers.

Acceptance criteria:

- Values and first/second derivatives match OFT on sample grids.
- Profile coefficient pytrees are differentiable.
- Profile save/load tests pass for text, TOML, and HDF5 where applicable.
- H-mode/bootstrap test reaches OFT reference diagnostics within tolerance.

### Grad-Shafranov Operator

Implementation choices:

- Build reference sparse matrices with `jax.experimental.sparse.BCOO`.
- Also implement a matrix-free element apply for large cases and gradient-friendly solves.
- Separate static boundary-condition setup from differentiable source assembly.
- Keep boundary edges, limiter markers, region masks, and coil/conductor maps static under JIT.

Acceptance criteria:

- `apply_operator` matches assembled matrix multiplication on random fields.
- Fixed-boundary manufactured solutions converge at expected order.
- Solov'ev and spheromak tests match OFT tolerances.

### Coils and Green's Functions

Required capabilities:

- Axisymmetric current filament Green's function and gradient.
- Coil regions from meshes and dictionary definitions.
- Coil set grouping, turns, signs, bounds, sub-coils, distributed coils.
- Virtual coil `#VSC` behavior.
- Voltage coils and conductor coupling metadata.
- Coil current dict/vector conversion with stable ordering.
- Self and mutual inductance matrices.

Acceptance criteria:

- Green's function values match OFT and brute-force quadrature.
- Coil mutual tests pass for point and distributed coils.
- Coil bounds/regularization behave identically on ITER and LTX examples.

### Static Solvers

Solver ladder:

1. Fixed-boundary linear solve.
2. Fixed-boundary nonlinear Picard parity with TokaMaker.
3. Free-boundary vacuum solve and boundary-condition matrix.
4. Free-boundary inverse/isoflux coil solve.
5. Newton/Newton-Krylov optional solvers for robustness and speed.
6. Implicit differentiation for production gradients.

Solve modes:

- `forward`: fixed coil currents and profiles.
- `inverse`: coil currents chosen to match shape/constraints.
- `vacuum`: field solve without plasma.
- `reconstruction`: fit equilibrium to diagnostics.

Acceptance criteria:

- Solov'ev and spheromak tests pass for orders 2, 3, 4.
- ITER and LTX free-boundary diagnostics match OFT within documented tolerance.
- Convergence failures produce structured diagnostics, not opaque exceptions.

### Free-Boundary Topology

Required logic:

- O-point location.
- X-point/saddle detection.
- limiter-point handling.
- LCFS and diverted/limited classification.
- flux surface tracing.
- optional limited-only mode and limiter z cutoff.

Differentiability policy:

- Hard topology detection is not considered differentiable.
- Smooth surrogate losses should be available for shape optimization.
- Gradients are valid only within a fixed topology region unless explicitly using a smoothed objective.

Acceptance criteria:

- O/X point locations match OFT regression values.
- LCFS contours are stable under small perturbations away from topology transitions.
- Shape losses are differentiable with respect to coil currents and smooth target parameters.

### Reconstruction

Constraints to port:

- Ip.
- dFlux.
- flux loops.
- Mirnov probes.
- saddle/X-point constraints.
- pressure constraints.
- q constraints.
- coil current regularization.

Implementation plan:

- Express each diagnostic as a differentiable residual function.
- Combine residuals with weights into a least-squares objective.
- Use JAX Jacobian-vector products for Gauss-Newton/Newton-Krylov paths.
- Preserve file read/write compatibility for `fit.in`/`fit.out` only if useful; prefer TOML/HDF5 for new workflows.

Acceptance criteria:

- ITER synthetic reconstruction test matches OFT diagnostics within tolerance.
- Diagnostic residuals can be differentiated with respect to profile coefficients, coil currents, and selected diagnostic positions.

### Time-Dependent and Wall Physics

Required capabilities:

- Passive conductor currents.
- Resistivity profiles.
- Wall source terms from `dpsi_dt`.
- `setup_td` and `step_td`.
- wall eigenmodes.
- linear stability eigenmodes.
- VDE and pulse workflows.

Implementation plan:

- First port wall and conductor matrices for static diagnostics.
- Add eigenvalue problems and parity against `compute_wall_modes` and `compute_linear_stability`.
- Add time stepping with `lax.scan` for differentiable trajectories.
- Add implicit differentiation or checkpointed scan for long pulses.

Acceptance criteria:

- ITER and LTX wall-mode/eigenvalue tests match OFT references.
- CUTE VDE example can generate a movie from TOML.
- Pulse design workflows run in batch mode and GUI mode.

### Bootstrap and Non-Inductive Currents

Required capabilities:

- H-mode profile helper equivalent.
- Sauter trapped particle fraction path.
- Redl bootstrap current path.
- conversion from parallel/bootstrap current to toroidal current source.

Acceptance criteria:

- ITER H-mode/bootstrap test matches reference diagnostic dictionary.
- Redl direct current test matches reference values.
- Bootstrap workflow is differentiable with respect to kinetic profile coefficients except where formula branches are nonsmooth.

## Differentiability Plan

### Transform Targets

Must work under:

- `jax.jit` for repeated solves on a fixed mesh.
- `jax.grad` for scalar objectives such as beta, coil power, shape residual, diagnostic chi-square, and stored energy.
- `jax.jacfwd`/`jax.jacrev` for small verification cases.
- `jax.vmap` for parameter sweeps and multi-case optimization.
- `jax.lax.scan` for time-dependent trajectories.

### Gradient Modes

| Mode | Use | Pros | Cons |
| --- | --- | --- | --- |
| Unrolled iterative differentiation | small tests, debugging | exact through the implemented iterations | memory heavy, compile heavy |
| Implicit linear solve VJP | production static solves | fast, low memory | requires careful residual definition |
| Custom VJP for nonlinear fixed point | production nonlinear solves | scalable gradients | must validate against finite differences and unrolled solves |
| Smooth surrogate objectives | shape optimization | avoids topology discontinuities | only approximates event-based LCFS metrics |

### Gradient Validation

- Compare `jax.grad` to finite differences for seed, fixed-boundary, and free-boundary smooth cases.
- Compare implicit gradients to unrolled gradients on small meshes.
- Check gradient signs against physical intuition for coil current perturbations.
- Add `vmap` tests over profile scale and coil current sweeps.
- Add documentation warnings for nondifferentiable topology events.

## Python API Plan

### Preferred Modern API

```python
from tokamaker_jax import Case, Machine, solve

case = Case.from_toml("iter_baseline.toml")
result = solve(case)
result.plot.flux().save("iter_flux.png")
result.io.save_eqdsk("iter.g")
```

Optimization:

```python
import jax
from tokamaker_jax import objective_from_case

case = Case.from_toml("shape.toml")
objective = objective_from_case(case, metric="shape_error")
grad = jax.grad(objective)(case.parameters.coil_currents)
```

### Compatibility API

Provide a compatibility layer for common old calls:

```python
from tokamaker_jax.compat import TokaMaker

tm = TokaMaker()
tm.setup_mesh(r, lc, reg)
tm.setup_regions(cond_dict=cond, coil_dict=coil)
tm.setup(order=2, F0=5.3)
tm.set_targets(Ip=15e6, R0=6.2, Z0=0.0)
tm.solve()
```

Policy:

- Compatibility layer can be stateful for migration convenience.
- Core solver remains functional and pytree-based.
- Compatibility methods should be thin wrappers with tests.

## TOML Schema Plan

Every TOML case should be self-contained and portable.

### Static Free-Boundary Sketch

```toml
schema_version = "0.1"

[machine]
name = "ITER-like"
cocos = 7
units = "si"

[mesh]
source = "hdf5"
path = "ITER_mesh.h5"
order = 2

[settings]
mode = "free-boundary"
solver = "picard"
max_iterations = 40
tolerance = 1.0e-6
float = "float64"

[field]
F0 = 17.3

[profiles.ffprime]
type = "linterp"
x = [0.0, 0.25, 0.5, 0.75, 1.0]
y = [1.0, 0.8, 0.4, 0.1, 0.0]

[profiles.pprime]
type = "power"
alpha = 1.5
gamma = 2.0

[targets]
Ip = 15.0e6
R0 = 6.2
Z0 = 0.0

[[constraints.isoflux]]
r = 6.2
z = 3.4
weight = 1.0

[[constraints.saddle]]
r = 5.0
z = -3.4
weight = 1.0

[output]
eqdsk = "outputs/iter.g"
hdf5 = "outputs/iter.h5"
plot = "outputs/iter.png"
```

### Time-Dependent Sketch

```toml
[time]
t_start = 0.0
t_end = 0.5
dt = 1.0e-3
method = "implicit-euler"

[[waveform.coil_voltage]]
name = "PF1"
times = [0.0, 0.1, 0.5]
values = [0.0, 10.0, 0.0]
```

Schema rules:

- Every schema version has a migration path.
- Missing optional fields get documented defaults.
- Validation errors should include a file path and key path.
- CLI should support `tokamaker-jax validate case.toml`.

## CLI Plan

Commands:

- `tokamaker-jax`: launch GUI.
- `tokamaker-jax case.toml`: run case.
- `tokamaker-jax run case.toml`: explicit run alias.
- `tokamaker-jax validate case.toml`: schema, mesh, profiles, and output path validation.
- `tokamaker-jax plot state.h5 --out figure.png`: plot existing state.
- `tokamaker-jax benchmark case.toml --backend cpu|gpu`: benchmark solve and gradients.
- `tokamaker-jax compare case.toml --oft /path/to/OpenFUSIONToolkit`: run OFT parity comparison when OFT is installed.
- `tokamaker-jax examples list`: list built-in examples.
- `tokamaker-jax examples run iter_baseline`: run and export a bundled example.
- `tokamaker-jax assets`: regenerate README/docs PNG/GIF/MP4 artifacts.

Exit code policy:

- `0`: success.
- `1`: validation or solve failure.
- `2`: CLI usage error.
- `3`: parity mismatch.

## GUI Plan

### Screens

1. Case browser:
   - new case, open TOML, example gallery, recent runs.
2. Machine/mesh:
   - region table, coil table, conductor table, limiter preview, mesh preview, mesh quality diagnostics.
3. Profiles:
   - profile type selector, coefficient editor, plot of `p'`, `FF'`, `j_phi`, bootstrap terms.
4. Constraints and targets:
   - Ip/R0/Z0/pax/beta targets, isoflux points, saddle points, flux loops, Mirnov probes, q/pressure constraints.
5. Solve:
   - solve mode, solver settings, backend, progress, convergence history, structured warnings.
6. Equilibrium:
   - flux contours, LCFS, O/X points, geometry overlay, coil currents, q profile, pressure/current profiles, global diagnostics.
7. Time-dependent:
   - waveform editor, time slider, animation export, conductor currents, VDE metrics.
8. Optimization:
   - objective terms, differentiable parameters, gradient checks, optimization traces.
9. Export:
   - TOML, HDF5, EQDSK, i-file, MUG-compatible output, PNG/SVG/PDF, GIF/MP4.

### GUI Acceptance Criteria

- User can load `examples/fixed_boundary.toml`, run, see a plot, and export a PNG.
- User can edit a profile parameter and rerun without restarting.
- User can open a free-boundary example once Phase 3 is complete.
- GUI-generated TOML validates and can be run headlessly.
- Browser smoke tests verify the page is nonblank, controls are visible, and plot updates after a run.

## Documentation Plan

Docs structure:

- Getting started: install, run GUI, run TOML, Python quickstart.
- Concepts: Grad-Shafranov equation, regions, coils, profiles, constraints, free-boundary solves.
- API reference: generated from docstrings.
- TOML schema reference with complete examples.
- Tutorials:
  - fixed-boundary Solov'ev.
  - free-boundary ITER-like scenario.
  - coil current optimization.
  - reconstruction from synthetic diagnostics.
  - time-dependent CUTE VDE.
  - bootstrap/H-mode scenario.
  - GUI walkthrough.
- Validation report: parity tables versus OFT and analytic cases.
- Performance report: CPU/GPU timings and gradient costs.
- Developer guide: porting conventions, testing, code style, release process.

Asset rules:

- README/docs images and movies are generated from scripts.
- No hand-edited plots.
- Artifacts should be small enough for GitHub, with larger movies stored as release assets if needed.

## Testing and Validation Plan

### Coverage Gates

- Unit coverage: at least 95% for core modules.
- GUI may use browser smoke tests and can be excluded from unit coverage until stable.
- Every feature port requires tests before it is marked complete.

### Test Types

| Test type | Purpose | Examples |
| --- | --- | --- |
| Unit | local behavior | profiles, basis functions, TOML validation, Green's functions |
| Property | invariants | partition of unity, symmetric matrices, positive areas |
| Manufactured solution | PDE correctness | fixed-boundary analytic fields |
| OFT parity | feature parity | Solov'ev, spheromak, ITER/LTX, reconstruction |
| Differentiability | gradient correctness | finite-difference vs `jax.grad`, implicit vs unrolled |
| Performance | regressions | solve time, compile time, memory |
| Docs/examples | user workflows | execute example TOML and notebooks |
| Browser | GUI behavior | open GUI, run, plot nonblank |

### Parity Matrix

| Capability | Upstream anchor | Target modules | Acceptance |
| --- | --- | --- | --- |
| Solov'ev fixed-boundary | `test_solo_*` | `fem`, `operator`, `solver` | OFT tolerances by order/resolution |
| Spheromak | `test_spheromak_*` | `profiles`, `solver` | relative psi error within OFT expectation |
| Coil mutuals | `test_coil_*` | `greens`, `coils`, `operator` | psi/mutual errors within OFT expectation |
| ITER equilibrium | `test_ITER_eq` | full static stack | diagnostic dictionary and EQDSK/i-file parity |
| ITER reconstruction | `test_ITER_recon` | `constraints`, `diagnostics`, `solver` | diagnostic dictionary parity |
| ITER wall modes | `test_ITER_eig` | `conductors`, `time` | first wall times within tolerance |
| ITER stability | `test_ITER_stability` | `time`, `operator` | growth rates within tolerance |
| ITER bootstrap | `test_ITER_bootstrap` | `bootstrap`, `profiles`, `solver` | diagnostic dictionary parity |
| Redl bootstrap | `test_Redl_jBS` | `bootstrap` | current/profile parity |
| LTX equilibrium | `test_LTX_eq` | static/free-boundary stack | diagnostic dictionary parity |
| LTX wall/stability | `test_LTX_eig`, `test_LTX_stability` | `conductors`, `time` | eigenvalue parity |
| EQDSK/i-file IO | `validate_eqdsk`, `validate_ifile` | `io` | round-trip relative errors below 1e-2 unless tighter is possible |
| Advanced workflows | NSTX-U/CUTE/DIII-D scripts | `gui`, `time`, `optimization` | examples run and generate docs assets |

### Tolerances

Initial parity tolerances:

- analytic manufactured tests: match or improve OFT reference error trends.
- diagnostic scalar parity: start at 1e-2 relative, tighten when numerical method is equivalent.
- arrays in EQDSK/i-file: 1e-2 relative norm, then tighten case by case.
- Green's functions: 1e-8 relative where formulas are smooth, relaxed near singular cases with documented handling.
- gradients: 1e-4 relative agreement with finite differences for smooth small cases.

## Performance Plan

Metrics to track:

- first-call compile time.
- steady-state solve time.
- nonlinear iterations.
- linear iterations.
- peak memory.
- gradient time relative to primal solve.
- CPU versus GPU behavior.
- scaling with nodes, elements, order, and active coils.

Benchmark cases:

- small Solov'ev p=1/p=2 for fast CI smoke.
- medium fixed-boundary p=2 for PR performance trend.
- ITER free-boundary p=2 for release benchmark.
- LTX conductor/stability for wall matrix benchmark.
- CUTE VDE time trajectory for scan/checkpointing benchmark.
- gradient benchmark for coil-current shape objective.

Optimization tactics:

- Keep connectivity static under JIT.
- Cache basis/quadrature and element maps by mesh signature.
- Use `vmap` for element kernels.
- Compare BCOO assembly against matrix-free apply.
- Use implicit VJPs for large solves.
- Consider checkpointing time scans.
- Keep GUI runs smaller by default, with explicit high-fidelity mode.

## Implementation Milestones

### M0: Bootstrap and Planning

Status: complete except ongoing plan/log maintenance.

Deliverables:

- public repo, docs, CI, coverage, README assets, seed solver, TOML CLI, GUI entry point, plan/log.

Acceptance:

- CI and docs green.
- local branch clean and pushed.

### M1: Mesh Import and Geometry Model

Deliverables:

- `geometry.py`, `mesh.py`, HDF5 mesh import/export, region dataclasses, mesh plotting.
- TOML schema for mesh source and region definitions.
- Import tests for upstream example meshes.

Acceptance:

- Import ITER, HBT, DIII-D, LTX, CUTE, MANTA, Dipole, NSTX-U meshes.
- Preserve node/cell counts, region ids, coil dictionaries, conductor dictionaries.
- Mesh preview plot generated for at least three machines.

### M2: FEM Basis and Operators

Deliverables:

- triangular basis orders 1-4.
- quadrature rules.
- mass/stiffness/Grad-Shafranov local assembly.
- BCOO global assembly and matrix-free apply.

Acceptance:

- partition-of-unity and derivative tests pass.
- manufactured operator tests converge.
- sparse and matrix-free apply agree.

### M3: Profiles and Fixed-Boundary Static Solve

Deliverables:

- full profile hierarchy.
- fixed-boundary solve on triangular mesh.
- diagnostic basics: psi normalization, O-point, simple stats.

Acceptance:

- Solov'ev tests pass for p=2, p=3, p=4.
- spheromak tests pass.
- gradients through profile coefficients pass finite-difference checks.

### M4: Coils and Free-Boundary Static Solve

Deliverables:

- Green's functions.
- coil source/mutual/self inductance.
- coil sets, bounds, regularization.
- free-boundary boundary matrix and inverse solve.

Acceptance:

- coil mutual tests pass.
- ITER and LTX baseline free-boundary examples pass scalar parity.
- simple GUI free-boundary example works.

### M5: Diagnostics, IO, and Plotting Parity

Deliverables:

- q profile, global quantities, beta/li, flux-surface tracing, O/X points, LCFS.
- EQDSK/i-file read/write.
- publication plotting and animations.

Acceptance:

- ITER EQDSK/i-file tests pass.
- plotting examples generate README/docs assets.
- docs include validation tables.

### M6: Reconstruction

Deliverables:

- diagnostic residual functions.
- reconstruction optimizer.
- fit import/export if needed.
- reconstruction tutorial.

Acceptance:

- ITER synthetic reconstruction parity passes.
- gradients through diagnostic residuals pass finite-difference checks.

### M7: Conductors, Wall Modes, and Stability

Deliverables:

- passive conductor matrices and sources.
- wall eigenmodes.
- linear stability eigenmodes.
- eddy-current plotting.

Acceptance:

- ITER and LTX wall/stability tests pass.
- conductor current diagnostics match OFT.

### M8: Time-Dependent and Pulse Workflows

Deliverables:

- `setup_td`/`step_td` equivalent.
- time scan API.
- CUTE VDE and pulse design examples.
- animation export.

Acceptance:

- representative VDE/pulse examples run from TOML and GUI.
- gradients through short fixed-topology trajectories pass checks.

### M9: Bootstrap and Advanced Workflows

Deliverables:

- Sauter/Redl bootstrap paths.
- ITER H-mode/bootstrap example.
- NSTX-U isoflux controller example.
- DIII-D/CUTE pulse design examples.

Acceptance:

- bootstrap and Redl tests pass.
- advanced workflows generate docs artifacts.

### M10: Release Hardening

Deliverables:

- full docs.
- benchmark report.
- API stability review.
- release checklist.
- first alpha tag and PyPI package if desired.

Acceptance:

- all CI, docs, parity, examples, and benchmarks pass.
- no known blocker issues for alpha users.

## Immediate Next Implementation Queue

Recommended next PRs:

1. Add `mesh.py` with `TriMesh`, HDF5 loader, region/coils/conductors containers, and tests importing `ITER_mesh.h5` copied from OFT fixtures.
2. Add `geometry.py` with `Region`, rectangle/polygon/annulus builders, and TOML schema validation.
3. Add p=1 triangular basis and quadrature tests.
4. Add p=1 mass/stiffness assembly with manufactured-solution tests.
5. Add `profiles.py` full flat/linterp functions and derivative tests against OFT samples.
6. Add `greens.py` axisymmetric Green's function and gradient with OFT/brute-force tests.
7. Add `docs/developer_porting.md` with a porting checklist and source-to-module map.
8. Add `tokamaker-jax validate` command and schema error reporting.

## Issue Backlog

Suggested GitHub issue labels:

- `physics-parity`
- `differentiability`
- `mesh-fem`
- `free-boundary`
- `gui`
- `docs`
- `testing`
- `performance`
- `compatibility`
- `good-first-issue`

Suggested initial issues:

- `M1-001`: Implement `TriMesh` dataclass and HDF5 loader.
- `M1-002`: Add TokaMaker example mesh fixtures with size policy.
- `M1-003`: Implement mesh plotting overlay by region type.
- `M2-001`: Implement triangular reference elements p=1.
- `M2-002`: Implement quadrature and partition-of-unity tests.
- `M2-003`: Implement BCOO mass matrix assembly.
- `M3-001`: Port flat and linterp flux profiles.
- `M3-002`: Add Solov'ev p=1 manufactured solve.
- `M4-001`: Port axisymmetric Green's function.
- `M4-002`: Implement coil dict/vector mapping.
- `M5-001`: Implement EQDSK reader/writer.
- `GUI-001`: Replace seed GUI with case browser and TOML editor.
- `DOC-001`: Add validation report page.

## Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| FEM parity takes longer than expected | delays all physics milestones | start p=1 with manufactured tests, then raise order |
| Free-boundary topology is nonsmooth | invalid gradients near events | document topology boundaries and provide smooth surrogate losses |
| Sparse solve gradients are memory-heavy | unusable optimization workflows | implement implicit VJPs and compare with unrolled small cases |
| JAX compile times are large | poor GUI/interactive UX | cache compiled functions, use smaller GUI defaults, separate preview/high-fidelity modes |
| OFT behavior is hard to infer from Fortran state | parity mismatches | port one regression family at a time and write comparison harnesses |
| Optional mesh/GUI dependencies become fragile | install pain | keep optional extras and a lightweight core |
| License conflicts from reference libraries | legal/redistribution issue | do not vendor GPL code; keep references conceptual unless license-compatible |
| GitHub assets become too large | slow clones | generate large movies as release artifacts, keep repo assets small |

## Open Questions

- Should the project remain `LGPL-3.0-only` to mirror OFT, or use `LGPL-3.0-or-later`?
- Should p=1 be exposed as a supported low-order option, or only used internally during development?
- Which production sparse solver backend should be default: JAX BCOO iterative, matrix-free Krylov, optional PETSc, optional GPU direct solvers, or a hybrid?
- How much old `TokaMaker` API compatibility should be guaranteed after the first alpha?
- Should mesh generation use Triangle, Gmsh, Shapely, or a minimal internal builder for common cases?
- Should GUI remain NiceGUI after the full workflow is clearer, or move to a different web stack?
- What should be the first public validation target: fixed-boundary only, ITER free-boundary, or a broader alpha bundle?

