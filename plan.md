# tokamaker-jax Porting Plan and Log

This file is both the technical plan and the running implementation log for a JAX-native port of TokaMaker from OpenFUSIONToolkit.

## Log

### 2026-05-12 15:45 WEST

- Found an existing `/Users/rogeriojorge/local/OpenFUSIONToolkit` checkout and updated it to upstream `main` commit `729a5f9`.
- Confirmed `gh` is authenticated as `rogeriojorge` with repository and workflow scopes.
- Audited TokaMaker source, Python API, docs, examples, and tests.
- Created `/Users/rogeriojorge/local/tokamaker_jax` with package scaffold, TOML CLI, optional GUI entry point, docs, Read the Docs config, GitHub workflows, tests, README badges, citation metadata, and a differentiable JAX fixed-boundary seed solver.
- The seed solver is not the full port. It is the executable base used to validate project packaging, differentiability, CLI/GUI shape, plotting, and coverage infrastructure while the full triangular FEM port proceeds.
- Published public repository: <https://github.com/rogeriojorge/tokamaker_jax>.
- Validation performed: Ruff formatting/linting passed, `pytest --cov=tokamaker_jax --cov-fail-under=95` passed with 98.66% coverage, and `sphinx -W -b html docs docs/_build/html` passed.
- Initial GitHub Actions CI and Docs runs succeeded. Added Node 24 opt-in to workflows to avoid the upcoming GitHub Actions Node 20 deprecation.

## Porting Goal

Build `tokamaker-jax` as a public repository at `github.com/rogeriojorge/tokamaker_jax`:

- Fully JAX-native and end-to-end differentiable for static solves, reconstruction objectives, coil/shape optimization, and time-dependent workflows.
- Feature parity with TokaMaker's fixed-boundary, free-boundary, reconstruction, passive conductor, wall-mode, and time-dependent Grad-Shafranov capabilities.
- Python 3.10+ support, using `tomli` on Python 3.10 and standard `tomllib` on Python 3.11+.
- GUI-first CLI: `tokamaker-jax` launches the GUI, while `tokamaker-jax case.toml` runs a reproducible TOML case.
- High-quality examples, plotting, generated images/movies, and comprehensive Read the Docs documentation.
- At least 95% test coverage enforced in CI.
- Performance suitable for optimization loops through `jit`, `vmap`, sparse JAX operators, accelerator support, and implicit differentiation where appropriate.

## Source Audit

Upstream source: `/Users/rogeriojorge/local/OpenFUSIONToolkit`, upstream `main` commit `729a5f9`, latest release observed online as `v1.0.0-beta7`.

Core TokaMaker source footprint:

- `src/physics/grad_shaf.F90` (~6215 lines): main equilibrium factory and solver. Includes setup, wall/conductor handling, coil sources, plasma mutuals, isoflux fitting, nonlinear solve, linear solve, vacuum solve, source assembly, O/X point and LCFS updates, q profiles, field interpolation, MUG export, boundary condition matrix, and Grad-Shafranov matrix construction.
- `src/physics/grad_shaf_profiles.F90` (~1198 lines): flux-function hierarchy for zero, flat, polynomial, spline, linear interpolation, Wesson, non-inductive profile forms, derivatives, coefficient updates, and save/load.
- `src/physics/grad_shaf_td.F90` (~1410 lines): time-dependent stepping, passive conductor/current diffusion operators, linearized stability/eigenvalue operators, wall-mode matrices.
- `src/physics/grad_shaf_fit.F90` (~1714 lines): reconstruction and fitting machinery.
- `src/physics/grad_shaf_util.F90` (~1420 lines): global quantities, loop voltage, save/load, EQDSK/i-file export, Sauter bootstrap helpers.
- `src/physics/axi_green.F90` (~286 lines): axisymmetric Green's functions and derivatives.
- `src/physics/gs_eq.F90` (~174 lines): equilibrium interpolation.
- `src/python/wrappers/tokamaker_f.F90`: C/Python wrapper layer.
- `src/python/OpenFUSIONToolkit/TokaMaker/_core.py`: public Python API. Important methods include `setup_mesh`, `setup_regions`, `setup`, `init_psi`, `set_profiles`, `solve`, `vac_solve`, `get_stats`, `set_isoflux`, `set_flux`, `set_saddles`, `set_targets`, `get_psi`, `get_q`, `get_profiles`, `get_xpoints`, `set_coil_currents`, plotting, EQDSK/i-file/MUG save, wall modes, linear stability, `setup_td`, and `step_td`.
- `src/python/OpenFUSIONToolkit/TokaMaker/meshing.py`: `gs_Domain`, region definitions, rectangle/polygon/annulus/enclosed geometry, Triangle/Cubit meshing, HDF5 mesh IO, plotting.
- `src/python/OpenFUSIONToolkit/TokaMaker/reconstruction.py`: Ip, flux-loop, Mirnov, pressure, q, saddle, and dFlux constraints.
- `src/python/OpenFUSIONToolkit/TokaMaker/util.py`: isoflux creation, flux functions, EQDSK/i-file readers, Green's evaluation, MHDIN/K-file helpers.
- `src/examples/TokaMaker`: fixed-boundary, ITER, HBT, DIII-D, LTX, CUTE notebooks and mesh/geometry files.
- `src/tests/physics/test_TokaMaker.py`: regression coverage for Solov'ev, spheromak, coil mutuals, ITER/LTX equilibria, reconstruction, concurrency, wall modes, time-dependent/stability, bootstrap current, Redl current, EQDSK and i-file IO.

## External References Reviewed

- OpenFUSIONToolkit documentation describes TokaMaker as a time-dependent free-boundary Grad-Shafranov equilibrium code and OFT as a high-order FEM framework for unstructured triangular/tetrahedral and quadrilateral/hexahedral meshes: <https://openfusiontoolkit.github.io/OpenFUSIONToolkit/>.
- OpenFUSIONToolkit GitHub README identifies TokaMaker as the axisymmetric static and time-dependent ideal MHD equilibrium component and the project license as LGPL-3.0: <https://github.com/OpenFUSIONToolkit/OpenFUSIONToolkit>.
- Hansen et al. describe TokaMaker as a static and time-dependent MHD equilibrium code based on the Grad-Shafranov equation, using finite elements on unstructured triangular grids, Python/Fortran/C/C++, novel free-boundary conditions, and analytic/cross-code validation: <https://arxiv.org/abs/2311.07719> and <https://doi.org/10.1016/j.cpc.2024.109111>.
- PPPL SULI 2024 TokaMaker workshop emphasizes FEM Grad-Shafranov solves, toroidal coil current sources, and fixed/free-boundary exercises: <https://suli.pppl.gov/2024/course/CJH-TokaMaker_SULI24.pdf>.
- FreeGS provides a Python free-boundary Grad-Shafranov reference with coil constraints, profiles, X-points, Picard iteration, direct/multigrid linear solves, plotting, EQDSK IO, and easy examples: <https://github.com/freegs-plasma/freegs> and <https://freegs.readthedocs.io/en/stable/creating_equilibria.html>.
- FreeGSNKE extends FreeGS toward static and evolutive free-boundary Newton-Krylov workflows: <https://github.com/FusionComputingLab/freegsnke>.
- JAX-FEM demonstrates differentiable FEM, GPU acceleration, and automatic inverse design patterns relevant to this port: <https://github.com/deepmodeling/jax-fem> and <https://arxiv.org/abs/2212.00964>.
- TORAX demonstrates modern JAX practice in tokamak simulation, including JAX autodiff, compilation, gradient-based nonlinear PDE solvers, sensitivities, and coupling to ML surrogates: <https://github.com/google-deepmind/torax>.

## Architecture Plan

### Public API

Target user-facing API:

```python
from tokamaker_jax import Machine, TokaMaker, load_case

case = load_case("iter_baseline.toml")
eq = TokaMaker(case).solve()
grads = eq.grad(lambda result: result.beta_p, wrt=["coil_currents", "profiles"])
eq.plot().save("iter.png")
eq.save_eqdsk("iter.g")
```

Design rules:

- Pure dataclasses or Equinox-style pytrees for machine, mesh, profiles, solver state, constraints, and outputs.
- No hidden Fortran-style mutable global state.
- Every numerical kernel accepts arrays and static metadata, returns arrays or pytrees, and can be composed under `jit`, `grad`, `vmap`, and `lax.scan`.
- Separate compatibility layer from new ergonomic layer. Preserve familiar names where helpful, but expose a simpler TOML/Python workflow.

### Modules

- `config`: typed TOML schemas, validation, defaults, versioned config migration.
- `geometry`: polygons, rectangles, annuli, enclosed regions, limiter and wall descriptions.
- `mesh`: differentiable-friendly triangular mesh containers; Triangle/Gmsh import; static mesh generation outside JAX; mesh quality diagnostics.
- `fem`: Lagrange basis p=1 through p=4, quadrature, element maps, gather/scatter, BCOO sparse assembly.
- `operator`: Grad-Shafranov operator, mass/current operators, vacuum/free-boundary boundary matrix, coil mutuals, conductor sources.
- `profiles`: flat, linterp, spline, polynomial, Wesson, Sauter/Redl bootstrap and non-inductive profiles.
- `solver`: fixed-boundary, free-boundary, vacuum, Picard, Newton, Newton-Krylov, optimization-based solve, implicit differentiation.
- `constraints`: isoflux, flux, saddle/X-point, targets, coil bounds, coil regularization, reconstruction diagnostics.
- `time`: passive conductor, wall modes, linear stability, time-dependent stepping.
- `io`: EQDSK, i-file, HDF5 mesh/state, MUG export, TokaMaker compatibility save/load.
- `plotting`: Matplotlib and Plotly visualizations, publication defaults, animations.
- `gui`: NiceGUI or equivalent interactive app for machine editing, TOML import/export, solve monitoring, plots, and optimization sliders.
- `cli`: GUI default, TOML execution, batch sweeps, benchmark commands, docs/example generation.

## Differentiability Strategy

- Use JAX arrays for all differentiable state: flux, profile coefficients, coil currents, conductor currents, target locations, and diagnostic weights.
- Treat mesh connectivity and polynomial order as static JIT metadata.
- Use `jax.experimental.sparse.BCOO` initially for sparse FEM matrices; benchmark against matrix-free element application.
- For linear solves:
  - Start with differentiating through iterative solves for small cases.
  - Add custom VJP/implicit differentiation for production sparse solves.
  - Expose both exact unrolled gradients and implicit gradients for verification.
- For nonlinear solves:
  - Implement Picard parity first.
  - Add Newton/Newton-Krylov with JAX Jacobian-vector products.
  - Use `jaxopt`-style implicit differentiation patterns where they improve compile time and memory.
- For free-boundary topology operations:
  - Keep hard LCFS/X-point detection outside gradient paths at first.
  - Provide smooth surrogate objectives for shape optimization.
  - Document nondifferentiable events such as topology changes, limiter transitions, and separatrix bifurcations.

## Performance Plan

- Static-shape compiled kernels for each mesh/order/case.
- Vectorized quadrature and element assembly with `vmap`.
- BCOO or matrix-free operators depending on benchmark result.
- Cache compiled solve functions by mesh signature, order, dtype, and active physics.
- Support float64 by default for verification; allow float32/mixed precision for exploratory GUI runs.
- Benchmarks:
  - Analytic Solov'ev convergence versus OFT orders 2, 3, 4.
  - Coil mutual Green's function accuracy and speed.
  - ITER and LTX parity cases versus OFT.
  - CPU and GPU solve/gradient timing.
  - Memory usage for unrolled versus implicit gradients.

## GUI Plan

The GUI should be a real workflow, not just a plot wrapper:

- First screen: machine/case editor with mesh preview, profile controls, coil currents, targets, constraints, and solve button.
- TOML import/export always visible.
- Live convergence plot, equilibrium contours, machine geometry, LCFS/X-points, coil currents, profile plots, q profile, and diagnostic residuals.
- Examples browser for fixed boundary, ITER-like free boundary, reconstruction, VDE/time-dependent, and differentiable optimization.
- Export buttons for images, movies, EQDSK, i-file, HDF5 state, and reproducible TOML.
- Use Plotly for interactive contours and sliders; Matplotlib for publication static figures.
- Generate README/docs assets from tested scripts so screenshots and movies never drift from code.

## CLI/TOML Plan

CLI behavior:

- `tokamaker-jax`: launch GUI.
- `tokamaker-jax case.toml`: run a case from TOML.
- `tokamaker-jax case.toml --plot out.png --output out.h5`: noninteractive artifacts.
- `tokamaker-jax validate case.toml`: schema and mesh checks.
- `tokamaker-jax benchmark case.toml`: timing and memory report.
- `tokamaker-jax examples --write docs/_static`: regenerate images/movies for docs and README.

TOML must remain Python 3.10 compatible by depending on `tomli` only when needed.

## Validation and Coverage Plan

Coverage policy:

- CI enforces `pytest --cov=tokamaker_jax --cov-fail-under=95`.
- GUI code may have separate browser/screenshot tests and can be omitted from unit coverage until stable, but core solver/API code may not be omitted.
- Every ported source feature gets a unit test, regression test, or notebook smoke test.

Regression matrix:

- Analytic fixed-boundary Solov'ev convergence at orders 2, 3, 4.
- Spheromak/dipole analytic-style cases.
- Coil Green's function and mutual inductance parity.
- Free-boundary ITER baseline and H-mode examples.
- HBT, DIII-D, LTX, CUTE, MANTA cases.
- EQDSK and i-file round-trips.
- Reconstruction from synthetic Ip, flux loop, Mirnov, pressure, q, dFlux, and saddle constraints.
- Passive conductor and wall-mode eigenvalue parity.
- Time-dependent step parity for representative VDE/passive cases.
- Differentiability tests for profile coefficients, coil currents, target locations, and regularization weights.
- JIT/vmap tests for parameter sweeps.

## Implementation Phases

### Phase 0: Repository Bootstrap

Status: started.

- [x] Clone/update OpenFUSIONToolkit in `/Users/rogeriojorge/local`.
- [x] Create `/Users/rogeriojorge/local/tokamaker_jax`.
- [x] Add package scaffold, README badges, docs, Read the Docs config, CI/CD workflows.
- [x] Add a differentiable JAX fixed-boundary seed solver to validate infrastructure.
- [x] Add TOML loader, CLI, optional GUI entry point, plotting, tests.
- [x] Publish public GitHub repo.

### Phase 1: Mesh and FEM Core

- [ ] Port `gs_Domain` concepts into a typed geometry model.
- [ ] Support rectangle, polygon, annulus, enclosed regions, limiter metadata, coil/conductor region dictionaries.
- [ ] Import TokaMaker HDF5 meshes and region definitions.
- [ ] Implement triangular Lagrange basis orders 1-4.
- [ ] Implement quadrature and local-to-global sparse assembly.
- [ ] Verify mass/stiffness operators against manufactured solutions.

### Phase 2: Static Fixed-Boundary Solver

- [ ] Port flux-function profile hierarchy.
- [ ] Assemble Grad-Shafranov `Delta*` FEM operator.
- [ ] Implement fixed-boundary solve with TokaMaker-compatible settings.
- [ ] Match Solov'ev and spheromak regression tests.
- [ ] Implement `get_psi`, `set_psi`, `get_stats`, `get_profiles`, `get_q`, `trace_surf`, `get_xpoints`.

### Phase 3: Free-Boundary, Coils, and Conductors

- [ ] Port axisymmetric Green's functions and gradients.
- [ ] Implement coil source, mutual inductance, distributed coils, coil sets, virtual coils, voltage coils.
- [ ] Implement vacuum/free-boundary boundary condition matrix.
- [ ] Implement isoflux, flux, saddle, coil bounds, coil regularization, and target constraints.
- [ ] Match ITER/HBT/DIII-D/LTX/CUTE/MANTA notebook cases.

### Phase 4: Reconstruction and IO

- [ ] Port reconstruction constraints and fitting objectives.
- [ ] Implement differentiable synthetic diagnostics.
- [ ] Implement EQDSK/i-file read/write and parity tests.
- [ ] Implement HDF5 state save/load and TokaMaker compatibility import.
- [ ] Add reconstruction examples and docs.

### Phase 5: Time-Dependent and Wall Physics

- [ ] Port passive conductor source terms and resistivity profiles.
- [ ] Implement wall-mode eigenvalue and linear stability workflows.
- [ ] Implement `setup_td`, `step_td`, and differentiable time scans.
- [ ] Add VDE/time-dependent examples and movies.

### Phase 6: UX, Docs, and Examples

- [ ] Build full GUI workflow.
- [ ] Add simple examples for all major use cases.
- [ ] Generate README images and docs movies from reproducible scripts.
- [ ] Add tutorial notebooks and TOML case gallery.
- [ ] Publish full Read the Docs documentation with API, equations, validation, and performance pages.

### Phase 7: Release Hardening

- [ ] Complete 95%+ coverage across core modules.
- [ ] Add pre-commit and release checklists.
- [ ] Add benchmark suite and performance dashboard.
- [ ] Validate Python 3.10 through 3.13.
- [ ] Tag first alpha release and publish wheels.

## Risks and Mitigations

- Free-boundary topology changes are not smoothly differentiable. Mitigation: expose smooth surrogate objectives and document event boundaries.
- Sparse linear solve differentiation can be memory-heavy. Mitigation: use implicit custom VJPs and benchmark unrolled solves only for small verification cases.
- JAX compilation on large unstructured meshes can be slow. Mitigation: static metadata, shape-stable kernels, cached compiled functions, and matrix-free operator paths.
- Exact TokaMaker parity may require careful interpretation of Fortran-side state and boundary condition conventions. Mitigation: port one regression family at a time and compare against OFT outputs.
- GUI dependencies can be heavy. Mitigation: keep GUI optional under the `gui` extra; core CLI/TOML remains lightweight.

## Open Questions

- Whether to keep LGPL-3.0-only or use LGPL-3.0-or-later for the new port. The current scaffold uses LGPL-3.0-only to match OFT headers.
- Which sparse solver backend should be default for production: JAX BCOO iterative, jaxopt implicit solves, PETSc-through-callback for non-differentiable reference mode, or optional GPU direct solvers.
- Whether the first full FEM port should directly implement p=2 default parity or start with p=1 for easier operator verification.
- Whether GUI should remain NiceGUI or move to a notebook-native app once the workflow is clearer.
