# Progress

`tokamaker-jax` is still a seed port, not a full TokaMaker replacement. The
current executable solver is a small fixed-boundary rectangular-grid
implementation used to validate project structure, JAX differentiation,
configuration loading, plotting, examples, and CI behavior before the
TokaMaker-equivalent triangular FEM stack is ported.

## Status

| Lane | Approx. status | Current state |
| --- | ---: | --- |
| M1 mesh/geometry | 80% | `TriMesh`, OFT-style mesh I/O, region primitives, TOML region parsing, canonical sample regions, rectangular triangular refinement fixtures, and geometry previews exist; region-aware production mesh generation remains open. |
| M2 FEM core | 62% | p=1 reference kernels, dense/sparse/global assembly, matrix-free applies, weighted mass/stiffness matrices, axisymmetric Grad-Shafranov weak-form assembly, load-vector/profile-source assembly, Dirichlet reduction, and manufactured convergence gates now have validation tests; nonlinear free-boundary coupling remains open. |
| Plotting | 62% | Seed equilibrium plots, mesh/region plots, animation outputs, `FigureRecipe`, JSON metadata, region tables, annotated seed plots, Poisson convergence, Grad-Shafranov convergence, and reduced coil-response figures exist; literature reproduction galleries remain pending. |
| Docs/examples | 64% | Installation, architecture, porting map, API, examples, progress, validation equations, validation manifest, Poisson/Grad-Shafranov derivations, reduced coil Green's-function derivation, and generated artifacts are present; full derivation atlas and broad tutorials remain pending. |
| Config/CLI | 56% | TOML runs support solver/grid/source/coil/output settings plus region geometry, `tokamaker-jax validate` checks configs without solving, and `tokamaker-jax verify` runs manufactured and reduced coil Green's-function gates; production schema export and richer case management are pending. |
| Test infra | 68% | Pytest, coverage, lint, docs, FEM analytic gates, weighted/axisymmetric assembly gates, true quadrature error convergence gates, reduced coil Green's-function gates, validation CLI tests, GUI metadata tests, and benchmark schema tests exist; OpenFUSIONToolkit parity and literature figure gates remain pending. |
| Differentiability | 45% | Seed solver paths, local FEM kernels, dense/sparse assembly, weighted axisymmetric assembly, matrix-free applies, and reduced coil Green's-function responses are JAX-compatible with focused gradient and finite-difference checks; implicit solver VJPs, topology policies, and optimization workflows remain open. |
| GUI | 42% | NiceGUI seed equilibrium controls, region-geometry preview, metadata summaries, region tables, validation convergence plots, and reduced coil-response plots exist; case browser, TOML editor, literature gallery, and solver workflow dashboards remain pending. |
| Performance | 38% | JSON-friendly benchmark helpers exist for the seed fixed-boundary solve, local p=1 FEM kernels, axisymmetric global assembly/matrix-free apply, and reduced coil Green's response; CI baselines and full free-boundary benchmarks remain pending. |
| Overall | 58% | Repository scaffolding, seed solver, mesh/geometry foundation, p=1 local/global/weighted FEM kernels, axisymmetric Grad-Shafranov weak form, reduced free-boundary coil Green's-function fixture, sparse and matrix-free assembly paths, source/profile loads, Dirichlet solves, true-error manufactured convergence, verification CLI, docs, examples, tests, GUI previews, plotting metadata, and benchmark helpers are in place; the project is still not a full TokaMaker replacement. |

Percentages are approximate planning markers, not validation metrics.

## Next Steps

The next implementation milestone should keep the scope narrow and measurable:

1. Add p=1 nonlinear profile/source iteration around pressure and FF' terms,
   using the implemented profile-load assembly and manufactured gates as
   oracles.
2. Replace the reduced coil Green's-function fixture with a full circular-loop
   elliptic-kernel path and compare it against OpenFUSIONToolkit fixtures.
3. Add the first literature-anchored TokaMaker figure reproduction script and
   artifact with a manifest entry, command, tolerance, and generated docs output.
4. Add CI-recorded benchmark baselines for sparse assembly, matrix-free apply,
   reduced/full coil response, and the manufactured Grad-Shafranov solve.

After those pieces pass focused tests, the seed rectangular-grid solver can be
replaced incrementally by the triangular FEM path described in the architecture
and porting map.
