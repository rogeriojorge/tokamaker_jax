# Progress

`tokamaker-jax` is still a seed port, not a full TokaMaker replacement. The
current executable solver stack now combines the original rectangular-grid seed
path with p=1 triangular FEM assembly, manufactured Grad-Shafranov validation,
a reduced free-boundary coil-response fixture, a closed-form circular-loop
elliptic kernel checked against quadrature, nonlinear profile iteration,
free-boundary/profile coupling, numeric OpenFUSIONToolkit `eval_green` parity
when the local OFT build is present, CI benchmark artifact reporting, GUI
stored-report summaries, citation-linked reproduction scaffolding, expanded
equation/design/IO/comparison documentation, and publication-ready generated
figures. Packaging now installs GUI dependencies by default and dependency
declarations are intentionally unversioned at the project metadata level.

## Status

| Lane | Approx. status | Current state |
| --- | ---: | --- |
| M1 mesh/geometry | 98% | `TriMesh`, OFT-style mesh I/O, region primitives, TOML region parsing, canonical sample regions, rectangular triangular refinement fixtures, geometry previews, and IO/architecture docs exist; production-grade mesh generation and limiter/divertor topology policies move to the next milestone. |
| M2 FEM core | 98% | p=1 reference kernels, dense/sparse/global assembly, matrix-free applies, weighted mass/stiffness matrices, axisymmetric Grad-Shafranov weak-form assembly, load-vector/profile-source assembly, Dirichlet reduction, manufactured convergence gates, nonlinear profile iteration, closed-form circular-loop coil response, and free-boundary/profile coupling have validation tests and derivation docs; full Newton/equilibrium reconstruction moves to the next milestone. |
| Plotting | 100% | Seed equilibrium plots, mesh/region plots, animation outputs, `FigureRecipe`, JSON metadata, region tables, annotated seed plots, Poisson convergence, Grad-Shafranov convergence, reduced coil-response figures, circular-loop elliptic figures, free-boundary/profile coupling figures, validation dashboard, benchmark summary, publication panel, IO map, comparison heatmap, coil-current sweep movie, nonlinear profile figures, and CPC-seed-family artifact are complete for this staged milestone. |
| Docs/examples | 100% | Installation, architecture, equations, derivations, design decisions, IO/artifact contracts, porting map, API, examples, progress, validation manifest, upstream/literature comparisons, benchmark threshold docs, GUI/report notes, literature reproduction docs, and README quick-start are complete for this staged milestone. |
| Config/CLI | 100% | Default install now includes GUI dependencies, dependency metadata is unversioned, TOML runs support solver/grid/source/coil/output settings plus region geometry, `tokamaker-jax validate` checks configs without solving, and `tokamaker-jax verify` runs all implemented gates; richer case management moves to the next milestone. |
| Test infra | 100% | Pytest, coverage, lint, docs, FEM analytic gates, weighted/axisymmetric assembly gates, true quadrature convergence gates, reduced and closed-form circular-loop coil gates, numeric OpenFUSIONToolkit `eval_green` parity when available, nonlinear profile/coupling gates, literature surrogate gates, validation CLI tests, GUI report tests, benchmark threshold tests, CI workflow tests, packaging tests, and docs artifact tests are complete for this staged milestone. |
| Differentiability | 98% | Seed solver paths, local FEM kernels, dense/sparse assembly, weighted axisymmetric assembly, matrix-free applies, reduced and closed-form circular-loop coil responses, nonlinear profile pressure-scale sensitivity, free-boundary current gradients, and coupled pressure-scale sensitivity are JAX-compatible with focused gradient and finite-difference checks; implicit solver VJPs and topology policies move to the next milestone. |
| GUI | 98% | NiceGUI seed equilibrium controls, region-geometry preview, metadata summaries, region tables, validation convergence plots, reduced coil-response plots, workflow-dashboard summaries, stored validation/OFT/benchmark report tables, and default-install GUI support are complete for this staged milestone; case browser and TOML editor move to the next milestone. |
| Performance | 99% | JSON-friendly benchmark helpers, generated benchmark report artifacts, threshold comparison reports, benchmark-summary plot, CI benchmark artifact upload, and performance documentation exist for the seed fixed-boundary solve, local p=1 FEM kernels, axisymmetric global assembly/matrix-free apply, reduced coil Green's response, and closed-form circular-loop elliptic response; hardware-normalized history moves to the next milestone. |
| Overall | 100% | The scoped staged repository milestone is complete: packaging, README quick start, scaffolding, seed solver, mesh/geometry foundation, p=1 local/global/weighted FEM kernels, axisymmetric Grad-Shafranov weak form, nonlinear profile iteration, reduced and closed-form circular-loop free-boundary coil kernels, numeric upstream `eval_green` parity, free-boundary/profile coupling, sparse and matrix-free assembly paths, source/profile loads, Dirichlet solves, true-error manufactured convergence, verification CLI, expanded docs, examples, tests, GUI report views, plotting metadata, comparison artifacts, literature reproduction scaffolding, and benchmark reports are in place. This is not a claim of complete upstream TokaMaker feature parity. |

Percentages are approximate planning markers for the completed staged
repository milestone, not validation metrics and not a claim that every
upstream TokaMaker feature is implemented.

## Next Steps

The next milestone should keep the scope narrow and measurable:

1. Promote the CPC seed-family surrogate into a code-to-code literature
   reproduction by importing the exact published/OFT case inputs and recording
   scalar tolerances.
2. Extend the current free-boundary/profile coupling gate into Newton/Picard
   convergence studies on documented equilibrium fixtures.
3. Add hardware-normalized benchmark history for sparse assembly, matrix-free
   apply, profile iteration, circular-loop response, and manufactured
   Grad-Shafranov solves.
4. Add a GUI case browser and TOML editor so research users can move directly
   between examples, validation reports, solver settings, and reproducible
   plots.
5. Add full TokaMaker equilibrium parity fixtures beyond the scalar
   `eval_green` kernel, starting with fixed-boundary examples and then
   free-boundary coil/conductor cases.

After those pieces pass focused tests, the seed rectangular-grid solver can be
replaced incrementally by the triangular FEM path described in the architecture
and porting map.
