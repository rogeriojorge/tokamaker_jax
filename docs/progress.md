# Progress

`tokamaker-jax` is still a seed port, not a full TokaMaker replacement. The
current executable solver stack now combines the original rectangular-grid seed
path with p=1 triangular FEM assembly, manufactured Grad-Shafranov validation,
a reduced free-boundary coil-response fixture, a closed-form circular-loop
elliptic kernel checked against quadrature, nonlinear profile iteration,
free-boundary/profile coupling, numeric OpenFUSIONToolkit `eval_green` parity
when the local OFT build is present, CI benchmark artifact reporting, GUI
stored-report summaries, and citation-linked reproduction scaffolding.

## Status

| Lane | Approx. status | Current state |
| --- | ---: | --- |
| M1 mesh/geometry | 86% | `TriMesh`, OFT-style mesh I/O, region primitives, TOML region parsing, canonical sample regions, rectangular triangular refinement fixtures, and geometry previews exist; region-aware production mesh generation and limiter/divertor topology policies remain open. |
| M2 FEM core | 84% | p=1 reference kernels, dense/sparse/global assembly, matrix-free applies, weighted mass/stiffness matrices, axisymmetric Grad-Shafranov weak-form assembly, load-vector/profile-source assembly, Dirichlet reduction, manufactured convergence gates, nonlinear profile iteration, closed-form circular-loop coil response, and a first free-boundary/profile coupling gate now have validation tests; full Newton/equilibrium reconstruction remains open. |
| Plotting | 89% | Seed equilibrium plots, mesh/region plots, animation outputs, `FigureRecipe`, JSON metadata, region tables, annotated seed plots, Poisson convergence, Grad-Shafranov convergence, reduced coil-response figures, circular-loop elliptic figures, free-boundary/profile coupling figures, validation dashboard, benchmark summary, coil-current sweep movie, nonlinear profile figures, and a CPC-seed-family reproduction artifact exist. |
| Docs/examples | 89% | Installation, architecture, porting map, API, examples, progress, validation equations, validation manifest, manufactured derivations, reduced/circular coil notes with AGM elliptic equations, nonlinear profile-iteration notes, free-boundary/profile coupling docs, benchmark threshold docs, upstream comparison probe docs, and literature reproduction docs are present; full derivation atlas and broad tutorials remain pending. |
| Config/CLI | 84% | TOML runs support solver/grid/source/coil/output settings plus region geometry, `tokamaker-jax validate` checks configs without solving, and `tokamaker-jax verify` runs manufactured, reduced coil Green's-function, circular-loop elliptic, OpenFUSIONToolkit parity, nonlinear profile-iteration, and free-boundary/profile coupling gates; production schema export and richer case management are pending. |
| Test infra | 90% | Pytest, coverage, lint, docs, FEM analytic gates, weighted/axisymmetric assembly gates, true quadrature convergence gates, reduced and closed-form circular-loop coil gates, numeric OpenFUSIONToolkit `eval_green` parity when available, nonlinear profile/coupling gates, literature surrogate gates, validation CLI tests, GUI report tests, and benchmark threshold tests exist; full OFT equilibrium parity remains pending. |
| Differentiability | 84% | Seed solver paths, local FEM kernels, dense/sparse assembly, weighted axisymmetric assembly, matrix-free applies, reduced and closed-form circular-loop coil responses, nonlinear profile pressure-scale sensitivity, free-boundary current gradients, and coupled pressure-scale sensitivity are JAX-compatible with focused gradient and finite-difference checks; implicit solver VJPs, topology policies, and optimization workflows remain open. |
| GUI | 82% | NiceGUI seed equilibrium controls, region-geometry preview, metadata summaries, region tables, validation convergence plots, reduced coil-response plots, workflow-dashboard summaries, and stored validation/OFT/benchmark report tables exist; case browser, TOML editor, and literature gallery controls remain pending. |
| Performance | 88% | JSON-friendly benchmark helpers, generated benchmark report artifacts, threshold comparison reports, benchmark-summary plot, and CI benchmark artifact upload exist for the seed fixed-boundary solve, local p=1 FEM kernels, axisymmetric global assembly/matrix-free apply, reduced coil Green's response, and closed-form circular-loop elliptic response; hardware-normalized baseline history and full free-boundary benchmarks remain pending. |
| Overall | 90% | Repository scaffolding, seed solver, mesh/geometry foundation, p=1 local/global/weighted FEM kernels, axisymmetric Grad-Shafranov weak form, nonlinear profile iteration, reduced and closed-form circular-loop free-boundary coil kernels, numeric upstream `eval_green` parity, free-boundary/profile coupling, sparse and matrix-free assembly paths, source/profile loads, Dirichlet solves, true-error manufactured convergence, verification CLI, docs, examples, tests, GUI report views, plotting metadata, literature reproduction scaffolding, and benchmark reports are in place; the project is still not a full TokaMaker replacement. |

Percentages are approximate planning markers, not validation metrics.

## Next Steps

The next implementation milestone should keep the scope narrow and measurable:

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
