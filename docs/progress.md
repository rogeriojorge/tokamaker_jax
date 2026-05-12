# Progress

`tokamaker-jax` is still a seed port, not a full TokaMaker replacement. The
current executable solver stack now combines the original rectangular-grid seed
path with p=1 triangular FEM assembly, manufactured Grad-Shafranov validation,
a reduced free-boundary coil-response fixture, a closed-form circular-loop
elliptic kernel checked against quadrature, nonlinear profile iteration,
benchmark reporting, upstream comparison probing, GUI workflow summaries, and
citation-linked reproduction scaffolding.

## Status

| Lane | Approx. status | Current state |
| --- | ---: | --- |
| M1 mesh/geometry | 84% | `TriMesh`, OFT-style mesh I/O, region primitives, TOML region parsing, canonical sample regions, rectangular triangular refinement fixtures, and geometry previews exist; region-aware production mesh generation and limiter/divertor topology policies remain open. |
| M2 FEM core | 76% | p=1 reference kernels, dense/sparse/global assembly, matrix-free applies, weighted mass/stiffness matrices, axisymmetric Grad-Shafranov weak-form assembly, load-vector/profile-source assembly, Dirichlet reduction, manufactured convergence gates, nonlinear profile iteration, and closed-form circular-loop coil response now have validation tests; full Newton/free-boundary coupling remains open. |
| Plotting | 82% | Seed equilibrium plots, mesh/region plots, animation outputs, `FigureRecipe`, JSON metadata, region tables, annotated seed plots, Poisson convergence, Grad-Shafranov convergence, reduced coil-response figures, circular-loop elliptic figures, validation dashboard, benchmark summary, coil-current sweep movie, nonlinear profile figures, and a CPC-seed-family reproduction artifact exist. |
| Docs/examples | 83% | Installation, architecture, porting map, API, examples, progress, validation equations, validation manifest, manufactured derivations, reduced/circular coil notes with AGM elliptic equations, nonlinear profile-iteration notes, benchmark reporting, upstream comparison probe docs, and literature reproduction docs are present; full derivation atlas and broad tutorials remain pending. |
| Config/CLI | 76% | TOML runs support solver/grid/source/coil/output settings plus region geometry, `tokamaker-jax validate` checks configs without solving, and `tokamaker-jax verify` runs manufactured, reduced coil Green's-function, circular-loop elliptic, OpenFUSIONToolkit parity-probe, and nonlinear profile-iteration gates; production schema export and richer case management are pending. |
| Test infra | 84% | Pytest, coverage, lint, docs, FEM analytic gates, weighted/axisymmetric assembly gates, true quadrature convergence gates, reduced and closed-form circular-loop coil gates, OpenFUSIONToolkit availability-gated comparison tests, nonlinear profile gates, literature surrogate gates, validation CLI tests, GUI workflow tests, and benchmark schema tests exist; full OFT equilibrium parity remains pending. |
| Differentiability | 73% | Seed solver paths, local FEM kernels, dense/sparse assembly, weighted axisymmetric assembly, matrix-free applies, reduced and closed-form circular-loop coil responses, and nonlinear profile pressure-scale sensitivity are JAX-compatible with focused gradient and finite-difference checks; implicit solver VJPs, topology policies, and optimization workflows remain open. |
| GUI | 68% | NiceGUI seed equilibrium controls, region-geometry preview, metadata summaries, region tables, validation convergence plots, reduced coil-response plots, and workflow-dashboard summaries exist; case browser, TOML editor, and literature gallery controls remain pending. |
| Performance | 72% | JSON-friendly benchmark helpers, generated benchmark report artifacts, and benchmark-summary plot exist for the seed fixed-boundary solve, local p=1 FEM kernels, axisymmetric global assembly/matrix-free apply, reduced coil Green's response, and closed-form circular-loop elliptic response; CI baseline comparison, artifact upload, and full free-boundary benchmarks remain pending. |
| Overall | 82% | Repository scaffolding, seed solver, mesh/geometry foundation, p=1 local/global/weighted FEM kernels, axisymmetric Grad-Shafranov weak form, nonlinear profile iteration, reduced and closed-form circular-loop free-boundary coil kernels, upstream comparison probe, sparse and matrix-free assembly paths, source/profile loads, Dirichlet solves, true-error manufactured convergence, verification CLI, docs, examples, tests, GUI previews, plotting metadata, literature reproduction scaffolding, and benchmark reports are in place; the project is still not a full TokaMaker replacement. |

Percentages are approximate planning markers, not validation metrics.

## Next Steps

The next implementation milestone should keep the scope narrow and measurable:

1. Build or locate the OpenFUSIONToolkit shared library locally so the
   availability-gated `oft-parity` probe can run numeric `eval_green`
   comparisons instead of reporting `skipped_unavailable`.
2. Couple the nonlinear p=1 profile iteration to the free-boundary coil
   response and add Picard/Newton convergence gates on at least one documented
   equilibrium fixture.
3. Promote the CPC seed-family surrogate into a code-to-code literature
   reproduction by importing the exact published/OFT case inputs and recording
   scalar tolerances.
4. Add CI-recorded benchmark baselines for sparse assembly, matrix-free apply,
   profile iteration, circular-loop response, and manufactured
   Grad-Shafranov solves.
5. Add a GUI case browser and TOML editor so research users can move directly
   between examples, validation reports, solver settings, and reproducible
   plots.

After those pieces pass focused tests, the seed rectangular-grid solver can be
replaced incrementally by the triangular FEM path described in the architecture
and porting map.
