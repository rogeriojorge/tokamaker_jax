# Progress

`tokamaker-jax` is still a seed port, not a full TokaMaker replacement. The
current executable solver is a small fixed-boundary rectangular-grid
implementation used to validate project structure, JAX differentiation,
configuration loading, plotting, examples, and CI behavior before the
TokaMaker-equivalent triangular FEM stack is ported.

## Status

| Lane | Approx. status | Current state |
| --- | ---: | --- |
| M1 mesh/geometry | 76% | `TriMesh`, OFT-style mesh I/O, region primitives, TOML region parsing, canonical sample regions, and geometry previews exist; region-aware mesh generation and production FEM coupling remain open. |
| M2 FEM core | 45% | p=1 reference kernels, dense and sparse global assembly, matrix-free applies, load-vector assembly, Dirichlet reduction, and manufactured Poisson convergence now have validation tests; axisymmetric weights and Grad-Shafranov weak forms are pending. |
| Plotting | 42% | Seed equilibrium plots, mesh/region plots, animation outputs, `FigureRecipe`, JSON metadata, region tables, annotated seed plots, and the manufactured Poisson convergence figure exist; literature figure reproduction scripts remain pending. |
| Docs/examples | 45% | Installation, architecture, porting map, API, examples, progress, validation equations, validation manifest, and manufactured convergence documentation are present; full derivation atlas and broad tutorials remain pending. |
| Config/CLI | 42% | TOML runs support solver/grid/source/coil/output settings plus region geometry, and `tokamaker-jax validate` checks configs without solving; production schema export and richer examples are pending. |
| Test infra | 48% | Pytest, coverage, lint, docs, FEM analytic gates, manufactured convergence gates, validation CLI tests, GUI metadata tests, and benchmark schema tests exist; OpenFUSIONToolkit parity and literature figure gates remain pending. |
| Differentiability | 25% | Seed solver paths, local FEM kernels, dense/sparse assembly, and matrix-free applies are JAX-compatible with focused gradient tests; implicit solver VJPs, topology policies, and optimization workflows remain open. |
| GUI | 22% | NiceGUI seed equilibrium controls, region-geometry preview, metadata summaries, and region tables exist; case browser, TOML editor, literature gallery, and validation dashboards remain pending. |
| Performance | 18% | JSON-friendly benchmark helpers exist for the seed fixed-boundary solve and local p=1 FEM kernels, with sparse/matrix-free assembly paths ready for benchmark baselines; CI baselines and free-boundary benchmarks remain pending. |
| Overall | 30% | Repository scaffolding, seed solver, mesh/geometry foundation, p=1 local/global FEM kernels, sparse and matrix-free assembly paths, load vectors, Dirichlet solves, manufactured convergence, validation CLI, docs, examples, tests, GUI previews, plotting metadata, and benchmark helpers are in place; the project is not yet TokaMaker feature complete. |

Percentages are approximate planning markers, not validation metrics.

## Next Steps

The next implementation milestone should keep the scope narrow and measurable:

1. Add axisymmetric Grad-Shafranov weak-form weights and verify them against
   analytic manufactured cylindrical-coordinate cases.
2. Add p=1 profile/source assembly for pressure and FF' terms on triangular
   meshes, using the dense Poisson path as the oracle.
3. Add reduced free-boundary coil Green's-function coupling behind a small
   fixture before trying full OFT parity.
4. Add the first literature-anchored TokaMaker figure reproduction script and
   artifact with a manifest entry, command, tolerance, and generated docs output.

After those pieces pass focused tests, the seed rectangular-grid solver can be
replaced incrementally by the triangular FEM path described in the architecture
and porting map.
