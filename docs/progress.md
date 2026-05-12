# Progress

`tokamaker-jax` is still a seed port, not a full TokaMaker replacement. The
current executable solver is a small fixed-boundary rectangular-grid
implementation used to validate project structure, JAX differentiation,
configuration loading, plotting, examples, and CI behavior before the
TokaMaker-equivalent triangular FEM stack is ported.

## Status

| Lane | Approx. status | Current state |
| --- | ---: | --- |
| M1 mesh/geometry | 74% | `TriMesh`, OFT-style mesh I/O, region primitives, TOML region parsing, canonical sample regions, and geometry previews exist; region-aware mesh generation and production FEM coupling remain open. |
| M2 FEM core | 22% | p=1 reference-triangle kernels and dense global p=1 mass/stiffness assembly now have exact analytic tests; sparse assembly, boundary conditions, load vectors, axisymmetric weights, and Grad-Shafranov weak forms are pending. |
| Plotting | 35% | Seed equilibrium plots, mesh/region plots, animation outputs, `FigureRecipe`, JSON metadata, region tables, and annotated seed plots exist; literature figure reproduction scripts remain pending. |
| Docs/examples | 30% | Installation, architecture, porting map, API, examples, progress, validation equations, and the first validation manifest are present; full derivation atlas and broad tutorials remain pending. |
| Config/CLI | 42% | TOML runs support solver/grid/source/coil/output settings plus region geometry, and `tokamaker-jax validate` checks configs without solving; production schema export and richer examples are pending. |
| Test infra | 34% | Pytest, coverage, lint, docs, 70 tests, 97% coverage, first FEM precision tests, validation CLI tests, GUI metadata tests, and benchmark schema tests exist; OpenFUSIONToolkit parity and literature figure gates remain pending. |
| Differentiability | 14% | Seed solver paths, local FEM kernels, and dense assembly are JAX-compatible with focused gradient tests; sparse assembly, implicit solver VJPs, topology policies, and optimization workflows remain open. |
| GUI | 22% | NiceGUI seed equilibrium controls, region-geometry preview, metadata summaries, and region tables exist; case browser, TOML editor, literature gallery, and validation dashboards remain pending. |
| Performance | 12% | JSON-friendly benchmark helpers exist for the seed fixed-boundary solve and local p=1 FEM kernels; CI baselines, sparse assembly benchmarks, and free-boundary benchmarks remain pending. |
| Overall | 15% | Repository scaffolding, seed solver, mesh/geometry foundation, p=1 local and dense global FEM kernels, validation CLI, docs, examples, tests, GUI previews, plotting metadata, and benchmark helpers are in place; the project is not yet TokaMaker feature complete. |

Percentages are approximate planning markers, not validation metrics.

## Next Steps

The next implementation milestone should keep the scope narrow and measurable:

1. Add p=1 load-vector assembly and Dirichlet boundary-condition handling.
2. Add manufactured-solution tests that verify operator convergence before
   expanding to TokaMaker-specific source terms and free-boundary behavior.
3. Replace dense global matrices with sparse `BCOO` assembly and matrix-free
   operator paths while preserving dense tests as analytic oracles.
4. Add the first literature-anchored figure reproduction script and artifact
   with a manifest entry, command, tolerance, and generated docs output.

After those pieces pass focused tests, the seed rectangular-grid solver can be
replaced incrementally by the triangular FEM path described in the architecture
and porting map.
