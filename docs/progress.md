# Progress

`tokamaker-jax` is still a seed port, not a full TokaMaker replacement. The
current executable solver is a small fixed-boundary rectangular-grid
implementation used to validate project structure, JAX differentiation,
configuration loading, plotting, examples, and CI behavior before the
TokaMaker-equivalent triangular FEM stack is ported.

## Status

| Lane | Approx. status | Current state |
| --- | ---: | --- |
| M1 mesh/geometry | 68% | `TriMesh`, OFT-style mesh I/O, region primitives, TOML region parsing, and geometry previews exist; region-aware mesh generation and full FEM coupling remain open. |
| M2 FEM core | 8% | p=1 reference-triangle nodes, basis functions, gradients, quadrature, and local mass/stiffness matrices now have exactness tests; global sparse assembly and Grad-Shafranov weak forms are pending. |
| Plotting | 22% | Seed equilibrium plots, mesh/region plots, animation outputs, and `FigureRecipe` metadata exports exist; literature figure reproduction and validation manifests remain pending. |
| Docs/examples | 18% | Installation, architecture, porting map, API, seed examples, and this progress page are present; full equation derivations, validation reports, and broad tutorials remain pending. |
| Config/CLI | 18% | TOML runs support solver/grid/source/coil/output settings plus region geometry; schema validation commands and production examples are pending. |
| Test infra | 20% | Pytest, coverage, lint, docs, seed differentiability tests, geometry tests, plotting tests, and first FEM precision tests exist; OpenFUSIONToolkit parity, literature gates, and performance gates remain pending. |
| Differentiability | 6% | Seed solver paths and local FEM kernels are JAX-compatible; sparse assembly, implicit solver VJPs, topology policies, and optimization workflows remain open. |
| GUI | 10% | NiceGUI seed equilibrium controls and region-geometry preview exist; case browser, TOML editor, literature gallery, and validation dashboards remain pending. |
| Overall | 6% | Repository scaffolding, seed solver, mesh/geometry foundation, p=1 local FEM kernels, examples, docs, and infrastructure are in place; the project is not yet TokaMaker feature complete. |

Percentages are approximate planning markers, not validation metrics.

## Next Steps

The next implementation milestone should keep the scope narrow and measurable:

1. Assemble global sparse mass/stiffness operators for triangular cells.
2. Add manufactured-solution tests that verify operator convergence before
   expanding to TokaMaker-specific source terms and free-boundary behavior.
3. Wire TOML-defined region geometry into meshing examples and CLI validation.
4. Add the first validation manifest with figure recipes, numeric tolerances,
   citation metadata, and generated docs artifacts.

After those pieces pass focused tests, the seed rectangular-grid solver can be
replaced incrementally by the triangular FEM path described in the architecture
and porting map.
