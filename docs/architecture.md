# Architecture

The final architecture is organized around pure, differentiable data flow:

```{image} _static/io_artifact_map.png
:alt: tokamaker-jax input/output artifact map
```

1. Geometry and machine definition from TOML, Python, or GUI state.
2. Mesh construction and region labeling.
3. FEM operator assembly in JAX sparse/BCOO form.
4. Profile, coil, conductor, and diagnostic models as JAX pytrees.
5. Static, reconstruction, and time-dependent solvers.
6. Post-processing, plotting, export, and optimization interfaces.

The seed implementation originally used a rectangular finite-difference grid
to establish project structure, differentiability tests, TOML loading,
plotting, and CLI behavior. The current milestone now also contains p=1
triangular FEM kernels, weighted axisymmetric assembly, nonlinear profile
iteration, reduced and closed-form coil Green's functions, and a coupled
free-boundary/profile validation gate. The remaining production work is to
replace the seed path incrementally with the TokaMaker-equivalent triangular
FEM/free-boundary solve described in the porting map.

For design rationale, see [](design_decisions.md). For the mathematical weak
forms and sign conventions, see [](equations.md).
