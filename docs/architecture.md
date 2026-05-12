# Architecture

The final architecture is organized around pure, differentiable data flow:

1. Geometry and machine definition from TOML, Python, or GUI state.
2. Mesh construction and region labeling.
3. FEM operator assembly in JAX sparse/BCOO form.
4. Profile, coil, conductor, and diagnostic models as JAX pytrees.
5. Static, reconstruction, and time-dependent solvers.
6. Post-processing, plotting, export, and optimization interfaces.

The seed implementation currently uses a rectangular finite-difference grid to establish project structure, differentiability tests, TOML loading, plotting, and CLI behavior. It is deliberately small so the port can replace it module by module with TokaMaker-equivalent triangular finite elements.

