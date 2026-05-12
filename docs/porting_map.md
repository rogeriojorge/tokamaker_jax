# Porting Map

The source audit is tracked in `plan.md`. The core OpenFUSIONToolkit files to port or match are:

- `src/physics/grad_shaf.F90`: equilibrium factory, source assembly, nonlinear solve, fixed/free-boundary behavior, coil coupling, wall sources, q profiles, field interpolation, plotting/export backends.
- `src/physics/grad_shaf_profiles.F90`: flat, polynomial, spline, linear-interpolation, Wesson, and non-inductive profile functions.
- `src/physics/grad_shaf_td.F90`: time-dependent stepping and wall-mode operators.
- `src/physics/grad_shaf_fit.F90`: reconstruction fit machinery.
- `src/physics/axi_green.F90`: axisymmetric Green's functions for coil/plasma mutuals.
- `src/python/OpenFUSIONToolkit/TokaMaker`: Python API, meshing, reconstruction helpers, plotting, and IO.
- `src/tests/physics/test_TokaMaker.py`: regression targets for Solov'ev, spheromak, coil mutuals, ITER/LTX equilibria, reconstruction, wall modes, bootstrap current, and IO.

The intended JAX modules are:

- `domain` and future `mesh`: differentiable mesh and region data.
- `profiles`: flux functions and bootstrap/non-inductive sources.
- `operator`: FEM basis, quadrature, and Grad-Shafranov assembly.
- `solver`: static, free-boundary, reconstruction, and time-dependent solvers.
- `io`: EQDSK, i-file, HDF5, and TokaMaker-compatible state.
- `plotting` and `gui`: publication-quality visualization and workflow UI.

