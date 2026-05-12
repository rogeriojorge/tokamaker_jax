# Validation

This page defines the current validation gates for `tokamaker-jax`. The goal is
to make every physics, differentiability, figure, and performance claim
traceable to an executable command, a fixture, a tolerance, and an artifact.

The first machine-readable manifest is
[`docs/validation/physics_gates_manifest.json`](validation/physics_gates_manifest.json).
It separates implemented gates from planned gates so the project can document
the roadmap without claiming full TokaMaker parity before the relevant tests
exist.

## Implemented Gates

The current implemented gates cover the seed infrastructure and the first local
triangular FEM path:

- p=1 triangular basis, gradients, quadrature, affine maps, local mass matrices,
  and local Laplace stiffness matrices.
- dense global p=1 triangular mass and Laplace stiffness assembly for fixed
  mesh topology.
- TOML parsing and `tokamaker-jax validate` checks for grid, solver, coil,
  output, and region-geometry inputs.
- fixed-boundary seed solver tests, including JAX differentiation checks on the
  current rectangular-grid path.
- figure metadata exports and GUI-ready summary data.
- small executable benchmark helpers for seed-solver and local-FEM timings.

## P1 Triangle Equations

On the reference triangle
$(\hat x_1,\hat x_2,\hat x_3)=((0,0),(1,0),(0,1))$, the p=1 basis is

$$
\hat \phi_1 = 1-\xi-\eta,\qquad
\hat \phi_2 = \xi,\qquad
\hat \phi_3 = \eta,
$$

with gradients

$$
\nabla_{\xi,\eta}\hat\phi_1=(-1,-1),\qquad
\nabla_{\xi,\eta}\hat\phi_2=(1,0),\qquad
\nabla_{\xi,\eta}\hat\phi_3=(0,1).
$$

For physical vertices $x_1,x_2,x_3$, the affine map is

$$
x(\xi,\eta)=x_1+\begin{bmatrix}x_2-x_1 & x_3-x_1\end{bmatrix}
\begin{bmatrix}\xi\\ \eta\end{bmatrix}.
$$

The local mass matrix used as an exact validation oracle is

$$
M^K = \frac{|K|}{12}
\begin{bmatrix}
2 & 1 & 1\\
1 & 2 & 1\\
1 & 1 & 2
\end{bmatrix}.
$$

The local Laplace stiffness matrix is

$$
A^K_{ij} = |K|\,\nabla\phi_i\cdot\nabla\phi_j.
$$

Implemented tests check interpolation at nodes, partition of unity, exact
degree-2 reference-triangle quadrature, affine mapping consistency, matrix
symmetry, positive mass eigenvalues, stiffness positive semidefiniteness, and
the constant-vector stiffness nullspace.

## Global Assembly Gates

For a fixed-connectivity triangular mesh, local element contributions are
scattered into a global nodal matrix:

$$
A_{IJ} = \sum_{K}\sum_{i,j=1}^{3}
\mathbf{1}_{T_{K,i}=I}\mathbf{1}_{T_{K,j}=J} A^K_{ij}.
$$

The first assembly gate uses a two-triangle unit square with exact global mass
and stiffness matrices. It verifies symmetry, mass total, stiffness row sums,
constant nullspace, eigenvalue signs, JIT compatibility, and gradients with
respect to node coordinates for fixed topology.

## Manufactured-Solution Plan

The next physics gate is a manufactured Poisson problem:

$$
-\Delta u=f,\qquad u|_{\partial\Omega}=g.
$$

For shape-regular p=1 triangular refinement, the expected rates are

$$
\|u-u_h\|_{H^1(\Omega)}=O(h),\qquad
\|u-u_h\|_{L^2(\Omega)}=O(h^2).
$$

The observed convergence rate will be computed as

$$
p_\mathrm{obs} =
\frac{\log(e(h_m)/e(h_{m+1}))}{\log(h_m/h_{m+1})}.
$$

This gate is still planned because the repository now has local and dense
global FEM operators, but not boundary-condition application, load-vector
assembly, or a triangular FEM solve path.

## Differentiability Gates

Differentiability gates compare JAX automatic differentiation with central
finite differences for scalar objectives:

$$
D_vF(x)_\mathrm{fd} =
\frac{F(x+\epsilon v)-F(x-\epsilon v)}{2\epsilon},
\qquad
D_vF(x)_\mathrm{ad}=\nabla F(x)\cdot v.
$$

The reported discrepancy is

$$
\eta_\nabla =
\frac{|D_vF(x)_\mathrm{ad}-D_vF(x)_\mathrm{fd}|}
{\max(1, |D_vF(x)_\mathrm{fd}|)}.
$$

Every differentiability gate must record dtype, backend, seed, perturbation
scale, topology assumptions, and whether gradients pass through dense matrices,
matrix-free operators, iterative solves, or implicit VJPs.

## Performance Gates

Performance gates are regression checks, not hardware-independent speed claims.
Each benchmark record must include command, fixture, dtype, backend, warmup
policy, repeat count, and hardware metadata when run in CI or release reports.

For timing baselines, the regression ratio is

$$
\rho_t = \frac{t_\mathrm{new}}{t_\mathrm{baseline}}.
$$

The current benchmark helpers produce JSON-friendly timing dictionaries for the
seed fixed-boundary solve and local p=1 FEM kernels. Future gates should add
global sparse assembly, matrix-free operator, and free-boundary solve timings.

## Literature-Anchored Figure Gates

Figure reproduction gates are anchored to the sources listed on the
[](references.md) page, especially the OpenFUSIONToolkit source/docs and the
TokaMaker CPC paper/preprint. A figure gate must include:

- source label and figure identifier,
- executable reproduction command,
- input fixture or case TOML,
- generated image/movie/table artifact,
- numerical comparison rule or documented visual-comparison criterion,
- tolerances and failure mode.

Until a literature figure has a checked-in script and artifact, its status
remains `planned`.
