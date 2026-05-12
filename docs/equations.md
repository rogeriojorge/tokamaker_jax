# Equations and Derivations

This page is the equation atlas for the implemented seed solver and the
remaining TokaMaker port. It documents the mathematical conventions used by
the JAX code, the weak forms that the tests exercise, and the places where the
current implementation is intentionally narrower than upstream TokaMaker.

## Axisymmetric Fields

Use cylindrical coordinates $(R,\phi,Z)$ and assume
$\partial/\partial\phi=0$. The standard Grad-Shafranov representation writes
the magnetic field as

$$
\mathbf{B}
= \frac{1}{R}\nabla\psi\times\mathbf{e}_\phi
+ \frac{F(\psi)}{R}\mathbf{e}_\phi,
$$

where $\psi$ is the poloidal flux divided by $2\pi$ in the convention used by
the current code paths, and $F=RB_\phi$. The corresponding current density is

$$
\mu_0\mathbf{J}
= -\frac{1}{R}\Delta^*\psi\,\mathbf{e}_\phi
+ \frac{1}{R}\nabla F\times\mathbf{e}_\phi,
$$

with

$$
\Delta^*\psi
= R\frac{\partial}{\partial R}
\left(\frac{1}{R}\frac{\partial\psi}{\partial R}\right)
+\frac{\partial^2\psi}{\partial Z^2}
= \frac{\partial^2\psi}{\partial R^2}
-\frac{1}{R}\frac{\partial\psi}{\partial R}
+\frac{\partial^2\psi}{\partial Z^2}.
$$

The MHD force balance $\mathbf{J}\times\mathbf{B}=\nabla p$ implies that
$p=p(\psi)$ and $F=F(\psi)$. The Grad-Shafranov equation is

$$
\Delta^*\psi
= -\mu_0 R^2 \frac{dp}{d\psi}
-\frac{1}{2}\frac{dF^2}{d\psi}.
$$

This is the form used by the source-profile helpers and validation notes. Sign
and normalization conventions are checked explicitly against local
OpenFUSIONToolkit/TokaMaker reports because equilibrium codes differ in their
choice of $\psi$, current sign, and toroidal-angle orientation. The COCOS
paper is the convention reference used for future EQDSK and reconstruction
interfaces.

## Self-Adjoint Weak Form

The implemented triangular FEM gate solves the self-adjoint form

$$
-\nabla\cdot\left(\frac{1}{R}\nabla\psi\right)=q
\quad\hbox{in}\quad\Omega,
$$

where the profile source for the negated self-adjoint equation is

$$
q(R,Z,\psi)
= \frac{1}{2R}\frac{dF^2}{d\psi}
+\mu_0 R\frac{dp}{d\psi}.
$$

For a test function $v$ that vanishes on Dirichlet boundaries, the weak form is

$$
a(\psi,v)=\ell(v),
\qquad
a(\psi,v)=\int_\Omega \frac{1}{R}\nabla\psi\cdot\nabla v\,dR\,dZ,
$$

and

$$
\ell(v)=\int_\Omega q(R,Z,\psi)v\,dR\,dZ.
$$

The current nonlinear profile iteration uses Picard-style fixed-point updates:

$$
A\psi^{k+1}=b(\psi^k),
\qquad
\psi^{k+1}\leftarrow (1-\omega)\psi^k+\omega A^{-1}b(\psi^k),
$$

with relaxation $\omega$. This is enough to validate source assembly,
Dirichlet treatment, residual histories, and differentiability through a fixed
number of iterations. Production parity will replace this with TokaMaker-like
nonlinear equilibrium solves and then add implicit differentiation.

## P1 Triangular Element

For the reference triangle
$\hat K=\{(\xi,\eta):\xi\ge0,\eta\ge0,\xi+\eta\le1\}$, the p=1 basis is

$$
\hat\phi_1=1-\xi-\eta,\qquad
\hat\phi_2=\xi,\qquad
\hat\phi_3=\eta.
$$

For physical vertices $x_1,x_2,x_3$, define

$$
J_K=\begin{bmatrix}x_2-x_1 & x_3-x_1\end{bmatrix},
\qquad
x(\xi,\eta)=x_1+J_K\begin{bmatrix}\xi\\\eta\end{bmatrix}.
$$

Physical gradients are

$$
\nabla\phi_i=J_K^{-T}\nabla_{\xi,\eta}\hat\phi_i.
$$

The exact p=1 mass matrix is

$$
M^K=\frac{|K|}{12}
\begin{bmatrix}
2&1&1\\
1&2&1\\
1&1&2
\end{bmatrix},
$$

and the weighted axisymmetric stiffness matrix is

$$
A^K_{ij}
=\int_K \frac{1}{R}\nabla\phi_i\cdot\nabla\phi_j\,dR\,dZ.
$$

The implemented code evaluates weighted integrals by quadrature and verifies
unweighted matrix identities exactly on simple meshes.

## Manufactured Solutions

Two manufactured gates provide numeric physics checks:

1. Poisson on the unit square:

   $$
   -\Delta u=2\pi^2\sin(\pi x)\sin(\pi y),
   \qquad
   u=\sin(\pi x)\sin(\pi y).
   $$

2. Axisymmetric Grad-Shafranov weak form on
   $\Omega=[1,2]\times[-0.5,0.5]$:

   $$
   \psi(R,Z)=
   \sin(\pi(R-1))\sin(\pi(Z+0.5)).
   $$

   The corresponding right-hand side is

   $$
   q(R,Z)=
   \frac{(k_R^2+k_Z^2)\psi}{R}
   +\frac{\partial\psi/\partial R}{R^2},
   \qquad k_R=k_Z=\pi.
   $$

For shape-regular p=1 refinement, the expected asymptotic rates are

$$
\|\psi-\psi_h\|_{L^2}=O(h^2),
\qquad
|\psi-\psi_h|_{H^1_R}=O(h).
$$

The tests measure true quadrature errors, not only nodal differences.

## Coil Green's Functions

The reduced large-aspect-ratio fixture uses

$$
G_\mathrm{red}
=-\frac{\mu_0}{4\pi}
\log\left(
\frac{(R-R_c)^2+(Z-Z_c)^2+\epsilon^2}{a_\mathrm{ref}^2}
\right).
$$

It is not a full TokaMaker free-boundary kernel. It exists to validate
linearity, symmetry, analytic derivatives, and JAX automatic differentiation.

The circular-filament kernel uses the standard complete-elliptic-integral
vector-potential form. With source radius $a$, field point $(R,Z)$, and
$dZ=Z-Z_c$,

$$
k^2=\frac{4aR}{(a+R)^2+dZ^2}.
$$

The azimuthal vector potential of a unit-current circular loop is

$$
A_\phi =
\frac{\mu_0}{\pi k}
\sqrt{\frac{a}{R}}
\left[\left(1-\frac{k^2}{2}\right)K(k^2)-E(k^2)\right].
$$

The JAX flux convention used by `circular_loop_elliptic_flux` is

$$
\psi_\mathrm{JAX}=R A_\phi.
$$

OpenFUSIONToolkit/TokaMaker `eval_green` currently reports the opposite sign
for the same unit-current check in the local comparison harness, so the parity
gate compares

$$
\psi_\mathrm{OFT}\approx-\psi_\mathrm{JAX}.
$$

This explicit sign gate is part of the future COCOS/EQDSK policy.

## Free-Boundary/Profile Coupling Gate

The first coupled validation constructs a response matrix

$$
\Psi_{ic}=G_\mathrm{loop}(R_i,Z_i;R_c,Z_c)
$$

and coil flux

$$
\psi_c(\mathbf{x}_i)=\sum_c\Psi_{ic}I_c.
$$

Boundary nodes receive Dirichlet values from $\psi_c$. The interior solve uses
the nonlinear profile iteration above. The validation checks

$$
\|\Psi I-\psi_c^\mathrm{direct}\|_2/\|\psi_c^\mathrm{direct}\|_2,
$$

exact boundary enforcement,

$$
\max_{i\in\partial\Omega}|\psi_i-\psi_{c,i}|,
$$

and current-gradient consistency for

$$
J(I)=\frac{1}{N}\|\Psi I\|_2^2,
\qquad
\nabla_IJ=\frac{2}{N}\Psi^T\Psi I.
$$

It also differentiates a fixed-iteration nonlinear solve with respect to the
pressure scale. This is a concrete differentiability gate, not a smoke test.

## Differentiability Policy

The implemented differentiable paths are fixed-topology and fixed-iteration:

- profile coefficients;
- coil currents;
- coil locations for smooth Green's-function calls;
- mesh node coordinates when connectivity is fixed;
- source scales through a fixed number of Picard iterations.

The project does not claim differentiability through mesh regeneration,
limiter contact changes, X-point creation/destruction, active-set changes, or
topology changes. Those events will be exposed as nonsmooth events in the GUI
and reports rather than hidden inside automatic differentiation.
