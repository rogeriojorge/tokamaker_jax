# Examples

The first executable example is a fixed-boundary seed solve:

```bash
tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png
```

```{image} _static/fixed_boundary_seed.png
:alt: Fixed-boundary seed equilibrium contours
```

```{image} _static/pressure_sweep.gif
:alt: Pressure sweep animation
```

Future examples tracked in `plan.md`:

- Fixed-boundary analytic Solov'ev convergence.
- Free-boundary ITER-like equilibrium.
- HBT-EP, DIII-D, LTX, CUTE, MANTA examples.
- Reconstruction from synthetic flux loops, Mirnov probes, pressure, Ip, q, and saddle constraints.
- Time-dependent passive conductor and VDE examples.
- Differentiable shape and coil optimization notebooks.
