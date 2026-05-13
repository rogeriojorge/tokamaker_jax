# Grad-Shafranov Explorer

The static browser explorer is a GitHub Pages companion for teaching
axisymmetric MHD equilibrium and the Grad-Shafranov equation:

![Grad-Shafranov browser explorer](_static/tokamaker_jax_explorer_screenshot.png)

```{raw} html
<p><a class="reference external" href="_static/tokamaker_jax_explorer.html">Open the Grad-Shafranov explorer</a></p>
<iframe src="_static/tokamaker_jax_explorer.html" title="tokamaker-jax Grad-Shafranov explorer" style="width: 100%; min-height: 930px; border: 1px solid #d7dde8; border-radius: 8px;"></iframe>
```

It runs entirely in the browser, so it is suitable for GitHub Pages. The page is
intentionally simple: pressure, FF prime, coil current, elongation, and
triangularity controls update a reduced fixed-boundary Grad-Shafranov solve,
flux-surface plot, source-term plot, profile plot, and relaxation curve. The
browser numerics are educational proxies; validated runs remain in the
Python/JAX CLI and NiceGUI dashboard.

Use the CLI for validated runs:

```bash
tokamaker-jax init-example fixed-boundary --output fixed_boundary.toml
tokamaker-jax fixed_boundary.toml --plot outputs/fixed_boundary.png
tokamaker-jax verify --gate all --subdivisions 4 8 16
```
