# MHD Solver Explorer

The static browser explorer is a GitHub Pages companion for teaching the main
tokamaker-jax model surfaces in one self-contained page:

![tokamaker-jax MHD browser explorer](_static/tokamaker_jax_explorer_screenshot.png)

```{raw} html
<p><a class="reference external" href="_static/tokamaker_jax_explorer.html">Open the MHD solver explorer</a></p>
<iframe src="_static/tokamaker_jax_explorer.html" title="tokamaker-jax MHD solver explorer" style="width: 100%; min-height: 930px; border: 1px solid #d7dde8; border-radius: 8px;"></iframe>
```

It runs entirely in the browser, so it is suitable for GitHub Pages. The
Grad-Shafranov tab remains the primary teaching view, with pressure, FF prime,
coil current, elongation, triangularity, flux-surface, source-term, profile, and
relaxation controls. Additional tabs expose the same conceptual surfaces as the
Python package: free-boundary coil Green functions, nonlinear profile
iterations, triangular FEM/mesh assembly, verification and IO contracts, and
copyable examples/CLI commands. The browser numerics are educational proxies;
validated runs remain in the Python/JAX CLI and NiceGUI dashboard.

Use the CLI for validated runs:

```bash
tokamaker-jax init-example fixed-boundary --output fixed_boundary.toml
tokamaker-jax fixed_boundary.toml --plot outputs/fixed_boundary.png
tokamaker-jax verify --gate all --subdivisions 4 8 16
```
