# Browser Research Workbench

The static browser research workbench is a GitHub Pages companion for the
tokamaker-jax solver stack:

![Static browser research workbench](_static/tokamaker_jax_explorer_screenshot.png)

```{raw} html
<p><a class="reference external" href="_static/tokamaker_jax_explorer.html">Open the browser research workbench</a></p>
<iframe src="_static/tokamaker_jax_explorer.html" title="tokamaker-jax browser research workbench" style="width: 100%; min-height: 980px; border: 1px solid #d7dde8; border-radius: 8px;"></iframe>
```

It runs entirely in the browser, so it is suitable for GitHub Pages. The tabs
cover the same research lanes as the Python package: case setup, equilibrium
preview, region geometry and mesh topology, pressure and FF' profiles, coil
Green-function response, physics validation gates, differentiability checks,
benchmark history, and reproducible export. The browser numerics are reduced
proxies; validated solves, gradients, parity checks, and publication artifacts
remain in the Python/JAX CLI and NiceGUI dashboard.

Use the CLI for validated runs:

```bash
tokamaker-jax init-example fixed-boundary --output fixed_boundary.toml
tokamaker-jax fixed_boundary.toml --plot outputs/fixed_boundary.png
tokamaker-jax verify --gate all --subdivisions 4 8 16
```
