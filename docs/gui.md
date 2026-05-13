# GUI

The GUI is a research dashboard for the same objects used by the Python API and
CLI. It is not a separate solver path: every runnable case is backed by a TOML
file or by an explicitly recorded command in the case manifest.

## Launching

The default command opens the GUI on `127.0.0.1:8080`:

```bash
tokamaker-jax
```

The explicit GUI subcommand exposes platform-friendly server controls:

```bash
tokamaker-jax gui --host 127.0.0.1 --port 8080
tokamaker-jax gui --host 0.0.0.0 --port 8081 --no-browser
```

Use `--no-browser` for remote Linux sessions, CI smoke checks, WSL, containers,
or any machine where the server should start without opening a local browser
window. Use `--port` when another local service already occupies port 8080.

For static hosting, use the browser-side MHD solver explorer instead of the
NiceGUI server: [](browser_explorer.md). It can be served by GitHub Pages
because it is a self-contained HTML teaching tool. The Grad-Shafranov tab is
first, and companion tabs cover coil Green functions, nonlinear profiles,
FEM/mesh assembly, verification/IO, and examples/CLI commands.

## First Screen

The first screen is the workflow dashboard. It shows:

- a seed-equilibrium workbench with pressure, FF', and iteration sliders;
- an animated flux/residual preview that can be replayed with the Plotly frame
  controls;
- the current seed-equilibrium residual, flux range, and grid settings;
- validation-gate status for Poisson, Grad-Shafranov, coil Green-function, and
  fixed-boundary gEQDSK checks;
- command lines that reproduce each visible workflow item.

The dashboard is intentionally dense enough for repeated research use. It avoids
marketing panels and keeps solver status, validation metrics, and provenance in
the first viewport. The `Run preview` button recomputes the reduced
fixed-boundary solve in the same path used by the CLI example, while slider moves
update displayed input values before a solve is requested.

```{image} _static/gui_workflow_dashboard.png
:alt: tokamaker-jax GUI workflow dashboard
```

## Tabs

| Tab | Purpose |
| --- | --- |
| Workflow | Current solve, validation, and next-step status tables. |
| Seed equilibrium | Interactive pressure/FF' controls, convergence summary, and flux plot. |
| Region geometry | Sample plasma, wall, and coil region plot plus geometric metadata. |
| Validation | Log-log convergence plots with fitted numerical rates. |
| Coil response | Reduced coil Green-function response preview with coil markers. |
| Cases | Shared manifest browser, TOML source preview/editor, validation rows, and one-click saved-TOML execution. |
| Reports | Stored validation, OpenFUSIONToolkit parity, benchmark, and upstream-fixture artifacts. |

## Case Runs

The `Cases` tab has two distinct operations:

- `Validate editor text` parses the current editor contents and reports TOML or
  schema errors without modifying the case file.
- `Run saved TOML` validates the editor, checks that it matches the manifest
  file on disk, then runs the saved file with `shell=False` and captures stdout,
  stderr, return code, duration, and artifact existence.

If the editor contains valid but unsaved changes, the GUI refuses to run and
asks the user to keep the manifest-backed reproducibility contract explicit. A
run from the GUI is therefore equivalent to the printed command, for example:

```bash
tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png
```

## Validation Use

The GUI surfaces the same validation gates as the CLI. The recommended command
for headless reproduction is:

```bash
tokamaker-jax verify --gate all --subdivisions 4 8 16
```

The fixed-boundary gEQDSK card uses committed upstream evidence and checks grid
shape, profile length, plasma current, central field, and magnetic-axis
diagnostics. It is a bounded diagnostic gate, not a claim of complete
fixed-boundary equilibrium parity.
