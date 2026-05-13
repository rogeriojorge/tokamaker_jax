# Case Manifest

The case manifest is the shared inventory used by the CLI, GUI, docs, and
tests. It separates runnable examples from planned upstream parity fixtures so
users can see what is executable today and what still needs physics acceptance
gates.

```bash
tokamaker-jax cases
tokamaker-jax cases --runnable-only
tokamaker-jax cases --json --output outputs/case_manifest.json
```

```{image} _static/case_manifest_status.png
:alt: Case manifest status counts
```

The committed JSON artifact is
`docs/_static/case_manifest.json`.

## Current Entries

| Case | Status | Level | Purpose |
| --- | --- | --- | --- |
| `fixed-boundary-seed` | `runnable` | `manufactured_validation` | README/docs/GUI seed equilibrium and manufactured Grad-Shafranov gate. |
| `case-manifest-browser` | `runnable` | `workflow_fixture` | CLI/GUI/docs case browser and JSON manifest. |
| `cpc-seed-family` | `runnable` | `surrogate_fixture` | Citation-linked CPC seed-family reproduction workflow. |
| `openfusiontoolkit-green-parity` | `validation_gate` | `kernel_parity` | Local OFT `eval_green` comparison when the upstream shared library is available. |
| `free-boundary-target-schema` | `schema_preview` | `source_audit` | Human-readable TOML target for the future free-boundary workflow. |
| `iter-baseline-upstream` | `planned_upstream_fixture` | `source_audit` | ITER baseline/H-mode fixture mapping. |
| `diiid-baseline-upstream` | `planned_upstream_fixture` | `source_audit` | DIII-D baseline plus g-file fixture mapping. |
| `hbt-equilibrium-upstream` | `planned_upstream_fixture` | `source_audit` | HBT equilibrium and vacuum-coil fixture mapping. |
| `cute-vde-upstream` | `planned_upstream_fixture` | `source_audit` | CUTE VDE/time-dependent fixture mapping. |
| `nstxu-isoflux-controller-upstream` | `planned_upstream_fixture` | `source_audit` | NSTX-U isoflux/controller fixture mapping. |

## Acceptance Levels

The manifest uses the comparison levels defined in [](comparisons.md):

- `source_audit`: the upstream source or paper has been mapped, but no numeric
  parity claim is made.
- `surrogate_fixture`: a reproducible artifact exists, but the exact upstream
  or literature data-level comparison remains future work.
- `kernel_parity`: a scalar/vector kernel has a numeric gate against an
  analytic or upstream oracle.
- `equilibrium_parity`: a full equilibrium state matches upstream or
  literature diagnostics within tolerance.
- `workflow_parity`: a complete case has input files, command, GUI/docs
  coverage, plots, reports, and CI validation.

No manifest entry currently claims `equilibrium_parity` or `workflow_parity`
for a full upstream TokaMaker equilibrium. Those remain explicit next
milestones.

## GUI Use

The GUI opens with:

```bash
tokamaker-jax
```

For explicit server control:

```bash
tokamaker-jax gui --host 127.0.0.1 --port 8081 --no-browser
```

The `Cases` tab shows the same table as the CLI, including status,
category, parity level, command, and validation gate. It includes a source
preview/editor for local TOML files, validates editor text without mutating the
case file, and runs only the saved manifest TOML after confirming the editor
matches disk. The result table captures process status, stdout/stderr, duration,
return code, and artifact existence so GUI runs remain reproducible from the
printed CLI command.

## Next Numeric Gates

The planned upstream fixtures should graduate in this order:

1. Import exact upstream mesh/geometry summaries and assert element counts,
   region ids, areas, and boundary tags.
2. Add fixed-boundary equilibrium parity for the upstream
   `fixed_boundary_ex1.ipynb` and `fixed_boundary_ex2.ipynb` cases.
3. Add coil and passive-structure response matrices for ITER, HBT, and DIII-D
   fixtures.
4. Add free-boundary equilibrium diagnostics: magnetic axis, X/O points, LCFS
   contour, plasma current, boundary flux residual, and selected probe fluxes.
5. Add reconstruction/control/time-dependent cases only after the static
   fixture gates are passing with documented tolerances.
