# Upstream Fixed-Boundary Evidence

This page records source-level evidence extracted from the upstream
OpenFUSIONToolkit/TokaMaker fixed-boundary examples. It is a preparation step
for future numeric fixed-boundary equilibrium parity, not a full parity claim.

```bash
tokamaker-jax fixed-boundary-evidence
tokamaker-jax fixed-boundary-evidence --json --output outputs/fixed_boundary_evidence.json
```

```{image} _static/fixed_boundary_upstream_geqdsk.png
:alt: Upstream fixed-boundary gEQDSK source flux map
```

The committed JSON artifact is
`docs/_static/fixed_boundary_upstream_evidence.json`.
The richer stored-output audit used by the physics-gate manifest is
`docs/validation/fixed_boundary_upstream_evidence.json`.
A second, validation-scoped artifact,
`docs/validation/fixed_boundary_upstream_evidence.json`, records the stored
upstream notebook mesh counts, `print_info` scalars, `gNT_example` header and
profile ranges, and the `fixed_boundary_ex2.ipynb` coil-current fit. It is also
bounded evidence, not a full parity claim.

## Extracted Sources

| Source | Evidence |
| --- | --- |
| `fixed_boundary_ex1.ipynb` | Analytic fixed-boundary solve plus gEQDSK-backed profile matching workflow. |
| `fixed_boundary_ex2.ipynb` | Fixed-boundary solve followed by fitted coil currents and free-boundary comparison workflow. |
| `gNT_example` | 129 x 129 gEQDSK source grid with profiles, q, pressure, current, and magnetic-axis metadata. |

## Acceptance Meaning

The evidence report records notebook hashes, code/markdown cell counts,
fixed-boundary assignments, solve calls, mesh-size values, target-current
expressions, profile-matching indicators, gEQDSK dimensions, flux range, q
range, magnetic axis, center field, and plasma current.

The next numeric gate is to convert this evidence into a reproducible parity
case:

1. Build the same analytic fixed-boundary mesh and compare geometry summaries.
2. Match the fixed-boundary solve diagnostics from `fixed_boundary_ex1.ipynb`.
3. Load the `gNT_example` profiles and compare pressure, FF', q, plasma
   current, magnetic axis, and boundary flux diagnostics.
4. Promote the result from `source_evidence_only` to `equilibrium_parity` only
   after explicit tolerances pass in CI.
