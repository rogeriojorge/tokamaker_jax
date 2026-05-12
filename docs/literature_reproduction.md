# Literature Reproduction Fixtures

This page tracks executable literature-anchored artifacts. The current fixture
is a surrogate for the Hansen et al. CPC/TokaMaker equilibrium-family workflow,
not an exact reproduction of a numbered paper figure and not a parity claim
against OpenFUSIONToolkit.

## Hansen CPC/TokaMaker Seed-Family Surrogate

The first fixture runs the existing rectangular fixed-boundary seed solver over
a small pressure-scale family and writes both a JSON report and a PNG figure.
It anchors the artifact metadata to:

- Hansen et al., "TokaMaker: An open-source time-dependent Grad-Shafranov tool
  for the design and modeling of axisymmetric fusion devices", *Computer
  Physics Communications* 298, 109111 (2024), DOI:
  `10.1016/j.cpc.2024.109111`.

Run it from the repository root:

```bash
python examples/reproduce_cpc_seed_family.py outputs/literature/cpc_seed_family
```

The output directory receives:

- `cpc_seed_family_report.json`: JSON report with source/citation metadata,
  generated command, seed-solver diagnostics, limitations, and an embedded
  `FigureRecipe` for the representative family member.
- `cpc_seed_family.png`: contour-panel PNG for the generated pressure-scale
  family.

![CPC seed-family surrogate](_static/cpc_seed_family.png)

The report uses `artifact_id: tokamaker-cpc-seed-equilibrium-family` and
`status: surrogate_fixture`. Its comparison rule is intentionally limited to
artifact generation and seed diagnostics. It does not assert a numeric tolerance
against Hansen et al. CPC figures, OpenFUSIONToolkit examples, free-boundary
coil responses, passive structures, reconstruction constraints, or
time-dependent behavior.

The fixture performs no network access at runtime. All citation/source metadata
is embedded in the script so the command remains reproducible offline.

Future OFT-backed reproduction work should replace this surrogate with checked
fixtures, figure identifiers, scalar parity diagnostics, and explicit tolerance
records in the validation manifest.
