# Inputs, Outputs, and Artifact Contracts

This page defines the files and runtime artifacts that users and tests can rely
on. It is deliberately explicit so examples, GUI actions, CI artifacts, and
publication figures remain reproducible.

## Input Surfaces

| Surface | Example | Contract |
| --- | --- | --- |
| CLI GUI launch | `tokamaker-jax` | Opens the NiceGUI workflow using built-in seed defaults. |
| CLI TOML run | `tokamaker-jax examples/fixed_boundary.toml --plot outputs/fixed_boundary.png` | Loads a TOML config, solves, prints JSON stats, optionally writes a PNG. |
| CLI validation | `tokamaker-jax verify --gate all --subdivisions 4 8 16` | Runs implemented validation gates and prints JSON. |
| CLI config check | `tokamaker-jax validate case.toml` | Validates TOML structure without solving. |
| Python API | `load_config`, `solve_from_config` | Uses the same dataclasses as the CLI. |
| Docs asset script | `python examples/generate_assets.py` | Regenerates committed docs images and JSON reports. |
| Benchmark script | `python examples/benchmark_report.py --output outputs/benchmark_report.json` | Writes benchmark JSON and optional threshold comparison. |
| Literature script | `python examples/reproduce_cpc_seed_family.py outputs/literature/cpc_seed_family` | Writes surrogate figure and JSON recipe. |

## TOML Schema Summary

The current TOML schema is intentionally small:

```toml
[grid]
r_min = 1.0
r_max = 2.0
z_min = -0.5
z_max = 0.5
nr = 65
nz = 65

[source]
pressure_scale = 5000.0
ffp_scale = -0.35

[solver]
iterations = 700
relaxation = 0.75
dtype = "float64"

[[coils]]
name = "PF_A"
r = 1.35
z = 0.45
current = 200000.0
sigma = 0.06
```

Region geometry is also supported for preview and future mesh generation. The
full production schema will add plasma profiles, conductors, diagnostics,
constraints, output policies, reconstruction settings, and time-dependent
waveforms.

## JSON Outputs

### Verification Report

Produced by:

```bash
tokamaker-jax verify --gate all --subdivisions 4 8 16
```

Shape:

```json
{
  "subdivisions": [4, 8, 16],
  "gates": {
    "poisson": {"l2_rates": [], "h1_rates": [], "results": []},
    "grad_shafranov": {"l2_rates": [], "weighted_h1_rates": [], "results": []},
    "coil_green": {},
    "circular_loop": {},
    "profile_iteration": {},
    "free_boundary_profile": {},
    "openfusiontoolkit": {}
  }
}
```

The exact fields are documented by `verification.py` dataclasses and the
physics-gate manifest.

### Benchmark Report

Produced by:

```bash
python examples/benchmark_report.py \
  --output outputs/benchmark_report.json \
  --thresholds docs/validation/benchmark_thresholds.json \
  --comparison-output outputs/benchmark_threshold_report.json
```

The report contains:

- schema version;
- suite name;
- time unit;
- lane names;
- median, best, worst, repeats, warmups;
- lane metadata such as grid sizes and number of elements.

Threshold comparison reports contain one pass/fail record per lane.

### Figure Recipe

Publication-style examples should include:

- artifact id;
- status and parity claim;
- citation and source link;
- command used to generate the artifact;
- input grid/config values;
- output array shapes and scalar diagnostics;
- comparison rule and tolerance if parity is claimed.

The CPC seed-family surrogate already follows this pattern.

## Committed Static Assets

| Asset | Producer | Meaning |
| --- | --- | --- |
| `fixed_boundary_seed.png` | `examples/generate_assets.py` | Seed fixed-boundary equilibrium. |
| `manufactured_poisson_convergence.png` | `examples/generate_assets.py` | p=1 Poisson convergence. |
| `manufactured_grad_shafranov_convergence.png` | `examples/generate_assets.py` | Axisymmetric GS convergence. |
| `coil_green_response.png` | `examples/generate_assets.py` | Reduced coil Green's function. |
| `circular_loop_elliptic_response.png` | `examples/generate_assets.py` | Closed-form circular-loop response. |
| `profile_iteration.png` | `examples/generate_assets.py` | Nonlinear profile iteration. |
| `free_boundary_profile_coupling.png` | `examples/generate_assets.py` | Coupled coil boundary/profile solve. |
| `validation_dashboard.png` | `examples/generate_assets.py` | Gate margin dashboard. |
| `benchmark_summary.png` | `examples/generate_assets.py` | Median benchmark lanes. |
| `cpc_seed_family.png` | `examples/reproduce_cpc_seed_family.py` | Literature reproduction surrogate. |
| `publication_validation_panel.png` | `examples/generate_assets.py` | Four-panel publication summary. |
| `upstream_comparison_matrix.png` | `examples/generate_assets.py` | Upstream/literature comparison status. |
| `io_artifact_map.png` | `examples/generate_assets.py` | Input/output artifact flow. |

## Reproducibility Policy

Every generated docs artifact should satisfy:

1. The producing command is documented.
2. Inputs are either in TOML, code constants, or JSON.
3. Outputs are deterministic enough for review.
4. Numeric claims are backed by tests or validation gates.
5. Parity claims state the upstream code, commit or source, tolerance, and sign
   convention.

Figures can be visually polished, but they must not imply physics capability
that the current code does not yet implement.
