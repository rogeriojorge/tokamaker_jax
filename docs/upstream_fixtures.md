# Upstream Fixture Inventory

This page records exact mesh and geometry summaries extracted from the local
OpenFUSIONToolkit/TokaMaker example files. This is a fixture inventory, not a full equilibrium parity claim.

```bash
tokamaker-jax upstream-fixtures
tokamaker-jax upstream-fixtures --json --output outputs/upstream_fixture_summary.json
```

```{image} _static/upstream_fixture_mesh_sizes.png
:alt: Upstream TokaMaker mesh fixture sizes
```

The committed JSON artifact is
`docs/_static/upstream_fixture_summary.json`.

## Scope

The inventory currently covers these upstream example families:

| Fixture | Category | Files inventoried |
| --- | --- | --- |
| `nstxu-isoflux-controller` | control | `NSTXU_mesh.h5`, `NSTXU_geom.json`, controller notebooks |
| `cute` | time-dependent | `CUTE_mesh.h5`, `CUTE_geom.json`, VDE notebook |
| `diiid` | reconstruction | `DIIID_mesh.h5`, `DIIID_geom.json`, `g192185.02440` |
| `dipole` | non-tokamak | `dipole_mesh.h5`, equilibrium notebook |
| `hbt` | free-boundary | `HBT_mesh.h5`, `HBT_geom.json`, equilibrium/vacuum-coil notebooks |
| `iter` | free-boundary | `ITER_mesh.h5`, `ITER_geom.json`, baseline/H-mode notebooks |
| `ltx` | free-boundary | `LTX_mesh.h5`, `LTX_geom.json`, equilibrium notebook |
| `manta` | free-boundary | `MANTA_mesh.h5`, `MANTA_geom.json`, baseline notebook |

For each HDF5 mesh the summary records node count, triangular cell count,
region count, coil count, conductor/vacuum-region count, boundary-edge count,
bounds, area diagnostics, region-cell counts, region areas, and SHA-256. For
each geometry JSON it records top-level sections, coil/vessel counts,
coordinate-pair counts, bounds, and SHA-256.

## Current Extracted Counts

| Fixture | Nodes | Cells | Regions | Coils | Conductors |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nstxu-isoflux-controller` | 16122 | 32138 | 40 | 30 | 8 |
| `cute` | 5796 | 11488 | 31 | 28 | 1 |
| `diiid` | 8911 | 17660 | 85 | 58 | 24 |
| `dipole` | 8546 | 16912 | 6 | 2 | 1 |
| `hbt` | 3736 | 7352 | 35 | 30 | 2 |
| `iter` | 4757 | 9400 | 20 | 14 | 2 |
| `ltx` | 3128 | 6114 | 28 | 17 | 9 |
| `manta` | 8001 | 15766 | 19 | 12 | 4 |

## Acceptance Meaning

This inventory establishes that `tokamaker-jax` can parse and summarize the
upstream TokaMaker mesh files without mutating them. It does not yet compare a
solved equilibrium state. The next gate is to promote selected entries into
numeric tests with exact tolerances:

1. Mesh/geometry structural parity: counts, bounds, region ids, and SHA-256.
2. Fixed-boundary equilibrium parity for upstream fixed-boundary notebooks.
3. Free-boundary coil/passive-structure response matrix parity.
4. Full equilibrium diagnostics: magnetic axis, X/O points, LCFS, plasma
   current, boundary residuals, and selected probe fluxes.
