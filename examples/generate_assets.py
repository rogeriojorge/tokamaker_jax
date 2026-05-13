"""Generate README and documentation visual assets."""

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from jax import config as jax_config
from matplotlib.animation import FuncAnimation, PillowWriter
from reproduce_cpc_seed_family import generate_cpc_seed_family_artifacts

from tokamaker_jax.assembly import boundary_nodes_from_coordinates
from tokamaker_jax.benchmarks import (
    DEFAULT_BENCHMARK_THRESHOLDS,
    benchmark_baseline_report,
    benchmark_report_to_json,
    benchmark_threshold_report,
)
from tokamaker_jax.cases import default_case_manifest, write_case_manifest
from tokamaker_jax.cli import run_verification_gates
from tokamaker_jax.comparison import run_openfusiontoolkit_green_comparison
from tokamaker_jax.config import CoilConfig, GridConfig, RunConfig, SolverConfig, SourceConfig
from tokamaker_jax.domain import RectangularGrid
from tokamaker_jax.fem_equilibrium import (
    NonlinearProfileParameters,
    PowerProfile,
    solve_profile_iteration,
    solve_profile_iteration_on_rectangle,
)
from tokamaker_jax.free_boundary import circular_loop_elliptic_coil_flux
from tokamaker_jax.geometry import sample_regions
from tokamaker_jax.plotting import (
    plot_equilibrium,
    save_coil_green_response_plot,
    save_equilibrium_plot,
    save_region_plot,
)
from tokamaker_jax.solver import solve_from_config
from tokamaker_jax.verification import (
    rectangular_triangles,
    run_grad_shafranov_convergence_study,
    run_poisson_convergence_study,
)

ASSET_DIR = Path("docs/_static")

jax_config.update("jax_enable_x64", True)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    base = RunConfig(
        grid=GridConfig(nr=65, nz=65),
        source=SourceConfig(pressure_scale=5000.0, ffp_scale=-0.35),
        solver=SolverConfig(iterations=700, relaxation=0.75, dtype="float64"),
    )
    save_equilibrium_plot(solve_from_config(base), ASSET_DIR / "fixed_boundary_seed.png")
    write_region_geometry_preview()
    write_manufactured_poisson_convergence()
    write_grad_shafranov_convergence()
    write_coil_green_response()
    write_circular_loop_elliptic_response()
    write_profile_iteration()
    write_free_boundary_profile_coupling()
    write_validation_dashboard()
    write_benchmark_summary()
    write_case_manifest_assets()
    write_upstream_comparison_matrix()
    write_io_artifact_map()
    write_publication_validation_panel()
    write_openfusiontoolkit_comparison_report()
    write_cpc_seed_family()
    write_coil_current_sweep()
    write_pressure_sweep()


def write_region_geometry_preview() -> None:
    save_region_plot(sample_regions(), ASSET_DIR / "region_geometry_seed.png")


def write_manufactured_poisson_convergence() -> None:
    study = run_poisson_convergence_study((4, 8, 16))
    h = np.asarray([result.h for result in study.results])
    l2 = np.asarray([result.l2_error for result in study.results])
    h1 = np.asarray([result.h1_error for result in study.results])

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ax.loglog(h, l2, "o-", label="L2 error")
    ax.loglog(h, h1, "s-", label="H1 seminorm error")
    ax.loglog(h, l2[-1] * (h / h[-1]) ** 2, "--", color="C0", alpha=0.55, label="O(h^2)")
    ax.loglog(h, h1[-1] * (h / h[-1]), "--", color="C1", alpha=0.55, label="O(h)")
    ax.invert_xaxis()
    ax.set_xlabel("mesh size h")
    ax.set_ylabel("error")
    ax.set_title("p=1 manufactured Poisson convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(ASSET_DIR / "manufactured_poisson_convergence.png", dpi=180)
    plt.close(fig)


def write_grad_shafranov_convergence() -> None:
    study = run_grad_shafranov_convergence_study((4, 8, 16))
    h = np.asarray([result.h for result in study.results])
    l2 = np.asarray([result.l2_error for result in study.results])
    h1 = np.asarray([result.weighted_h1_error for result in study.results])

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    ax.loglog(h, l2, "o-", label="L2 error")
    ax.loglog(h, h1, "s-", label="weighted H1 seminorm error")
    ax.loglog(h, l2[-1] * (h / h[-1]) ** 2, "--", color="C0", alpha=0.55, label="O(h^2)")
    ax.loglog(h, h1[-1] * (h / h[-1]), "--", color="C1", alpha=0.55, label="O(h)")
    ax.invert_xaxis()
    ax.set_xlabel("mesh size h")
    ax.set_ylabel("error")
    ax.set_title("axisymmetric Grad-Shafranov manufactured convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(ASSET_DIR / "manufactured_grad_shafranov_convergence.png", dpi=180)
    plt.close(fig)


def write_coil_green_response() -> None:
    grid = RectangularGrid(1.0, 2.8, -0.9, 0.9, 101, 101)
    coils = (
        CoilConfig(name="PF_A", r=1.35, z=0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_B", r=1.35, z=-0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_C", r=2.45, z=0.0, current=-1.2e5, sigma=0.08),
    )
    save_coil_green_response_plot(grid, coils, ASSET_DIR / "coil_green_response.png")


def write_circular_loop_elliptic_response() -> None:
    grid = RectangularGrid(1.0, 2.8, -0.9, 0.9, 121, 121)
    coils = (
        CoilConfig(name="PF_A", r=1.35, z=0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_B", r=1.35, z=-0.45, current=2.0e5, sigma=0.06),
        CoilConfig(name="PF_C", r=2.45, z=0.0, current=-1.2e5, sigma=0.08),
    )
    r, z = grid.mesh()
    points = np.column_stack((np.asarray(r).reshape(-1), np.asarray(z).reshape(-1)))
    flux = np.asarray(circular_loop_elliptic_coil_flux(points, coils)).reshape(grid.nr, grid.nz)

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    filled = ax.contourf(np.asarray(r), np.asarray(z), flux, levels=30, cmap="viridis")
    contours = ax.contour(
        np.asarray(r), np.asarray(z), flux, levels=14, colors="black", linewidths=0.5
    )
    ax.clabel(contours, inline=True, fontsize=6)
    ax.scatter(
        [coil.r for coil in coils],
        [coil.z for coil in coils],
        c=["tab:red" if coil.current >= 0.0 else "tab:blue" for coil in coils],
        marker="s",
        edgecolor="black",
        linewidth=0.7,
    )
    for coil in coils:
        ax.text(coil.r + 0.04, coil.z, coil.name, fontsize=8, va="center")
    fig.colorbar(filled, ax=ax, label="circular-loop flux")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("closed-form circular-loop elliptic response")
    fig.savefig(ASSET_DIR / "circular_loop_elliptic_response.png", dpi=180)
    plt.close(fig)


def write_profile_iteration() -> None:
    solution = solve_profile_iteration_on_rectangle(
        subdivisions=16,
        parameters=NonlinearProfileParameters(
            pressure=PowerProfile(scale=4.0e3, alpha=1.25, gamma=1.0),
            ffprime=PowerProfile(scale=-0.25, alpha=1.0, gamma=1.0),
        ),
        iterations=5,
        relaxation=0.85,
    )
    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    contour = ax.tricontourf(
        np.asarray(solution.nodes[:, 0]),
        np.asarray(solution.nodes[:, 1]),
        np.asarray(solution.triangles),
        np.asarray(solution.psi),
        levels=24,
        cmap="viridis",
    )
    ax.tricontour(
        np.asarray(solution.nodes[:, 0]),
        np.asarray(solution.nodes[:, 1]),
        np.asarray(solution.triangles),
        np.asarray(solution.psi),
        levels=12,
        colors="black",
        linewidths=0.45,
    )
    fig.colorbar(contour, ax=ax, label="psi")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("nonlinear p=1 FEM profile iteration")
    fig.savefig(ASSET_DIR / "profile_iteration.png", dpi=180)
    plt.close(fig)


def write_free_boundary_profile_coupling() -> None:
    nodes, triangles = rectangular_triangles(1.0, 2.0, -0.5, 0.5, 16)
    coils = (
        CoilConfig(name="PF_U", r=2.28, z=0.72, current=1.25e5, sigma=0.05),
        CoilConfig(name="PF_L", r=2.28, z=-0.72, current=1.25e5, sigma=0.05),
        CoilConfig(name="PF_C", r=0.72, z=0.0, current=-0.75e5, sigma=0.06),
    )
    coil_flux = circular_loop_elliptic_coil_flux(nodes, coils)
    boundary_nodes = boundary_nodes_from_coordinates(nodes)
    solution = solve_profile_iteration(
        nodes,
        triangles,
        NonlinearProfileParameters(
            pressure=PowerProfile(scale=2.5, alpha=1.0, gamma=1.0),
            ffprime=PowerProfile(scale=-0.05, alpha=1.0, gamma=1.0),
        ),
        iterations=4,
        relaxation=0.75,
        initial_psi=coil_flux,
        dirichlet_nodes=boundary_nodes,
        dirichlet_values=coil_flux[boundary_nodes],
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    for ax, values, title in (
        (axes[0], np.asarray(coil_flux), "coil boundary flux"),
        (axes[1], np.asarray(solution.psi), "profile-coupled solution"),
    ):
        contour = ax.tricontourf(
            np.asarray(nodes[:, 0]),
            np.asarray(nodes[:, 1]),
            np.asarray(triangles),
            values,
            levels=24,
            cmap="viridis",
        )
        ax.tricontour(
            np.asarray(nodes[:, 0]),
            np.asarray(nodes[:, 1]),
            np.asarray(triangles),
            values,
            levels=10,
            colors="black",
            linewidths=0.38,
        )
        ax.scatter(
            [coil.r for coil in coils],
            [coil.z for coil in coils],
            c=["tab:red" if coil.current >= 0.0 else "tab:blue" for coil in coils],
            marker="s",
            edgecolor="black",
            linewidth=0.7,
            clip_on=False,
        )
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        fig.colorbar(contour, ax=ax, label="psi")
    fig.suptitle("free-boundary coil response coupled to nonlinear profile iteration")
    fig.savefig(ASSET_DIR / "free_boundary_profile_coupling.png", dpi=180)
    plt.close(fig)


def write_validation_dashboard() -> None:
    report = run_verification_gates("all", (4, 8, 16))
    gates = report["gates"]
    oft_relative_error = gates["openfusiontoolkit"]["relative_error"]
    oft_margin = -np.log10(oft_relative_error) if oft_relative_error is not None else 0.0
    free_boundary_errors = [
        gates["free_boundary_profile"]["boundary_error"],
        gates["free_boundary_profile"]["coil_linearity_relative_error"],
        gates["free_boundary_profile"]["current_gradient_error"],
    ]
    values = {
        "Poisson L2 rate": min(gates["poisson"]["l2_rates"]),
        "GS L2 rate": min(gates["grad_shafranov"]["l2_rates"]),
        "Circular loop": -np.log10(gates["circular_loop"]["elliptic_quadrature_relative_error"]),
        "OFT parity": oft_margin,
        "Coil Green": -np.log10(max(gates["coil_green"]["log_ratio_error"], 1.0e-30)),
        "Free-boundary": -np.log10(max(free_boundary_errors + [1.0e-30])),
        "Profile residual drop": gates["profile_iteration"]["residual_initial"]
        / gates["profile_iteration"]["residual_final"],
    }
    labels = list(values)
    metric_values = np.asarray([values[label] for label in labels], dtype=float)
    targets = np.asarray([1.8, 1.8, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=float)
    normalized = np.minimum(metric_values / targets, 1.35)
    colors = ["#2ca25f" if value >= 1.0 else "#de2d26" for value in normalized]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, normalized, color=colors)
    ax.axvline(1.0, color="black", linewidth=1.0, linestyle="--", label="gate target")
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.35)
    ax.set_xlabel("normalized validation margin")
    ax.set_title("tokamaker-jax physics validation dashboard")
    for index, value in enumerate(metric_values):
        ax.text(
            min(normalized[index] + 0.03, 1.28),
            index,
            f"{value:.3g}",
            va="center",
            fontsize=9,
        )
    ax.legend(loc="lower right")
    fig.savefig(ASSET_DIR / "validation_dashboard.png", dpi=180)
    plt.close(fig)


def write_benchmark_summary() -> None:
    report = benchmark_baseline_report(
        repeats=1,
        warmups=0,
        seed_equilibrium={"nr": 17, "nz": 17, "iterations": 20},
        axisymmetric_fem={"subdivisions": 6},
        coil_green={"nr": 17, "nz": 17},
        circular_loop={"n_points": 48},
    )
    (ASSET_DIR / "benchmark_report.json").write_text(
        benchmark_report_to_json(report),
        encoding="utf-8",
    )
    threshold_report = benchmark_threshold_report(report, DEFAULT_BENCHMARK_THRESHOLDS)
    (ASSET_DIR / "benchmark_threshold_report.json").write_text(
        benchmark_report_to_json(threshold_report),
        encoding="utf-8",
    )
    labels = [entry["lane"].replace("_", " ") for entry in report["benchmarks"]]
    medians_ms = np.asarray(
        [entry["result"]["median_s"] * 1000.0 for entry in report["benchmarks"]],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.barh(np.arange(len(labels)), medians_ms, color="#3182bd")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("median runtime [ms]")
    ax.set_title("baseline benchmark lanes")
    for index, value in enumerate(medians_ms):
        ax.text(value + max(medians_ms) * 0.015, index, f"{value:.2f}", va="center", fontsize=9)
    ax.set_xlim(0.0, max(medians_ms) * 1.25 + 1.0e-6)
    fig.savefig(ASSET_DIR / "benchmark_summary.png", dpi=180)
    plt.close(fig)


def write_case_manifest_assets() -> None:
    manifest = default_case_manifest()
    write_case_manifest(ASSET_DIR / "case_manifest.json", manifest=manifest)

    statuses = manifest.status_counts()
    labels = list(statuses)
    counts = np.asarray([statuses[label] for label in labels], dtype=float)
    colors = {
        "runnable": "#2ca25f",
        "validation_gate": "#3182bd",
        "schema_preview": "#756bb1",
        "planned_upstream_fixture": "#fdae6b",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y, counts, color=[colors.get(label, "#9e9ac8") for label in labels])
    ax.set_yticks(y, [label.replace("_", " ") for label in labels])
    ax.set_xlabel("case count")
    ax.set_title("case manifest: runnable cases and parity targets")
    for index, value in enumerate(counts):
        ax.text(value + 0.08, index, f"{int(value)}", va="center", fontsize=9)
    ax.set_xlim(0.0, max(counts) + 1.0)
    fig.savefig(ASSET_DIR / "case_manifest_status.png", dpi=180)
    plt.close(fig)


def write_upstream_comparison_matrix() -> None:
    rows = [
        {
            "reference": "OFT/TokaMaker",
            "level": "kernel_parity",
            "scores": [0.80, 0.65, 0.25, 0.40, 0.90],
            "next_gate": "fixed/free-boundary equilibrium parity",
        },
        {
            "reference": "TokaMaker CPC paper",
            "level": "surrogate_fixture",
            "scores": [0.60, 0.45, 0.25, 0.35, 0.80],
            "next_gate": "published figure data-level reproduction",
        },
        {
            "reference": "FreeGS/FreeGSNKE",
            "level": "source_audit",
            "scores": [0.45, 0.45, 0.10, 0.30, 0.55],
            "next_gate": "static inverse/passive structure fixture",
        },
        {
            "reference": "JAX-FEM/TORAX",
            "level": "design_parity",
            "scores": [0.75, 0.40, 0.20, 0.85, 0.70],
            "next_gate": "implicit differentiation benchmark",
        },
        {
            "reference": "COCOS/EFIT/bootstrap literature",
            "level": "source_audit",
            "scores": [0.30, 0.20, 0.10, 0.25, 0.50],
            "next_gate": "EQDSK/reconstruction/bootstrap gates",
        },
    ]
    columns = ["FEM/mesh", "free-boundary", "time/recon", "AD", "docs/tests"]
    report = {
        "schema_version": 1,
        "artifact_id": "upstream-literature-comparison-matrix",
        "status_levels": {
            "source_audit": "requirements extracted, no numeric parity claim",
            "surrogate_fixture": "workflow artifact exists without numeric parity claim",
            "design_parity": "design pattern adopted, no code-to-code physics claim",
            "kernel_parity": "scalar/vector kernel matches reference within tolerance",
        },
        "columns": columns,
        "rows": rows,
    }
    (ASSET_DIR / "upstream_comparison_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    matrix = np.asarray([row["scores"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(rows)), [row["reference"] for row in rows])
    ax.set_title("comparison coverage against upstream codes and literature")
    for i, row in enumerate(rows):
        for j, score in enumerate(row["scores"]):
            ax.text(j, i, f"{score:.2f}", ha="center", va="center", fontsize=8)
        ax.text(len(columns) + 0.15, i, row["level"], va="center", fontsize=8)
    ax.set_xlim(-0.5, len(columns) + 1.85)
    ax.text(len(columns) + 0.15, -0.75, "current level", fontsize=9, fontweight="bold")
    fig.colorbar(image, ax=ax, label="current comparison coverage")
    fig.savefig(ASSET_DIR / "upstream_comparison_matrix.png", dpi=180)
    plt.close(fig)


def write_io_artifact_map() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    width = 0.20
    height = 0.15
    boxes = {
        "TOML": (0.04, 0.72, "TOML cases\nexamples/*.toml"),
        "Python": (0.04, 0.44, "Python API\nload_config/solve"),
        "GUI": (0.04, 0.16, "GUI workflow\ncase + report views"),
        "Config": (0.33, 0.48, "validated dataclasses\nRunConfig/CoilConfig"),
        "Physics": (0.57, 0.48, "JAX physics kernels\nFEM + profiles + coils"),
        "Artifacts": (0.78, 0.62, "JSON reports\nvalidation/benchmarks"),
        "Figures": (0.78, 0.32, "PNG/GIF figures\ndocs/_static"),
    }
    for key, (x, y, text) in boxes.items():
        color = "#e8f4f8" if key in {"TOML", "Python", "GUI"} else "#f4f1de"
        rect = patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.015",
            linewidth=1.1,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.075, text, ha="center", va="center", fontsize=9)

    arrows = [
        ("TOML", "Config"),
        ("Python", "Config"),
        ("GUI", "Config"),
        ("Config", "Physics"),
        ("Physics", "Artifacts"),
        ("Physics", "Figures"),
    ]
    for start, end in arrows:
        sx, sy, _ = boxes[start]
        ex, ey, _ = boxes[end]
        ax.annotate(
            "",
            xy=(ex, ey + height / 2.0),
            xytext=(sx + width, sy + height / 2.0),
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1},
        )
    ax.text(
        0.80,
        0.16,
        "The GUI reads the same reports and figures\nthat CI/docs generate.",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.text(
        0.5,
        0.94,
        "tokamaker-jax reproducible input/output flow",
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.text(
        0.5,
        0.05,
        "Every GUI or docs artifact should map back to a command, dataclass config, JSON report, or figure recipe.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.savefig(ASSET_DIR / "io_artifact_map.png", dpi=180)
    plt.close(fig)


def write_publication_validation_panel() -> None:
    panels = [
        ("manufactured_grad_shafranov_convergence.png", "a) Grad-Shafranov convergence"),
        ("free_boundary_profile_coupling.png", "b) coupled free-boundary/profile solve"),
        ("validation_dashboard.png", "c) physics gate margins"),
        ("benchmark_summary.png", "d) benchmark lanes"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    for ax, (filename, title) in zip(axes.reshape(-1), panels, strict=True):
        image = mpimg.imread(ASSET_DIR / filename)
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=11)
    fig.suptitle("tokamaker-jax current validation and performance evidence", fontsize=15)
    fig.savefig(ASSET_DIR / "publication_validation_panel.png", dpi=200)
    plt.close(fig)


def write_openfusiontoolkit_comparison_report() -> None:
    comparison = run_openfusiontoolkit_green_comparison().to_dict()
    (ASSET_DIR / "openfusiontoolkit_comparison_report.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_cpc_seed_family() -> None:
    generate_cpc_seed_family_artifacts(
        ASSET_DIR,
        nr=31,
        nz=31,
        iterations=120,
        report_name="cpc_seed_family_report.json",
        png_name="cpc_seed_family.png",
        command="python examples/reproduce_cpc_seed_family.py docs/_static",
    )


def write_coil_current_sweep() -> None:
    grid = RectangularGrid(1.0, 2.8, -0.9, 0.9, 81, 81)
    r, z = grid.mesh()
    points = np.column_stack((np.asarray(r).reshape(-1), np.asarray(z).reshape(-1)))
    currents = np.linspace(-1.6e5, 1.6e5, 13)
    frames = []
    for current in currents:
        coils = (
            CoilConfig(name="PF_A", r=1.35, z=0.45, current=2.0e5, sigma=0.06),
            CoilConfig(name="PF_B", r=1.35, z=-0.45, current=2.0e5, sigma=0.06),
            CoilConfig(name="PF_C", r=2.45, z=0.0, current=float(current), sigma=0.08),
        )
        frames.append(
            np.asarray(circular_loop_elliptic_coil_flux(points, coils)).reshape(grid.nr, grid.nz)
        )
    vmin = min(float(np.min(frame)) for frame in frames)
    vmax = max(float(np.max(frame)) for frame in frames)
    levels = np.linspace(vmin, vmax, 28)
    fig, ax = plt.subplots(figsize=(5.8, 4.6), constrained_layout=True)

    def update(frame: int):
        ax.clear()
        ax.contourf(np.asarray(r), np.asarray(z), frames[frame], levels=levels, cmap="viridis")
        ax.contour(
            np.asarray(r), np.asarray(z), frames[frame], levels=12, colors="black", linewidths=0.45
        )
        ax.scatter(
            [1.35, 1.35, 2.45],
            [0.45, -0.45, 0.0],
            c=["tab:red", "tab:red", "tab:blue"],
            marker="s",
            edgecolor="black",
        )
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"PF_C current = {currents[frame] / 1000.0:.0f} kA")
        return []

    animation = FuncAnimation(fig, update, frames=len(frames), blit=False)
    animation.save(ASSET_DIR / "coil_current_sweep.gif", writer=PillowWriter(fps=3))
    plt.close(fig)


def write_pressure_sweep() -> None:
    pressures = np.linspace(1500.0, 8500.0, 12)
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)

    def update(frame: int):
        ax.clear()
        config = RunConfig(
            grid=GridConfig(nr=41, nz=41),
            source=SourceConfig(pressure_scale=float(pressures[frame]), ffp_scale=-0.35),
            solver=SolverConfig(iterations=220, relaxation=0.75, dtype="float64"),
        )
        solution = solve_from_config(config)
        plot_equilibrium(solution, levels=18, ax=ax, show_source=False)
        ax.set_title(f"pressure scale = {pressures[frame]:.0f}")
        return []

    animation = FuncAnimation(fig, update, frames=len(pressures), blit=False)
    animation.save(ASSET_DIR / "pressure_sweep.gif", writer=PillowWriter(fps=3))
    plt.close(fig)


if __name__ == "__main__":
    main()
