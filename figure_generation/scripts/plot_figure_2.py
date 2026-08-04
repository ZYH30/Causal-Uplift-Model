"""Figure 2: systematic evidence from local structure to treatment effects."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tlcd_rite_mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from figure_style import (  # noqa: E402
    COLORS,
    FIGURE_WIDTH,
    configure_style,
    light_xgrid,
    light_ygrid,
    panel_label,
    save_figure,
    title_left,
)


DATA = HERE.parent / "data"
DEFAULT_OUTPUT = HERE.parent.parent / "figures"
SCENARIOS = [
    "nonlinear_reference",
    "linear_additive",
    "weak_edges",
    "noisy_missing",
]
SCENARIO_LABELS = ["Nonlinear", "Linear", "Weak edges", "Noise/missing"]

METHODS = ["TLCD", "GES", "PC", "FCI", "DirectLiNGAM"]
METHOD_LABELS = {
    "TLCD": "TLCD",
    "GES": "GES",
    "PC": "PC",
    "FCI": "FCI",
    "DirectLiNGAM": "DirectLiNGAM",
}
METHOD_COLORS = {
    "TLCD": COLORS["tlcd"],
    "GES": COLORS["ges"],
    "PC": "#5F9E89",
    "FCI": "#B48A38",
    "DirectLiNGAM": COLORS["gray"],
}
METHOD_MARKERS = {
    "TLCD": "D",
    "GES": "o",
    "PC": "s",
    "FCI": "^",
    "DirectLiNGAM": "v",
}

INTERFACES = [
    "OracleRoles",
    "TLCDRoles",
    "GESRoles",
    "AllTruePretreatment",
    "AllRecorded",
]
INTERFACE_LABELS = {
    "OracleRoles": "True roles",
    "TLCDRoles": "TLCD roles",
    "GESRoles": "GES roles",
    "AllTruePretreatment": "All true baseline",
    "AllRecorded": "All recorded",
}
INTERFACE_COLORS = {
    "OracleRoles": COLORS["black"],
    "TLCDRoles": COLORS["tlcd"],
    "GESRoles": COLORS["ges"],
    "AllTruePretreatment": COLORS["purple"],
    "AllRecorded": COLORS["gray"],
}
INTERFACE_MARKERS = {
    "OracleRoles": "o",
    "TLCDRoles": "D",
    "GESRoles": "s",
    "AllTruePretreatment": "^",
    "AllRecorded": "X",
}


def mean_and_se(table: pd.DataFrame, key: tuple[str, str]) -> tuple[float, float]:
    row = table.loc[key]
    return float(row["Mean"]), float(row["SD"]) / np.sqrt(float(row["Runs"]))


def plot_relative_to_oracle(
    parent_spec,
    fig: plt.Figure,
    cate: pd.DataFrame,
    metric: str,
    title: str,
    left_xlim: tuple[float, float],
    right_xlim: tuple[float, float],
    label: str,
) -> None:
    """Compare means near the true-role reference and isolate the collider stress test."""

    rows = cate[(cate["N"] == 10000) & (cate["Metric"] == metric)].set_index(
        ["Scenario", "Interface"]
    )
    subgrid = parent_spec.subgridspec(1, 2, width_ratios=[0.76, 0.24], wspace=0.10)
    ax = fig.add_subplot(subgrid[0, 0])
    stress = fig.add_subplot(subgrid[0, 1], sharey=ax)
    y = np.arange(len(SCENARIOS))[::-1].astype(float)
    role_aware = ["TLCDRoles", "GESRoles", "AllTruePretreatment"]
    offsets = [0.18, 0.0, -0.18]

    ax.axvspan(0.95, 1.05, color=COLORS["tlcd_light"], alpha=0.65, lw=0)
    ax.axvline(1.0, color=COLORS["black"], lw=0.9, ls="--", zorder=1)
    for interface, offset in zip(role_aware, offsets):
        ratios = []
        for scenario in SCENARIOS:
            oracle = mean_and_se(rows, (scenario, "OracleRoles"))[0]
            value = mean_and_se(rows, (scenario, interface))[0]
            ratios.append(value / oracle)
        ax.scatter(
            ratios,
            y + offset,
            marker=INTERFACE_MARKERS[interface],
            s=28,
            color=INTERFACE_COLORS[interface],
            zorder=3,
        )

    stress_ratios = []
    for scenario in SCENARIOS:
        oracle = mean_and_se(rows, (scenario, "OracleRoles"))[0]
        value = mean_and_se(rows, (scenario, "AllRecorded"))[0]
        stress_ratios.append(value / oracle)
    stress.scatter(
        stress_ratios,
        y,
        marker=INTERFACE_MARKERS["AllRecorded"],
        s=34,
        facecolor="none",
        edgecolor=INTERFACE_COLORS["AllRecorded"],
        linewidth=1.1,
        zorder=3,
    )

    ax.set_xlim(*left_xlim)
    stress.set_xlim(*right_xlim)
    ax.set_ylim(-0.48, 3.48)
    ax.set_yticks(y, SCENARIO_LABELS)
    stress.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlabel("Ratio to true-role model")
    stress.set_xlabel("")
    ax.text(
        1.0,
        3.40,
        "True roles",
        ha="center",
        va="top",
        fontsize=7.0,
        color=COLORS["black"],
        fontweight="bold",
    )
    stress.text(
        0.98,
        0.03,
        "Lower is better",
        transform=stress.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=COLORS["gray"],
    )
    title_left(ax, title)
    light_xgrid(ax)
    light_xgrid(stress)
    ax.spines["right"].set_visible(False)
    stress.spines["left"].set_visible(False)
    stress.spines["top"].set_visible(False)
    break_kwargs = dict(color=COLORS["black"], clip_on=False, lw=0.8)
    ax.plot((0.985, 1.015), (-0.018, 0.018), transform=ax.transAxes, **break_kwargs)
    stress.plot((-0.045, 0.045), (-0.018, 0.018), transform=stress.transAxes, **break_kwargs)
    panel_label(ax, label, x=-0.34)


def build_figure(output_dir: Path) -> None:
    graph = pd.read_csv(DATA / "systematic_graph_summary.csv")
    cate = pd.read_csv(DATA / "systematic_cate_summary.csv")
    ablation = pd.read_csv(DATA / "tlcd_component_ablation_summary.csv")

    fig = plt.figure(figsize=(FIGURE_WIDTH, 6.12))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.955,
        wspace=0.72,
        hspace=0.48,
    )

    # a: equal-spaced categorical sample-size trajectory.
    ax = fig.add_subplot(grid[0, 0])
    subset = graph[
        (graph["Scenario"] == "nonlinear_reference")
        & (graph["Metric"] == "Directed_F1")
    ]
    x = np.arange(4, dtype=float)
    x_offsets = np.linspace(-0.045, 0.045, len(METHODS))
    for method, x_offset in zip(METHODS, x_offsets):
        rows = subset[subset["Method"] == method].sort_values("N")
        se = rows["SD"].to_numpy(float) / np.sqrt(rows["Runs"].to_numpy(float))
        priority = method in {"TLCD", "GES"}
        ax.errorbar(
            x + x_offset,
            rows["Mean"],
            yerr=se,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            alpha=1.0 if priority else 0.78,
            lw=1.35 if priority else 0.85,
            capsize=2.0,
            elinewidth=0.8,
            markersize=5.1 if priority else 4.2,
            zorder=3 if priority else 2,
        )
        end_value = float(rows.iloc[-1]["Mean"])
        end_label = "LiNGAM" if method == "DirectLiNGAM" else METHOD_LABELS[method]
        ax.text(
            3.15,
            end_value,
            end_label,
            ha="left",
            va="center",
            fontsize=7.0,
            color=METHOD_COLORS[method],
            fontweight="bold" if priority else "normal",
            clip_on=True,
        )
    ax.set_xticks(x, ["500", "1k", "5k", "10k"])
    ax.set_xlim(-0.20, 4.12)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Directed F1")
    title_left(ax, "Nonlinear recovery improves\nwith sample size")
    light_ygrid(ax)
    panel_label(ax, "a")

    # b: paired comparison across data-generating mechanisms.
    ax = fig.add_subplot(grid[0, 1])
    directed = graph[
        (graph["N"] == 10000)
        & (graph["Metric"] == "Directed_F1")
        & (graph["Method"].isin(["TLCD", "GES"]))
    ].pivot(index="Scenario", columns="Method", values="Mean")
    y = np.arange(len(SCENARIOS))[::-1]
    for y_value, scenario in zip(y, SCENARIOS):
        tlcd = float(directed.loc[scenario, "TLCD"])
        ges = float(directed.loc[scenario, "GES"])
        favored = COLORS["tlcd_light"] if tlcd > ges else COLORS["ges_light"]
        ax.axhspan(y_value - 0.36, y_value + 0.36, color=favored, alpha=0.65, lw=0)
        ax.plot([tlcd, ges], [y_value, y_value], color=COLORS["mid_gray"], lw=1.8)
        ax.scatter(tlcd, y_value, marker="D", s=31, color=COLORS["tlcd"], zorder=3)
        ax.scatter(ges, y_value, marker="o", s=30, color=COLORS["ges"], zorder=3)
        delta = tlcd - ges
        ax.text(
            max(tlcd, ges) + 0.018,
            y_value,
            rf"$\Delta$ {delta:+.2f}",
            ha="left",
            va="center",
            fontsize=7.0,
            color=COLORS["tlcd"] if delta > 0 else COLORS["ges"],
            fontweight="bold",
        )
    ax.set_yticks(y, SCENARIO_LABELS)
    ax.set_xlim(0.66, 1.075)
    ax.set_xlabel(r"Directed F1 at $n=10{,}000$")
    title_left(ax, "Recovery varies\nacross mechanisms")
    ax.legend(
        handles=[
            Line2D([], [], marker="D", color="none", markerfacecolor=COLORS["tlcd"], markeredgecolor=COLORS["tlcd"], label="TLCD"),
            Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["ges"], markeredgecolor=COLORS["ges"], label="GES"),
        ],
        frameon=False,
        ncol=2,
        loc="lower right",
    )
    panel_label(ax, "b")

    # c: assignment and precision roles are visually separated.
    ax = fig.add_subplot(grid[0, 2])
    role_values = []
    for scenario in SCENARIOS:
        row = []
        for metric, method in [
            ("C_Proximal_F1", "TLCD"),
            ("C_Proximal_F1", "GES"),
            ("S_Precision_F1", "TLCD"),
            ("S_Precision_F1", "GES"),
        ]:
            value = graph[
                (graph["Scenario"] == scenario)
                & (graph["N"] == 10000)
                & (graph["Metric"] == metric)
                & (graph["Method"] == method)
            ]["Mean"].iloc[0]
            row.append(float(value))
        role_values.append(row)
    role_array = np.asarray(role_values)
    role_map = LinearSegmentedColormap.from_list(
        "role_f1", ["#F5F7F8", "#91C4DA", COLORS["tlcd"]]
    )
    image = ax.imshow(role_array, aspect="auto", cmap=role_map, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(4), ["TLCD", "GES", "TLCD", "GES"])
    ax.set_yticks(np.arange(4), SCENARIO_LABELS)
    ax.axvline(1.5, color="white", lw=4.0)
    for row in range(4):
        for col in range(4):
            value = role_array[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if value >= 0.72 else COLORS["black"],
                fontweight="bold" if value == role_array[row, (col // 2) * 2 : (col // 2) * 2 + 2].max() else "normal",
            )
    ax.text(0.5, 1.025, "Adjustment C", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    ax.text(2.5, 1.025, "Precision S", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.0, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.035, ticks=[0, 0.5, 1])
    colorbar.set_label("F1", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=7.0)
    title_left(ax, "Covariate-role recovery at $n=10{,}000$", pad=21)
    panel_label(ax, "c", y=1.17)

    plot_relative_to_oracle(
        grid[1, 0],
        fig,
        cate,
        "CATE_MAE",
        "CATE MAE relative to true roles",
        (0.975, 1.07),
        (3.5, 7.1),
        "d",
    )

    plot_relative_to_oracle(
        grid[1, 1],
        fig,
        cate,
        "Policy_Regret",
        "Policy regret relative to true roles",
        (0.92, 1.55),
        (4.0, 22.0),
        "e",
    )

    # f: component losses are shown relative to the complete TLCD method.
    ax = fig.add_subplot(grid[1, 2])
    variants = [
        ("Complete_TLCD", "Complete TLCD"),
        ("PairwiseANM_FullRepair", "Pairwise ANM\n+ endpoint checks"),
        ("TaskAdaptive_NoRepair", "No endpoint\nchecks"),
        ("TaskAdaptive_NoDAGGate", "No cycle check"),
    ]
    y = np.arange(4)[::-1]
    ax.axhspan(2.62, 3.38, color=COLORS["tlcd_light"], alpha=0.72, lw=0)
    complete = ablation[ablation["Variant"] == "Complete_TLCD"].set_index("Metric")
    complete_d = float(complete.loc["Directed_F1", "Mean"])
    complete_c = float(complete.loc["C_Proximal_F1", "Mean"])
    for y_value, (variant, label) in zip(y, variants):
        rows = ablation[ablation["Variant"] == variant].set_index("Metric")
        directed_f1 = float(rows.loc["Directed_F1", "Mean"])
        c_f1 = float(rows.loc["C_Proximal_F1", "Mean"])
        ax.plot([directed_f1, c_f1], [y_value, y_value], color=COLORS["light_gray"], lw=2.4)
        ax.scatter(directed_f1, y_value, s=29, marker="o", color=COLORS["black"], zorder=3)
        ax.scatter(c_f1, y_value, s=31, marker="D", color=COLORS["tlcd"], zorder=3)
        if variant == "Complete_TLCD":
            delta_label = "Reference"
        elif variant == "TaskAdaptive_NoDAGGate":
            delta_label = "$\\Delta D$ +0.00  |  $\\Delta C$ +0.00"
        else:
            delta_label = rf"$\Delta D$ {directed_f1-complete_d:+.2f}  |  $\Delta C$ {c_f1-complete_c:+.2f}"
        ax.text(1.025, y_value, delta_label, ha="left", va="center", fontsize=7.0, color=COLORS["gray"])
    ax.set_yticks(y, [label for _, label in variants])
    ax.set_ylim(-0.45, 3.45)
    ax.set_xlim(0.35, 1.39)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"F1 at $n=5{,}000$")
    title_left(ax, "Endpoint checks reduce\ndirection reversals")
    ax.scatter([0.45], [3.37], marker="o", s=25, color=COLORS["black"], zorder=3)
    ax.text(0.50, 3.37, "Directed F1", ha="left", va="center", fontsize=7.0)
    ax.scatter([0.98], [3.37], marker="D", s=27, color=COLORS["tlcd"], zorder=3)
    ax.text(1.03, 3.37, "C-set F1", ha="left", va="center", fontsize=7.0)
    panel_label(ax, "f")

    fig.legend(
        handles=[
            Line2D([], [], marker=INTERFACE_MARKERS[name], color=INTERFACE_COLORS[name], markerfacecolor="none" if name == "AllRecorded" else INTERFACE_COLORS[name], lw=0, label=INTERFACE_LABELS[name])
            for name in ["TLCDRoles", "GESRoles", "AllTruePretreatment", "AllRecorded"]
        ]
        + [Line2D([], [], color=COLORS["black"], ls="--", lw=0.9, label="True-role reference")],
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.018),
    )
    save_figure(fig, output_dir, "figure_2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    build_figure(args.output_dir)


if __name__ == "__main__":
    main()
