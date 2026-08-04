"""Figure 3: MIMIC-IV design, balance, and held-out evidence."""

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
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from figure_style import (  # noqa: E402
    COLORS,
    FIGURE_WIDTH,
    configure_style,
    light_xgrid,
    panel_label,
    save_figure,
    title_left,
)


DATA = HERE.parent / "data"
DEFAULT_OUTPUT = HERE.parent.parent / "figures"


BALANCE_LABELS = {
    "baseline_hb_min": "Minimum Hb",
    "chart_gcs_combined_last": "GCS",
    "chart_rass_last": "RASS",
    "chart_temp_c_last": "Temperature",
    "lab_glucose_last_missing": "Glucose unrecorded",
    "lab_lactate_last": "Lactate",
    "service_medical": "Medical service",
    "trigger_hour": "Trigger time",
    "trigger_value": "Trigger Hb",
    "unit_cardiac": "Cardiac ICU",
    "urine_observed_6h": "Urine recorded",
}


def format_interval(point: float, low: float, high: float, decimals: int) -> str:
    return f"{point:.{decimals}f}\n[{low:.{decimals}f}, {high:.{decimals}f}]"


def plot_gain_panel(
    parent_spec,
    fig: plt.Figure,
    label: str,
    title: str,
    xlabel: str,
    repeated: pd.Series,
    temporal: pd.Series,
    scale: float,
    decimals: int,
) -> None:
    subgrid = parent_spec.subgridspec(1, 2, width_ratios=[0.64, 0.36], wspace=0.05)
    ax = fig.add_subplot(subgrid[0, 0])
    evidence = fig.add_subplot(subgrid[0, 1])

    points = [
        (
            "Repeated",
            float(repeated["Mean"]) * scale,
            float(repeated["CI_Low"]) * scale,
            float(repeated["CI_High"]) * scale,
            COLORS["tlcd"],
            "o",
        ),
        (
            "Temporal",
            float(temporal["Point_Estimate"]) * scale,
            float(temporal["CI_Low"]) * scale,
            float(temporal["CI_High"]) * scale,
            COLORS["teal"],
            "D",
        ),
    ]
    raw_min = min(point[2] for point in points)
    raw_max = max(point[3] for point in points)
    span = max(raw_max - raw_min, 1e-6)
    x_min = min(raw_min - 0.08 * span, -0.06 * span)
    x_max = max(raw_max + 0.08 * span, 0.06 * span)
    ax.axvspan(x_min, 0, color=COLORS["pale_gray"], alpha=0.95, lw=0)
    ax.axvspan(0, x_max, color=COLORS["tlcd_light"], alpha=0.68, lw=0)

    for y_value, (_, point, low, high, color, marker) in zip([1, 0], points):
        ax.errorbar(
            point,
            y_value,
            xerr=[[point - low], [high - point]],
            fmt=marker,
            markersize=5.0,
            color=color,
            ecolor=color,
            capsize=2.8,
            elinewidth=1.05,
            zorder=3,
        )
    ax.axvline(0, color=COLORS["gray"], lw=0.85, ls="--", zorder=2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.50, 1.63)
    ax.set_yticks([1, 0], ["Repeated", "Temporal"])
    ax.set_xlabel(xlabel)
    ax.text(
        0.02,
        0.90,
        "All variables",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        color=COLORS["gray"],
    )
    ax.text(
        0.98,
        0.90,
        "RITE",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color=COLORS["tlcd"],
        fontweight="bold",
    )
    light_xgrid(ax)
    title_left(ax, title)
    panel_label(ax, label, x=-0.27)

    evidence.set_axis_off()
    evidence.set_xlim(0, 1)
    evidence.set_ylim(-0.50, 1.63)
    evidence.text(
        0.02,
        1.52,
        "Gain\n[95% interval]",
        ha="left",
        va="center",
        fontsize=6.7,
        color=COLORS["gray"],
        fontweight="bold",
    )
    evidence.text(
        0.02,
        1.02,
        format_interval(points[0][1], points[0][2], points[0][3], decimals),
        ha="left",
        va="center",
        fontsize=6.7,
        color=COLORS["tlcd"],
        fontweight="bold",
    )
    evidence.text(
        0.02,
        0.70,
        f"{int(repeated['Wins'])}/20 wins",
        ha="left",
        va="center",
        fontsize=6.7,
        color=COLORS["gray"],
    )
    evidence.text(
        0.02,
        0.02,
        format_interval(points[1][1], points[1][2], points[1][3], decimals),
        ha="left",
        va="center",
        fontsize=6.7,
        color=COLORS["teal"],
        fontweight="bold",
    )
    evidence.text(
        0.02,
        -0.30,
        f"P(gain > 0)={100 * float(temporal['Probability_Positive']):.1f}%",
        ha="left",
        va="center",
        fontsize=6.5,
        color=COLORS["gray"],
    )


def build_figure(output_dir: Path) -> None:
    era = pd.read_csv(DATA / "mimic_rbc_analysis_eras.csv")
    balance = pd.read_csv(DATA / "mimic_rbc_temporal_balance.csv")
    diagnostics = pd.read_csv(DATA / "mimic_rbc_temporal_diagnostics.csv").iloc[0]
    flow = pd.read_csv(DATA / "mimic_rbc_cohort_flow.csv").set_index("stage")["count"]
    repeated = pd.read_csv(DATA / "mimic_rbc_repeated_summary.csv").set_index("Metric")
    temporal = pd.read_csv(DATA / "mimic_rbc_temporal_comparison.csv").set_index("Metric")

    fig = plt.figure(figsize=(FIGURE_WIDTH, 6.10))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.10, 0.90],
        left=0.070,
        right=0.985,
        bottom=0.090,
        top=0.950,
        wspace=0.63,
        hspace=0.40,
    )

    # a: target-trial timing and transparent cohort restriction.
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(-0.35, 4.25)
    ax.set_ylim(-0.37, 1.17)
    ax.add_patch(Rectangle((0.00, 0.91), 1.00, 0.13, color=COLORS["pale_gray"], lw=0))
    ax.add_patch(Rectangle((1.00, 0.74), 1.00, 0.13, color="#F9E7B6", lw=0))
    ax.add_patch(Rectangle((2.00, 0.58), 1.00, 0.11, color="#DDEEF7", lw=0))
    ax.add_patch(Rectangle((2.00, 0.43), 2.00, 0.11, color=COLORS["teal_light"], lw=0))
    timeline_x = np.arange(5, dtype=float)
    ax.plot(timeline_x, [0.35] * 5, color=COLORS["black"], lw=0.95)
    ax.scatter(timeline_x, [0.35] * 5, color=COLORS["tlcd"], s=23, zorder=3)
    for x_value, tick in zip(timeline_x, ["-24 h", "0 h", "6 h", "24 h", "90 d"]):
        ax.text(x_value, 0.29, tick, ha="center", va="top", fontsize=6.6)
    ax.text(0.92, 0.20, "Trigger", ha="right", va="top", fontsize=6.2, color=COLORS["gray"])
    ax.text(2.08, 0.20, "Landmark", ha="left", va="top", fontsize=6.2, color=COLORS["gray"])
    ax.text(0.50, 0.975, "Baseline\nlookback", ha="center", va="center", fontsize=6.1, fontweight="bold", linespacing=0.95)
    ax.text(1.50, 0.805, "RBC initiation\n0-6 h", ha="center", va="center", fontsize=6.1, fontweight="bold", linespacing=0.95)
    ax.text(2.50, 0.635, "Hb response", ha="center", va="center", fontsize=6.5, fontweight="bold")
    ax.text(3.00, 0.485, "90-day mortality", ha="center", va="center", fontsize=6.5, fontweight="bold")

    ax.text(
        1.95,
        0.02,
        f"{int(flow['qualifying_triggers']):,} triggers  |  {int(flow['target_trial_eligible']):,} landmark-eligible",
        ha="center",
        va="center",
        fontsize=6.7,
        color=COLORS["gray"],
    )
    ax.text(
        1.95,
        -0.085,
        f"{int(flow['excluded_prior_rbc']):,} prior RBC  |  {int(flow['excluded_unavailable_landmark']):,} unavailable",
        ha="center",
        va="center",
        fontsize=6.7,
        color=COLORS["gray"],
    )
    ax.text(
        1.95,
        -0.190,
        f"{int(flow['excluded_no_support']):,} excluded: no treated temporal support",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#8A6100",
    )
    ax.text(
        1.95,
        -0.310,
        f"{int(flow['analysis_cohort']):,} analyzed  |  {int(flow['early_rbc']):,} RBC  |  {int(flow['deaths_90d']):,} deaths",
        ha="center",
        va="center",
        fontsize=6.8,
        color=COLORS["black"],
        fontweight="bold",
    )
    ax.set_axis_off()
    title_left(ax, "Target trial and analysis cohort")
    panel_label(ax, "a")

    # b: non-overlapping discovery, selection, and temporal-test eras.
    ax = fig.add_subplot(grid[0, 1])
    x = np.arange(len(era))
    background = [COLORS["tlcd_light"], "#F9EBC9", COLORS["teal_light"]]
    for x_value, color in zip(x, background):
        ax.axvspan(x_value - 0.46, x_value + 0.46, color=color, alpha=0.72, lw=0)
    treatment_rate = 100 * era["treatment_rate"].to_numpy(float)
    mortality_rate = 100 * era["outcome_rate"].to_numpy(float)
    ax.plot(x, treatment_rate, "D-", color=COLORS["tlcd"], lw=1.35, label="Early RBC")
    ax.plot(x, mortality_rate, "o--", color=COLORS["ges"], lw=1.25, label="90-day mortality")
    for index, x_value in enumerate(x):
        treatment_offset = 0.9 if index != 1 else -1.0
        treatment_va = "bottom" if treatment_offset > 0 else "top"
        mortality_offset = -0.9 if index != 1 else 1.0
        mortality_va = "top" if mortality_offset < 0 else "bottom"
        ax.text(x_value, treatment_rate[index] + treatment_offset, f"{treatment_rate[index]:.1f}%", ha="center", va=treatment_va, fontsize=6.7, color=COLORS["tlcd"], fontweight="bold")
        ax.text(x_value, mortality_rate[index] + mortality_offset, f"{mortality_rate[index]:.1f}%", ha="center", va=mortality_va, fontsize=6.7, color=COLORS["ges"], fontweight="bold")
        ax.text(x_value, 1.1, f"n={int(era.iloc[index]['n']):,}", ha="center", va="bottom", fontsize=6.5, color=COLORS["gray"])
    tick_labels = [
        f"{row.calendar_era}\n{row.analysis_role}"
        for row in era.itertuples(index=False)
    ]
    ax.set_xticks(x, tick_labels)
    ax.set_ylim(0, 29)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylabel("Cohort rate (%)")
    ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    title_left(ax, "Treatment and outcome shift\nacross eras")
    panel_label(ax, "b")

    # c: every variable in the final treatment-adjustment set is shown.
    ax = fig.add_subplot(grid[0, 2])
    balance = balance.sort_values("abs_smd_weighted", ascending=True).reset_index(drop=True)
    y = np.arange(len(balance))
    pre = balance["abs_smd_pre"].to_numpy(float)
    weighted = balance["abs_smd_weighted"].to_numpy(float)
    for y_value, before, after in zip(y, pre, weighted):
        ax.plot([after, before], [y_value, y_value], color=COLORS["light_gray"], lw=1.8, zorder=1)
    ax.scatter(pre, y, marker="o", s=20, color=COLORS["gray"], label="Before", zorder=2)
    ax.scatter(weighted, y, marker="D", s=22, color=COLORS["tlcd"], label="Weighted", zorder=3)
    ax.axvline(0.10, color=COLORS["ges"], ls="--", lw=0.9)
    ax.text(0.103, len(balance) - 0.15, "0.10", ha="left", va="top", fontsize=6.2, color=COLORS["ges"])
    ax.set_yticks(y, [BALANCE_LABELS[value] for value in balance["feature"]])
    ax.tick_params(axis="y", labelsize=5.9, pad=2.0)
    ax.set_xlim(0, 0.66)
    ax.set_ylim(-0.75, len(balance) + 0.72)
    ax.set_xlabel("Absolute standardized mean difference")
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.925),
        fontsize=6.2,
    )
    ax.text(
        0.02,
        0.99,
        f"Mean 0.386 to 0.111  |  ESS/n={float(diagnostics['ess_ratio']):.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.4,
        color=COLORS["black"],
        bbox={"boxstyle": "square,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.90},
    )
    light_xgrid(ax)
    title_left(ax, "Temporal balance across 11\nadjustment variables")
    panel_label(ax, "c", x=-0.25)

    metric_specs = [
        ("R_Loss_Improvement", "Effect risk: R-loss", r"Gain ($\times10^{-4}$)", 10000.0, 2),
        ("DR_Qini_Improvement", "Ranking: DR-Qini", "Gain", 1.0, 3),
        ("Policy_Risk_Reduction_Improvement", "Policy value", "Gain (pp)", 100.0, 2),
    ]
    for col, (metric, title, xlabel, scale, decimals) in enumerate(metric_specs):
        plot_gain_panel(
            grid[1, col],
            fig,
            chr(ord("d") + col),
            title,
            xlabel,
            repeated.loc[metric],
            temporal.loc[metric],
            scale,
            decimals,
        )

    save_figure(fig, output_dir, "figure_3")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    build_figure(args.output_dir)


if __name__ == "__main__":
    main()
