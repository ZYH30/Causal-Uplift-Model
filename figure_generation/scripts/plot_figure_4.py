"""Figure 4: clinical interpretation of personalized transfusion benefit."""

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


def build_figure(output_dir: Path) -> None:
    profiles = pd.read_csv(DATA / "mimic_rbc_temporal_benefit_profiles.csv")
    chart = pd.read_csv(DATA / "mimic_rbc_temporal_chart_profiles.csv")
    hb = pd.read_csv(DATA / "mimic_rbc_temporal_hb_mechanism.csv")
    budget = pd.read_csv(DATA / "mimic_rbc_temporal_policy_budget_curve.csv")

    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.82), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.03, 0.97])
    fig.set_constrained_layout_pads(w_pad=0.055, h_pad=0.055, wspace=0.09, hspace=0.11)
    quartiles = np.arange(1, 5)

    # a: predicted and DR-estimated mortality benefit with aligned treatment uptake.
    subgrid = grid[0, 0].subgridspec(1, 2, width_ratios=[0.78, 0.22], wspace=0.06)
    ax = fig.add_subplot(subgrid[0, 0])
    uptake_ax = fig.add_subplot(subgrid[0, 1], sharey=ax)
    observed = profiles["dr_observed_benefit_mean"].to_numpy(float) * 100
    low = profiles["dr_observed_benefit_ci_low"].to_numpy(float) * 100
    high = profiles["dr_observed_benefit_ci_high"].to_numpy(float) * 100
    predicted = profiles["predicted_benefit_mean"].to_numpy(float) * 100
    treatment = profiles["observed_treatment_rate"].to_numpy(float) * 100
    y = np.arange(4)[::-1]
    ax.errorbar(
        observed,
        y - 0.10,
        xerr=[observed - low, high - observed],
        fmt="D",
        color=COLORS["tlcd"],
        ecolor=COLORS["tlcd"],
        capsize=2.8,
        elinewidth=1.0,
        markersize=4.8,
        label="DR-estimated",
        zorder=3,
    )
    ax.scatter(
        predicted,
        y + 0.10,
        marker="o",
        s=20,
        color=COLORS["ges"],
        label="Predicted mean",
        zorder=4,
    )
    ax.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax.set_yticks(y, [f"Q{i}" for i in quartiles])
    ax.set_ylim(-0.60, 3.58)
    ax.set_xlim(-78, 57)
    ax.set_xlabel("90-day mortality benefit (percentage points)")
    ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 1.01))
    light_xgrid(ax)
    title_left(ax, "Predicted and DR-estimated benefit")
    panel_label(ax, "a", x=-0.18)

    uptake_ax.barh(y, treatment, height=0.48, color=COLORS["mid_gray"], alpha=0.88)
    for y_value, rate in zip(y, treatment):
        uptake_ax.text(rate + 0.35, y_value, f"{rate:.1f}%", ha="left", va="center", fontsize=6.5, color=COLORS["gray"], fontweight="bold")
    uptake_ax.set_xlim(0, 14.0)
    uptake_ax.set_xlabel("Treated (%)")
    uptake_ax.tick_params(axis="y", left=False, labelleft=False)
    uptake_ax.spines["left"].set_visible(False)
    title_left(uptake_ax, "Historical uptake")

    # b: clinical and pre-trigger bedside phenotypes in original units.
    ax = fig.add_subplot(grid[0, 1])
    phenotype = pd.DataFrame(
        {
            "Age (y)": profiles["age_mean"],
            "BUN (mg/dL)": profiles["lab_bun_last_mean"],
            "Mortality (%)": 100 * profiles["observed_mortality_90d"],
            "Medical service (%)": 100 * profiles["service_medical_mean"],
            "Cardiac ICU (%)": 100 * profiles["unit_cardiac_mean"],
            "GCS": chart["chart_gcs_combined_last_mean"],
            "Minimum MAP (mmHg)": chart["chart_map_min_mean"],
            "FiO2 charted (%)": 100 * chart["chart_fio2_recorded_mean"],
            "Ventilator measure (%)": 100 * chart["chart_ventilator_measurement_recorded_mean"],
        }
    )
    standardized = (phenotype - phenotype.mean()) / phenotype.std(ddof=0)
    clinical_map = LinearSegmentedColormap.from_list(
        "clinical_diverging", [COLORS["tlcd"], "#F7F7F5", COLORS["ges"]]
    )
    image = ax.imshow(
        standardized.to_numpy().T,
        aspect="auto",
        cmap=clinical_map,
        vmin=-1.6,
        vmax=1.6,
    )
    ax.set_xticks(np.arange(4), [f"Q{i}" for i in quartiles])
    ax.set_yticks(np.arange(len(phenotype.columns)), phenotype.columns)
    ax.tick_params(axis="y", labelsize=5.9)
    for separator in (1.5, 4.5):
        ax.axhline(separator, color="white", lw=2.0)
    one_decimal = {"Age (y)", "BUN (mg/dL)", "GCS", "Minimum MAP (mmHg)"}
    for row, column in enumerate(phenotype.columns):
        for col, value in enumerate(phenotype[column]):
            label = f"{value:.1f}" if column in one_decimal else f"{value:.0f}"
            z_value = float(standardized.iloc[col, row])
            text_color = "white" if abs(z_value) > 1.02 else COLORS["black"]
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=6.1,
                color=text_color,
                fontweight="bold" if abs(z_value) > 1.25 else "normal",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025, ticks=[-1, 0, 1])
    colorbar.set_label("Within-row z score", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6.2)
    title_left(ax, "Clinical and bedside phenotype")
    panel_label(ax, "b", x=-0.24)

    # c: overall physiologic response and the ordered quartile gradient.
    ax = fig.add_subplot(grid[1, 0])
    order = ["overall"] + [f"benefit_quartile_{i}" for i in range(1, 5)]
    labels = ["Overall", "Q1", "Q2", "Q3", "Q4"]
    rows = hb.set_index("group").loc[order]
    y = np.asarray([5.0, 3.0, 2.0, 1.0, 0.0])
    quartile_points = rows.iloc[1:]["aipw_hb_change_difference"].to_numpy(float)
    ax.plot(quartile_points, y[1:], color=COLORS["ges"], lw=0.75, alpha=0.70, zorder=1)
    ax.axhline(4.0, color=COLORS["light_gray"], lw=0.9)
    for index, (y_value, (_, row)) in enumerate(zip(y, rows.iterrows())):
        point = float(row["aipw_hb_change_difference"])
        low_value = float(row["ci_low"])
        high_value = float(row["ci_high"])
        color = COLORS["tlcd"] if index == 0 else COLORS["ges"]
        marker = "D" if index == 0 else "o"
        ax.errorbar(
            point,
            y_value,
            xerr=[[point - low_value], [high_value - point]],
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2.6,
            elinewidth=1.0,
            markersize=4.8,
            zorder=3,
        )
    ax.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax.text(
        0.98,
        0.95,
        "Q4 - Q1 = 1.11 g/dL",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color=COLORS["ges"],
        fontweight="bold",
    )
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.55, 5.55)
    ax.set_xlim(-0.78, 2.55)
    ax.set_xlabel("AIPW-adjusted hemoglobin change (g/dL)")
    title_left(ax, "Hemoglobin response across benefit strata")
    light_xgrid(ax)
    panel_label(ax, "c", x=-0.18)

    # d: decision value and the pointwise RITE advantage share budget positions.
    subgrid = grid[1, 1].subgridspec(2, 1, height_ratios=[0.73, 0.27], hspace=0.05)
    ax = fig.add_subplot(subgrid[0, 0])
    diff_ax = fig.add_subplot(subgrid[1, 0], sharex=ax)
    rite = budget[budget["Interface"].eq("RITE_CPlusS")].sort_values("Budget")
    all_vars = budget[budget["Interface"].eq("AllVariables")].sort_values("Budget")
    x_budget = 100 * rite["Budget"].to_numpy(float)
    rite_gain = 100 * rite["Risk_Reduction"].to_numpy(float)
    all_gain = 100 * all_vars["Risk_Reduction"].to_numpy(float)
    difference = rite_gain - all_gain
    better_count = int((difference > 0).sum())
    positive_count = int((rite_gain > 0).sum())
    ax.plot(x_budget, rite_gain, "D-", color=COLORS["tlcd"], lw=1.35, markersize=4.7)
    ax.plot(x_budget, all_gain, "o--", color=COLORS["gray"], lw=1.15, markersize=4.2)
    observed_budget = 100 * float(budget.loc[budget["Observed_Budget"], "Budget"].iloc[0])
    ax.axvline(observed_budget, color=COLORS["gold"], lw=0.95, ls="--")
    ax.axhline(0, color=COLORS["gray"], lw=0.75, ls=":")
    ax.text(x_budget[-1] + 0.45, all_gain[-1], "All variables", color=COLORS["gray"], ha="left", va="center", fontsize=6.7, clip_on=False)
    ax.text(observed_budget + 0.35, 5.50, f"Observed budget\n{observed_budget:.1f}%", ha="left", va="top", fontsize=6.5, color="#8A6100")
    ax.text(
        0.98,
        0.95,
        f"RITE > all: {better_count}/{len(x_budget)}\nRITE > random: {positive_count}/{len(x_budget)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        color=COLORS["tlcd"],
        bbox={"boxstyle": "square,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )
    ax.set_xlim(2.0, 33.6)
    ax.set_ylim(-2.1, 6.25)
    ax.set_ylabel("Risk reduction vs random (pp)")
    ax.tick_params(axis="x", labelbottom=False)
    title_left(ax, "Decision value across treatment capacity")
    light_ygrid(ax)
    panel_label(ax, "d", x=-0.24)

    diff_colors = np.where(difference >= 0, COLORS["tlcd"], COLORS["ges"])
    diff_ax.vlines(x_budget, 0, difference, color=diff_colors, lw=1.05, alpha=0.88)
    diff_ax.scatter(x_budget, difference, c=diff_colors, marker="s", s=19, zorder=3)
    diff_ax.axhline(0, color=COLORS["gray"], lw=0.75)
    diff_ax.axvline(observed_budget, color=COLORS["gold"], lw=0.95, ls="--")
    diff_ax.set_ylim(min(-1.4, difference.min() - 0.4), difference.max() + 0.65)
    diff_ax.set_xlabel("Treatment budget (% of patients)")
    diff_ax.set_ylabel(r"$\Delta$ risk" + "\n(pp)", fontsize=6.5)
    diff_ax.tick_params(axis="y", labelsize=6.1)

    save_figure(fig, output_dir, "figure_4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    build_figure(args.output_dir)


if __name__ == "__main__":
    main()
