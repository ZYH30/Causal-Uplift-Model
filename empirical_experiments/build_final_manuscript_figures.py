"""Build publication-ready figures for the final manuscript.

The script reads immutable experiment artifacts and writes vector PDF plus
high-resolution PNG files. It does not refit any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"

COLORS = {
    "navy": "#264653",
    "teal": "#178F8A",
    "gold": "#E9A23B",
    "coral": "#E76F51",
    "gray": "#747A80",
    "light_gray": "#D7DBDE",
    "black": "#202326",
}

TASKS = {
    "antihypertensive_sbp": {
        "short": "Antihypertensive / SBP",
        "color": COLORS["teal"],
    },
    "chol_med_tchol": {
        "short": "Cholesterol medication / TC",
        "color": COLORS["coral"],
    },
}

MIMIC_RESULTS = RESULTS / "mimic_rbc_target_trial_v3"
MIMIC_AUDIT = RESULTS / "mimic_rbc_target_trial_v5_product_audit"
SYSTEMATIC_RESULTS = RESULTS / "systematic_simulation_v1"
ABLATION_RESULTS = RESULTS / "ccia_component_ablation_v1"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.dpi": 400,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label.lower(),
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        color=COLORS["black"],
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def build_synthetic_evidence(output_dir: Path) -> None:
    graph = pd.read_csv(SYSTEMATIC_RESULTS / "systematic_graph_summary.csv")
    cate = pd.read_csv(SYSTEMATIC_RESULTS / "systematic_cate_summary.csv")
    ablation = pd.read_csv(
        ABLATION_RESULTS / "ccia_component_ablation_summary.csv"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.45), constrained_layout=True)
    method_order = ["CCIA", "GES", "PC", "FCI", "DirectLiNGAM"]
    method_labels = {
        "CCIA": "CCIA",
        "GES": "GES",
        "PC": "PC",
        "FCI": "FCI",
        "DirectLiNGAM": "DirectLiNGAM",
    }
    method_colors = {
        "CCIA": COLORS["teal"],
        "GES": COLORS["coral"],
        "PC": COLORS["navy"],
        "FCI": COLORS["gold"],
        "DirectLiNGAM": COLORS["gray"],
    }
    method_markers = {
        "CCIA": "D",
        "GES": "o",
        "PC": "s",
        "FCI": "^",
        "DirectLiNGAM": "v",
    }

    def plot_graph_curve(
        ax: plt.Axes,
        scenario: str,
        metric: str,
        title: str,
        ylabel: str,
    ) -> None:
        subset = graph[
            (graph["Scenario"] == scenario) & (graph["Metric"] == metric)
        ]
        for method in method_order:
            rows = subset[subset["Method"] == method].sort_values("N")
            if rows.empty:
                continue
            error = rows["SD"].to_numpy(dtype=float) / np.sqrt(
                rows["Runs"].to_numpy(dtype=float)
            )
            ax.errorbar(
                rows["N"],
                rows["Mean"],
                yerr=error,
                marker=method_markers[method],
                markersize=4.5,
                capsize=2.2,
                lw=1.45,
                color=method_colors[method],
                label=method_labels[method],
            )
        ax.set_xscale("log")
        ax.set_xticks([500, 1000, 5000, 10000], ["500", "1k", "5k", "10k"])
        ax.set_ylim(-0.03, 1.04)
        ax.set_xlabel("Sample size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(color=COLORS["light_gray"], lw=0.55)

    ax = axes[0, 0]
    plot_graph_curve(
        ax,
        "nonlinear_reference",
        "Directed_F1",
        "Directed recovery under nonlinear mechanisms",
        "Directed F1",
    )
    ax.legend(frameon=False, ncol=2, loc="lower right")
    panel_label(ax, "a")

    ax = axes[0, 1]
    plot_graph_curve(
        ax,
        "noisy_missing",
        "C_Proximal_F1",
        "Assignment-role recovery with measurement noise",
        r"Assignment-proximal $C$ F1",
    )
    panel_label(ax, "b")

    ax = axes[1, 0]
    scenario_order = [
        "nonlinear_reference",
        "linear_additive",
        "weak_edges",
        "noisy_missing",
    ]
    scenario_labels = ["Nonlinear", "Linear", "Weak edges", "Noisy/missing"]
    interface_order = [
        "OracleRoles",
        "CCIARoles",
        "GESRoles",
        "AllTruePretreatment",
        "AllRecorded",
    ]
    interface_labels = {
        "OracleRoles": "Oracle roles",
        "CCIARoles": "CCIA roles",
        "GESRoles": "GES roles",
        "AllTruePretreatment": "All true baseline",
        "AllRecorded": "All recorded",
    }
    interface_colors = {
        "OracleRoles": COLORS["navy"],
        "CCIARoles": COLORS["teal"],
        "GESRoles": COLORS["coral"],
        "AllTruePretreatment": COLORS["gold"],
        "AllRecorded": COLORS["gray"],
    }
    interface_markers = {
        "OracleRoles": "o",
        "CCIARoles": "D",
        "GESRoles": "s",
        "AllTruePretreatment": "^",
        "AllRecorded": "X",
    }
    n10k = cate[
        (cate["N"] == 10000) & (cate["Metric"] == "CATE_MAE")
    ].set_index(["Scenario", "Interface"])
    x = np.arange(len(scenario_order), dtype=float)
    offsets = np.linspace(-0.20, 0.20, len(interface_order))
    for offset, interface in zip(offsets, interface_order):
        means = [
            float(n10k.loc[(scenario, interface), "Mean"])
            for scenario in scenario_order
        ]
        ses = [
            float(n10k.loc[(scenario, interface), "SD"])
            / np.sqrt(float(n10k.loc[(scenario, interface), "Runs"]))
            for scenario in scenario_order
        ]
        ax.errorbar(
            x + offset,
            means,
            yerr=ses,
            fmt=interface_markers[interface],
            markersize=5,
            capsize=2,
            color=interface_colors[interface],
            label=interface_labels[interface],
        )
    ax.set_yscale("log")
    ax.set_xticks(x, scenario_labels)
    ax.set_ylim(0.25, 3.1)
    ax.set_yticks([0.3, 0.5, 1.0, 2.0, 3.0], ["0.3", "0.5", "1", "2", "3"])
    ax.set_ylabel("CATE MAE at $n=10{,}000$ (log scale)")
    ax.set_title("Local roles preserve oracle downstream accuracy")
    ax.grid(axis="y", color=COLORS["light_gray"], lw=0.55)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    panel_label(ax, "c")

    ax = axes[1, 1]
    variant_order = [
        "Complete_CCIA",
        "PairwiseANM_FullRepair",
        "TaskAdaptive_NoRepair",
        "TaskAdaptive_NoDAGGate",
    ]
    variant_labels = [
        "Complete CCIA",
        "Pairwise ANM + repair",
        "Task-adaptive, no repair",
        "Task-adaptive, no DAG gate",
    ]
    y = np.arange(len(variant_order))[::-1]
    for y_value, variant in zip(y, variant_order):
        rows = ablation[ablation["Variant"] == variant].set_index("Metric")
        directed = float(rows.loc["Directed_F1", "Mean"])
        proximal = float(rows.loc["C_Proximal_F1", "Mean"])
        ax.plot(
            [directed, proximal],
            [y_value, y_value],
            color=COLORS["light_gray"],
            lw=2.8,
            zorder=1,
        )
        ax.scatter(
            directed,
            y_value,
            s=43,
            marker="o",
            color=COLORS["navy"],
            zorder=3,
        )
        ax.scatter(
            proximal,
            y_value,
            s=46,
            marker="D",
            color=COLORS["teal"],
            zorder=3,
        )
    ax.set_yticks(y, variant_labels)
    ax.set_xlim(0.35, 1.03)
    ax.set_xlabel("F1 score")
    ax.set_title("Direction and repair components are complementary")
    ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55)
    ax.scatter([], [], s=43, color=COLORS["navy"], label="Directed graph")
    ax.scatter(
        [],
        [],
        s=46,
        marker="D",
        color=COLORS["teal"],
        label=r"Assignment-proximal $C$",
    )
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "d")

    save_figure(fig, output_dir, "figure_2_synthetic_evidence_chain")


def build_mimic_target_trial_evidence(output_dir: Path) -> None:
    era = pd.read_csv(MIMIC_RESULTS / "mimic_rbc_era_summary.csv")
    full_balance = pd.read_csv(
        MIMIC_AUDIT / "mimic_rbc_full_surface_balance_summary.csv"
    )
    repeated = pd.read_csv(
        MIMIC_AUDIT / "mimic_rbc_safety_validated_repeated_summary.csv"
    ).set_index("Metric")
    temporal = pd.read_csv(
        MIMIC_AUDIT / "mimic_rbc_independent_evaluator_audit.csv"
    )
    temporal = temporal.loc[temporal["evaluator"] == "SafetyPromotedRole"].iloc[0]

    fig = plt.figure(figsize=(11.2, 6.25), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.05, 0.95],
        width_ratios=[1.30, 1.02, 1.02],
    )

    # A: clinical target-trial design and cohort flow.
    ax = fig.add_subplot(grid[0, 0])
    ax.set_xlim(-0.15, 4.35)
    ax.set_ylim(-0.10, 1.10)
    ax.axhline(0.47, color=COLORS["black"], lw=1.1)
    ax.axvspan(0, 1, ymin=0.32, ymax=0.60, color=COLORS["light_gray"], alpha=0.75)
    ax.axvspan(1, 2, ymin=0.32, ymax=0.60, color=COLORS["gold"], alpha=0.55)
    ax.axvspan(2, 3, ymin=0.32, ymax=0.60, color=COLORS["teal"], alpha=0.18)
    ax.scatter([1, 2, 3, 4], [0.47] * 4, s=34, color=COLORS["navy"], zorder=3)
    ax.text(0.43, 0.78, "Baseline\nlookback", ha="center", va="center", fontsize=7.0)
    ax.text(1.52, 0.78, "RBC start\ngrace period", ha="center", va="center", fontsize=7.0)
    ax.text(2.60, 0.78, "Post-landmark\nHb response", ha="center", va="center", fontsize=7.0)
    ax.text(4.00, 0.78, "90-day\nmortality", ha="center", va="center", fontsize=7.0)
    ax.text(0.92, 0.25, "Hb 7--9 g/dL\ntrigger", ha="center", va="top", fontsize=6.9)
    ax.text(2.08, 0.25, "Treatment\nlandmark", ha="center", va="top", fontsize=6.9)
    ax.text(
        -0.05,
        1.04,
        "18,742 triggers  -  2,432 prior RBC  -  742 not at landmark",
        ha="left",
        va="top",
        fontsize=7.5,
        color=COLORS["gray"],
    )
    ax.text(
        -0.05,
        -0.02,
        r"Final $n=15{,}568$: 3,017 early RBC; 3,312 deaths",
        ha="left",
        va="bottom",
        fontsize=7.8,
        fontweight="bold",
    )
    ax.set_xticks([0, 1, 2, 3, 4], ["-24 h", "0", "6 h", "24 h", "90 d"])
    ax.set_yticks([])
    ax.set_title("Longitudinal transfusion target trial")
    ax.text(
        1.0,
        -0.16,
        "Timeline shown schematically",
        transform=ax.transAxes,
        ha="right",
        fontsize=6.9,
        color=COLORS["gray"],
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "A")

    # B: discovery, model selection, and temporally held-out evaluation.
    ax = fig.add_subplot(grid[0, 1])
    periods = [
        ("Discover graph", "2008--2016", 10179, 0.2295, COLORS["navy"]),
        ("Select learner", "2017--2019", 3127, 0.1708, COLORS["gold"]),
        ("Temporal test", "2020--2022", 2262, 0.0650, COLORS["teal"]),
    ]
    y = np.arange(3)[::-1]
    for y_value, (stage, years, n_value, rate, color) in zip(y, periods):
        ax.barh(y_value, 0.62, height=0.50, color=color, alpha=0.90)
        ax.text(
            0.03,
            y_value,
            f"{stage}\n{years}",
            color="white",
            va="center",
            ha="left",
            fontsize=7.5,
            fontweight="bold",
        )
        ax.text(
            0.66,
            y_value,
            f"$n={n_value:,}$\nRBC {100 * rate:.1f}%",
            va="center",
            fontsize=7.4,
        )
    ax.set_xlim(0, 1.03)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Retrospective temporal holdout")
    ax.text(
        0.01,
        -0.04,
        "Non-overlapping eras separate discovery, selection, and evaluation.",
        transform=ax.transAxes,
        fontsize=7.2,
        color=COLORS["gray"],
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "B")

    # C: balance of the safety-validated roles in the temporal cohort.
    ax = fig.add_subplot(grid[0, 2])
    balance_lookup = full_balance.set_index(["surface", "audit_scope"])
    balance_sets = [
        (
            r"Assignment set $C$ (7)",
            float(
                balance_lookup.loc[
                    ("SafetyValidated_C", "SafetyC"), "mean_abs_smd_pre"
                ]
            ),
            float(
                balance_lookup.loc[
                    ("SafetyValidated_C", "SafetyC"), "mean_abs_smd_weighted"
                ]
            ),
            COLORS["teal"],
        ),
        (
            r"Retained $S$ (1)",
            float(
                balance_lookup.loc[
                    ("SafetyValidated_C", "SafetyS"), "mean_abs_smd_pre"
                ]
            ),
            float(
                balance_lookup.loc[
                    ("SafetyValidated_C", "SafetyS"), "mean_abs_smd_weighted"
                ]
            ),
            COLORS["gold"],
        ),
    ]
    y = np.array([1.0, 0.0])
    for y_value, (label, pre, post, color) in zip(y, balance_sets):
        ax.plot([post, pre], [y_value, y_value], color=COLORS["light_gray"], lw=3)
        ax.scatter(pre, y_value, s=48, color=COLORS["gray"], zorder=3)
        ax.scatter(post, y_value, s=55, marker="D", color=color, zorder=3)
    ax.text(
        0.98,
        0.94,
        "ESS/n=0.905",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
    )
    ax.axvline(0.10, color=COLORS["gray"], lw=0.8, ls="--")
    ax.set_yticks(y, [item[0] for item in balance_sets])
    ax.set_ylim(-0.35, 1.35)
    ax.set_xlim(0, 0.47)
    ax.set_xlabel(r"Mean $|\mathrm{SMD}|$")
    ax.set_title("Role safety and overlap")
    ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55)
    ax.scatter([], [], s=45, color=COLORS["gray"], label="Before")
    ax.scatter([], [], s=52, marker="D", color=COLORS["teal"], label="After")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "C")

    metric_specs = [
        (
            "R_Loss_Improvement",
            r"R-loss gain ($\times10^{-4}$)",
            10000.0,
            "D",
        ),
        (
            "DR_Qini_Improvement",
            "DR-Qini gain",
            1.0,
            "E",
        ),
        (
            "Policy_Risk_Reduction_Improvement",
            "Policy-risk gain (pp)",
            100.0,
            "F",
        ),
    ]
    for col, (metric, title, scale, label) in enumerate(metric_specs):
        ax = fig.add_subplot(grid[1, col])
        rep = repeated.loc[metric]
        points = [
            (
                float(rep["Mean"]) * scale,
                float(rep["CI_Low"]) * scale,
                float(rep["CI_High"]) * scale,
                COLORS["navy"],
            ),
            (
                float(temporal[metric]) * scale,
                float(temporal[f"{metric}_ci_low"]) * scale,
                float(temporal[f"{metric}_ci_high"]) * scale,
                COLORS["teal"],
            ),
        ]
        for y_value, (point, low, high, color) in zip([1, 0], points):
            ax.errorbar(
                point,
                y_value,
                xerr=[[point - low], [high - point]],
                fmt="D",
                markersize=5,
                color=color,
                ecolor=color,
                capsize=3,
                lw=1.35,
                zorder=3,
            )
            ax.text(
                point,
                y_value + 0.20,
                f"{point:.3f}" if scale == 1 else f"{point:.2f}",
                ha="center",
                fontsize=7.4,
            )
        ax.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
        ax.set_yticks([1, 0], ["20 repeated splits", "Temporal holdout"])
        if col > 0:
            ax.set_yticklabels([])
        ax.set_title(title)
        ax.set_xlabel("Positive values favor CCIA C+S")
        ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55)
        panel_label(ax, label)

    save_figure(fig, output_dir, "figure_3_mimic_target_trial_evidence")


def build_mimic_clinical_interpretation(output_dir: Path) -> None:
    profiles = pd.read_csv(
        MIMIC_RESULTS / "mimic_rbc_temporal_benefit_profiles.csv"
    )
    hb = pd.read_csv(MIMIC_RESULTS / "mimic_rbc_temporal_hb_mechanism.csv")
    budget = pd.read_csv(
        MIMIC_RESULTS / "mimic_rbc_temporal_policy_budget_curve.csv"
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.35), constrained_layout=True)
    quartiles = np.arange(1, 5)

    # A: calibration of predicted mortality benefit.
    ax = axes[0, 0]
    observed = profiles["dr_observed_benefit_mean"].to_numpy(dtype=float) * 100
    low = profiles["dr_observed_benefit_ci_low"].to_numpy(dtype=float) * 100
    high = profiles["dr_observed_benefit_ci_high"].to_numpy(dtype=float) * 100
    predicted = profiles["predicted_benefit_mean"].to_numpy(dtype=float) * 100
    ax.errorbar(
        quartiles,
        observed,
        yerr=[observed - low, high - observed],
        fmt="D-",
        color=COLORS["teal"],
        ecolor=COLORS["teal"],
        capsize=3,
        lw=1.5,
        markersize=5,
        label="DR-estimated benefit",
    )
    ax.plot(
        quartiles,
        predicted,
        "o--",
        color=COLORS["navy"],
        lw=1.3,
        markersize=4.5,
        label="Predicted benefit",
    )
    ax.axhline(0, color=COLORS["gray"], lw=0.8, ls="--")
    ax.set_xticks(quartiles, [f"Q{i}" for i in quartiles])
    ax.set_ylabel("Mortality risk reduction (percentage points)")
    ax.set_title("Benefit stratification in the temporal test")
    ax.grid(axis="y", color=COLORS["light_gray"], lw=0.55)
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "A")

    # B: clinical phenotype behind the benefit ranking.
    ax = axes[0, 1]
    phenotype = pd.DataFrame(
        {
            "Age": profiles["age_mean"],
            "BUN": profiles["lab_bun_last_mean"],
            "Urine output": profiles["urine_output_6h_mean"],
            "Mortality": profiles["observed_mortality_90d"],
        }
    )
    standardized = (phenotype - phenotype.mean()) / phenotype.std(ddof=0)
    clinical_map = LinearSegmentedColormap.from_list(
        "clinical_diverging",
        [COLORS["navy"], "#F4F5F3", COLORS["coral"]],
    )
    image = ax.imshow(
        standardized.to_numpy().T,
        aspect="auto",
        cmap=clinical_map,
        vmin=-1.55,
        vmax=1.55,
    )
    ax.set_xticks(np.arange(4), [f"Q{i}" for i in quartiles])
    ax.set_yticks(np.arange(4), phenotype.columns)
    ax.set_title("Clinical phenotype across benefit quartiles")
    labels = [
        [f"{value:.1f}" for value in phenotype["Age"]],
        [f"{value:.1f}" for value in phenotype["BUN"]],
        [f"{value:.0f}" for value in phenotype["Urine output"]],
        [f"{100 * value:.1f}%" for value in phenotype["Mortality"]],
    ]
    for row in range(4):
        for col in range(4):
            text_color = "white" if abs(standardized.iloc[col, row]) > 1.05 else COLORS["black"]
            ax.text(col, row, labels[row][col], ha="center", va="center", color=text_color, fontsize=7.4)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Within-feature standardized mean", fontsize=7.4)
    colorbar.ax.tick_params(labelsize=7)
    panel_label(ax, "B")

    # C: hemoglobin mechanism endpoint.
    ax = axes[1, 0]
    hb_quartiles = hb[hb["group"].str.startswith("benefit_quartile")].copy()
    point = hb_quartiles["aipw_hb_change_difference"].to_numpy(dtype=float)
    low = hb_quartiles["ci_low"].to_numpy(dtype=float)
    high = hb_quartiles["ci_high"].to_numpy(dtype=float)
    overall = float(hb.loc[hb["group"] == "overall", "aipw_hb_change_difference"].iloc[0])
    ax.errorbar(
        quartiles,
        point,
        yerr=[point - low, high - point],
        fmt="D",
        color=COLORS["coral"],
        ecolor=COLORS["coral"],
        capsize=3,
        markersize=5,
        lw=1.4,
    )
    ax.axhline(overall, color=COLORS["navy"], lw=1.2, ls="--", label=f"Overall: {overall:.2f} g/dL")
    ax.axhline(0, color=COLORS["gray"], lw=0.75)
    ax.set_xticks(quartiles, [f"Q{i}" for i in quartiles])
    ax.set_ylabel("Adjusted hemoglobin increase (g/dL)")
    ax.set_title("Physiologic response supports the ranking")
    ax.grid(axis="y", color=COLORS["light_gray"], lw=0.55)
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "C")

    # D: policy value over clinically relevant treatment budgets.
    ax = axes[1, 1]
    for interface, label, color, marker in [
        ("CCIA_CPlusS", "CCIA C+S", COLORS["teal"], "D"),
        ("AllVariables", "All variables", COLORS["gray"], "o"),
    ]:
        subset = budget[budget["Interface"] == interface].sort_values("Budget")
        ax.plot(
            100 * subset["Budget"],
            100 * subset["Risk_Reduction"],
            marker=marker,
            markersize=4.5,
            lw=1.5,
            color=color,
            label=label,
        )
    observed_budget = 100 * float(
        budget.loc[budget["Observed_Budget"], "Budget"].iloc[0]
    )
    ax.axvline(
        observed_budget,
        color=COLORS["gold"],
        lw=1.0,
        ls="--",
        label=f"Observed budget: {observed_budget:.1f}%",
    )
    ax.axhline(0, color=COLORS["gray"], lw=0.75)
    ax.set_xlabel("Treatment budget (% of patients)")
    ax.set_ylabel("Estimated risk reduction vs random (percentage points)")
    ax.set_title("Allocation value across treatment capacity")
    ax.grid(color=COLORS["light_gray"], lw=0.55)
    ax.legend(frameon=False, loc="best")
    panel_label(ax, "D")

    save_figure(fig, output_dir, "figure_4_mimic_clinical_interpretation")


def build_locked_confirmation(output_dir: Path) -> None:
    per_seed = pd.read_csv(
        RESULTS / "nhanes_llm_fixed_confirmatory_v1" / "fixed_confirmatory_per_seed.csv"
    )
    summary = pd.read_csv(
        RESULTS / "nhanes_llm_fixed_confirmatory_v1" / "fixed_confirmatory_summary.csv"
    )
    metrics = [
        ("R_Loss_Improvement", "R-loss improvement"),
        ("DR_Qini_Improvement", "DR-Qini improvement"),
        ("ObservedBudgetOutcome_Improvement", "Budget outcome improvement"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.25), constrained_layout=True)
    rng = np.random.default_rng(4102)
    for row_idx, (task, task_info) in enumerate(TASKS.items()):
        task_values = per_seed[per_seed["Task"] == task]
        task_summary = summary[summary["Task"] == task].set_index("Metric")
        for col_idx, (metric, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            values = task_values[metric].to_numpy(dtype=float)
            y_jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(
                values,
                y_jitter,
                s=23,
                color=task_info["color"],
                alpha=0.68,
                edgecolor="white",
                linewidth=0.4,
                zorder=2,
            )
            row = task_summary.loc[metric]
            mean = float(row["Mean"])
            low = float(row["Bootstrap_CI_Low"])
            high = float(row["Bootstrap_CI_High"])
            ax.errorbar(
                mean,
                0.29,
                xerr=[[mean - low], [high - mean]],
                fmt="D",
                color=COLORS["black"],
                ecolor=COLORS["black"],
                markersize=4.5,
                capsize=3,
                lw=1.25,
                zorder=3,
            )
            ax.axvline(0, color=COLORS["gray"], lw=0.8, ls="--", zorder=1)
            ax.set_ylim(-0.22, 0.64)
            ax.set_yticks([])
            ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55)
            if row_idx == 0:
                ax.set_title(title)
                panel_label(ax, chr(ord("A") + col_idx))
            if col_idx == 0:
                row_label = "SBP" if task == "antihypertensive_sbp" else "Total cholesterol"
                ax.set_ylabel(row_label, labelpad=10, fontweight="bold")
            p_value = float(row["Holm_T_P"])
            p_text = f"$p_H={p_value:.3g}$" if p_value >= 0.001 else f"$p_H={p_value:.1e}$"
            ax.text(
                0.98,
                0.91,
                f"{int(row['Wins'])}/20 wins; {p_text}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2, "alpha": 0.9},
            )
            ax.text(
                mean,
                0.39,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=COLORS["black"],
            )
            if row_idx == 1:
                ax.set_xlabel("Positive values favor CCIA-OCU")

    save_figure(fig, output_dir, "figure_4_locked_real_world_confirmation")


def build_clinical_diagnostics(output_dir: Path) -> None:
    records = []
    for task, filename in [
        (
            "antihypertensive_sbp",
            "nhanes_llm_conservative_antihypertensive_sbp_proximal_v1/"
            "nhanes_antihypertensive_sbp_run_meta.json",
        ),
        (
            "chol_med_tchol",
            "nhanes_llm_conservative_chol_med_tchol_proximal_v1/"
            "nhanes_chol_med_tchol_run_meta.json",
        ),
    ]:
        meta = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
        records.append(
            {
                "task": task,
                "raw": float(meta["raw_diff_treated_minus_control"]),
                "aipw": float(meta["aipw_ate"]),
                "low": float(meta["aipw_ci_low"]),
                "high": float(meta["aipw_ci_high"]),
                "pre_smd": float(meta["balance_c_pre_mean_abs_smd"]),
                "post_smd": float(meta["balance_c_ipw_mean_abs_smd"]),
                "ess": float(meta["overlap"]["ipw_ess_ratio"]),
            }
        )
    data = pd.DataFrame(records).set_index("task")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.30), constrained_layout=True)
    contrast_specs = [
        ("antihypertensive_sbp", "SBP contrast", "Treated - control (mmHg)"),
        ("chol_med_tchol", "Total-cholesterol contrast", "Treated - control (mg/dL)"),
    ]
    for idx, (task, title, xlabel) in enumerate(contrast_specs):
        ax = axes[idx]
        row = data.loc[task]
        ax.scatter(
            row["raw"],
            1,
            s=50,
            color=COLORS["gray"],
            label="Unadjusted",
            zorder=3,
        )
        ax.errorbar(
            row["aipw"],
            0,
            xerr=[[row["aipw"] - row["low"]], [row["high"] - row["aipw"]]],
            fmt="D",
            color=TASKS[task]["color"],
            markersize=5,
            capsize=3,
            lw=1.4,
            label="AIPW: $e(C),m(C,S)$",
            zorder=3,
        )
        ax.axvline(0, color=COLORS["gray"], lw=0.8, ls="--")
        ax.set_yticks([0, 1], ["AIPW", "Unadjusted"])
        ax.set_ylim(-0.55, 1.55)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55)
        panel_label(ax, chr(ord("A") + idx))
        if idx == 0:
            ax.legend(frameon=False, loc="lower right")

    ax = axes[2]
    y_positions = np.array([1, 0], dtype=float)
    for y, (task, task_info) in zip(y_positions, TASKS.items()):
        row = data.loc[task]
        ax.plot(
            [row["pre_smd"], row["post_smd"]],
            [y, y],
            color=COLORS["light_gray"],
            lw=3,
        )
        ax.scatter(row["pre_smd"], y, s=45, color=COLORS["gray"], zorder=3)
        ax.scatter(
            row["post_smd"],
            y,
            s=52,
            marker="D",
            color=task_info["color"],
            zorder=3,
        )
        ax.text(
            row["post_smd"],
            y + 0.20,
            f"ESS/n={row['ess']:.3f}",
            ha="left",
            fontsize=7.6,
        )
    ax.set_yticks(y_positions, [info["short"] for info in TASKS.values()])
    ax.set_xlim(0, 0.43)
    ax.set_xlabel(r"Mean $|\mathrm{SMD}|$ in learned $C$-set")
    ax.set_title("Balance and overlap")
    ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55)
    ax.scatter([], [], s=45, color=COLORS["gray"], label="Before weighting")
    ax.scatter([], [], s=52, marker="D", color=COLORS["teal"], label="After weighting")
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "C")

    save_figure(fig, output_dir, "figure_3_clinical_diagnostics")


def build_neurosymbolic_audit(output_dir: Path) -> None:
    audit = pd.read_csv(
        RESULTS / "paper_tables" / "table_3_neurosymbolic_graph_audit.csv"
    )
    task_labels = ["Antihypertensive / SBP", "Cholesterol medication / TC"]

    parsed = []
    for _, row in audit.iterrows():
        edge_before, edge_after = [int(item.strip()) for item in row["Edges"].split("->")]
        violation_before, violation_after = [
            int(item.strip()) for item in row["Task-order violations"].split("->")
        ]
        c_before, c_after = [int(item.strip()) for item in row["|C_prox|"].split("->")]
        s_before, s_after = [int(item.strip()) for item in row["|S|"].split("->")]
        parsed.append(
            {
                "edge_before": edge_before,
                "edge_after": edge_after,
                "violation_before": violation_before,
                "violation_after": violation_after,
                "c_before": c_before,
                "c_after": c_after,
                "s_before": s_before,
                "s_after": s_after,
            }
        )
    data = pd.DataFrame(parsed)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0), constrained_layout=True)
    x = np.arange(2)
    width = 0.30

    ax = axes[0]
    ax.bar(
        x - width / 2,
        data["violation_before"],
        width,
        color=COLORS["gray"],
        label="Statistical graph",
    )
    ax.bar(
        x + width / 2,
        data["violation_after"],
        width,
        color=COLORS["gold"],
        label="After semantic audit",
    )
    for xpos, value in zip(x - width / 2, data["violation_before"]):
        ax.text(xpos, value + 0.25, f"{value}", ha="center", fontsize=8)
    for xpos, value in zip(x + width / 2, data["violation_after"]):
        ax.text(xpos, value + 0.25, f"{value}", ha="center", fontsize=8)
    ax.set_xticks(x, task_labels, rotation=12, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Task-order violations")
    ax.set_ylim(0, 10.5)
    ax.grid(axis="y", color=COLORS["light_gray"], lw=0.55)
    panel_label(ax, "A")

    ax = axes[1]
    ax.bar(x - width / 2, data["edge_before"], width, color=COLORS["gray"])
    ax.bar(x + width / 2, data["edge_after"], width, color=COLORS["gold"])
    for xpos, value in zip(x - width / 2, data["edge_before"]):
        ax.text(xpos, value + 0.8, f"{value}", ha="center", fontsize=8)
    for xpos, value in zip(x + width / 2, data["edge_after"]):
        ax.text(xpos, value + 0.8, f"{value}", ha="center", fontsize=8)
    ax.set_xticks(x, task_labels, rotation=12, ha="right")
    ax.set_ylabel("Directed edges")
    ax.set_title("Local graph size")
    ax.set_ylim(0, 50)
    ax.grid(axis="y", color=COLORS["light_gray"], lw=0.55)
    panel_label(ax, "B")

    ax = axes[2]
    categories = [
        ("SBP: $C_P$", data.loc[0, "c_before"], data.loc[0, "c_after"]),
        ("SBP: $S$", data.loc[0, "s_before"], data.loc[0, "s_after"]),
        ("TC: $C_P$", data.loc[1, "c_before"], data.loc[1, "c_after"]),
        ("TC: $S$", data.loc[1, "s_before"], data.loc[1, "s_after"]),
    ]
    y = np.arange(len(categories))[::-1]
    for y_value, (_, before, after) in zip(y, categories):
        ax.plot([before, after], [y_value, y_value], color=COLORS["light_gray"], lw=3)
        ax.scatter(before, y_value, s=42, color=COLORS["gray"], zorder=3)
        ax.scatter(after, y_value, s=50, marker="D", color=COLORS["gold"], zorder=3)
        ax.text(before, y_value + 0.19, f"{before}", ha="center", fontsize=7.6)
        ax.text(after, y_value - 0.28, f"{after}", ha="center", fontsize=7.6)
    ax.set_yticks(y, [label for label, _, _ in categories])
    ax.set_xlim(0, 10)
    ax.set_xlabel("Set size")
    ax.set_title("Graph-induced causal roles")
    ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55)
    panel_label(ax, "C")

    fig.legend(
        handles=[
            Patch(facecolor=COLORS["gray"], label="Statistical graph"),
            Patch(facecolor=COLORS["gold"], label="After semantic audit"),
        ],
        frameon=False,
        loc="outside lower center",
        ncol=2,
    )

    save_figure(fig, output_dir, "figure_5_neurosymbolic_graph_audit")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    configure_style()
    build_synthetic_evidence(output_dir)
    build_mimic_target_trial_evidence(output_dir)
    build_mimic_clinical_interpretation(output_dir)
    build_clinical_diagnostics(output_dir)
    build_locked_confirmation(output_dir)
    build_neurosymbolic_audit(output_dir)
    print(f"Wrote final manuscript figures to {output_dir}")


if __name__ == "__main__":
    main()
