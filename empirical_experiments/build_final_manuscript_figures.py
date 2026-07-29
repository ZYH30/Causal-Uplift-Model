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
        label,
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
    direction = pd.read_csv(
        RESULTS
        / "direction_benchmark_multiseed_all_v1"
        / "direction_method_summary_by_seed.csv"
    )
    graph = json.loads(
        (
            RESULTS
            / "ccia_synthetic_n10000_task_adaptive_allvars_v5_treatment_rescue"
            / "ccia_graph_metrics.json"
        ).read_text(encoding="utf-8")
    )
    downstream = pd.read_csv(
        RESULTS
        / "ccia_synthetic_n10000_task_adaptive_allvars_v5_treatment_rescue"
        / "ccia_downstream_uplift.csv"
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.05),
        gridspec_kw={"width_ratios": [0.90, 1.05, 1.18]},
        constrained_layout=True,
    )

    # A: direction accuracy over seeds.
    ax = axes[0]
    method_order = ["pair_spearman_anm", "task_adaptive"]
    labels = ["Pairwise ANM", "Task-adaptive"]
    colors = [COLORS["gray"], COLORS["teal"]]
    rng = np.random.default_rng(417)
    for x, (method, label, color) in enumerate(zip(method_order, labels, colors)):
        values = direction.loc[direction["method"] == method, "accuracy"].to_numpy()
        jitter = rng.uniform(-0.07, 0.07, size=len(values))
        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=25,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        mean = float(np.mean(values))
        ax.plot([x - 0.17, x + 0.17], [mean, mean], color=COLORS["black"], lw=2.2)
        ax.text(x, mean + 0.035, f"{mean:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0.36, 1.075)
    ax.set_ylabel("Direction accuracy")
    ax.set_title("Mixed-type edge orientation")
    ax.grid(axis="y", color=COLORS["light_gray"], lw=0.55)
    panel_label(ax, "A")

    # B: graph and causal-role recovery.
    ax = axes[1]
    f1_rows = [
        ("Directed graph", graph["directed"]["f1"]),
        (r"$\mathrm{Pa}(T)$", graph["local"]["Parents_T"]["f1"]),
        (r"$\mathrm{Pa}(Y)$", graph["local"]["Parents_Y"]["f1"]),
        (r"$C_{\mathrm{P}}$", graph["ocu_sets"]["C_Proximal"]["f1"]),
        (r"$S_{\mathrm{prec}}$", graph["ocu_sets"]["S_Prec"]["f1"]),
    ]
    y = np.arange(len(f1_rows))[::-1]
    values = np.array([value for _, value in f1_rows])
    ax.hlines(y, 0.75, values, color=COLORS["light_gray"], lw=2)
    ax.scatter(values, y, s=45, color=COLORS["navy"], zorder=3)
    for x_value, y_value in zip(values, y):
        ax.text(x_value - 0.012, y_value + 0.27, f"{x_value:.3f}", ha="right", fontsize=7.8)
    ax.set_yticks(y, [name for name, _ in f1_rows])
    ax.set_xlim(0.73, 1.025)
    ax.set_xlabel("F1 score")
    ax.set_title("Task-relevant graph recovery")
    ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55)
    panel_label(ax, "B")

    # C: downstream oracle-versus-learned graph gap.
    ax = axes[2]
    models = ["S-Learner", "ClassTransform"]
    y = np.arange(len(models))[::-1]
    for y_value, model in zip(y, models):
        subset = downstream[downstream["Model"] == model].set_index("Setting")
        oracle = float(subset.loc["OCU_Oracle", "MAE"])
        learned = float(subset.loc["OCU_CCIA", "MAE"])
        ax.plot([oracle, learned], [y_value, y_value], color=COLORS["light_gray"], lw=3)
        ax.scatter(
            oracle,
            y_value,
            s=48,
            color=COLORS["navy"],
            label="Oracle graph" if model == models[0] else None,
            zorder=3,
        )
        ax.scatter(
            learned,
            y_value,
            s=54,
            marker="D",
            color=COLORS["teal"],
            label="CCIA graph" if model == models[0] else None,
            zorder=3,
        )
        ax.text(oracle, y_value + 0.19, f"{oracle:.3f}", ha="center", fontsize=7.6)
        ax.text(learned, y_value - 0.27, f"{learned:.3f}", ha="center", fontsize=7.6)
    ax.set_yticks(y, models)
    ax.set_xlim(0.15, 0.46)
    ax.set_xlabel("CATE MAE (lower is better)")
    ax.set_title("Oracle-to-learned graph transfer")
    ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55)
    ax.legend(frameon=False, loc="lower right")
    panel_label(ax, "C")

    save_figure(fig, output_dir, "figure_2_synthetic_evidence_chain")


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
    build_clinical_diagnostics(output_dir)
    build_locked_confirmation(output_dir)
    build_neurosymbolic_audit(output_dir)
    print(f"Wrote final manuscript figures to {output_dir}")


if __name__ == "__main__":
    main()
