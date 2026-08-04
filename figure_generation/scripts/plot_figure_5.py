"""Figure 5: constrained language-model review of the NHANES task graphs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tlcd_rite_mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
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
TASKS = ["Blood pressure", "Cholesterol"]
AFTER_COLOR = COLORS["tlcd"]


def before_after_panel(
    ax: plt.Axes,
    audit: pd.DataFrame,
    before_col: str,
    after_col: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    show_reduction: bool,
) -> None:
    y = np.arange(2)[::-1]
    for y_value, (_, row) in zip(y, audit.iterrows()):
        before = float(row[before_col])
        after = float(row[after_col])
        ax.plot([after, before], [y_value, y_value], color=COLORS["light_gray"], lw=2.8)
        ax.scatter(before, y_value, marker="o", s=33, color=COLORS["gray"], zorder=3)
        ax.scatter(after, y_value, marker="D", s=35, color=AFTER_COLOR, zorder=3)
        if show_reduction:
            ax.text(before, y_value + 0.16, f"{before:.0f}", ha="center", fontsize=7.0, color=COLORS["gray"])
            ax.text(after, y_value - 0.23, f"{after:.0f}", ha="center", fontsize=7.0, color=AFTER_COLOR, fontweight="bold")
            ax.text(
                (before + after) / 2,
                y_value + 0.02,
                f"{abs(before-after):.0f} fewer",
                ha="center",
                va="center",
                fontsize=7.0,
                color=COLORS["black"],
                fontweight="bold",
                bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none"},
            )
        else:
            ax.text(before, y_value + 0.18, f"{before:.0f}", ha="center", fontsize=7.0, color=COLORS["gray"], fontweight="bold")
            ax.text(after, y_value - 0.24, f"{after:.0f}", ha="center", fontsize=7.0, color=AFTER_COLOR, fontweight="bold")
    ax.set_yticks(y, TASKS)
    ax.set_ylim(-0.45, 1.68)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    title_left(ax, title)
    light_xgrid(ax)


def build_figure(output_dir: Path) -> None:
    audit = pd.read_csv(DATA / "table_3_neurosymbolic_graph_audit.csv")
    decisions = pd.read_csv(DATA / "llm_edge_decisions.csv")
    usage = json.loads((DATA / "llm_usage_summary.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIGURE_WIDTH, 4.75),
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.055, h_pad=0.055, wspace=0.08, hspace=0.10)

    ax = axes[0, 0]
    before_after_panel(
        ax,
        audit,
        "Order_Violations_Before",
        "Order_Violations_After",
        "Review removes detected\ntreatment-order violations",
        "Detected violation count",
        (-1.0, 10.4),
        True,
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["gray"], markeredgecolor=COLORS["gray"], label="Before"),
            Line2D([], [], marker="D", color="none", markerfacecolor=COLORS["tlcd"], markeredgecolor=COLORS["tlcd"], label="After review"),
        ],
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
    )
    panel_label(ax, "a")

    ax = axes[0, 1]
    before_after_panel(
        ax,
        audit,
        "Edges_Before",
        "Edges_After",
        "Directed edge counts\nafter accepted edits",
        "Directed edges",
        (24.2, 48.7),
        False,
    )
    ax.text(
        0.98,
        0.93,
        "Both final graphs acyclic",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        fontweight="bold",
        color=COLORS["teal"],
        bbox={"boxstyle": "square,pad=0.25", "facecolor": COLORS["teal_light"], "edgecolor": "none"},
    )
    panel_label(ax, "b")

    ax = axes[1, 0]
    role_rows = [
        ("BP: C", audit.loc[0, "C_Proximal_Before"], audit.loc[0, "C_Proximal_After"], COLORS["tlcd"]),
        ("BP: S", audit.loc[0, "S_Before"], audit.loc[0, "S_After"], COLORS["tlcd"]),
        ("TC: C", audit.loc[1, "C_Proximal_Before"], audit.loc[1, "C_Proximal_After"], COLORS["ges"]),
        ("TC: S", audit.loc[1, "S_Before"], audit.loc[1, "S_After"], COLORS["ges"]),
    ]
    y = np.asarray([3.15, 2.15, 0.85, -0.15])
    ax.axhspan(1.60, 3.55, color=COLORS["tlcd_light"], alpha=0.48, lw=0)
    ax.axhspan(-0.55, 1.40, color=COLORS["ges_light"], alpha=0.48, lw=0)
    ax.axhline(1.50, color=COLORS["light_gray"], lw=0.9)
    for y_value, (_, before, after, color) in zip(y, role_rows):
        before = float(before)
        after = float(after)
        ax.plot([before, after], [y_value, y_value], color=COLORS["light_gray"], lw=2.8)
        ax.scatter(before, y_value, marker="o", s=31, color=COLORS["gray"], zorder=3)
        ax.scatter(after, y_value, marker="D", s=33, color=color, zorder=3)
        ax.text(before, y_value + 0.20, f"{before:.0f}", ha="center", va="center", fontsize=7.0, color=COLORS["gray"], fontweight="bold")
        ax.text(after, y_value - 0.22, f"{after:.0f}", ha="center", va="center", fontsize=7.0, color=color, fontweight="bold")
    ax.set_yticks(y, [row[0] for row in role_rows])
    ax.set_ylim(-0.58, 3.58)
    ax.set_xlim(0, 10.4)
    ax.set_xlabel("Variables in each role set")
    title_left(ax, "Graph edits update\ncovariate-role sets")
    light_xgrid(ax)
    panel_label(ax, "c")

    # d: counts are separated from the API-usage footer.
    ax = axes[1, 1]
    directed_count = int(decisions["consensus_judgement"].isin(["A_TO_B", "B_TO_A"]).sum())
    no_edge_count = int((decisions["consensus_judgement"] == "NO_DIRECT_EDGE").sum())
    unresolved_count = int((decisions["consensus_judgement"] == "UNRESOLVED").sum())
    rows = [
        ("All consensus labels", [directed_count, no_edge_count, unresolved_count], int(len(decisions))),
        ("BP accepted edits", [int(audit.loc[0, "Semantic_Orientations"]), int(audit.loc[0, "Semantic_Removals"]), 0], int(audit.loc[0, "Semantic_Orientations"] + audit.loc[0, "Semantic_Removals"])),
        ("TC accepted edits", [int(audit.loc[1, "Semantic_Orientations"]), int(audit.loc[1, "Semantic_Removals"]), 0], int(audit.loc[1, "Semantic_Orientations"] + audit.loc[1, "Semantic_Removals"])),
    ]
    segment_colors = [COLORS["tlcd"], COLORS["ges"], COLORS["gray"]]
    segment_labels = ["Directed", "No edge", "Unresolved"]
    y = np.arange(3)[::-1]
    for y_value, (_, counts, total) in zip(y, rows):
        left = 0.0
        for count, color in zip(counts, segment_colors):
            if count == 0:
                continue
            width = 100.0 * count / total
            ax.barh(y_value, width, left=left, height=0.48, color=color)
            if width >= 9:
                ax.text(left + width / 2, y_value, f"{count}", ha="center", va="center", fontsize=7.0, color="white", fontweight="bold")
            else:
                ax.text(left + width + 1.2, y_value, f"{count}", ha="left", va="center", fontsize=7.0, color=color, fontweight="bold")
            left += width
    row_labels = ["Consensus", "BP edits", "TC edits"]
    ax.set_yticks(y, [f"{name}\nn={total}" for name, (_, _, total) in zip(row_labels, rows)])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.82, 2.92)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share within each row (%)")
    title_left(ax, "Review decisions and\naccepted graph changes")
    for x_value, color, legend_label in zip([3, 36, 68], segment_colors, segment_labels):
        ax.add_patch(Rectangle((x_value, 2.57), 5.5, 0.18, color=color, clip_on=True))
        ax.text(x_value + 7.0, 2.66, legend_label, ha="left", va="center", fontsize=7.0, color=COLORS["black"])
    ax.text(
        50,
        -0.68,
        f"Usage: {usage['valid_calls']} calls | {usage['total_tokens']:,} tokens",
        ha="center",
        va="center",
        fontsize=7.0,
        color=COLORS["gray"],
    )
    panel_label(ax, "d")

    save_figure(fig, output_dir, "figure_5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    build_figure(args.output_dir)


if __name__ == "__main__":
    main()
