"""NHANES effect-learning gains and graph-based covariate assignment."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from collections import Counter
from matplotlib.patches import PathPatch, Rectangle, Patch
from matplotlib.path import Path as MplPath
from .figure_io import export_figure

STEM = "Figure_6_NHANES"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "tlcd-rite-figure-6",
    "mathtext.fontset": "dejavusans",
    "hatch.linewidth": 0.6,
    "savefig.bbox": None,
}

W, H = (162.0, 132.0)
ROLE_PLOT_WIDTH = 72.0
INK, GRAY, GRID = ("#25333D", "#687582", "#DEE5EB")
BLUE, TEAL, ORANGE = ("#0072B2", "#24B6A4", "#D55E00")
ROLE_COLORS = {"C": BLUE, "S": TEAL, "O": "#BCC5D0"}
ACTION_COLORS = [BLUE, "#F3A33B", "#EF7C83", "#CED6E0", "#9E80C8"]
ACTION_LABELS = [
    "Oriented/confirmed",
    "Semantic removal",
    "Task-order removal",
    "Retained",
    "Query/DAG rules",
]
TASKS = ["antihypertensive_sbp", "chol_med_tchol"]
TITLES = ["BP medication: SBP", "Cholesterol medication: TC"]
METRICS = [
    ("R-loss", "R_Loss_Improvement", BLUE),
    ("DR-Qini", "DR_Qini_Improvement", TEAL),
    ("Budget outcome", "ObservedBudgetOutcome_Improvement", ORANGE),
]
XLIMS = [[(-1.05, 2.0), (-0.8, 1.8), (-0.8, 1.8)], [(-20.0, 140.0), (-1.0, 2.6), (-3.0, 4.5)]]
XTICKS = [[[-1, 0, 2], [-0.5, 0.5, 1.5], [-0.5, 0.5, 1.5]], [[0, 60, 120], [0, 1, 2], [-2, 0, 4]]]
BOUNDS = {
    "a": (1.0, 1.0, 79.0, 53.0),
    "b": (83.0, 1.0, 78.0, 53.0),
    "c": (1.0, 60.5, 79.0, 63.8),
    "d": (83.0, 60.5, 78.0, 63.8),
}


def read_data(directory):
    raw = pd.read_csv(directory / "paired/fixed_confirmatory_per_seed.csv")
    summaries = pd.read_csv(directory / "nhanes_reported_comparisons.csv")
    metadata = json.loads((directory / "nhanes_task_metadata.json").read_text())
    result = {}
    for i, task in enumerate(TASKS):
        rows = raw[raw.Task.eq(task)].sort_values("Seed")
        assert len(rows) == 20 and rows.Seed.nunique() == 20
        assert rows.Method.eq("RITE_COnly").all()
        meta = json.loads(
            (directory / f"paired/nhanes_{task}_graph_llm_conservative_meta.json").read_text()
        )
        graph = json.loads(
            (directory / f"paired/nhanes_{task}_graph_llm_conservative.json").read_text()
        )
        universe = sorted(set(graph) - {"T", "Y"})
        assert len(universe) == 30
        sets = [meta[f"sets_{stage}"] for stage in ["before", "after"]]
        states = []
        for state in sets:
            c, s = (set(state["C_Proximal"]), set(state["S_Prec"]))
            assert not c & s
            states.append({v: "C" if v in c else "S" if v in s else "O" for v in universe})
        selected = [v for v in universe if states[0][v] != "O" or states[1][v] != "O"]
        flows = []
        for a in ["C", "S", "O"]:
            for b in ["C", "S", "O"]:
                variables = [v for v in selected if states[0][v] == a and states[1][v] == b]
                if variables:
                    flows.append(dict(before=a, after=b, count=len(variables), variables=variables))
        assert len(selected) == [10, 12][i]
        counts = meta["action_counts"]
        assert dict(Counter((a["action"] for a in meta["actions"]))) == counts
        hard_actions = [a for a in meta["actions"] if a["action"] == "removed_by_hard_task_order"]
        initial = {
            tuple(sorted([a["node_a"], a["node_b"]])): a
            for a in meta["actions"]
            if a["action"] != "removed_by_hard_task_order"
        }
        assert len(initial) == [37, 33][i]
        assert all(
            (
                initial[tuple(sorted([a["node_a"], a["node_b"]]))]["action"]
                == "retained_statistical"
                for a in hard_actions
            )
        )
        hard_count = len(hard_actions)
        assert hard_count == counts.get("removed_by_hard_task_order", 0)
        actions = [
            counts.get("oriented_by_semantic_consensus", 0),
            counts.get("removed_by_semantic_consensus", 0),
            hard_count,
            counts.get("retained_statistical", 0) - hard_count,
            counts.get("fixed_treatment_outcome", 0)
            + counts.get("removed_cycle_blocked_orientation", 0),
        ]
        assert sum(actions) == [37, 33][i]
        records = []
        for j, (metric, column, _) in enumerate(METRICS):
            record = summaries[summaries.Task.eq(task) & summaries.Metric.eq(metric)].iloc[0]
            values = rows[column].to_numpy()
            assert abs(values.mean() - record.Mean) <= 0.000501
            assert np.isfinite(values).all()
            records.append(
                dict(
                    metric=metric,
                    column=column,
                    seeds=rows.Seed.tolist(),
                    values=values.tolist(),
                    positive=int(sum(values > 0)),
                    negative=int(sum(values < 0)),
                    zero=int(sum(values == 0)),
                    mean=float(record.Mean),
                    low=float(record.CI_Low),
                    high=float(record.CI_High),
                    limits=list(XLIMS[i][j]),
                    unit=record.Units,
                )
            )
        result[task] = dict(
            records=records,
            metadata=metadata[task],
            meta=meta,
            states=states,
            selected=selected,
            flows=flows,
            action_counts=actions,
        )
    return result


def label(
    fig,
    x,
    y,
    s,
    panel,
    *,
    ha="left",
    va="top",
    size=7.2,
    weight="normal",
    color=INK,
    rotation=0,
    bbox=None,
):
    artist = fig.text(
        x / W,
        1 - y / H,
        s,
        ha=ha,
        va=va,
        fontsize=size,
        weight=weight,
        color=color,
        rotation=rotation,
        linespacing=1.05,
        bbox=bbox,
    )
    artist._panel = panel
    if ha == "center" and rotation == 0:
        artist._center_x_mm = x
    return artist


def title(fig, letter, value):
    x, y, w, _ = BOUNDS[letter]
    label(fig, x + 0.2, y + 0.2, letter, letter, weight="bold", size=9.3)
    label(fig, x + w / 2, y + 0.4, value, letter, ha="center", weight="bold", size=8.3)


def axes(fig, x, y, w, h, panel):
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax._panel = panel
    ax.tick_params(length=1.6, width=0.55, labelsize=7.0, pad=1.5, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.spines[["left", "bottom"]].set_linewidth(0.55)
    ax.set_axisbelow(True)
    return ax


def add_legend(fig, handles, x, y, ncol, panel):
    legend = fig.legend(
        handles=handles,
        ncol=ncol,
        frameon=False,
        loc="center",
        bbox_to_anchor=(x / W, 1 - y / H),
        bbox_transform=fig.transFigure,
        fontsize=7.0,
        handlelength=1.0,
        handleheight=0.8,
        handletextpad=0.45,
        columnspacing=1.55,
        borderpad=0,
        borderaxespad=0,
    )
    legend._center_x_mm = x
    for t in legend.get_texts():
        t._panel = panel
    return legend


def gain_panel(fig, data, i):
    letter = "ab"[i]
    task = TASKS[i]
    record = data[task]
    x, _, w, _ = BOUNDS[letter]
    title(fig, letter, TITLES[i])
    meta = record["metadata"]
    subtitle = f"n = {meta['cohort_n']:,}; {meta['learner']}"
    label(fig, x + w / 2, 6.6, subtitle, letter, ha="center", size=7.2)
    label(
        fig,
        x + 1.2,
        27.3,
        "Cumulative splits (%)",
        letter,
        rotation=90,
        ha="center",
        va="center",
        size=7.0,
    )
    unit = ["mmHg", "mg/dL"][i]
    for j, r in enumerate(record["records"]):
        x0, width = (x + 10.0 + j * 23.0, 20.0)
        cx = x0 + width / 2
        color = METRICS[j][2]
        label(fig, cx, 11.5, r["metric"], letter, ha="center", weight="bold", size=7.2)
        squared_unit = "mmHg$^2$" if i == 0 else "(mg/dL)$^2$"
        label(
            fig,
            cx,
            14.5,
            squared_unit if j == 0 else unit,
            letter,
            ha="center",
            color=GRAY,
            size=7.0,
        )
        ax = axes(fig, x0, 18.7, width, 17.3, letter)
        ax.set_xlim(r["limits"])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100], labels=["0", "50", "100"] if j == 0 else ["", "", ""])
        ax.set_xticks(XTICKS[i][j])
        ax.tick_params(axis="x", pad=3.0)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.axvspan(0, r["limits"][1], facecolor=color, alpha=0.055, lw=0)
        ax.axvline(0, color=INK, lw=0.7, ls=(0, (2, 2)), zorder=3)
        xs = np.r_[r["limits"][0], np.sort(r["values"]), r["limits"][1]]
        ys = np.r_[0, np.arange(1, 21) * 5, 100]
        (line,) = ax.step(xs, ys, where="post", color=color, lw=1.2, zorder=4)
        line._evidence = dict(task=task, metric=r["metric"], kind="empirical_CDF")
        ax.fill_between(xs, ys, step="post", color=color, alpha=0.12, lw=0)
        label(fig, cx, 42.2, f"{r['positive']}/20 positive", letter, ha="center", size=7.0)
        track = axes(fig, x0, 46.0, width, 3.8, letter)
        track.set_xlim(r["limits"])
        track.set_ylim(0, 1)
        track.axis("off")
        track.plot(r["limits"], [0.5, 0.5], color=GRID, lw=0.6)
        track.axvline(0, color=GRAY, lw=0.65, ls=(0, (2, 2)))
        ci = track.errorbar(
            r["mean"],
            0.5,
            xerr=[[r["mean"] - r["low"]], [r["high"] - r["mean"]]],
            fmt="D",
            ms=3.2,
            mew=0.5,
            mfc=color,
            mec="white",
            color=color,
            lw=1.3,
            capsize=2.2,
            zorder=4,
        )
        ci._evidence = dict(task=task, metric=r["metric"], kind="archived_mean_CI")
        label(fig, cx, 50.7, f"{r['mean']:+.3f}", letter, ha="center", weight="bold", size=7.1)


def role_coordinates(flows, n):
    roles = ["C", "S", "O"]
    gap = 0.064
    unit = (1 - 2 * gap) / n
    stages = []
    for name in ["before", "after"]:
        sizes = {r: sum((f["count"] for f in flows if f[name] == r)) for r in roles}
        active = [r for r in roles if sizes[r]]
        y = (1 - n * unit - gap * (len(active) - 1)) / 2
        coords = {}
        for r in active:
            coords[r] = [y, y + sizes[r] * unit, sizes[r]]
            y += sizes[r] * unit + gap
        stages.append(coords)
    slots = []
    for f in flows:
        before, after = (f["before"], f["after"])
        left_offset = sum(
            (
                g["count"]
                for g in flows
                if g["before"] == before and roles.index(g["after"]) < roles.index(after)
            )
        )
        right_offset = sum(
            (
                g["count"]
                for g in flows
                if g["after"] == after and roles.index(g["before"]) < roles.index(before)
            )
        )
        left = stages[0][before][0] + left_offset * unit
        right = stages[1][after][0] + right_offset * unit
        slots.append(dict(**f, left=left, right=right, thickness=f["count"] * unit))
    return (stages, slots)


def ribbon(ax, item):
    x0, x1, control = (0.147, 0.853, 0.28)
    y0, y1, h = (item["left"], item["right"], item["thickness"])
    vertices = [
        (x0, y0),
        (x0 + control, y0),
        (x1 - control, y1),
        (x1, y1),
        (x1, y1 + h),
        (x1 - control, y1 + h),
        (x0 + control, y0 + h),
        (x0, y0 + h),
        (x0, y0),
    ]
    codes = (
        [MplPath.MOVETO]
        + [MplPath.CURVE4] * 3
        + [MplPath.LINETO]
        + [MplPath.CURVE4] * 3
        + [MplPath.CLOSEPOLY]
    )
    changed = item["before"] != item["after"]
    patch = PathPatch(
        MplPath(vertices, codes),
        facecolor=ROLE_COLORS[item["after"]],
        edgecolor="white",
        lw=0.45,
        alpha=0.57 if changed else 0.22,
        zorder=3 if changed else 2,
    )
    patch._flow_evidence = item
    ax.add_patch(patch)


def flow_text(ax, text, item, panel, *, t=0.5, dy=0):
    q = 3 * t * t - 2 * t * t * t
    y = (1 - q) * item["left"] + q * item["right"] + item["thickness"] / 2 + dy
    x = 0.147 + 0.706 * t
    artist = ax.text(
        x, y, text, ha="center", va="center", fontsize=7.0, color=INK, linespacing=1.02, zorder=7
    )
    artist._panel = panel
    artist._flow_label = True
    return artist


def role_flow(fig, record, i):
    letter = "cd"[i]
    x, _, w, _ = BOUNDS[letter]
    ax = axes(fig, x + (w - ROLE_PLOT_WIDTH) / 2, 75.0, ROLE_PLOT_WIDTH, 33.0, letter)
    ax._role_flow = True
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis("off")
    coords, slots = role_coordinates(record["flows"], len(record["selected"]))
    for item in slots:
        ribbon(ax, item)
    for stage, xpos in zip(coords, [0, 0.853]):
        for role, (y0, y1, n) in stage.items():
            node = Rectangle(
                (xpos, y0),
                0.147,
                y1 - y0,
                facecolor=ROLE_COLORS[role],
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )
            ax.add_patch(node)
            t = ax.text(
                xpos + 0.147 / 2,
                (y0 + y1) / 2,
                f"{role} {n}",
                ha="center",
                va="center",
                color="white" if role == "C" else INK,
                fontsize=7.0,
                weight="bold",
                zorder=8,
            )
            t._panel = letter
    for item in slots:
        a, b = (item["before"], item["after"])
        if a == b and item["count"] >= 4:
            flow_text(ax, f"{item['count']} retained in {a}", item, letter, t=0.36 if i else 0.5)
        elif i == 0 and a == b == "S":
            flow_text(ax, "Weight", item, letter, t=0.72)
        elif a != b:
            if i == 0 and b == "S":
                s, t = ("Height", 0.46)
            elif i == 0:
                s, t = ("Insulin use;\nvigorous work", 0.28)
            elif a == "O":
                s, t = ("CHD history", 0.76)
            else:
                s, t = ("Height", 0.54)
            flow_text(ax, s, item, letter, t=t)
    return slots


def audit_strip(fig, record, i):
    letter = "cd"[i]
    x, _, w, _ = BOUNDS[letter]
    n = sum(record["action_counts"])
    label(fig, x + w / 2, 115.8, f"Review decisions: {n} pairs", letter, ha="center", size=7.1)
    ax = axes(fig, x + (w - ROLE_PLOT_WIDTH) / 2, 120.1, ROLE_PLOT_WIDTH, 4.3, letter)
    ax._review_strip = True
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    start = 0
    for count, color, label_ in zip(record["action_counts"], ACTION_COLORS, ACTION_LABELS):
        ax.barh(
            0.5,
            count,
            left=start,
            height=1,
            color=color,
            edgecolor="white",
            lw=0.6,
            hatch="///" if label_ == "Task-order removal" else None,
        )
        ax.text(
            start + count / 2,
            0.5,
            str(count),
            ha="center",
            va="center",
            fontsize=7,
            color="white" if color == BLUE else INK,
            weight="bold",
        )._panel = letter
        start += count


def build_figure(data):
    fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=160, facecolor="white")
    for i in range(2):
        gain_panel(fig, data, i)
    label(
        fig,
        W / 2,
        56.4,
        "Gain > 0 favors RITE; diamonds: mean and 95% CI",
        "shared",
        ha="center",
        size=7.1,
        color=GRAY,
    )
    for i, task in enumerate(TASKS):
        letter = "cd"[i]
        x, _, w, _ = BOUNDS[letter]
        title(fig, letter, ["BP task: variable roles", "Cholesterol task: variable roles"][i])
        record = data[task]
        label(
            fig,
            x + w / 2,
            65.5,
            f"{len(record['selected'])} variables in C/S at either stage",
            letter,
            ha="center",
            size=7.0,
        )
        left = x + (w - ROLE_PLOT_WIDTH) / 2
        label(
            fig, left + ROLE_PLOT_WIDTH * 0.0735, 70.7, "Statistical", letter, ha="center", size=7.0
        )
        label(fig, left + ROLE_PLOT_WIDTH * 0.9265, 70.7, "Reviewed", letter, ha="center", size=7.0)
        role_flow(fig, record, i)
        role_handles = [
            Patch(facecolor=ROLE_COLORS[role], label=meaning)
            for role, meaning in [
                ("C", "C: proximal"),
                ("S", "S: precision"),
                ("O", "O: outside C/S"),
            ]
        ]
        add_legend(fig, role_handles, x + w / 2, 111.5, 3, letter)
        audit_strip(fig, record, i)
    action_handles = [
        Patch(
            facecolor=c,
            edgecolor="white",
            label=s,
            hatch="///" if s == "Task-order removal" else None,
        )
        for c, s in zip(ACTION_COLORS, ACTION_LABELS)
    ]
    add_legend(fig, action_handles, W / 2, 128.3, 5, "shared")
    return fig


def render(data_dir, output_dir, *, dpi=900, panels=False):
    """Generate the manuscript figure and optional individual panels."""
    with plt.rc_context(STYLE):
        fig = build_figure(read_data(data_dir))
        try:
            return export_figure(
                fig,
                output_dir,
                STEM,
                dpi=dpi,
                panel_bounds=BOUNDS if panels else None,
                height_mm=H,
                figure_number=6,
                panel_padding=0.1,
            )
        finally:
            plt.close(fig)
