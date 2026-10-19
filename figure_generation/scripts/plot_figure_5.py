"""Known-role effect learning, local recovery and downstream treatment prioritization."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from .figure_io import export_figure, expanded_panel_bounds

STEM = "Figure_5_Role_mechanism"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "tlcd-rite-figure-5",
    "axes.unicode_minus": True,
    "axes.labelcolor": "#25333D",
    "text.color": "#25333D",
    "hatch.linewidth": 0.55,
    "lines.solid_capstyle": "butt",
}

W, H = (162.0, 150.0)
INK, GRAY, GRID = ("#25333D", "#687582", "#DCE3E9")
BLUE, TEAL, ORANGE = ("#0072B2", "#24B6A4", "#D55E00")
METHODS = ["TLCD", "GES", "DirectLiNGAM", "PC", "FCI"]
COLORS = {
    "TLCD": BLUE,
    "GES": "#E7B33D",
    "PC": "#29B9A1",
    "FCI": "#EE7B83",
    "DirectLiNGAM": "#A18ACB",
}
SCENARIOS = ["nonlinear_reference", "linear_additive", "weak_edges", "noisy_missing"]
SHORT = ["Nonlinear", "Linear", "Weak", "Noisy/miss."]
BOUNDS = {
    "a": (1, 1, 160, 38.5),
    "b": (1, 40, 79, 64),
    "c": (83, 40, 78, 64),
    "d": (1, 106.5, 64, 42),
    "e": (68, 106.5, 44.5, 43.2),
    "f": (115.5, 106.5, 45.5, 43.2),
}
CONFIGS = [
    ("Trad_IPW_NoPrec", "Total-C IPW", "#93A0AD"),
    ("Trad_IPW_WithPrec", "Total-C + S", TEAL),
    ("RITE_Ours", "Proximal-C + S (RITE)", BLUE),
]
BASELINE_RADIUS, POLAR_MAX = (1.15, 2.28)


def read_data(directory):
    known = pd.read_csv(directory / "known_roles.csv")
    graph = pd.read_csv(directory / "graph_recovery.csv")
    components = pd.read_csv(directory / "component_ablation.csv")
    raw = pd.read_csv(directory / "effect_estimates.csv")
    selected = raw[raw.N.eq(10000) & raw.Interface.isin(["TLCDRoles", "OracleRoles"])].copy()
    keys = ["Scenario", "N", "Seed"]
    assert not selected.duplicated(keys + ["Interface"]).any()
    assert selected.groupby(keys).Interface.nunique().eq(2).all()
    paired = selected.set_index(keys + ["Interface"]).unstack("Interface").sort_index()
    assert len(paired) == 40
    return (known, graph, components, paired)


def center(letter):
    x, _, w, _ = BOUNDS[letter]
    return x + w / 2


def text(
    fig,
    x,
    y,
    value,
    *,
    panel,
    size=7.2,
    ha="left",
    va="top",
    color=INK,
    weight="normal",
    scope=None,
):
    artist = fig.text(
        x / W,
        1 - y / H,
        value,
        ha=ha,
        va=va,
        color=color,
        fontsize=size,
        fontweight=weight,
        linespacing=1.08,
    )
    artist._panel = panel
    if ha == "center":
        artist._center_target = x
        artist._center_scope = scope or f"panel {panel} annotation"
    return artist


def title(fig, letter, label, size=8.3):
    x, y, _, _ = BOUNDS[letter]
    text(fig, x + 0.3, y + 0.4, letter, panel=letter, size=9.3, weight="bold")
    text(
        fig,
        center(letter),
        y + 0.5,
        label,
        panel=letter,
        size=size,
        weight="bold",
        ha="center",
        scope=f"panel {letter} title",
    )


def axes(fig, x, y, w, h, letter, grid="x"):
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax._panel = letter
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.spines["bottom"].set_linewidth(0.65)
    ax.tick_params(length=2, width=0.55, pad=1.5, labelsize=7.0, colors=INK)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    return ax


def legend(
    fig,
    handles,
    x,
    y,
    panel,
    *,
    ncol=1,
    size=7.0,
    handlelength=0.8,
    loc="center",
    labelspacing=0.22,
):
    obj = fig.legend(
        handles=handles,
        loc=loc,
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(x / W, 1 - y / H),
        bbox_transform=fig.transFigure,
        fontsize=size,
        handlelength=handlelength,
        handleheight=0.7,
        handletextpad=0.4,
        columnspacing=1.4,
        borderpad=0,
        borderaxespad=0,
        labelspacing=labelspacing,
    )
    obj._center_target = x
    obj._center_scope = f"panel {panel} complete legend"
    for item in obj.get_texts():
        item._panel = panel
    return obj


def rectangle(fig, x, y, w, h, face, edge="none", lw=0.6):
    obj = Rectangle(
        (x / W, 1 - (y + h) / H),
        w / W,
        h / H,
        transform=fig.transFigure,
        facecolor=face,
        edgecolor=edge,
        lw=lw,
    )
    fig.add_artist(obj)
    return obj


def circular_method_key(fig, letter, cx, cy):
    renderer = fig.canvas.get_renderer()
    properties = FontProperties(family="DejaVu Sans", size=7.0)
    if not hasattr(fig, "_legend_row_groups"):
        fig._legend_row_groups = []
    for i, method in enumerate(METHODS):
        text_width = (
            renderer.get_text_width_height_descent(method, properties, False)[0] / fig.dpi * 25.4
        )
        width = 1.8 + 0.7 + text_width
        left = cx - width / 2
        y = cy + (i - 2) * 3.05
        swatch = rectangle(fig, left, y - 0.75, 1.8, 1.5, COLORS[method])
        label = text(fig, left + 2.5, y, method, panel=letter, va="center", size=7.0)
        fig._legend_row_groups.append(dict(panel=letter, target=cx, artists=[swatch, label]))


def graph_records(summary, role=False):
    columns = [("Directed_F1", method, method) for method in METHODS]
    if role:
        columns = [
            ("C_Proximal_F1", "TLCD", "C"),
            ("C_Proximal_F1", "GES", "C"),
            ("S_Precision_F1", "TLCD", "S"),
            ("S_Precision_F1", "GES", "S"),
        ]
    result = []
    for scenario in SCENARIOS:
        for metric, method, label in columns:
            rows = summary[
                summary.Scenario.eq(scenario)
                & summary.N.eq(10000)
                & summary.Method.eq(method)
                & summary.Metric.eq(metric)
            ]
            assert len(rows) == 1
            row = rows.iloc[0]
            assert int(row.Runs) == 10
            value, se = (float(row.Mean), float(row.SD) / np.sqrt(int(row.Runs)))
            assert 0 <= value - se <= value + se <= 1.00000001
            result.append(
                dict(
                    scenario=scenario,
                    method=method,
                    metric=metric,
                    role=label,
                    mean=value,
                    SE=se,
                    runs=10,
                )
            )
    return result


def draw_a(fig, known):
    title(fig, "a", "Known covariate roles: effect-learning configuration")
    legend(
        fig,
        [Patch(facecolor=color, label=label) for _, label, color in CONFIGS],
        center("a"),
        7.2,
        "a",
        ncol=3,
        size=7.1,
        handlelength=1.4,
    )
    tracks = [(26, 55, "CATE MAE"), (87, 31.5, "MAE gain (%)"), (128, 31.8, "R² change")]
    for x, w, label in tracks:
        text(fig, x + w / 2, 10.4, label, panel="a", ha="center", size=7.5)
    native = axes(fig, 26, 14.5, 55, 17.8, "a")
    gain = axes(fig, 87, 14.5, 31.5, 17.8, "a")
    r2 = axes(fig, 128, 14.5, 31.8, 17.8, "a")
    native.set(xlim=(0, 1.5), xticks=[0, 0.5, 1, 1.5], ylim=(4.5, -0.5), yticks=[])
    gain.set(xlim=(0, 22.5), xticks=[0, 10, 20], ylim=(4.5, -0.5), yticks=[])
    r2.set(xlim=(-0.018, 0.17), xticks=[0, 0.05, 0.1], ylim=(4.5, -0.5), yticks=[])
    r2.set_xticklabels(["0", ".05", ".10"])
    r2.axvline(0, color=GRAY, lw=0.7)
    learners = ["S-Learner", "T-Learner", "X-Learner", "ClassTransform", "UpliftForest"]
    labels = ["S-learner", "T-learner", "X-learner", "Class transform", "Uplift forest"]
    records = []
    for i, (name, label) in enumerate(zip(learners, labels)):
        row = known[known.Model.eq(name)].set_index("Method")
        mae = [float(row.loc[key, "MAE"]) for key, *_ in CONFIGS]
        fits = [float(row.loc[key, "R2"]) for key, *_ in CONFIGS]
        for j, (_, _, color) in enumerate(CONFIGS):
            native.barh(i + (j - 1) * 0.27, mae[j], height=0.23, color=color, lw=0, zorder=3)
        y = 14.5 + (i + 0.5) / 5 * 17.8
        text(fig, 24.2, y, label, panel="a", ha="right", va="center", size=7.1)
        reduction = 100 * (mae[1] - mae[2]) / mae[1]
        r2delta = fits[2] - fits[1]
        gain.barh(i, reduction, height=0.58, color=BLUE, zorder=3)
        gain.text(
            reduction + 0.45, i, f"{reduction:.1f}", ha="left", va="center", fontsize=7.0, color=INK
        )
        r2.barh(i, r2delta, height=0.58, color=BLUE if r2delta >= 0 else ORANGE, zorder=3)
        r2.text(
            max(r2delta, 0) + 0.004,
            i,
            f"{r2delta:+.3f}".replace("-", "−"),
            ha="left",
            va="center",
            fontsize=7.0,
            color=INK,
        )
        records.append(
            dict(
                learner=name,
                MAE=dict(zip([c[0] for c in CONFIGS], mae)),
                R2=dict(zip([c[0] for c in CONFIGS], fits)),
                MAE_reduction_percent=reduction,
                R2_difference=r2delta,
                N=10000,
                single_dataset_seed=2026,
            )
        )
    text(
        fig,
        center("a"),
        37.0,
        "Gains: proximal-C + S versus total-C + S; n = 10,000; seed 2026",
        panel="a",
        ha="center",
        size=7.0,
        color=GRAY,
    )
    return records


def draw_radial(fig, records, role=False):
    letter = "c" if role else "b"
    title(fig, letter, "Covariate-role recovery" if role else "Directed-graph recovery")
    text(
        fig,
        center(letter),
        44.6,
        "n = 10,000; mean ± SE; 10 seeds",
        panel=letter,
        ha="center",
        size=7.0,
        color=GRAY,
    )
    cx, cy, radius = (center(letter), 76.5, 23.3)
    ax = fig.add_axes(
        [(cx - radius) / W, 1 - (cy + radius) / H, 2 * radius / W, 2 * radius / H],
        projection="polar",
    )
    ax._panel = letter
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, POLAR_MAX)
    ax.set_xticks([])
    ax.set_yticks([BASELINE_RADIUS, BASELINE_RADIUS + 0.5, BASELINE_RADIUS + 1])
    ax.set_yticklabels(["0", ".5", "1"], fontsize=7.0, color=GRAY)
    ax.set_rlabel_position(45)
    for label in ax.get_yticklabels():
        label.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.2))
    ax.spines["polar"].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.65)
    ax.set_axisbelow(True)
    n = 4 if role else 5
    offsets = np.linspace(-25, 25, n) if role else np.linspace(-28, 28, n)
    width = np.deg2rad(12.5 if role else 11)
    for i, scenario in enumerate(SCENARIOS):
        rows = [r for r in records if r["scenario"] == scenario]
        assert len(rows) == n
        for delta, row in zip(offsets, rows):
            theta = np.deg2rad(i * 90 + delta)
            hatch = "////" if role and row["role"] == "S" else None
            ax.bar(
                theta,
                row["mean"],
                bottom=BASELINE_RADIUS,
                width=width,
                color=COLORS[row["method"]],
                edgecolor="white",
                lw=0.45,
                hatch=hatch,
                zorder=3,
            )
            ax.errorbar(
                theta,
                BASELINE_RADIUS + row["mean"],
                yerr=row["SE"],
                fmt="none",
                color=INK,
                lw=0.8,
                capsize=1.3,
                capthick=0.65,
                zorder=4,
            )
    labels = [
        (cx, cy - 25.7, "Nonlinear", "center", "center"),
        (cx + 25.7, cy, "Linear", "left", "center"),
        (cx, cy + 25.7, "Weak edges", "center", "center"),
        (cx - 25.7, cy, "Noise +\nmissing", "right", "center"),
    ]
    for x, y, label, ha, va in labels:
        text(fig, x, y, label, panel=letter, ha=ha, va=va, size=7.1, weight="bold")
    if role:
        combinations = [("TLCD", "C"), ("GES", "C"), ("TLCD", "S"), ("GES", "S")]
        handles = [
            Patch(
                facecolor=COLORS[method],
                edgecolor="white",
                hatch="////" if kind == "S" else None,
                label=f"{method} ({kind})",
            )
            for method, kind in combinations
        ]
        legend(fig, handles, cx, cy, letter, size=7.0, handlelength=1.4, labelspacing=0.35)
    else:
        circular_method_key(fig, letter, cx, cy)
    return ax


def draw_d(fig, summary):
    title(fig, "d", "Component analysis", size=8.0)
    text(
        fig,
        center("d"),
        111.0,
        "n = 5,000; 10 seeds; mean ± SD",
        panel="d",
        ha="center",
        size=7.0,
        color=GRAY,
    )
    variant_ids = [
        "Complete_TLCD",
        "PairwiseANM_FullRepair",
        "TaskAdaptive_NoRepair",
        "TaskAdaptive_NoDAGGate",
    ]
    labels = ["Complete", "Pairwise ANM", "No endpoint", "No DAG gate"]
    switches = [[1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
    for x, label in zip([22.1, 27.6, 33.1], ["Mix", "End", "DAG"]):
        text(fig, x, 116.3, label, panel="d", ha="center", size=7.0)
    columns = [
        (37.8, 11.4, "Directed_F1", "Directed\nF1", GRAY, (0.55, 1.03), [0.6, 1.0]),
        (53.2, 10.6, "C_Proximal_F1", "C-set\nF1", BLUE, (0.15, 1.15), [0.2, 1.0]),
    ]
    tracks = []
    for x, w, metric, label, _, limits, ticks in columns:
        text(fig, x + w / 2, 114.7, label, panel="d", ha="center", size=7.0)
        ax = axes(fig, x, 121.0, w, 24.0, "d")
        ax._metric = metric
        ax.set(xlim=limits, ylim=(3.5, -0.5), yticks=[], xticks=ticks)
        ax.set_xticklabels([f"{value:g}" for value in ticks])
        tracks.append(ax)
    result = []
    for i, (variant, label, enabled) in enumerate(zip(variant_ids, labels, switches)):
        y = 121 + i * 6.0
        text(fig, 18.6, y + 3, label, panel="d", ha="right", va="center", size=7.0)
        for x, on in zip([22.1, 27.6, 33.1], enabled):
            rectangle(
                fig,
                x - 0.85,
                y + 2.15,
                1.7,
                1.7,
                BLUE if on else "white",
                edge=BLUE if on else GRAY,
                lw=0.6,
            )
        for ax, (_, _, metric, _, color, limits, _) in zip(tracks, columns):
            row = summary[summary.Variant.eq(variant) & summary.Metric.eq(metric)]
            assert len(row) == 1
            mean, sd = (float(row.iloc[0].Mean), float(row.iloc[0].SD))
            assert limits[0] < mean - sd <= mean + sd < limits[1]
            ax.errorbar(
                mean,
                i,
                xerr=sd,
                fmt="o",
                color=color,
                ms=3.0,
                capsize=1.6,
                capthick=0.7,
                lw=0.9,
                zorder=3,
            )
            result.append(
                dict(
                    variant=variant,
                    metric=metric,
                    mean=mean,
                    SD=sd,
                    enabled=enabled,
                    interval=[mean - sd, mean + sd],
                    axis_limits=list(limits),
                )
            )
    return result


def gap_records(paired, policy=False):
    metric = "Policy_Regret" if policy else "CATE_MAE"
    limits = [-0.005, 0.008] if policy else [-0.023, 0.031]
    records = []
    for scenario in SCENARIOS:
        block = paired.loc[scenario]
        values = block[metric, "TLCDRoles"] - block[metric, "OracleRoles"]
        assert len(values) == 10
        q = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
        records.append(
            dict(
                scenario=scenario,
                metric=metric,
                N=10000,
                mean=float(values.mean()),
                q25=float(q[0]),
                median=float(q[1]),
                q75=float(q[2]),
                seeds=[
                    dict(
                        seed=int(seed),
                        value=float(value),
                        offscale=not limits[0] <= value <= limits[1],
                    )
                    for (_, seed), value in values.items()
                ],
                limits=limits,
                display_scale=1000 if policy else 100,
                percentile_method="numpy linear; descriptive, not CI",
            )
        )
    return records


def categorical_offsets(group, plot_w, plot_h):
    limits = group["limits"]
    lanes = [-0.435, -0.305, -0.175, -0.045]
    placed, result = ([], [])
    for seed in group["seeds"]:
        if seed["offscale"]:
            result.append(None)
            continue
        x = (seed["value"] - limits[0]) / (limits[1] - limits[0]) * plot_w
        for lane in [lanes[1], lanes[2], lanes[0], lanes[3]]:
            y = lane * plot_h / 4.15
            if all((np.hypot(x - px, y - py) >= 0.68 for px, py in placed)):
                placed.append((x, y))
                result.append(lane)
                break
        else:
            raise ValueError("Insufficient categorical room for truthful raw-point placement")
    return result


def draw_gap(fig, records, policy=False):
    letter = "f" if policy else "e"
    title(fig, letter, "Policy regret gap" if policy else "CATE MAE gap", size=8.0)
    x, _, w, _ = BOUNDS[letter]
    text(
        fig,
        center(letter),
        111.0,
        "TLCD − true roles; 10 seeds\nBox: IQR; ◇: mean",
        panel=letter,
        ha="center",
        size=7.0,
        color=GRAY,
    )
    overflow = [r["value"] for group in records for r in group["seeds"] if r["offscale"]]
    overflow_label = ", ".join((f"{v * records[0]['display_scale']:.2f}" for v in sorted(overflow)))
    text(
        fig,
        center(letter),
        117.4,
        f"Off-axis ({len(overflow)}): " + overflow_label,
        panel=letter,
        ha="center",
        size=7.0,
        color=ORANGE,
    )
    plot_x, plot_w = (x + 17.2, w - 18.2)
    ax = axes(fig, plot_x, 120.3, plot_w, 22.2, letter)
    limits = records[0]["limits"]
    ax.set(xlim=limits, ylim=(3.6, -0.55), yticks=[])
    if policy:
        ax.set_xticks([-0.004, 0, 0.004, 0.008])
        ax.set_xticklabels(["−4", "0", "4", "8"])
    else:
        ax.set_xticks([-0.02, 0, 0.02])
        ax.set_xticklabels(["−2", "0", "2"])
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (2, 2)), zorder=1)
    for i, (group, label) in enumerate(zip(records, SHORT)):
        y_mm = 120.3 + (i + 0.55) / 4.15 * 22.2
        text(fig, plot_x - 1.6, y_mm, label, panel=letter, ha="right", va="center", size=7.0)
        rectangle_data = Rectangle(
            (group["q25"], i + 0.08),
            group["q75"] - group["q25"],
            0.12,
            facecolor="#BCE6F1",
            edgecolor=BLUE,
            lw=0.75,
            zorder=3,
        )
        ax.add_patch(rectangle_data)
        ax.vlines(group["median"], i + 0.08, i + 0.2, color=INK, lw=1, zorder=4)
        offscale_count = sum((s["offscale"] for s in group["seeds"]))
        offscale_index = 0
        offsets = categorical_offsets(group, plot_w, 22.2)
        for offset, seed in zip(offsets, group["seeds"]):
            value = seed["value"]
            if seed["offscale"]:
                y = i - 0.45 + 0.28 * offscale_index if offscale_count > 1 else i - 0.34
                offscale_index += 1
                edge = limits[1] if value > limits[1] else limits[0]
                ax.plot(
                    edge,
                    y,
                    marker=">" if value > limits[1] else "<",
                    ls="",
                    color=ORANGE,
                    ms=3.2,
                    markeredgewidth=0.4,
                    clip_on=False,
                    zorder=5,
                )
            else:
                y = i + offset
                ax.plot(
                    value,
                    y,
                    marker="o",
                    ls="",
                    color=BLUE,
                    alpha=0.62,
                    ms=1.7,
                    markeredgewidth=0,
                    zorder=4,
                )
            seed["categorical_offset"] = float(y - i)
        assert limits[0] < group["q25"] <= group["q75"] < limits[1]
        assert limits[0] < group["mean"] < limits[1]
        ax.plot(
            group["mean"],
            i + 0.38,
            marker="D",
            ls="",
            color=BLUE,
            ms=2.0,
            markeredgewidth=0.4,
            zorder=5,
        )
    label = "Regret gap (×10⁻³)" if policy else "MAE gap (×10⁻²)"
    text(fig, plot_x + plot_w / 2, 148.0, label, panel=letter, ha="center", va="center", size=7.1)
    return ax


def render(data_dir, output_dir, *, dpi=900, panels=False):
    """Generate the manuscript figure and optional individual panels."""
    with plt.rc_context(STYLE):
        known, summary, components, paired = read_data(data_dir)
        fig = plt.figure(figsize=(W / 25.4, H / 25.4), dpi=140, facecolor="white")
        draw_a(fig, known)
        draw_radial(fig, graph_records(summary))
        draw_radial(fig, graph_records(summary, role=True), role=True)
        draw_d(fig, components)
        draw_gap(fig, gap_records(paired))
        draw_gap(fig, gap_records(paired, policy=True), policy=True)
        try:
            bounds = expanded_panel_bounds(fig, BOUNDS, H) if panels else None
            return export_figure(
                fig,
                output_dir,
                STEM,
                dpi=dpi,
                panel_bounds=bounds if panels else None,
                height_mm=H,
                figure_number=5,
            )
        finally:
            plt.close(fig)
