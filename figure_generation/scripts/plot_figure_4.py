"""Transfusion-benefit strata, clinical profiles, hemoglobin response and allocation."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from .figure_io import export_figure

STEM = "Figure_4_Clinical_profiles"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.55,
    "text.color": "#26323B",
    "axes.labelcolor": "#26323B",
    "savefig.pad_inches": 0,
    "mathtext.fontset": "dejavusans",
}

WIDTH, HEIGHT = (162.0, 112.0)
BLUE, ORANGE = ("#0072B2", "#D55E00")
INK, GRAY, LIGHT = ("#26323B", "#687483", "#DCE2E8")
BOUNDS = {
    "a": (1, 1, 71, 53),
    "b": (74.5, 1, 86.5, 53),
    "c": (1, 55.5, 71, 55.5),
    "d": (74.5, 55.5, 86.5, 55.5),
}
PROFILE_CMAP = colors.LinearSegmentedColormap.from_list("clinical_z", [BLUE, "#FAFAFA", ORANGE])


def read_data(directory):
    benefit = pd.read_csv(directory / "mimic_rbc_temporal_benefit_profiles.csv").sort_values(
        "benefit_quartile"
    )
    chart = pd.read_csv(directory / "mimic_rbc_temporal_chart_profiles.csv").sort_values(
        "benefit_quartile"
    )
    assert benefit.benefit_quartile.tolist() == chart.benefit_quartile.tolist() == [1, 2, 3, 4]
    joined = benefit.merge(
        chart, on="benefit_quartile", validate="one_to_one", suffixes=("", "_chart")
    )
    assert joined.n.equals(joined.n_chart) and int(joined.n.sum()) == 2107
    hb = pd.read_csv(directory / "mimic_rbc_temporal_hb_mechanism.csv").set_index("group")
    hb = hb.loc[["overall"] + [f"benefit_quartile_{i}" for i in range(1, 5)]].reset_index()
    curve = pd.read_csv(directory / "mimic_rbc_temporal_policy_budget_curve.csv")
    assert not curve.duplicated(["Budget", "Interface"]).any()
    assert curve.groupby("Budget").Interface.nunique().eq(2).all()
    pair = curve.pivot(index="Budget", columns="Interface").sort_index()
    assert len(pair) == 8 and set(curve.Interface) == {"RITE_CPlusS", "AllVariables"}
    np.testing.assert_array_equal(
        pair["Selected_N", "RITE_CPlusS"], pair["Selected_N", "AllVariables"]
    )
    np.testing.assert_allclose(
        pair["Random_Budget_Risk", "RITE_CPlusS"], pair["Random_Budget_Risk", "AllVariables"]
    )
    return (joined, hb, pair)


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
    t = fig.text(
        x / WIDTH,
        1 - y / HEIGHT,
        value,
        ha=ha,
        va=va,
        color=color,
        fontsize=size,
        fontweight=weight,
        linespacing=1.08,
    )
    t._panel = panel
    if ha == "center":
        t._center_target = x
        t._center_scope = scope or f"panel {panel} annotation"
    return t


def title(fig, letter, label):
    x, y, _, _ = BOUNDS[letter]
    text(fig, x + 0.5, y + 0.4, letter, panel=letter, size=9.3, weight="bold")
    text(
        fig,
        center(letter),
        y + 0.5,
        label,
        panel=letter,
        size=8.4,
        weight="bold",
        ha="center",
        scope=f"panel {letter} title",
    )


def axes(fig, x, y, w, h, panel, grid="x"):
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    ax._panel = panel
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(length=2, width=0.55, pad=1.7, labelsize=7.1, colors=INK)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=0.5)
    return ax


def key(fig, letter, y, handles):
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(center(letter) / WIDTH, 1 - y / HEIGHT),
        bbox_transform=fig.transFigure,
        fontsize=7.2,
        handlelength=1.45,
        handletextpad=0.45,
        columnspacing=1.0,
        borderpad=0,
        borderaxespad=0,
    )
    legend._center_target = center(letter)
    legend._center_scope = f"panel {letter} complete legend"
    for item in legend.get_texts():
        item._panel = letter


def rectangle(fig, x, y, w, h, color):
    fig.add_artist(
        Rectangle(
            (x / WIDTH, 1 - (y + h) / HEIGHT),
            w / WIDTH,
            h / HEIGHT,
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
        )
    )


def line(fig, x, y0, y1):
    fig.add_artist(
        Line2D(
            [x / WIDTH, x / WIDTH],
            [1 - y0 / HEIGHT, 1 - y1 / HEIGHT],
            transform=fig.transFigure,
            color=LIGHT,
            lw=0.7,
        )
    )


def draw_a(fig, frame):
    title(fig, "a", "Benefit and treatment uptake")
    key(
        fig,
        "a",
        6.5,
        [
            Line2D([], [], color=ORANGE, ls="", marker="o", ms=4, label="Predicted"),
            Line2D([], [], color=BLUE, ls="", marker="D", ms=4, label="DR"),
        ],
    )
    text(
        fig,
        59.6,
        10.7,
        "Historical use",
        panel="a",
        size=7.1,
        ha="center",
        scope="historical-use block",
    )
    ax = axes(fig, 9.2, 15.0, 36.0, 28.0, "a")
    ux = axes(fig, 50.0, 15.0, 11.5, 28.0, "a", grid=None)
    ax.set(xlim=(-80, 60), ylim=(3.65, -0.45), xticks=[-60, 0, 60], yticks=[])
    ux.set(xlim=(0, 15), ylim=(3.65, -0.45), xticks=[0, 15], yticks=[])
    ax.spines["left"].set_visible(False)
    ux.spines["left"].set_visible(False)
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (2, 2)))
    records = []
    for i, r in enumerate(frame.itertuples()):
        mean, lo, hi = 100 * np.array(
            [
                r.dr_observed_benefit_mean,
                r.dr_observed_benefit_ci_low,
                r.dr_observed_benefit_ci_high,
            ]
        )
        assert -80 < lo < mean < hi < 60
        ax.errorbar(
            mean,
            i + 0.12,
            xerr=[[mean - lo], [hi - mean]],
            fmt="D",
            color=BLUE,
            ms=3.8,
            lw=1.1,
            capsize=2,
            capthick=0.8,
        )
        ax.plot(100 * r.predicted_benefit_mean, i - 0.16, "o", color=ORANGE, ms=3.8)
        yq = 15.0 + (i + 0.45) / 4.1 * 28.0
        text(fig, 6.5, yq, f"Q{i + 1}", panel="a", ha="right", va="center", size=7.4)
        rate = 100 * r.observed_treatment_rate
        treated = int(round(r.n * r.observed_treatment_rate))
        ux.barh(i - 0.15, rate, height=0.28, color="#BEDDEB", edgecolor=BLUE, lw=0.55)
        ybar = 15.0 + (i - 0.15 + 0.45) / 4.1 * 28.0
        ycount = 15.0 + (i + 0.3 + 0.45) / 4.1 * 28.0
        text(fig, 69.5, ybar, f"{rate:.1f}", panel="a", ha="right", va="center", size=7.1)
        text(
            fig,
            55.75,
            ycount,
            f"{treated}/{int(r.n)}",
            panel="a",
            ha="center",
            va="center",
            size=7.0,
        )
        records.append(
            dict(
                quartile=i + 1,
                n=int(r.n),
                treated=treated,
                rate_percent=rate,
                predicted_pp=float(100 * r.predicted_benefit_mean),
                DR_pp=float(mean),
                low_pp=float(lo),
                high_pp=float(hi),
            )
        )
    text(fig, 27.2, 48.7, "Mortality benefit (pp)", panel="a", ha="center", va="center", size=7.2)
    text(fig, 55.75, 48.7, "Use (%)", panel="a", ha="center", va="center", size=7.2)
    text(
        fig,
        center("a"),
        51.7,
        "Counts: treated / total",
        panel="a",
        ha="center",
        va="center",
        size=7.0,
        scope="panel a count note",
    )
    return records


def draw_b(fig, frame):
    title(fig, "b", "Clinical profiles by quartile")
    labels = [
        "Age (years)",
        "BUN (mg/dL)",
        "GCS",
        "MAP min (mmHg)",
        "Mortality (%)",
        "Medical svc (%)",
        "Cardiac ICU (%)",
        "FiO₂ rec. (%)",
        "Vent. rec. (%)",
    ]
    values = np.array(
        [
            frame.age_mean,
            frame.lab_bun_last_mean,
            frame.chart_gcs_combined_last_mean,
            frame.chart_map_min_mean,
            100 * frame.observed_mortality_90d,
            100 * frame.service_medical_mean,
            100 * frame.unit_cardiac_mean,
            100 * frame.chart_fio2_recorded_mean,
            100 * frame.chart_ventilator_measurement_recorded_mean,
        ]
    )
    z = (values - values.mean(axis=1, keepdims=True)) / values.std(axis=1, keepdims=True, ddof=0)
    assert np.max(np.abs(z)) < 2
    x0, cell_w, row_h, top = (100.5, 11.05, 3.7, 11.3)
    extra = np.array([0, 0, 0, 0, 0.8, 1.6, 1.6, 2.4, 2.4])
    for j in range(4):
        text(fig, x0 + (j + 0.5) * cell_w, 7.2, f"Q{j + 1}", panel="b", ha="center", size=7.3)
    for i, label in enumerate(labels):
        y = top + i * row_h + extra[i]
        text(fig, 98.7, y + row_h / 2, label, panel="b", ha="right", va="center", size=7.1)
        for j in range(4):
            rgba = PROFILE_CMAP((z[i, j] + 2) / 4)
            rgb = np.array(rgba[:3])
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            fg = "white" if linear @ np.array([0.2126, 0.7152, 0.0722]) < 0.183 else INK
            rectangle(fig, x0 + j * cell_w, y, cell_w, row_h, rgba)
            text(
                fig,
                x0 + (j + 0.5) * cell_w,
                y + row_h / 2,
                f"{values[i, j]:.1f}",
                panel="b",
                ha="center",
                va="center",
                size=7.3,
                color=fg,
            )
    for lo, hi, name in [(0, 3, "Baseline"), (4, 4, "Outcome"), (5, 6, "Care"), (7, 8, "Recorded")]:
        start = top + lo * row_h + extra[lo]
        end = top + (hi + 1) * row_h + extra[hi]
        line(fig, 146.0, start + 0.4, end - 0.4)
        text(fig, 153.8, (start + end) / 2, name, panel="b", ha="center", va="center", size=7.0)
    text(fig, 98.7, 48.5, "Within-row z", panel="b", ha="right", va="center", size=7.0)
    for k in range(128):
        rectangle(fig, 102.2 + k * 41.4 / 128, 47.9, 41.4 / 128 + 0.05, 1.2, PROFILE_CMAP(k / 127))
    for x, value in [(102.2, "−2"), (122.9, "0"), (143.6, "2")]:
        text(fig, x, 50.8, value, panel="b", ha="center", va="center", size=7.0)
    return dict(labels=labels, raw=values.tolist(), row_z=z.tolist(), z_ddof=0, z_limits=[-2, 2])


def draw_c(fig, frame):
    title(fig, "c", "Subsequent Hb response")
    text(
        fig,
        center("c"),
        61.1,
        "Early RBC 0–6 h; follow-up Hb 6–24 h",
        panel="c",
        ha="center",
        size=7.1,
        color=GRAY,
        scope="panel c timing note",
    )
    ax = axes(fig, 15.0, 67.6, 55.8, 35.0, "c")
    ax.spines["left"].set_visible(False)
    ax.set(xlim=(-0.75, 2.5), ylim=(4.65, -0.35), yticks=[], xticks=[-0.5, 0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels(["−0.5", "0", "0.5", "1", "1.5", "2"])
    ax.axvline(0, color=GRAY, lw=0.8, ls=(0, (2, 2)))
    ax.axhline(0.65, color=LIGHT, lw=0.7)
    positions = [0, 1.3, 2.3, 3.3, 4.3]
    for i, (r, pos) in enumerate(zip(frame.itertuples(), positions)):
        value, lo, hi = (float(r.aipw_hb_change_difference), float(r.ci_low), float(r.ci_high))
        assert -0.75 < lo < value < hi < 2.5
        ax.errorbar(
            value,
            pos,
            xerr=[[value - lo], [hi - value]],
            fmt="o" if i == 0 else "D",
            color=BLUE,
            ms=4.0 if i == 0 else 3.7,
            capsize=2,
            lw=1.1,
            capthick=0.8,
            zorder=3,
        )
        y = 67.6 + (pos + 0.35) / 5 * 35.0
        text(
            fig,
            12.0,
            y,
            "Overall" if i == 0 else f"Q{i}",
            panel="c",
            ha="right",
            va="center",
            size=7.2,
        )
    ax.plot(
        frame.aipw_hb_change_difference.to_numpy()[1:],
        positions[1:],
        color="#A1C8DC",
        lw=0.85,
        zorder=1,
    )
    text(
        fig,
        42.9,
        109.4,
        "Hb change difference (g/dL)",
        panel="c",
        ha="center",
        va="center",
        size=7.2,
    )
    return frame.to_dict("records")


def draw_d(fig, pair):
    title(fig, "d", "Capacity-constrained allocation")
    key(
        fig,
        "d",
        60.5,
        [
            Line2D([], [], color=BLUE, marker="D", lw=1.1, ms=3.7, label="RITE"),
            Line2D([], [], color=GRAY, marker="o", ls="--", lw=1.1, ms=3.7, label="All variables"),
        ],
    )
    budgets = pair.index.to_numpy() * 100
    proposed = pair["Risk_Reduction", "RITE_CPlusS"].to_numpy() * 100
    reference = pair["Risk_Reduction", "AllVariables"].to_numpy() * 100
    delta = proposed - reference
    observed_mask = pair["Observed_Budget", "RITE_CPlusS"].to_numpy(dtype=bool)
    assert observed_mask.sum() == 1
    observed = float(budgets[observed_mask][0])
    text(
        fig,
        center("d"),
        64.7,
        "Risk reduction vs random (pp)",
        panel="d",
        ha="center",
        size=7.1,
        scope="panel d absolute-track note",
    )
    ax = axes(fig, 83.4, 68.0, 77.0, 18.5, "d", grid="y")
    ax.set(xlim=(0, 32), ylim=(-1.8, 6.5), yticks=[0, 3, 6], xticks=[])
    ax.axhline(0, color=GRAY, ls=(0, (2, 2)), lw=0.7)
    ax.axvline(observed, color=GRAY, ls=(0, (2, 2)), lw=0.7)
    ax.plot(budgets, proposed, color=BLUE, marker="D", ms=3.7, lw=1.1)
    ax.plot(budgets, reference, color=GRAY, marker="o", ls="--", ms=3.7, lw=1.1)
    text(
        fig,
        center("d"),
        89.8,
        f"RITE − all (pp)   ·   Observed capacity {observed:.1f}%",
        panel="d",
        ha="center",
        va="center",
        size=7.1,
        scope="panel d difference-track note",
    )
    lower = axes(fig, 83.4, 93.0, 77.0, 9.6, "d", grid=None)
    lower.set(xlim=(0, 32), ylim=(-2.0, 5.7), yticks=[0, 4], xticks=[0, 10, 20, 30])
    lower.axhline(0, color=GRAY, ls=(0, (2, 2)), lw=0.7)
    lower.axvline(observed, color=GRAY, ls=(0, (2, 2)), lw=0.7)
    for x, value in zip(budgets, delta):
        col = BLUE if value >= 0 else ORANGE
        lower.vlines(x, 0, value, color=col, lw=0.8)
        lower.plot(x, value, "o", color=col, ms=3)
    text(fig, 121.9, 109.4, "Treatment capacity (%)", panel="d", ha="center", va="center", size=7.2)
    assert proposed.min() > -1.8 and reference.min() > -1.8
    assert proposed.max() < 6.5 and reference.max() < 6.5
    assert delta.min() > -2.0 and delta.max() < 5.7
    return dict(
        budgets_percent=budgets.tolist(),
        RITE_pp=proposed.tolist(),
        all_variables_pp=reference.tolist(),
        difference_pp=delta.tolist(),
        observed_percent=observed,
        selected_n=pair["Selected_N", "RITE_CPlusS"].astype(int).tolist(),
    )


def render(data_dir, output_dir, *, dpi=900, panels=False):
    """Generate the manuscript figure and optional individual panels."""
    with plt.rc_context(STYLE):
        profiles, hb, pair = read_data(data_dir)
        fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor="white")
        draw_a(fig, profiles)
        draw_b(fig, profiles)
        draw_c(fig, hb)
        draw_d(fig, pair)
        try:
            return export_figure(
                fig,
                output_dir,
                STEM,
                dpi=dpi,
                panel_bounds=BOUNDS if panels else None,
                height_mm=HEIGHT,
                figure_number=4,
            )
        finally:
            plt.close(fig)
