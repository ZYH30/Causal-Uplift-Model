"""MIMIC-IV estimation gains, covariate balance and clinical eras."""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.lines import Line2D
from .figure_io import export_figure

STEM = "Figure_3_MIMIC"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 7.2,
    "axes.linewidth": 0.55,
    "axes.labelcolor": "#26323B",
    "text.color": "#26323B",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "xtick.labelsize": 7.1,
    "ytick.labelsize": 7.1,
    "savefig.pad_inches": 0,
    "mathtext.fontset": "dejavusans",
}

WIDTH, HEIGHT = (162.0, 109.0)
PANEL_BOUNDS = {
    "a": (1, 5.5, 52.2, 44.3),
    "b": (55, 5.5, 52.2, 44.3),
    "c": (109, 5.5, 52.2, 44.3),
    "d": (1, 55.1, 94.7, 53.4),
    "e": (98.7, 55.1, 62.5, 53.4),
}
BLUE = "#0072B2"
WARM = "#C86A4A"
INK = "#26323B"
GRAY = "#63717E"
LIGHT = "#DDE3E8"
SMDCMAP = colors.LinearSegmentedColormap.from_list(
    "absolute_smd", ["#F0FAFD", "#B8E7EF", "#73CCD9", "#31AAC6"]
)
SPECS = [
    dict(
        letter="a",
        column="R_Loss",
        key="R_Loss_Improvement",
        title="R-loss",
        unit="Gain ($\\times 10^{-4}$)",
        scale=10000.0,
        lim=(-8, 18),
        ticks=[-5, 0, 5, 10, 15],
    ),
    dict(
        letter="b",
        column="DR_Qini",
        key="DR_Qini_Improvement",
        title="DR-Qini",
        unit="Gain",
        scale=1,
        lim=(-0.025, 0.055),
        ticks=[-0.02, 0, 0.02, 0.04],
    ),
    dict(
        letter="c",
        column="Policy_Risk_Reduction",
        key="Policy_Risk_Reduction_Improvement",
        title="Policy risk reduction",
        unit="Gain (pp)",
        scale=100,
        lim=(-6.5, 8),
        ticks=[-5, 0, 5],
    ),
]
NAMES = {
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


def read_data(directory):
    raw = pd.read_csv(directory / "post_discovery_repeated_splits.csv")
    assert len(raw) == 40
    assert not raw.duplicated(["Split", "Seed", "Interface"]).any()
    assert set(raw.Interface) == {"AllVariables", "RITE_CPlusS"}
    assert raw.groupby(["Split", "Seed"]).Interface.nunique().eq(2).all()
    pair = raw.set_index(["Split", "Seed", "Interface"]).unstack("Interface").sort_index()
    summary = pd.read_csv(directory / "mimic_rbc_repeated_summary.csv").set_index("Metric")
    temporal = pd.read_csv(directory / "mimic_rbc_temporal_comparison.csv").set_index("Metric")
    gains = []
    records = []
    for spec in SPECS:
        col = spec["column"]
        a, b = (pair[col, "AllVariables"], pair[col, "RITE_CPlusS"])
        value = (a - b if col == "R_Loss" else b - a).to_numpy() * spec["scale"]
        sm, tm = (summary.loc[spec["key"]], temporal.loc[spec["key"]])
        mean = np.array([sm.Mean, sm.CI_Low, sm.CI_High], dtype=float) * spec["scale"]
        time = np.array([tm.Point_Estimate, tm.CI_Low, tm.CI_High], dtype=float) * spec["scale"]
        np.testing.assert_allclose(value.mean(), mean[0], atol=1e-10, rtol=0)
        assert (value > 0).sum() == int(sm.Wins)
        assert len(value) == int(sm.Repeats) == 20
        assert spec["lim"][0] <= min(value.min(), mean[1], time[1])
        assert spec["lim"][1] >= max(value.max(), mean[2], time[2])
        assert time[1] < 0 < time[2]
        gains.append(dict(spec=spec, values=value, mean=mean, temporal=time, wins=int(sm.Wins)))
        for ordinal, ((split, seed), val) in enumerate(zip(pair.index, value), 1):
            records.append(
                dict(
                    panel=spec["letter"],
                    metric=col,
                    ordinal=ordinal,
                    split=split,
                    seed=int(seed),
                    gain=float(val),
                    scale=spec["scale"],
                )
            )
    balance = pd.read_csv(directory / "mimic_rbc_temporal_balance.csv")
    eras = pd.read_csv(directory / "mimic_rbc_analysis_eras.csv")
    diagnostics = pd.read_csv(directory / "mimic_rbc_temporal_diagnostics.csv").iloc[0]
    assert len(balance) == 11 and set(balance.feature) == set(NAMES)
    np.testing.assert_allclose(balance.abs_smd_pre.mean(), diagnostics.mean_abs_smd_pre)
    np.testing.assert_allclose(balance.abs_smd_weighted.mean(), diagnostics.mean_abs_smd_ipw)
    np.testing.assert_allclose(eras.treated / eras.n, eras.treatment_rate)
    np.testing.assert_allclose(eras.outcomes / eras.n, eras.outcome_rate)
    return (gains, balance, eras, diagnostics, pd.DataFrame(records))


def axes(fig, x, y, w, h, panel):
    ax = fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])
    ax._panel = panel
    return ax


def text(
    fig,
    x,
    y,
    value,
    *,
    size=7.2,
    color=INK,
    ha="left",
    va="top",
    weight="normal",
    panel="shared",
    center_scope=None,
    **kwargs,
):
    artist = fig.text(
        x / WIDTH,
        1 - y / HEIGHT,
        value,
        size=size,
        color=color,
        ha=ha,
        va=va,
        weight=weight,
        **kwargs,
    )
    artist._panel = panel
    if ha == "center":
        artist._center_target_mm = x
        artist._center_scope = center_scope or panel
    return artist


def title(fig, x, y, letter, label):
    text(fig, x, y, letter, size=9.3, weight="bold", panel=letter)
    center = PANEL_BOUNDS[letter][0] + PANEL_BOUNDS[letter][2] / 2
    text(
        fig,
        center,
        y + 0.1,
        label,
        size=8.5,
        weight="bold",
        panel=letter,
        ha="center",
        center_scope=f"panel {letter} title",
    )


def format_number(val):
    return f"{val:g}"


def style_axis(ax, *, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(length=2, width=0.55, pad=1.7, labelsize=7.1, colors=INK)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=LIGHT, lw=0.45)


def interval(ax, x, triple, marker, *, vertical=True):
    point, low, high = triple
    err = [[point - low], [high - point]]
    options = dict(
        fmt=marker,
        color=BLUE,
        ms=4.0,
        lw=1.0,
        capsize=2,
        capthick=0.8,
        mfc=BLUE if marker == "o" else "white",
        mec=BLUE,
        mew=0.9,
        zorder=5,
    )
    if vertical:
        ax.errorbar(x, point, yerr=err, **options)
    else:
        ax.errorbar(point, x, xerr=err, **options)


def draw_gains(fig, groups):
    for left, group in zip([1.5, 55.5, 109.5], groups):
        spec = group["spec"]
        letter = spec["letter"]
        title(fig, left, 6.1, letter, spec["title"])
        center = PANEL_BOUNDS[letter][0] + PANEL_BOUNDS[letter][2] / 2
        text(
            fig,
            center,
            10.7,
            f"{spec['unit']}   ·   {group['wins']}/20 positive",
            size=7.1,
            panel=letter,
            ha="center",
            center_scope=f"panel {letter} subtitle",
        )
        ax = axes(fig, left + 7.8, 15.0, 43.0, 30.6, letter)
        style_axis(ax)
        ax.set(xlim=(0.25, 27.7), ylim=spec["lim"], yticks=spec["ticks"])
        ax.set_yticklabels([format_number(t) for t in spec["ticks"]])
        ax.bar(
            np.arange(1, 21),
            group["values"],
            width=0.79,
            color=np.where(group["values"] >= 0, BLUE, WARM),
            lw=0,
            zorder=3,
        )
        ax.axhline(0, color=INK, lw=0.8, zorder=4)
        ax.axvline(21.5, color=GRAY, lw=0.6, ls=(0, (1, 2)))
        interval(ax, 23, group["mean"], "o")
        interval(ax, 26, group["temporal"], "D")
        ax.set_xticks([1, 10, 20, 23, 26], ["1", "10", "20", "M", "T"])
    text(
        fig,
        WIDTH / 2,
        50.0,
        "Bars: paired splits   ·   M: mean   ·   T: temporal test   ·   Whiskers: 95% CI   ·   pp: percentage points",
        size=7.1,
        ha="center",
        center_scope="upper row shared key",
    )


def draw_balance(fig, frame, diagnostics):
    title(fig, 1.5, 55.6, "d", "Temporal covariate balance")
    matrix = axes(fig, 31.5, 63.7, 25.7, 33.6, "d")
    values = frame[["abs_smd_pre", "abs_smd_weighted"]].to_numpy()
    matrix.pcolormesh(
        np.arange(3) - 0.5,
        np.arange(12) - 0.5,
        values,
        vmin=0,
        vmax=0.65,
        cmap=SMDCMAP,
        shading="flat",
        edgecolors="face",
        linewidth=0,
    )
    matrix.set(xlim=(-0.5, 1.5), ylim=(10.5, -0.5), xticks=[], yticks=np.arange(11))
    matrix.set_yticklabels([NAMES[name] for name in frame.feature], fontsize=7.1)
    matrix.tick_params(axis="y", length=0, pad=2.2)
    matrix.spines[:].set_visible(False)
    matrix.axvline(0.5, color="white", lw=1.1)
    for r in range(1, 11):
        matrix.axhline(r - 0.5, color="white", lw=0.5)
    for i in range(11):
        for j in range(2):
            val = float(values[i, j])
            rgba = SMDCMAP(val / 0.65)
            rgb = np.array(rgba[:3])
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            foreground = "white" if linear @ np.array([0.2126, 0.7152, 0.0722]) < 0.19 else INK
            matrix.text(j, i, f"{val:.3f}", ha="center", va="center", size=7.1, color=foreground)
    text(fig, 31.5 + 25.7 / 4, 60.3, "Before", ha="center", panel="d")
    text(fig, 31.5 + 3 * 25.7 / 4, 60.3, "Weighted", ha="center", panel="d", color=BLUE)
    change = axes(fig, 63.0, 63.7, 31.9, 33.6, "d")
    style_axis(change, grid="x")
    reductions = frame.abs_smd_pre.to_numpy() - frame.abs_smd_weighted.to_numpy()
    change.barh(
        np.arange(11), reductions, height=0.59, color=np.where(reductions >= 0, BLUE, WARM), lw=0
    )
    change.axvline(0, color=INK, lw=0.7)
    change.set(xlim=(-0.1, 0.55), ylim=(10.5, -0.5), yticks=[], xticks=[-0.1, 0, 0.2, 0.4])
    change.set_xticklabels(["−0.1", "0", "0.2", "0.4"])
    change.spines["left"].set_visible(False)
    text(fig, 78.95, 60.3, "Reduction", ha="center", panel="d")
    cb = axes(fig, 32.6, 98.7, 18.0, 1.25, "d")
    cb.pcolormesh(
        np.linspace(0, 1, 129),
        [0, 1],
        np.linspace(0, 0.65, 128)[None, :],
        cmap=SMDCMAP,
        vmin=0,
        vmax=0.65,
        shading="flat",
        edgecolors="face",
        linewidth=0,
    )
    cb.set_axis_off()
    text(fig, 30.5, 99.32, "0", ha="right", va="center", size=7.0, panel="d")
    text(fig, 52.2, 99.32, "0.65", va="center", size=7.0, panel="d")
    center = PANEL_BOUNDS["d"][0] + PANEL_BOUNDS["d"][2] / 2
    text(
        fig,
        center,
        103.8,
        f"Mean |SMD| {values[:, 0].mean():.3f} → {values[:, 1].mean():.3f}    ESS/n = {diagnostics.ess_ratio:.3f}",
        size=7.2,
        panel="d",
        ha="center",
        va="center",
        center_scope="panel d summary",
    )
    below = (values <= 0.1).sum(axis=0)
    text(
        fig,
        center,
        106.8,
        f"Variables with |SMD| ≤ 0.10: {below[0]}/11 → {below[1]}/11",
        size=7.0,
        panel="d",
        ha="center",
        va="center",
        center_scope="panel d threshold counts",
    )


def draw_eras(fig, frame):
    title(fig, 99.2, 55.6, "e", "Clinical eras")
    center = PANEL_BOUNDS["e"][0] + PANEL_BOUNDS["e"][2] / 2
    handles = [
        Line2D([], [], color=BLUE, marker="D", lw=1.1, ms=3.8, label="Early RBC"),
        Line2D([], [], color=WARM, marker="o", ls="--", lw=1.1, ms=3.8, label="90-day mortality"),
    ]
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(center / WIDTH, 1 - 60.0 / HEIGHT),
        bbox_transform=fig.transFigure,
        fontsize=7.1,
        borderaxespad=0,
        borderpad=0,
        handlelength=1.35,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    legend._center_target_mm = center
    legend._center_scope = "panel e legend (handles and text together)"
    for item in legend.get_texts():
        item._panel = "e"
    ax = axes(fig, 106.7, 65.1, 53.7, 25.2, "e")
    style_axis(ax)
    ax.set(ylim=(0, 32), yticks=[0, 10, 20, 30], xlim=(-0.35, 2.35))
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    for column, color, marker, linestyle in [
        ("treatment_rate", BLUE, "D", "-"),
        ("outcome_rate", WARM, "o", "--"),
    ]:
        x = np.arange(3)
        vals = 100 * frame[column].to_numpy()
        ax.plot(x, vals, marker=marker, ms=4.0, color=color, lw=1.1, ls=linestyle, zorder=3)
        for i, (xx, val) in enumerate(zip(x, vals)):
            other = 100 * float(
                frame.iloc[i]["outcome_rate" if column == "treatment_rate" else "treatment_rate"]
            )
            higher = val > other
            ax.text(
                xx,
                val + (2.6 if higher else -2.6),
                f"{val:.1f}%",
                ha="center",
                va="bottom" if higher else "top",
                size=7.0,
                color=INK,
            )
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(
        [
            f"{year}\n{role}"
            for year, role in zip(
                ["2008–16", "2017–19", "2020–22"], ["Discovery", "Selection", "Temporal"]
            )
        ],
        fontsize=7.1,
        linespacing=1.1,
    )
    ax.tick_params(axis="x", length=0, pad=2.5)
    for y, key, label in [
        (99.5, "n", "n"),
        (102.8, "treated", "RBC"),
        (106.1, "outcomes", "Deaths"),
    ]:
        text(fig, 107.6, y, label, ha="right", va="center", size=7.0, panel="e")
        for i, value in enumerate(frame[key]):
            xx = 106.7 + (i + 0.35) / 2.7 * 53.7
            text(
                fig,
                xx,
                y,
                f"{int(value):,}",
                ha="center",
                va="center",
                size=7.0,
                panel="e",
                center_scope=f"panel e era {i + 1} {key} count",
            )


def make_figure(groups, balance, eras, diagnostics):
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), facecolor="white")
    text(
        fig,
        WIDTH / 2,
        1.1,
        "RITE versus all-variable baseline   ·   Positive gain favours RITE",
        size=7.4,
        ha="center",
        center_scope="whole figure header",
    )
    draw_gains(fig, groups)
    draw_balance(fig, balance, diagnostics)
    draw_eras(fig, eras)
    return fig


def render(data_dir, output_dir, *, dpi=900, panels=False):
    """Generate the manuscript figure and optional individual panels."""
    with plt.rc_context(STYLE):
        groups, balance, eras, diagnostics, _ = read_data(data_dir)
        fig = make_figure(groups, balance, eras, diagnostics)
        try:
            return export_figure(
                fig,
                output_dir,
                STEM,
                dpi=dpi,
                panel_bounds=PANEL_BOUNDS if panels else None,
                height_mm=HEIGHT,
                figure_number=3,
            )
        finally:
            plt.close(fig)
