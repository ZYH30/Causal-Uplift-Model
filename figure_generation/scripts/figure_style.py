"""Shared Nature Portfolio-style settings for Figures 2--5."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tlcd_rite_mplconfig")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


SEED = 2026
FIGURE_WIDTH = 7.09  # 180 mm, the full manuscript text width.

COLORS = {
    "tlcd": "#0072B2",
    "tlcd_light": "#DCEEF7",
    "ges": "#D55E00",
    "ges_light": "#F8E5D9",
    "teal": "#009E73",
    "teal_light": "#DDF2EB",
    "gold": "#E69F00",
    "purple": "#8C6BB1",
    "black": "#202124",
    "gray": "#6B7280",
    "mid_gray": "#A7AFB8",
    "light_gray": "#D9DEE3",
    "pale_gray": "#F3F5F6",
    "white": "#FFFFFF",
}


def configure_style() -> None:
    """Apply one journal-scale visual system to all figures."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Liberation Sans",
            "mathtext.it": "Liberation Sans:italic",
            "mathtext.bf": "Liberation Sans:bold",
            "mathtext.cal": "Liberation Sans:italic",
            "mathtext.sf": "Liberation Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.4,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.0,
            "axes.labelcolor": COLORS["black"],
            "axes.edgecolor": COLORS["black"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "xtick.color": COLORS["black"],
            "ytick.color": COLORS["black"],
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 6.7,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.9,
            "lines.linewidth": 1.1,
            "lines.markersize": 5.0,
            "errorbar.capsize": 2.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.14,
    y: float = 1.10,
) -> None:
    ax.text(
        x,
        y,
        label.lower(),
        transform=ax.transAxes,
        fontsize=8.0,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLORS["black"],
        clip_on=False,
    )


def title_left(ax: plt.Axes, title: str, *, pad: float = 5.0) -> None:
    ax.set_title(title, loc="left", pad=pad)


def light_xgrid(ax: plt.Axes) -> None:
    ax.grid(axis="x", color=COLORS["light_gray"], lw=0.55, alpha=0.65, zorder=0)


def light_ygrid(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["light_gray"], lw=0.55, alpha=0.65, zorder=0)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    """Save publication, editing, and review formats from one render."""

    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "bbox_inches": "tight",
        "pad_inches": 0.035,
        "facecolor": "white",
    }
    fig.savefig(
        output_dir / f"{stem}.pdf",
        metadata={"Creator": f"{stem}.py"},
        **common,
    )
    fig.savefig(output_dir / f"{stem}.svg", **common)
    fig.savefig(output_dir / f"{stem}.png", dpi=600, **common)
    plt.close(fig)
