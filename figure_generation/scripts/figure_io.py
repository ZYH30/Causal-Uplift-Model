"""Shared file export for the manuscript's fixed-size result figures."""

from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tlcd-rite-figures"))

import matplotlib

matplotlib.use("Agg")
from matplotlib.text import Text
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUT = ROOT.parent / "figures"
EXPORT_DATE = datetime(2026, 10, 19, tzinfo=timezone.utc)


def expanded_panel_bounds(fig, bounds, height_mm):
    """Include panel annotations when cropping the synthetic result panels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item._panel = ax._panel
    texts = {letter: [] for letter in bounds}
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        panel = getattr(artist, "_panel", getattr(artist.axes, "_panel", None))
        if panel not in texts:
            continue
        box = artist.get_window_extent(renderer)
        texts[panel].append(
            (
                box.x0 / fig.dpi * 25.4,
                height_mm - box.y1 / fig.dpi * 25.4,
                box.x1 / fig.dpi * 25.4,
                height_mm - box.y0 / fig.dpi * 25.4,
            )
        )
    expanded = {}
    for letter, (x, y, w, h) in bounds.items():
        related = texts[letter]
        x0 = min([x] + [t[0] - 0.2 for t in related])
        y0 = min([y] + [t[1] - 0.2 for t in related])
        x1 = max([x + w] + [t[2] + 0.2 for t in related])
        y1 = max([y + h] + [t[3] + 0.2 for t in related])
        expanded[letter] = (x0, y0, x1 - x0, y1 - y0)
    return expanded


def export_figure(
    fig, output_dir, stem, *, dpi, height_mm, figure_number, panel_bounds=None, panel_padding=0
):
    """Save PDF, editable-text SVG and high-resolution PNG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pdf": {"Creator": "Matplotlib", "CreationDate": EXPORT_DATE, "ModDate": EXPORT_DATE},
        "svg": {"Creator": "Matplotlib", "Date": EXPORT_DATE.date().isoformat()},
        "png": {"Software": "Matplotlib"},
    }
    outputs = []
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=dpi, facecolor="white", metadata=metadata[suffix])
        outputs.append(path)
    if panel_bounds:
        directory = output_dir / "panels"
        directory.mkdir(exist_ok=True)
        for letter, (x, y, w, h) in panel_bounds.items():
            box = Bbox.from_bounds(x / 25.4, (height_mm - y - h) / 25.4, w / 25.4, h / 25.4)
            for suffix in ("pdf", "svg", "png"):
                path = directory / f"Figure_{figure_number}{letter}.{suffix}"
                fig.savefig(
                    path,
                    dpi=dpi,
                    facecolor="white",
                    bbox_inches=box,
                    pad_inches=panel_padding,
                    metadata=metadata[suffix],
                )
                outputs.append(path)
    return outputs
