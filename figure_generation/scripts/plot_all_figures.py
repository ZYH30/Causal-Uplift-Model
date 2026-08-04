"""Regenerate Figures 2--5 from frozen aggregate inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from figure_style import configure_style
from plot_figure_2 import build_figure as build_figure_2
from plot_figure_3 import build_figure as build_figure_3
from plot_figure_4 import build_figure as build_figure_4
from plot_figure_5 import build_figure as build_figure_5


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "figures"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure_style()
    build_figure_2(args.output_dir)
    build_figure_3(args.output_dir)
    build_figure_4(args.output_dir)
    build_figure_5(args.output_dir)


if __name__ == "__main__":
    main()
