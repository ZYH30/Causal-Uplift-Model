"""Generate the manuscript result figures from the included aggregate data."""

import argparse
import importlib
from pathlib import Path

from scripts.figure_io import DEFAULT_DATA, DEFAULT_OUTPUT

DATA_FOLDERS = {3: "mimic", 4: "mimic", 5: "synthetic", 6: "nhanes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=DATA_FOLDERS,
        default=list(DATA_FOLDERS),
        metavar="N",
        help="Figure numbers to generate; default: 3 4 5 6",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help="Directory containing mimic/, synthetic/ and nhanes/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for figure files; default: repository figures/",
    )
    parser.add_argument(
        "--dpi", type=int, default=900, help="PNG resolution, greater than 600; default: 900"
    )
    parser.add_argument(
        "--panels", action="store_true", help="Also export each individual panel to panels/"
    )
    args = parser.parse_args()
    if args.dpi <= 600:
        parser.error("--dpi must be greater than 600 for publication exports")
    for number in dict.fromkeys(args.figures):
        module = importlib.import_module(f"scripts.plot_figure_{number}")
        outputs = module.render(
            args.data_dir / DATA_FOLDERS[number], args.output_dir, dpi=args.dpi, panels=args.panels
        )
        print(
            f"Figure {number}: {len(outputs)} files written to {args.output_dir.resolve()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
