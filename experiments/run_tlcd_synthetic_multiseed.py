import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from experiments.run_tlcd_synthetic import parse_args as parse_single_args
from experiments.run_tlcd_synthetic import run


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def extract_summary(seed: int, seed_dir: Path) -> dict:
    with (seed_dir / "tlcd_graph_metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)

    row = {"seed": seed, "runtime_sec": metrics["runtime_sec"]}
    row.update(metrics["flat"])

    uplift_path = seed_dir / "tlcd_downstream_uplift.csv"
    if uplift_path.exists():
        uplift = pd.read_csv(uplift_path)
        for _, result in uplift.iterrows():
            prefix = f"{result['Model']}_{result['Setting']}".replace("-", "_")
            for metric in ("MAE", "MSE", "R2", "Spearman", "Bias"):
                row[f"{prefix}_{metric}"] = result[metric]
    return row


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in per_seed.columns if column != "seed"]
    rows = []
    for column in metric_columns:
        values = pd.to_numeric(per_seed[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "metric": column,
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "min": values.min(),
                "max": values.max(),
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def main():
    wrapper = argparse.ArgumentParser(
        description="Run the locked synthetic TLCD configuration over multiple seeds.",
        add_help=True,
    )
    wrapper.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=parse_seed_list("2026,2027,2028,2029,2030"),
    )
    wrapper.add_argument(
        "--multiseed-out-dir",
        default="results/tlcd_synthetic_multiseed",
    )
    wrapper.add_argument("--force", action="store_true")
    wrapper_args, remaining = wrapper.parse_known_args()

    base_args = parse_single_args(remaining)
    out_dir = Path(wrapper_args.multiseed_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in wrapper_args.seeds:
        seed_dir = out_dir / f"seed_{seed}"
        metrics_path = seed_dir / "tlcd_graph_metrics.json"
        if metrics_path.exists() and not wrapper_args.force:
            print(f"Reusing completed seed={seed}: {seed_dir}", flush=True)
        else:
            args = copy.deepcopy(base_args)
            args.seed = seed
            args.out_dir = str(seed_dir)
            print(f"Running seed={seed}: {seed_dir}", flush=True)
            run(args)
        rows.append(extract_summary(seed, seed_dir))

    per_seed = pd.DataFrame(rows).sort_values("seed")
    summary = summarize(per_seed)
    per_seed.to_csv(out_dir / "tlcd_multiseed_per_seed.csv", index=False)
    summary.to_csv(out_dir / "tlcd_multiseed_summary.csv", index=False)

    manifest = {
        "seeds": wrapper_args.seeds,
        "single_run_config": vars(base_args),
        "per_seed_file": "tlcd_multiseed_per_seed.csv",
        "summary_file": "tlcd_multiseed_summary.csv",
    }
    with (out_dir / "tlcd_multiseed_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    focus = summary[
        summary["metric"].isin(
            [
                "directed_f1",
                "SHD",
                "C_Proximal_f1",
                "S_Prec_f1",
                "S_Learner_TLCD_RITE_MAE",
                "ClassTransform_TLCD_RITE_MAE",
            ]
        )
    ]
    print("\n=== Multi-seed summary ===", flush=True)
    print(focus.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
