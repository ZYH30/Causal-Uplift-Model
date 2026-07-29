import argparse
import os
import sys
import time
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.benchmark_direction_methods import run as run_single_seed


def parse_seed_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    all_rows = []
    for seed in args.seeds:
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        single_args = SimpleNamespace(
            n=args.n,
            seed=seed,
            test_size=args.test_size,
            out_dir=seed_dir,
            methods=args.methods,
            edge_tags=args.edge_tags,
            max_context=args.max_context,
            max_hsic_samples=args.max_hsic_samples,
            lgb_estimators=args.lgb_estimators,
            gain_estimators=args.gain_estimators,
            treatment=args.treatment,
            outcome=args.outcome,
            include_true_tau=False,
        )
        seed_rows = run_single_seed(single_args)
        seed_rows["seed"] = seed
        all_rows.append(seed_rows)

    result_df = pd.concat(all_rows, ignore_index=True)
    result_df.to_csv(os.path.join(args.out_dir, "direction_edge_results_all_seeds.csv"), index=False)

    summary = (
        result_df.groupby("method")
        .agg(
            accuracy_mean=("correct", "mean"),
            ambiguous_rate_mean=("ambiguous", "mean"),
            n_decisions=("correct", "size"),
        )
        .reset_index()
    )
    per_seed = (
        result_df.groupby(["method", "seed"])
        .agg(
            accuracy=("correct", "mean"),
            ambiguous_rate=("ambiguous", "mean"),
            n_edges=("correct", "size"),
        )
        .reset_index()
    )
    stability = (
        per_seed.groupby("method")
        .agg(
            accuracy_seed_mean=("accuracy", "mean"),
            accuracy_seed_std=("accuracy", "std"),
            accuracy_seed_min=("accuracy", "min"),
            accuracy_seed_max=("accuracy", "max"),
            ambiguous_seed_mean=("ambiguous_rate", "mean"),
        )
        .reset_index()
    )
    tag_summary = (
        result_df.groupby(["method", "tag"])
        .agg(
            accuracy_mean=("correct", "mean"),
            ambiguous_rate_mean=("ambiguous", "mean"),
            n_decisions=("correct", "size"),
        )
        .reset_index()
    )
    failures = result_df.loc[~result_df["correct"]].copy()

    summary.to_csv(os.path.join(args.out_dir, "direction_method_summary_all_seeds.csv"), index=False)
    per_seed.to_csv(os.path.join(args.out_dir, "direction_method_summary_by_seed.csv"), index=False)
    stability.to_csv(os.path.join(args.out_dir, "direction_method_stability.csv"), index=False)
    tag_summary.to_csv(os.path.join(args.out_dir, "direction_method_tag_summary_all_seeds.csv"), index=False)
    failures.to_csv(os.path.join(args.out_dir, "direction_failures_all_seeds.csv"), index=False)

    print("\n=== Multi-seed direction summary ===")
    print(summary.to_string(index=False))
    print("\n=== Multi-seed stability ===")
    print(stability.to_string(index=False))
    print("\n=== Multi-seed tag summary ===")
    print(tag_summary.to_string(index=False))
    print(f"\nRuntime: {time.time() - started:.1f}s")
    return result_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--seeds", type=parse_seed_list, default="2026,2027,2028,2029,2030")
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--out-dir", default="results/direction_benchmark_multiseed")
    parser.add_argument("--methods", nargs="*", default=["pair_spearman_anm", "task_adaptive"])
    parser.add_argument("--edge-tags", nargs="*", default=None)
    parser.add_argument("--max-context", type=int, default=3)
    parser.add_argument("--max-hsic-samples", type=int, default=1000)
    parser.add_argument("--lgb-estimators", type=int, default=80)
    parser.add_argument("--gain-estimators", type=int, default=50)
    parser.add_argument("--treatment", default="T")
    parser.add_argument("--outcome", default="Y")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
