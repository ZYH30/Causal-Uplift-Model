import argparse
import json
import os
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ccia.direction_methods import (
    anm_direction,
    ensemble_direction,
    infer_var_types,
    predictive_gain_direction,
    select_context,
    standardize_frame,
    task_adaptive_direction,
)
from data_generation import generate_complex_environment


DIFFICULT_EDGE_TAGS = {
    ("T", "Y"): "treatment_outcome",
    ("T", "Collid_1"): "collider_parent",
    ("Y", "Collid_1"): "collider_parent",
    ("T", "Collid_2"): "collider_parent",
    ("Y", "Collid_2"): "collider_parent",
    ("C_prox_1", "T"): "confounder_to_treatment",
    ("C_prox_2", "T"): "confounder_to_treatment",
    ("C_prox_3", "T"): "confounder_to_treatment",
    ("C_prox_1", "Y"): "confounder_to_outcome",
    ("C_prox_2", "Y"): "confounder_to_outcome",
    ("C_prox_3", "Y"): "confounder_to_outcome",
    ("P_1", "Y"): "precision_to_outcome",
    ("P_2", "Y"): "precision_to_outcome",
    ("Z_1", "T"): "instrument_to_treatment",
    ("Z_2", "T"): "instrument_to_treatment",
}


def directed_edges(graph):
    return [(parent, child) for child, parents in graph.items() for parent in parents]


def edge_tag(parent, child):
    if (parent, child) in DIFFICULT_EDGE_TAGS:
        return DIFFICULT_EDGE_TAGS[(parent, child)]
    if parent.startswith("C_root_") and child.startswith("C_prox_"):
        return "root_to_proximal"
    return "other"


def evaluate_relation(result, true_parent, true_child, x, y):
    if result.relation == "ambiguous":
        pred_parent, pred_child = None, None
        correct = False
    elif result.relation == "x_to_y":
        pred_parent, pred_child = x, y
        correct = pred_parent == true_parent and pred_child == true_child
    else:
        pred_parent, pred_child = y, x
        correct = pred_parent == true_parent and pred_child == true_child
    return pred_parent, pred_child, correct


def run_method(df, var_types, x, y, method, context, args):
    if method == "pair_spearman_anm":
        return anm_direction(
            df,
            x,
            y,
            var_types=var_types,
            context=[],
            dep_method="spearman",
            random_state=args.seed,
            max_samples=args.max_hsic_samples,
            n_estimators=args.lgb_estimators,
        )
    if method == "pair_hsic_anm":
        return anm_direction(
            df,
            x,
            y,
            var_types=var_types,
            context=[],
            dep_method="hsic",
            random_state=args.seed,
            max_samples=args.max_hsic_samples,
            n_estimators=args.lgb_estimators,
        )
    if method == "auto_cond_hsic_anm":
        return anm_direction(
            df,
            x,
            y,
            var_types=var_types,
            context=context,
            dep_method="hsic",
            random_state=args.seed,
            max_samples=args.max_hsic_samples,
            n_estimators=args.lgb_estimators,
        )
    if method == "auto_predictive_gain":
        return predictive_gain_direction(
            df,
            x,
            y,
            var_types=var_types,
            context=context,
            random_state=args.seed,
            n_estimators=args.gain_estimators,
        )
    if method == "auto_ensemble":
        return ensemble_direction(
            df,
            x,
            y,
            var_types=var_types,
            context=context,
            random_state=args.seed,
            max_samples=args.max_hsic_samples,
            n_estimators=args.lgb_estimators,
        )
    if method == "task_adaptive":
        return task_adaptive_direction(
            df,
            x,
            y,
            treatment=args.treatment,
            outcome=args.outcome,
            var_types=var_types,
            random_state=args.seed,
            max_samples=args.max_hsic_samples,
            n_estimators=args.lgb_estimators,
        )
    raise ValueError(method)


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    df, graph, _ = generate_complex_environment(n=args.n, seed=args.seed)
    if not args.include_true_tau:
        df = df.drop(columns=["true_tau"])
    df_train, _ = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["T"],
    )
    var_types = infer_var_types(df_train)
    df_train = standardize_frame(df_train, var_types)

    methods = args.methods or [
        "pair_spearman_anm",
        "pair_hsic_anm",
        "auto_cond_hsic_anm",
        "auto_predictive_gain",
        "auto_ensemble",
        "task_adaptive",
    ]

    edges = directed_edges(graph)
    if args.edge_tags:
        edges = [edge for edge in edges if edge_tag(*edge) in set(args.edge_tags)]

    rows = []
    for parent, child in edges:
        if parent not in df_train.columns or child not in df_train.columns:
            continue
        x, y = parent, child
        context = select_context(
            df_train,
            x,
            y,
            max_context=args.max_context,
            forbidden={"true_tau"},
        )
        for method in methods:
            result = run_method(df_train, var_types, x, y, method, context, args)
            pred_parent, pred_child, correct = evaluate_relation(result, parent, child, x, y)
            rows.append(
                {
                    "method": method,
                    "parent": parent,
                    "child": child,
                    "tag": edge_tag(parent, child),
                    "x": x,
                    "y": y,
                    "context": json.dumps(context, ensure_ascii=False),
                    "relation": result.relation,
                    "pred_parent": pred_parent,
                    "pred_child": pred_child,
                    "correct": correct,
                    "ambiguous": result.relation == "ambiguous",
                    "score_xy": result.score_xy,
                    "score_yx": result.score_yx,
                    "confidence": result.confidence,
                    "details": json.dumps(result.details, default=str),
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(args.out_dir, "direction_edge_results.csv"), index=False)

    summary = (
        result_df.groupby("method")
        .agg(
            accuracy=("correct", "mean"),
            ambiguous_rate=("ambiguous", "mean"),
            n_edges=("correct", "size"),
        )
        .reset_index()
    )
    tag_summary = (
        result_df.groupby(["method", "tag"])
        .agg(
            accuracy=("correct", "mean"),
            ambiguous_rate=("ambiguous", "mean"),
            n_edges=("correct", "size"),
        )
        .reset_index()
    )
    summary.to_csv(os.path.join(args.out_dir, "direction_method_summary.csv"), index=False)
    tag_summary.to_csv(os.path.join(args.out_dir, "direction_method_tag_summary.csv"), index=False)

    metadata = {
        "n": args.n,
        "seed": args.seed,
        "test_size": args.test_size,
        "runtime_sec": time.time() - started,
        "methods": methods,
        "max_context": args.max_context,
        "edge_tags": args.edge_tags,
    }
    with open(os.path.join(args.out_dir, "direction_benchmark_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== Direction method summary ===")
    print(summary.to_string(index=False))
    print("\n=== Direction tag summary ===")
    print(tag_summary.to_string(index=False))
    return result_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--out-dir", default="results/direction_benchmark")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--edge-tags", nargs="*", default=None)
    parser.add_argument("--max-context", type=int, default=3)
    parser.add_argument("--max-hsic-samples", type=int, default=1200)
    parser.add_argument("--lgb-estimators", type=int, default=100)
    parser.add_argument("--gain-estimators", type=int, default=60)
    parser.add_argument("--include-true-tau", action="store_true", default=False)
    parser.add_argument("--treatment", default="T")
    parser.add_argument("--outcome", default="Y")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
