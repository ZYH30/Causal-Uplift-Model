import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from empirical_experiments.nhanes_empirical import (
    TASKS,
    compute_stage1,
    fit_nuisance,
    load_all_nhanes,
    make_models,
    policy_outcome,
    r_loss,
    task_frame,
    uplift_ranking_metrics,
)
from experiments.reproduce_synthetic import fit_predict_stage2


SPECS = [
    {
        "task": "antihypertensive_sbp",
        "conf_type": "proximal",
        "graph_variant": "Statistical+LLM-Conservative",
        "graph": "results/llm_adjudication/nhanes_antihypertensive_sbp_graph_llm_conservative.json",
        "seeds": [2026, 2027, 2028, 2029],
    },
    {
        "task": "chol_med_tchol",
        "conf_type": "proximal",
        "graph_variant": "Statistical+LLM-Conservative",
        "graph": "results/llm_adjudication/nhanes_chol_med_tchol_graph_llm_conservative.json",
        "seeds": [2026, 2027, 2028, 2029],
    },
]


def candidate_tau(model_name: str, base_model, method: str, train_df: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], stage1, args):
    if method == "Naive_AllFeatures":
        model = copy.deepcopy(base_model)
        model.fit(train_df[feature_cols].values, train_df["Y"].values, train_df["T"].values)
        return model.predict(eval_df[feature_cols].values)
    if method == "RITE_COnly":
        return fit_predict_stage2(base_model, eval_df, "T", "Y", stage1, use_precision=False, args=args)
    if method == "RITE_CPlusS":
        return fit_predict_stage2(base_model, eval_df, "T", "Y", stage1, use_precision=True, args=args)
    raise ValueError(method)


def evaluate_tau(task: str, eval_df: pd.DataFrame, tau: np.ndarray, e, mu, m0, m1) -> dict:
    lower_is_better = TASKS[task]["lower_is_better"]
    policy = (tau < 0).astype(int) if lower_is_better else (tau > 0).astype(int)
    ranking, _curve = uplift_ranking_metrics(
        eval_df["Y"].to_numpy(),
        eval_df["T"].to_numpy(),
        tau,
        e,
        m0,
        m1,
        lower_is_better=lower_is_better,
        observed_treat_rate=float(eval_df["T"].mean()),
    )
    return {
        "R_Loss": r_loss(eval_df["Y"].to_numpy(), eval_df["T"].to_numpy(), tau, mu, e),
        "PolicyOutcome": policy_outcome(eval_df["Y"].to_numpy(), eval_df["T"].to_numpy(), policy, e, m0, m1),
        "TauMean": float(np.mean(tau)),
        "TauStd": float(np.std(tau)),
        "TreatRatePolicy": float(policy.mean()),
        **ranking,
    }


def naive_delta(test_rows: list[dict], selected: dict) -> dict:
    base = next(
        row
        for row in test_rows
        if row["Model"] == selected["Model"] and row["Method"] == "Naive_AllFeatures"
    )
    return {
        "Test_R_Loss_Improvement": base["R_Loss"] - selected["R_Loss"],
        "Test_DR_Qini_Improvement": selected["DR_Qini"] - base["DR_Qini"],
        "Test_ObservedBudgetOutcome_Improvement": base["PolicyOutcome_ObservedBudget"] - selected["PolicyOutcome_ObservedBudget"],
        "Test_BestGridOutcome_Improvement": base["PolicyOutcome_BestGrid"] - selected["PolicyOutcome_BestGrid"],
    }


def select_validation_candidate(group: pd.DataFrame, rule: str) -> pd.Series:
    """Select one candidate using validation metrics only."""
    ranked = group.copy()
    if rule == "lexicographic_policy":
        return ranked.sort_values(
            ["PolicyOutcome_ObservedBudget", "DR_Qini", "R_Loss"],
            ascending=[True, False, True],
        ).iloc[0]
    if rule != "composite_rank":
        raise ValueError(f"Unknown selection rule: {rule}")

    # The rank sum is scale-free and gives equal weight to calibration,
    # uplift ranking, and fixed-budget policy performance.
    ranked["R_Loss_Rank"] = ranked["R_Loss"].rank(method="average", ascending=True)
    ranked["DR_Qini_Rank"] = ranked["DR_Qini"].rank(method="average", ascending=False)
    ranked["Budget_Rank"] = ranked["PolicyOutcome_ObservedBudget"].rank(
        method="average", ascending=True
    )
    ranked["Composite_Rank_Sum"] = ranked[
        ["R_Loss_Rank", "DR_Qini_Rank", "Budget_Rank"]
    ].sum(axis=1)
    return ranked.sort_values(
        ["Composite_Rank_Sum", "PolicyOutcome_ObservedBudget", "DR_Qini", "R_Loss"],
        ascending=[True, True, False, True],
    ).iloc[0]


def run_one(df_all: pd.DataFrame, spec: dict, seed: int, args) -> tuple[list[dict], list[dict]]:
    task = spec["task"]
    graph_path = Path(spec["graph"])
    if not graph_path.is_absolute():
        graph_path = Path(PROJECT_ROOT) / graph_path
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph_variant = spec.get("graph_variant", "statistical")
    df_task, meta = task_frame(df_all, task, max_missing=args.max_missing)
    feature_cols = [c for c in df_task.columns if c not in {"SEQN", "T", "Y"}]

    train_full, test = train_test_split(df_task, test_size=args.test_size, random_state=seed, stratify=df_task["T"])
    inner_train, val = train_test_split(train_full, test_size=args.val_size, random_state=seed + 1000, stratify=train_full["T"])

    stage_args = SimpleNamespace(seed=seed, conf_type=spec["conf_type"], n_trials=0)
    val_stage1 = compute_stage1(inner_train, graph, stage_args)
    test_stage1 = compute_stage1(train_full, graph, stage_args)

    val_c = sorted(val_stage1["c_set"])
    val_cs = sorted(set(val_c) | set(val_stage1["s_prec"]))
    test_c = sorted(test_stage1["c_set"])
    test_cs = sorted(set(test_c) | set(test_stage1["s_prec"]))
    e_val, mu_val, m0_val, m1_val = fit_nuisance(
        inner_train,
        val,
        val_c,
        seed,
        outcome_feature_cols=val_cs,
    )
    e_test, mu_test, m0_test, m1_test = fit_nuisance(
        train_full,
        test,
        test_c,
        seed,
        outcome_feature_cols=test_cs,
    )

    methods = ["Naive_AllFeatures", "RITE_COnly", "RITE_CPlusS"]
    val_rows = []
    test_rows = []
    model_args = SimpleNamespace(seed=seed)

    for model_name, base_model in make_models(seed, args.uplift_estimators).items():
        for method in methods:
            tau_val = candidate_tau(model_name, base_model, method, inner_train, val, feature_cols, val_stage1, model_args)
            val_metrics = evaluate_tau(task, val, tau_val, e_val, mu_val, m0_val, m1_val)
            val_rows.append(
                {
                    "Task": task,
                    "GraphVariant": graph_variant,
                    "ConfType": spec["conf_type"],
                    "Seed": seed,
                    "Split": "validation",
                    "Model": model_name,
                    "Method": method,
                    **val_metrics,
                }
            )

            tau_test = candidate_tau(model_name, base_model, method, train_full, test, feature_cols, test_stage1, model_args)
            test_metrics = evaluate_tau(task, test, tau_test, e_test, mu_test, m0_test, m1_test)
            test_rows.append(
                {
                    "Task": task,
                    "GraphVariant": graph_variant,
                    "ConfType": spec["conf_type"],
                    "Seed": seed,
                    "Split": "test",
                    "Model": model_name,
                    "Method": method,
                    "N": meta["n"],
                    "Treated": meta["treated"],
                    "Control": meta["control"],
                    "C_Size": len(test_stage1["c_set"]),
                    "S_Size": len(test_stage1["s_prec"]),
                    "C_Set": ";".join(test_stage1["c_set"]),
                    "S_Prec": ";".join(test_stage1["s_prec"]),
                    **test_metrics,
                }
            )

    selected_rows = []
    for model_name, group in pd.DataFrame(val_rows).groupby("Model"):
        chosen = select_validation_candidate(group, args.selection_rule)
        selected_test = next(
            row
            for row in test_rows
            if row["Model"] == model_name and row["Method"] == chosen["Method"]
        )
        selected_rows.append(
            {
                **selected_test,
                "SelectionMetric": args.selection_rule,
                "Validation_Selected_Method": chosen["Method"],
                "Validation_ObservedBudgetOutcome": chosen["PolicyOutcome_ObservedBudget"],
                "Validation_DR_Qini": chosen["DR_Qini"],
                "Validation_R_Loss": chosen["R_Loss"],
                "Validation_Composite_Rank_Sum": chosen.get("Composite_Rank_Sum", np.nan),
                **naive_delta(test_rows, selected_test),
            }
        )
    return val_rows + test_rows, selected_rows


def summarize_selected(selected: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Test_R_Loss_Improvement",
        "Test_DR_Qini_Improvement",
        "Test_ObservedBudgetOutcome_Improvement",
        "Test_BestGridOutcome_Improvement",
        "DR_Qini",
        "PolicyOutcome_ObservedBudget",
        "R_Loss",
    ]
    summary = (
        selected.groupby(["Task", "GraphVariant", "ConfType", "Model", "Validation_Selected_Method"])[metric_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = ["_".join([str(x) for x in col if x]) for col in summary.columns]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/nhanes_raw")
    parser.add_argument("--out-dir", default="results/nhanes_validation_selection_v1")
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--val-size", type=float, default=0.25)
    parser.add_argument("--max-missing", type=float, default=0.35)
    parser.add_argument("--uplift-estimators", type=int, default=80)
    parser.add_argument("--spec-file", default="")
    parser.add_argument(
        "--selection-rule",
        choices=["composite_rank", "lexicographic_policy"],
        default="composite_rank",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all = load_all_nhanes(Path(args.cache_dir))

    all_rows = []
    selected_rows = []
    specs = SPECS
    if args.spec_file:
        spec_path = Path(args.spec_file)
        if not spec_path.is_absolute():
            spec_path = Path(PROJECT_ROOT) / spec_path
        specs = json.loads(spec_path.read_text(encoding="utf-8"))

    for spec in specs:
        for seed in spec["seeds"]:
            print(f"Running validation selection task={spec['task']} conf={spec['conf_type']} seed={seed}", flush=True)
            rows, selected = run_one(df_all, spec, seed, args)
            all_rows.extend(rows)
            selected_rows.extend(selected)
            pd.DataFrame(all_rows).to_csv(out_dir / "validation_selection_all_metrics.partial.csv", index=False)
            pd.DataFrame(selected_rows).to_csv(out_dir / "validation_selection_selected.partial.csv", index=False)

    all_df = pd.DataFrame(all_rows)
    selected_df = pd.DataFrame(selected_rows)
    summary_df = summarize_selected(selected_df)
    all_df.to_csv(out_dir / "validation_selection_all_metrics.csv", index=False)
    selected_df.to_csv(out_dir / "validation_selection_selected.csv", index=False)
    summary_df.to_csv(out_dir / "validation_selection_summary.csv", index=False)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
