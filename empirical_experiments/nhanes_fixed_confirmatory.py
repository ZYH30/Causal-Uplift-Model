import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from empirical_experiments.nhanes_empirical import (
    compute_stage1,
    fit_nuisance,
    load_all_nhanes,
    make_models,
    task_frame,
)
from empirical_experiments.nhanes_validation_selection import candidate_tau, evaluate_tau


LOCKED_SPECS = [
    {
        "task": "antihypertensive_sbp",
        "model": "S-Learner",
        "method": "RITE_COnly",
        "conf_type": "proximal",
        "graph_variant": "Statistical+LLM-Conservative",
        "graph": "results/llm_adjudication/nhanes_antihypertensive_sbp_graph_llm_conservative.json",
    },
    {
        "task": "chol_med_tchol",
        "model": "ClassTransform",
        "method": "RITE_COnly",
        "conf_type": "proximal",
        "graph_variant": "Statistical+LLM-Conservative",
        "graph": "results/llm_adjudication/nhanes_chol_med_tchol_graph_llm_conservative.json",
    },
]

IMPROVEMENT_COLUMNS = [
    "R_Loss_Improvement",
    "DR_Qini_Improvement",
    "ObservedBudgetOutcome_Improvement",
]


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * float(p_values[idx]))
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted.tolist()


def run_locked_split(df_all: pd.DataFrame, spec: dict, seed: int, args) -> dict:
    graph_path = Path(spec["graph"])
    if not graph_path.is_absolute():
        graph_path = Path(PROJECT_ROOT) / graph_path
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    frame, meta = task_frame(df_all, spec["task"], max_missing=args.max_missing)
    feature_cols = [c for c in frame.columns if c not in {"SEQN", "T", "Y"}]
    train, test = train_test_split(
        frame,
        test_size=args.test_size,
        random_state=seed,
        stratify=frame["T"],
    )

    stage_args = SimpleNamespace(seed=seed, conf_type=spec["conf_type"], n_trials=0)
    stage1 = compute_stage1(train, graph, stage_args)
    c_set = sorted(stage1["c_set"])
    cs_set = sorted(set(c_set) | set(stage1["s_prec"]))
    e, mu, m0, m1 = fit_nuisance(
        train,
        test,
        c_set,
        seed,
        outcome_feature_cols=cs_set,
    )

    model = make_models(seed, args.uplift_estimators)[spec["model"]]
    model_args = SimpleNamespace(seed=seed)
    tau_naive = candidate_tau(
        spec["model"], model, "Naive_AllFeatures", train, test, feature_cols, stage1, model_args
    )
    tau_rite = candidate_tau(
        spec["model"], model, spec["method"], train, test, feature_cols, stage1, model_args
    )
    naive = evaluate_tau(spec["task"], test, tau_naive, e, mu, m0, m1)
    rite = evaluate_tau(spec["task"], test, tau_rite, e, mu, m0, m1)

    return {
        "Task": spec["task"],
        "GraphVariant": spec["graph_variant"],
        "ConfType": spec["conf_type"],
        "Model": spec["model"],
        "Method": spec["method"],
        "Seed": seed,
        "N": meta["n"],
        "Treated": meta["treated"],
        "Control": meta["control"],
        "C_Size": len(c_set),
        "S_Size": len(stage1["s_prec"]),
        "C_Set": ";".join(c_set),
        "S_Prec": ";".join(stage1["s_prec"]),
        "Naive_R_Loss": naive["R_Loss"],
        "RITE_R_Loss": rite["R_Loss"],
        "R_Loss_Improvement": naive["R_Loss"] - rite["R_Loss"],
        "Naive_DR_Qini": naive["DR_Qini"],
        "RITE_DR_Qini": rite["DR_Qini"],
        "DR_Qini_Improvement": rite["DR_Qini"] - naive["DR_Qini"],
        "Naive_ObservedBudgetOutcome": naive["PolicyOutcome_ObservedBudget"],
        "RITE_ObservedBudgetOutcome": rite["PolicyOutcome_ObservedBudget"],
        "ObservedBudgetOutcome_Improvement": (
            naive["PolicyOutcome_ObservedBudget"] - rite["PolicyOutcome_ObservedBudget"]
        ),
    }


def summarize(rows: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    summaries = []
    for keys, group in rows.groupby(["Task", "GraphVariant", "ConfType", "Model", "Method"]):
        common = dict(zip(["Task", "GraphVariant", "ConfType", "Model", "Method"], keys))
        for metric_idx, column in enumerate(IMPROVEMENT_COLUMNS):
            values = group[column].to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, 7319 + metric_idx, n_bootstrap)
            t_result = ttest_1samp(values, popmean=0.0, alternative="greater")
            try:
                w_result = wilcoxon(values, alternative="greater", zero_method="wilcox")
                wilcoxon_p = float(w_result.pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            summaries.append(
                {
                    **common,
                    "Metric": column,
                    "Repeats": len(values),
                    "Mean": float(np.mean(values)),
                    "Std": float(np.std(values, ddof=1)),
                    "Median": float(np.median(values)),
                    "Bootstrap_CI_Low": low,
                    "Bootstrap_CI_High": high,
                    "Wins": int(np.sum(values > 0)),
                    "Win_Rate": float(np.mean(values > 0)),
                    "OneSided_T_P": float(t_result.pvalue),
                    "OneSided_Wilcoxon_P": wilcoxon_p,
                }
            )
    result = pd.DataFrame(summaries)
    result["Holm_T_P"] = holm_adjust(result["OneSided_T_P"].tolist())
    result["Holm_Wilcoxon_P"] = holm_adjust(result["OneSided_Wilcoxon_P"].tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/nhanes_raw")
    parser.add_argument("--out-dir", default="results/nhanes_llm_fixed_confirmatory_v1")
    parser.add_argument("--seed-start", type=int, default=2040)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--max-missing", type=float, default=0.35)
    parser.add_argument("--uplift-estimators", type=int, default=80)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all = load_all_nhanes(Path(args.cache_dir))
    rows = []
    seeds = list(range(args.seed_start, args.seed_start + args.repeats))
    for spec in LOCKED_SPECS:
        for seed in seeds:
            print(
                f"Locked confirmation task={spec['task']} model={spec['model']} seed={seed}",
                flush=True,
            )
            rows.append(run_locked_split(df_all, spec, seed, args))
            pd.DataFrame(rows).to_csv(out_dir / "fixed_confirmatory_per_seed.partial.csv", index=False)

    per_seed = pd.DataFrame(rows)
    summary = summarize(per_seed, args.bootstrap)
    per_seed.to_csv(out_dir / "fixed_confirmatory_per_seed.csv", index=False)
    summary.to_csv(out_dir / "fixed_confirmatory_summary.csv", index=False)
    manifest = {
        "protocol": "locked_before_seeds_2040_2059",
        "specs": LOCKED_SPECS,
        "seeds": seeds,
        "test_size": args.test_size,
        "evaluation_propensity": "e(C)",
        "evaluation_outcome_nuisance": "m(C union S)",
        "comparison": "fixed RITE_COnly versus Naive_AllFeatures",
        "bootstrap_resamples": args.bootstrap,
    }
    (out_dir / "fixed_confirmatory_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
