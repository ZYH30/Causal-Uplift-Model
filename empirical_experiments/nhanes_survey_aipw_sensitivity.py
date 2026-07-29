import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from empirical_experiments.nhanes_empirical import (
    aipw_scores_crossfit,
    load_all_nhanes,
    task_frame,
)


TASKS = {
    "antihypertensive_sbp": (
        "results/nhanes_llm_conservative_antihypertensive_sbp_proximal_v1/"
        "nhanes_antihypertensive_sbp_run_meta.json"
    ),
    "chol_med_tchol": (
        "results/nhanes_llm_conservative_chol_med_tchol_proximal_v1/"
        "nhanes_chol_med_tchol_run_meta.json"
    ),
}


def design_linearized_mean(
    full_design: pd.DataFrame,
    domain_scores: pd.DataFrame,
) -> dict:
    design = full_design.merge(domain_scores, on="SEQN", how="left")
    design["domain"] = design["aipw_score"].notna()
    domain = design[design["domain"]]
    weights = domain["survey_weight"].to_numpy(dtype=float)
    scores = domain["aipw_score"].to_numpy(dtype=float)
    theta = float(np.sum(weights * scores) / np.sum(weights))

    residual = np.where(
        design["domain"],
        design["survey_weight"] * (design["aipw_score"].fillna(theta) - theta),
        0.0,
    )
    design = design.assign(linearized=residual / np.sum(weights))
    psu_totals = (
        design.groupby(["cycle_year", "SDMVSTRA", "SDMVPSU"], observed=True)["linearized"]
        .sum()
        .reset_index()
    )
    variance = 0.0
    strata_count = 0
    psu_count = 0
    for _, stratum in psu_totals.groupby(["cycle_year", "SDMVSTRA"], observed=True):
        values = stratum["linearized"].to_numpy(dtype=float)
        if len(values) < 2:
            continue
        variance += len(values) / (len(values) - 1) * np.sum((values - values.mean()) ** 2)
        strata_count += 1
        psu_count += len(values)
    degrees_freedom = psu_count - strata_count
    standard_error = float(np.sqrt(variance))
    critical = float(student_t.ppf(0.975, degrees_freedom))
    effective_n = float(np.sum(weights) ** 2 / np.sum(weights**2))
    return {
        "survey_aipw_ate": theta,
        "survey_aipw_se": standard_error,
        "survey_aipw_ci_low": theta - critical * standard_error,
        "survey_aipw_ci_high": theta + critical * standard_error,
        "survey_design_df": degrees_freedom,
        "survey_weighted_effective_n": effective_n,
        "survey_domain_weight_sum": float(np.sum(weights)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data/nhanes_raw")
    parser.add_argument("--out-dir", default="results/nhanes_survey_aipw_sensitivity_v1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-missing", type=float, default=0.35)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_data = load_all_nhanes(Path(args.cache_dir))
    design = all_data.loc[
        (all_data["RIDSTATR"] == 2)
        & all_data["WTMEC2YR"].notna()
        & (all_data["WTMEC2YR"] > 0),
        ["SEQN", "cycle_year", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"],
    ].drop_duplicates("SEQN")
    design["survey_weight"] = design["WTMEC2YR"] / 4.0

    rows = []
    for task, meta_path in TASKS.items():
        frame, meta = task_frame(all_data, task, max_missing=args.max_missing)
        run_meta = json.loads((Path(PROJECT_ROOT) / meta_path).read_text(encoding="utf-8"))
        c_set = run_meta["stage1_c_set"]
        cs_set = sorted(set(c_set) | set(run_meta["stage1_s_prec"]))
        frame = frame.merge(design[["SEQN", "survey_weight"]], on="SEQN", how="inner")
        scores = aipw_scores_crossfit(
            frame,
            c_set,
            args.seed,
            outcome_feature_cols=cs_set,
            sample_weight_col="survey_weight",
        )
        domain_scores = pd.DataFrame({"SEQN": frame["SEQN"], "aipw_score": scores})
        estimate = design_linearized_mean(design, domain_scores)
        weighted_treatment_rate = float(
            np.average(frame["T"].to_numpy(dtype=float), weights=frame["survey_weight"])
        )
        rows.append(
            {
                "Task": task,
                "N": meta["n"],
                "Weight": "WTMEC2YR / 4",
                "Strata": "cycle_year x SDMVSTRA",
                "PSU": "SDMVPSU",
                "PropensityFeatures": ";".join(c_set),
                "OutcomeFeatures": ";".join(cs_set),
                "survey_weighted_treatment_rate": weighted_treatment_rate,
                "unweighted_main_aipw_ate": run_meta["aipw_ate"],
                **estimate,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "nhanes_survey_weighted_aipw_sensitivity.csv", index=False)
    manifest = {
        "cycles": [2011, 2013, 2015, 2017],
        "weight": "WTMEC2YR divided by four two-year cycles",
        "variance": "Taylor linearization over cycle-specific masked strata and PSU; full MEC sample retained for domain variance",
        "nuisance": "survey-weighted LightGBM with e(C) and m(C union S)",
        "target": "US civilian noninstitutionalized population represented by combined 2011-2018 NHANES MEC cycles",
    }
    (out_dir / "nhanes_survey_weighted_aipw_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
