"""Clinical interpretation and uncertainty analyses for the locked MIMIC RBC study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from empirical_experiments.mimic_rbc_target_trial import (
    AdjustmentInterface,
    DEFAULT_DATABASE,
    DEFAULT_MIMIC_ROOT,
    DEFAULT_OUT_DIR,
    cohort_artifacts,
    crossfit_training_nuisance,
    dr_effect_score,
    evaluate_tau,
    fit_binary_or_constant,
    fit_propensity_or_constant,
    fit_tau,
    impute_from_training,
    joint_strata,
    policy_risk,
    propensity_clip_for_dimension,
    regressor,
    weighted_smd,
    write_json,
)


def locked_analysis_inputs(
    database: Path,
    out_dir: Path,
    max_missing: float,
    seed: int,
    mimic_root: Path = DEFAULT_MIMIC_ROOT,
) -> dict:
    frame, feature_cols, _ = cohort_artifacts(
        database,
        out_dir,
        max_missing,
        mimic_root,
    )
    roles = json.loads(
        (out_dir / "mimic_rbc_ccia_roles.json").read_text(encoding="utf-8")
    )
    lock = json.loads(
        (out_dir / "mimic_rbc_locked_spec.json").read_text(encoding="utf-8")
    )
    train_raw = frame[frame["era_index"].le(3)].copy()
    evaluation_raw = frame[frame["era_index"].eq(4)].copy()
    train, evaluation, _ = impute_from_training(
        train_raw,
        evaluation_raw,
        feature_cols,
    )
    c_set = tuple(
        feature for feature in roles["c_proximal"] if feature in feature_cols
    )
    s_set = tuple(
        feature
        for feature in roles["s_precision"]
        if feature in feature_cols and feature not in c_set
    )
    ccia = AdjustmentInterface("CCIA_CPlusS", c_set, s_set)
    all_variables = AdjustmentInterface("AllVariables", tuple(feature_cols))
    analysis_seed = seed + 1
    evaluator = crossfit_training_nuisance(
        evaluation,
        list(ccia.c_set),
        ccia.outcome_set,
        analysis_seed + 700,
        folds=5,
    )
    learner = lock["primary_learner"]
    tau = {
        "AllVariables": fit_tau(
            learner,
            train,
            evaluation,
            all_variables,
            analysis_seed,
        ),
        "CCIA_CPlusS": fit_tau(
            learner,
            train,
            evaluation,
            ccia,
            analysis_seed,
        ),
    }
    return {
        "frame": frame,
        "train": train,
        "evaluation": evaluation,
        "feature_cols": feature_cols,
        "ccia": ccia,
        "all_variables": all_variables,
        "learner": learner,
        "analysis_seed": analysis_seed,
        "evaluator": evaluator,
        "tau": tau,
    }


def verify_locked_result(payload: dict, out_dir: Path) -> None:
    expected = pd.read_csv(out_dir / "mimic_rbc_temporal_test_metrics.csv")
    expected = expected[
        expected["Learner"].eq(payload["learner"])
        & expected["Interface"].isin(["AllVariables", "CCIA_CPlusS"])
    ].set_index("Interface")
    outcome = payload["evaluation"]["Y"].to_numpy(dtype=int)
    treatment = payload["evaluation"]["T"].to_numpy(dtype=int)
    for interface, tau in payload["tau"].items():
        observed = evaluate_tau(
            outcome,
            treatment,
            tau,
            payload["evaluator"],
        )
        for metric in ("R_Loss", "DR_Qini", "Policy_Risk_Reduction"):
            if not np.isclose(
                observed[metric],
                expected.loc[interface, metric],
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    f"Locked result mismatch for {interface}/{metric}"
                )


def paired_patient_bootstrap(
    payload: dict,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = payload["evaluation"]
    outcome = evaluation["Y"].to_numpy(dtype=int)
    treatment = evaluation["T"].to_numpy(dtype=int)
    nuisance = payload["evaluator"]
    tau = payload["tau"]
    n = len(evaluation)
    rng = np.random.default_rng(seed)
    records = []
    for draw in range(draws):
        index = rng.integers(0, n, size=n)
        nuisance_draw = {
            key: values[index]
            for key, values in nuisance.items()
        }
        metrics = {
            interface: evaluate_tau(
                outcome[index],
                treatment[index],
                values[index],
                nuisance_draw,
            )
            for interface, values in tau.items()
        }
        records.append(
            {
                "draw": draw,
                "R_Loss_Improvement": (
                    metrics["AllVariables"]["R_Loss"]
                    - metrics["CCIA_CPlusS"]["R_Loss"]
                ),
                "DR_Qini_Improvement": (
                    metrics["CCIA_CPlusS"]["DR_Qini"]
                    - metrics["AllVariables"]["DR_Qini"]
                ),
                "Policy_Risk_Reduction_Improvement": (
                    metrics["CCIA_CPlusS"]["Policy_Risk_Reduction"]
                    - metrics["AllVariables"]["Policy_Risk_Reduction"]
                ),
            }
        )
    samples = pd.DataFrame(records)
    full_metrics = {
        interface: evaluate_tau(outcome, treatment, values, nuisance)
        for interface, values in tau.items()
    }
    point = {
        "R_Loss_Improvement": (
            full_metrics["AllVariables"]["R_Loss"]
            - full_metrics["CCIA_CPlusS"]["R_Loss"]
        ),
        "DR_Qini_Improvement": (
            full_metrics["CCIA_CPlusS"]["DR_Qini"]
            - full_metrics["AllVariables"]["DR_Qini"]
        ),
        "Policy_Risk_Reduction_Improvement": (
            full_metrics["CCIA_CPlusS"]["Policy_Risk_Reduction"]
            - full_metrics["AllVariables"]["Policy_Risk_Reduction"]
        ),
    }
    summary = []
    for metric, estimate in point.items():
        values = samples[metric].to_numpy()
        low, high = np.quantile(values, [0.025, 0.975])
        summary.append(
            {
                "Metric": metric,
                "Point_Estimate": estimate,
                "Bootstrap_Mean": float(values.mean()),
                "CI_Low": float(low),
                "CI_High": float(high),
                "Probability_Positive": float(np.mean(values > 0)),
                "Draws": draws,
            }
        )
    return samples, pd.DataFrame(summary)


def _crossfit_linear_propensity(
    frame: pd.DataFrame,
    features: list[str],
    seed: int,
    folds: int = 5,
) -> np.ndarray:
    treatment = frame["T"].to_numpy(dtype=int)
    outcome = frame["Y"].to_numpy(dtype=int)
    propensity = np.zeros(len(frame), dtype=float)
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(
        split.split(frame, joint_strata(treatment, outcome))
    ):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.3,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
                random_state=seed + fold,
            ),
        )
        model.fit(frame.iloc[train_idx][features], treatment[train_idx])
        propensity[test_idx] = model.predict_proba(
            frame.iloc[test_idx][features]
        )[:, 1]
    return np.clip(propensity, 0.005, 0.995)


def conservative_total_set_sensitivity(
    payload: dict,
    out_dir: Path,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roles = json.loads(
        (out_dir / "mimic_rbc_ccia_roles.json").read_text(encoding="utf-8")
    )
    c_total = [
        feature
        for feature in roles["c_total"]
        if feature in payload["feature_cols"]
    ]
    evaluation = payload["evaluation"]
    treatment = evaluation["T"].to_numpy(dtype=int)
    outcome = evaluation["Y"].to_numpy(dtype=int)
    nuisance = {
        key: value.copy()
        for key, value in payload["evaluator"].items()
    }
    nuisance["e"] = _crossfit_linear_propensity(
        evaluation,
        c_total,
        seed,
    )

    prevalence = float(np.mean(treatment))
    weights = np.where(
        treatment == 1,
        prevalence / nuisance["e"],
        (1 - prevalence) / (1 - nuisance["e"]),
    )
    balance = pd.DataFrame(
        [
            {
                "feature": feature,
                "smd_pre": weighted_smd(
                    evaluation[feature].to_numpy(dtype=float),
                    treatment,
                ),
                "smd_ipw": weighted_smd(
                    evaluation[feature].to_numpy(dtype=float),
                    treatment,
                    weights,
                ),
            }
            for feature in c_total
        ]
    )

    metrics = {
        interface: evaluate_tau(
            outcome,
            treatment,
            tau,
            nuisance,
        )
        for interface, tau in payload["tau"].items()
    }
    point = {
        "R_Loss_Improvement": (
            metrics["AllVariables"]["R_Loss"]
            - metrics["CCIA_CPlusS"]["R_Loss"]
        ),
        "DR_Qini_Improvement": (
            metrics["CCIA_CPlusS"]["DR_Qini"]
            - metrics["AllVariables"]["DR_Qini"]
        ),
        "Policy_Risk_Reduction_Improvement": (
            metrics["CCIA_CPlusS"]["Policy_Risk_Reduction"]
            - metrics["AllVariables"]["Policy_Risk_Reduction"]
        ),
    }

    rng = np.random.default_rng(seed + 1)
    records = []
    for draw in range(draws):
        index = rng.integers(0, len(evaluation), size=len(evaluation))
        nuisance_draw = {
            key: value[index]
            for key, value in nuisance.items()
        }
        draw_metrics = {
            interface: evaluate_tau(
                outcome[index],
                treatment[index],
                tau[index],
                nuisance_draw,
            )
            for interface, tau in payload["tau"].items()
        }
        records.append(
            {
                "draw": draw,
                "R_Loss_Improvement": (
                    draw_metrics["AllVariables"]["R_Loss"]
                    - draw_metrics["CCIA_CPlusS"]["R_Loss"]
                ),
                "DR_Qini_Improvement": (
                    draw_metrics["CCIA_CPlusS"]["DR_Qini"]
                    - draw_metrics["AllVariables"]["DR_Qini"]
                ),
                "Policy_Risk_Reduction_Improvement": (
                    draw_metrics["CCIA_CPlusS"]["Policy_Risk_Reduction"]
                    - draw_metrics["AllVariables"]["Policy_Risk_Reduction"]
                ),
            }
        )
    bootstrap = pd.DataFrame(records)
    summary_row = {
        "analysis": "conservative_c_total_propensity",
        "c_total_size": len(c_total),
        "ess_ratio": float(
            weights.sum() ** 2 / np.sum(weights**2) / len(weights)
        ),
        "mean_abs_smd_pre": float(balance["smd_pre"].abs().mean()),
        "mean_abs_smd_ipw": float(balance["smd_ipw"].abs().mean()),
        "max_abs_smd_pre": float(balance["smd_pre"].abs().max()),
        "max_abs_smd_ipw": float(balance["smd_ipw"].abs().max()),
        **point,
    }
    for metric in point:
        values = bootstrap[metric].to_numpy()
        summary_row[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
        summary_row[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
        summary_row[f"{metric}_positive_probability"] = float(
            np.mean(values > 0)
        )
    return pd.DataFrame([summary_row]), balance, bootstrap


def benefit_quartile_profiles(
    payload: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = payload["evaluation"].copy()
    benefit = -payload["tau"]["CCIA_CPlusS"]
    dr_benefit = -dr_effect_score(
        evaluation["Y"].to_numpy(dtype=int),
        evaluation["T"].to_numpy(dtype=int),
        payload["evaluator"]["e"],
        payload["evaluator"]["m0"],
        payload["evaluator"]["m1"],
    )
    evaluation["predicted_benefit"] = benefit
    evaluation["dr_benefit"] = dr_benefit
    evaluation["benefit_quartile"] = (
        pd.qcut(
            pd.Series(benefit).rank(method="first"),
            q=4,
            labels=False,
        ).to_numpy()
        + 1
    )
    profile_features = [
        "age",
        "trigger_value",
        "baseline_hb_min",
        "lab_lactate_last",
        "lab_bun_last",
        "urine_output_6h",
        "lab_glucose_last_missing",
        "service_medical",
        "unit_cardiac",
        "unit_medical",
        "pretrigger_invasive_respiratory_support",
        "pretrigger_dialysis",
        "pretrigger_operating_room",
        "pretrigger_mechanical_support",
    ]
    rows = []
    for quartile, group in evaluation.groupby("benefit_quartile"):
        observed = group["dr_benefit"].to_numpy()
        se = float(observed.std(ddof=1) / np.sqrt(len(group)))
        row = {
            "benefit_quartile": int(quartile),
            "n": int(len(group)),
            "predicted_benefit_mean": float(group["predicted_benefit"].mean()),
            "dr_observed_benefit_mean": float(observed.mean()),
            "dr_observed_benefit_se": se,
            "dr_observed_benefit_ci_low": float(observed.mean() - 1.96 * se),
            "dr_observed_benefit_ci_high": float(observed.mean() + 1.96 * se),
            "observed_treatment_rate": float(group["T"].mean()),
            "observed_mortality_90d": float(group["Y"].mean()),
            "followup_hb_observed_rate": float(
                group["followup_hb_observed"].mean()
            ),
        }
        row.update(
            {
                f"{feature}_mean": float(group[feature].mean())
                for feature in profile_features
            }
        )
        rows.append(row)
    profiles = pd.DataFrame(rows)

    budget = float(evaluation["T"].mean())
    k = max(1, int(round(budget * len(evaluation))))
    selected = np.zeros(len(evaluation), dtype=bool)
    selected[np.argsort(-benefit)[:k]] = True
    allocation_rows = []
    for label, mask in (
        ("selected_at_observed_budget", selected),
        ("not_selected", ~selected),
    ):
        group = evaluation.loc[mask]
        row = {
            "allocation_group": label,
            "n": int(mask.sum()),
            "share": float(mask.mean()),
            "predicted_benefit_mean": float(group["predicted_benefit"].mean()),
            "dr_observed_benefit_mean": float(group["dr_benefit"].mean()),
        }
        row.update(
            {
                f"{feature}_mean": float(group[feature].mean())
                for feature in profile_features
            }
        )
        allocation_rows.append(row)
    return profiles, pd.DataFrame(allocation_rows)


def policy_budget_curve(payload: dict) -> pd.DataFrame:
    evaluation = payload["evaluation"]
    outcome = evaluation["Y"].to_numpy(dtype=int)
    treatment = evaluation["T"].to_numpy(dtype=int)
    nuisance = payload["evaluator"]
    risk_none = policy_risk(
        outcome,
        treatment,
        np.zeros(len(evaluation), dtype=int),
        nuisance["e"],
        nuisance["m0"],
        nuisance["m1"],
    )
    risk_all = policy_risk(
        outcome,
        treatment,
        np.ones(len(evaluation), dtype=int),
        nuisance["e"],
        nuisance["m0"],
        nuisance["m1"],
    )
    observed_budget = float(treatment.mean())
    budgets = sorted(
        {
            0.025,
            0.05,
            observed_budget,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
        }
    )
    rows = []
    for budget in budgets:
        random_budget_risk = budget * risk_all + (1 - budget) * risk_none
        k = max(1, int(round(budget * len(evaluation))))
        for interface, tau in payload["tau"].items():
            policy = np.zeros(len(evaluation), dtype=int)
            policy[np.argsort(tau)[:k]] = 1
            learned_risk = policy_risk(
                outcome,
                treatment,
                policy,
                nuisance["e"],
                nuisance["m0"],
                nuisance["m1"],
            )
            rows.append(
                {
                    "Interface": interface,
                    "Budget": budget,
                    "Selected_N": k,
                    "Estimated_Risk": learned_risk,
                    "Random_Budget_Risk": random_budget_risk,
                    "Risk_Reduction": random_budget_risk - learned_risk,
                    "Observed_Budget": np.isclose(budget, observed_budget),
                }
            )
    return pd.DataFrame(rows)


def pairwise_robustness_summary(
    out_dir: Path,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rows = pd.read_csv(out_dir / "mimic_rbc_random_robustness.csv")
    pivot = rows.pivot(
        index="Seed",
        columns="Interface",
        values=["R_Loss", "DR_Qini", "Policy_Risk_Reduction"],
    )
    rng = np.random.default_rng(seed)
    records = []
    for comparator in (
        "AllVariables",
        "ExpertClinical",
        "DoubleSelection",
        "CCIA_COnly",
    ):
        improvements = {
            "R_Loss_Improvement": (
                pivot["R_Loss"][comparator]
                - pivot["R_Loss"]["CCIA_CPlusS"]
            ).to_numpy(),
            "DR_Qini_Improvement": (
                pivot["DR_Qini"]["CCIA_CPlusS"]
                - pivot["DR_Qini"][comparator]
            ).to_numpy(),
            "Policy_Risk_Reduction_Improvement": (
                pivot["Policy_Risk_Reduction"]["CCIA_CPlusS"]
                - pivot["Policy_Risk_Reduction"][comparator]
            ).to_numpy(),
        }
        for metric, values in improvements.items():
            sampled = rng.choice(
                values,
                size=(draws, len(values)),
                replace=True,
            ).mean(axis=1)
            low, high = np.quantile(sampled, [0.025, 0.975])
            records.append(
                {
                    "Comparator": comparator,
                    "Metric": metric,
                    "Repeats": len(values),
                    "Mean": float(values.mean()),
                    "CI_Low": float(low),
                    "CI_High": float(high),
                    "Wins": int(np.sum(values > 0)),
                    "Win_Rate": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(records)


def _fit_continuous_or_constant(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    if len(y_train) == 0:
        raise ValueError("No observed continuous outcomes in training fold")
    if np.std(y_train) < 1e-12:
        return np.full(len(x_eval), float(np.mean(y_train)))
    model = regressor(seed, estimators=220)
    model.fit(x_train, y_train)
    return model.predict(x_eval)


def crossfit_missing_continuous_effect(
    evaluation: pd.DataFrame,
    interface: AdjustmentInterface,
    seed: int,
    folds: int = 5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    treatment = evaluation["T"].to_numpy(dtype=int)
    observed = evaluation["followup_hb_observed"].to_numpy(dtype=int)
    outcome = evaluation["followup_hb_change"].to_numpy(dtype=float)
    c_set = list(interface.c_set)
    outcome_set = interface.outcome_set
    e = np.zeros(len(evaluation), dtype=float)
    r0 = np.zeros(len(evaluation), dtype=float)
    r1 = np.zeros(len(evaluation), dtype=float)
    m0 = np.zeros(len(evaluation), dtype=float)
    m1 = np.zeros(len(evaluation), dtype=float)
    strata = 2 * treatment + observed
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(split.split(evaluation, strata)):
        train = evaluation.iloc[train_idx]
        test = evaluation.iloc[test_idx]
        e[test_idx] = fit_propensity_or_constant(
            train[c_set],
            treatment[train_idx],
            test[c_set],
            seed + fold,
        )
        observation_train = train[outcome_set].copy()
        observation_train["assigned_treatment"] = treatment[train_idx]
        for arm, store in ((0, r0), (1, r1)):
            observation_test = test[outcome_set].copy()
            observation_test["assigned_treatment"] = arm
            store[test_idx] = fit_binary_or_constant(
                observation_train,
                observed[train_idx],
                observation_test,
                seed + 20 + 2 * fold + arm,
            )
        for arm, store in ((0, m0), (1, m1)):
            arm_observed = train_idx[
                (treatment[train_idx] == arm) & (observed[train_idx] == 1)
            ]
            store[test_idx] = _fit_continuous_or_constant(
                evaluation.iloc[arm_observed][outcome_set],
                outcome[arm_observed],
                test[outcome_set],
                seed + 40 + 2 * fold + arm,
            )
    e = np.clip(
        e,
        propensity_clip_for_dimension(len(c_set)),
        1 - propensity_clip_for_dimension(len(c_set)),
    )
    r0 = np.clip(r0, 0.05, 0.995)
    r1 = np.clip(r1, 0.05, 0.995)
    residual_outcome = np.where(observed == 1, outcome, 0.0)
    score = (
        m1
        - m0
        + treatment
        * observed
        * (residual_outcome - m1)
        / (e * r1)
        - (1 - treatment)
        * observed
        * (residual_outcome - m0)
        / ((1 - e) * r0)
    )
    return score, {"e": e, "r0": r0, "r1": r1, "m0": m0, "m1": m1}


def hemoglobin_mechanism_analysis(
    payload: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = payload["evaluation"].copy()
    score, nuisance = crossfit_missing_continuous_effect(
        evaluation,
        payload["ccia"],
        payload["analysis_seed"] + 1200,
    )
    benefit = -payload["tau"]["CCIA_CPlusS"]
    quartile = (
        pd.qcut(
            pd.Series(benefit).rank(method="first"),
            q=4,
            labels=False,
        ).to_numpy()
        + 1
    )
    rows = []
    for label, mask in [
        ("overall", np.ones(len(evaluation), dtype=bool)),
        *[
            (f"benefit_quartile_{group}", quartile == group)
            for group in range(1, 5)
        ],
    ]:
        values = score[mask]
        se = float(values.std(ddof=1) / np.sqrt(mask.sum()))
        rows.append(
            {
                "group": label,
                "n": int(mask.sum()),
                "aipw_hb_change_difference": float(values.mean()),
                "se": se,
                "ci_low": float(values.mean() - 1.96 * se),
                "ci_high": float(values.mean() + 1.96 * se),
            }
        )
    treatment = evaluation["T"].to_numpy(dtype=int)
    diagnostics = pd.DataFrame(
        [
            {
                "treated_followup_observed": float(
                    evaluation.loc[treatment == 1, "followup_hb_observed"].mean()
                ),
                "control_followup_observed": float(
                    evaluation.loc[treatment == 0, "followup_hb_observed"].mean()
                ),
                "observation_probability_p01": float(
                    np.quantile(
                        np.where(treatment == 1, nuisance["r1"], nuisance["r0"]),
                        0.01,
                    )
                ),
                "observation_probability_p99": float(
                    np.quantile(
                        np.where(treatment == 1, nuisance["r1"], nuisance["r0"]),
                        0.99,
                    )
                ),
            }
        ]
    )
    return pd.DataFrame(rows), diagnostics


def late_transfusion_sensitivity(payload: dict) -> pd.DataFrame:
    evaluation = payload["evaluation"]
    keep = ~(
        evaluation["T"].eq(0)
        & evaluation["late_rbc_24h"].astype(bool)
    )
    subset = evaluation.loc[keep].copy()
    index = np.flatnonzero(keep.to_numpy())
    evaluator = crossfit_training_nuisance(
        subset,
        list(payload["ccia"].c_set),
        payload["ccia"].outcome_set,
        payload["analysis_seed"] + 1500,
        folds=5,
    )
    metrics = {
        interface: evaluate_tau(
            subset["Y"].to_numpy(dtype=int),
            subset["T"].to_numpy(dtype=int),
            tau[index],
            evaluator,
        )
        for interface, tau in payload["tau"].items()
    }
    return pd.DataFrame(
        [
            {
                "analysis": "exclude_controls_transfused_6_to_24h",
                "n": int(len(subset)),
                "excluded_late_controls": int((~keep).sum()),
                "R_Loss_Improvement": (
                    metrics["AllVariables"]["R_Loss"]
                    - metrics["CCIA_CPlusS"]["R_Loss"]
                ),
                "DR_Qini_Improvement": (
                    metrics["CCIA_CPlusS"]["DR_Qini"]
                    - metrics["AllVariables"]["DR_Qini"]
                ),
                "Policy_Risk_Reduction_Improvement": (
                    metrics["CCIA_CPlusS"]["Policy_Risk_Reduction"]
                    - metrics["AllVariables"]["Policy_Risk_Reduction"]
                ),
            }
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=DEFAULT_MIMIC_ROOT,
        help="Path containing the MIMIC-IV v3.1 hosp/ and icu/ directories.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-missing", type=float, default=0.70)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = locked_analysis_inputs(
        args.database,
        args.out_dir,
        args.max_missing,
        args.seed,
        args.mimic_root,
    )
    verify_locked_result(payload, args.out_dir)

    samples, bootstrap_summary = paired_patient_bootstrap(
        payload,
        args.bootstrap_draws,
        args.seed + 4000,
    )
    samples.to_csv(
        args.out_dir / "mimic_rbc_temporal_paired_bootstrap_draws.csv",
        index=False,
    )
    bootstrap_summary.to_csv(
        args.out_dir / "mimic_rbc_temporal_paired_bootstrap_summary.csv",
        index=False,
    )

    profiles, allocation = benefit_quartile_profiles(payload)
    profiles.to_csv(
        args.out_dir / "mimic_rbc_temporal_benefit_profiles.csv",
        index=False,
    )
    allocation.to_csv(
        args.out_dir / "mimic_rbc_temporal_allocation_profile.csv",
        index=False,
    )
    policy_curve = policy_budget_curve(payload)
    policy_curve.to_csv(
        args.out_dir / "mimic_rbc_temporal_policy_budget_curve.csv",
        index=False,
    )
    pairwise = pairwise_robustness_summary(
        args.out_dir,
        args.bootstrap_draws,
        args.seed + 4500,
    )
    pairwise.to_csv(
        args.out_dir / "mimic_rbc_random_robustness_pairwise_summary.csv",
        index=False,
    )

    mechanism, observation = hemoglobin_mechanism_analysis(payload)
    mechanism.to_csv(
        args.out_dir / "mimic_rbc_temporal_hb_mechanism.csv",
        index=False,
    )
    observation.to_csv(
        args.out_dir / "mimic_rbc_temporal_hb_observation_diagnostics.csv",
        index=False,
    )

    sensitivity = late_transfusion_sensitivity(payload)
    sensitivity.to_csv(
        args.out_dir / "mimic_rbc_temporal_late_transfusion_sensitivity.csv",
        index=False,
    )
    total_set, total_set_balance, total_set_bootstrap = (
        conservative_total_set_sensitivity(
            payload,
            args.out_dir,
            args.bootstrap_draws,
            args.seed + 5000,
        )
    )
    total_set.to_csv(
        args.out_dir / "mimic_rbc_temporal_c_total_sensitivity.csv",
        index=False,
    )
    total_set_balance.to_csv(
        args.out_dir / "mimic_rbc_temporal_c_total_balance.csv",
        index=False,
    )
    total_set_bootstrap.to_csv(
        args.out_dir / "mimic_rbc_temporal_c_total_bootstrap.csv",
        index=False,
    )
    write_json(
        {
            "locked_result_reproduced": True,
            "bootstrap_draws": args.bootstrap_draws,
            "row_level_data_exported": False,
            "analyses": [
                "paired patient bootstrap",
                "benefit quartile clinical profiles",
                "policy budget curve",
                "pairwise repeated-split comparator summary",
                "missingness-adjusted hemoglobin mechanism endpoint",
                "late-transfusion contamination sensitivity",
                "conservative total-common-cause propensity sensitivity",
            ],
        },
        args.out_dir / "mimic_rbc_interpretation_manifest.json",
    )
    print(bootstrap_summary.to_string(index=False), flush=True)
    print(mechanism.to_string(index=False), flush=True)
    print(sensitivity.to_string(index=False), flush=True)
    print(total_set.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
