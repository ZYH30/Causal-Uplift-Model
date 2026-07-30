"""Robustness audits for the locked MIMIC-IV red-cell target-trial study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from empirical_experiments.mimic_rbc_interpretation import locked_analysis_inputs
from empirical_experiments.mimic_rbc_target_trial import (
    AdjustmentInterface,
    DEFAULT_DATABASE,
    DEFAULT_OUT_DIR,
    EXPERT_CANDIDATE_C,
    EXPERT_CANDIDATE_S,
    aipw_summary,
    crossfit_training_nuisance,
    dr_effect_score,
    evaluate_tau,
    fit_propensity_or_constant,
    fit_tau,
    joint_strata,
    propensity_clip_for_dimension,
    run_comparison,
    summarize_robustness,
    weighted_smd,
    write_json,
)


def crossfit_propensity(
    frame: pd.DataFrame,
    features: list[str],
    seed: int,
    folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    treatment = frame["T"].to_numpy(dtype=int)
    outcome = frame["Y"].to_numpy(dtype=int)
    raw = np.zeros(len(frame), dtype=float)
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(
        split.split(frame, joint_strata(treatment, outcome))
    ):
        raw[test_idx] = fit_propensity_or_constant(
            frame.iloc[train_idx][features],
            treatment[train_idx],
            frame.iloc[test_idx][features],
            seed + fold,
        )
    clip = propensity_clip_for_dimension(len(features))
    return raw, np.clip(raw, clip, 1.0 - clip), clip


def stabilized_weights(treatment: np.ndarray, propensity: np.ndarray) -> np.ndarray:
    prevalence = float(np.mean(treatment))
    weights = np.where(
        treatment == 1,
        prevalence / propensity,
        (1.0 - prevalence) / (1.0 - propensity),
    )
    for level in (0, 1):
        mask = treatment == level
        weights[mask] *= mask.sum() / weights[mask].sum()
    return weights


def propensity_surfaces(
    payload: dict,
    roles: dict,
    safe_c: list[str],
) -> dict[str, list[str]]:
    features = payload["feature_cols"]
    available = set(features)
    c_set = list(payload["ccia"].c_set)
    s_set = list(payload["ccia"].s_set)
    c_total = [feature for feature in roles["c_total"] if feature in available]
    expert_c = [feature for feature in EXPERT_CANDIDATE_C if feature in available]
    return {
        "CCIA_C": c_set,
        "SafetyValidated_C": list(safe_c),
        "CCIA_CPlusS": sorted(set(c_set) | set(s_set)),
        "CCIA_CTotal": c_total,
        "ExpertClinical": expert_c,
        "AllBaseline": list(features),
    }


def treatment_neutrality_safety_gate(
    payload: dict,
    seed: int,
    min_log_loss_gain: float = 0.0005,
    max_weighted_smd: float = 0.10,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    development = payload["train"]
    treatment = development["T"].to_numpy(dtype=int)
    c_set = list(payload["ccia"].c_set)
    s_set = list(payload["ccia"].s_set)
    _, baseline_e, _ = crossfit_propensity(
        development,
        c_set,
        seed,
    )
    baseline_loss = float(log_loss(treatment, baseline_e))
    baseline_auc = float(roc_auc_score(treatment, baseline_e))
    weights = stabilized_weights(treatment, baseline_e)
    rows = []
    promoted = []
    retained = []
    for offset, feature in enumerate(s_set):
        _, augmented_e, _ = crossfit_propensity(
            development,
            sorted(set(c_set) | {feature}),
            seed + 100 * (offset + 1),
        )
        augmented_loss = float(log_loss(treatment, augmented_e))
        augmented_auc = float(roc_auc_score(treatment, augmented_e))
        weighted_association = abs(
            weighted_smd(
                development[feature].to_numpy(dtype=float),
                treatment,
                weights,
            )
        )
        log_loss_gain = baseline_loss - augmented_loss
        promote = (
            log_loss_gain > min_log_loss_gain
            or weighted_association > max_weighted_smd
        )
        (promoted if promote else retained).append(feature)
        rows.append(
            {
                "feature": feature,
                "development_n": len(development),
                "baseline_c_size": len(c_set),
                "baseline_log_loss": baseline_loss,
                "augmented_log_loss": augmented_loss,
                "log_loss_gain": log_loss_gain,
                "baseline_auc": baseline_auc,
                "augmented_auc": augmented_auc,
                "auc_gain": augmented_auc - baseline_auc,
                "abs_weighted_smd_under_c": weighted_association,
                "min_log_loss_gain": min_log_loss_gain,
                "max_weighted_smd": max_weighted_smd,
                "decision": "promote_to_C" if promote else "retain_in_S",
            }
        )
    safe_c = sorted(set(c_set) | set(promoted))
    safe_s = sorted(retained)
    return pd.DataFrame(rows), safe_c, safe_s


def full_surface_balance_audit(
    payload: dict,
    roles: dict,
    safe_c: list[str],
    safe_s: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = payload["evaluation"]
    treatment = evaluation["T"].to_numpy(dtype=int)
    all_features = [
        feature
        for feature in payload["feature_cols"]
        if evaluation[feature].nunique(dropna=False) > 1
    ]
    detail_rows = []
    summary_rows = []
    c_set = list(payload["ccia"].c_set)
    s_set = list(payload["ccia"].s_set)
    c_total = [
        feature for feature in roles["c_total"] if feature in all_features
    ]
    for offset, (surface, propensity_features) in enumerate(
        propensity_surfaces(payload, roles, safe_c).items()
    ):
        raw, propensity, clip = crossfit_propensity(
            evaluation,
            propensity_features,
            seed + 100 * offset,
        )
        weights = stabilized_weights(treatment, propensity)
        surface_rows = []
        for feature in all_features:
            values = evaluation[feature].to_numpy(dtype=float)
            surface_rows.append(
                {
                    "surface": surface,
                    "feature": feature,
                    "in_propensity_surface": feature in propensity_features,
                    "in_original_c": feature in c_set,
                    "in_original_s": feature in s_set,
                    "in_safety_c": feature in safe_c,
                    "in_safety_s": feature in safe_s,
                    "in_c_total": feature in c_total,
                    "smd_pre": weighted_smd(values, treatment),
                    "smd_weighted": weighted_smd(values, treatment, weights),
                }
            )
        detail_rows.extend(surface_rows)
        surface_frame = pd.DataFrame(surface_rows)
        audit_scopes = {
            "AllBaseline": all_features,
            "PropensitySurface": propensity_features,
            "OriginalC": c_set,
            "OriginalS": s_set,
            "SafetyC": safe_c,
            "SafetyS": safe_s,
            "CTotal": c_total,
        }
        for scope, scope_features in audit_scopes.items():
            scoped = surface_frame[
                surface_frame["feature"].isin(scope_features)
            ]
            if scoped.empty:
                continue
            absolute = scoped["smd_weighted"].abs()
            max_row = scoped.loc[absolute.idxmax()]
            summary_rows.append(
                {
                    "surface": surface,
                    "audit_scope": scope,
                    "propensity_feature_count": len(propensity_features),
                    "audited_feature_count": len(scoped),
                    "mean_abs_smd_pre": float(scoped["smd_pre"].abs().mean()),
                    "mean_abs_smd_weighted": float(absolute.mean()),
                    "max_abs_smd_weighted": float(absolute.max()),
                    "max_imbalance_feature": max_row["feature"],
                    "features_abs_smd_gt_0_10": int((absolute > 0.10).sum()),
                    "features_abs_smd_gt_0_20": int((absolute > 0.20).sum()),
                    "ess_ratio": float(
                        weights.sum() ** 2 / np.sum(weights**2) / len(weights)
                    ),
                    "propensity_min": float(propensity.min()),
                    "propensity_p01": float(np.quantile(propensity, 0.01)),
                    "propensity_p05": float(np.quantile(propensity, 0.05)),
                    "propensity_p50": float(np.quantile(propensity, 0.50)),
                    "propensity_p95": float(np.quantile(propensity, 0.95)),
                    "propensity_p99": float(np.quantile(propensity, 0.99)),
                    "propensity_max": float(propensity.max()),
                    "clip_threshold": clip,
                    "raw_below_clip_fraction": float(np.mean(raw < clip)),
                    "raw_above_clip_fraction": float(
                        np.mean(raw > 1.0 - clip)
                    ),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def evaluator_surfaces(
    payload: dict,
    roles: dict,
    safe_c: list[str],
    safe_s: list[str],
) -> dict[str, tuple[list[str], list[str]]]:
    available = set(payload["feature_cols"])
    c_total = [feature for feature in roles["c_total"] if feature in available]
    c_total_outcome = sorted(
        set(c_total)
        | {
            feature
            for feature in roles["s_precision"]
            if feature in available
        }
    )
    expert_c = [
        feature for feature in EXPERT_CANDIDATE_C if feature in available
    ]
    expert_outcome = sorted(
        set(expert_c)
        | {
            feature
            for feature in EXPERT_CANDIDATE_S
            if feature in available
        }
    )
    all_features = list(payload["feature_cols"])
    return {
        "RoleSeparatedPrimary": (
            list(payload["ccia"].c_set),
            payload["ccia"].outcome_set,
        ),
        "SafetyPromotedRole": (
            safe_c,
            sorted(set(safe_c) | set(safe_s)),
        ),
        "CCIA_CTotal": (c_total, c_total_outcome),
        "ExpertClinical": (expert_c, expert_outcome),
        "AllBaseline": (all_features, all_features),
    }


def metric_improvement(
    metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    return {
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


def bootstrap_metric_improvement(
    evaluation: pd.DataFrame,
    tau: dict[str, np.ndarray],
    nuisance: dict[str, np.ndarray],
    draws: int,
    seed: int,
) -> dict[str, float]:
    outcome = evaluation["Y"].to_numpy(dtype=int)
    treatment = evaluation["T"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(draws):
        index = rng.integers(0, len(evaluation), size=len(evaluation))
        draw_nuisance = {name: values[index] for name, values in nuisance.items()}
        metrics = {
            interface: evaluate_tau(
                outcome[index],
                treatment[index],
                values[index],
                draw_nuisance,
            )
            for interface, values in tau.items()
        }
        records.append(metric_improvement(metrics))
    draws_frame = pd.DataFrame(records)
    result = {}
    for metric in draws_frame:
        values = draws_frame[metric].to_numpy(dtype=float)
        result[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
        result[f"{metric}_probability_positive"] = float(np.mean(values > 0))
    return result


def independent_evaluator_audit(
    payload: dict,
    roles: dict,
    safe_c: list[str],
    safe_s: list[str],
    draws: int,
    seed: int,
) -> pd.DataFrame:
    evaluation = payload["evaluation"]
    outcome = evaluation["Y"].to_numpy(dtype=int)
    treatment = evaluation["T"].to_numpy(dtype=int)
    rows = []
    for offset, (name, (c_set, outcome_set)) in enumerate(
        evaluator_surfaces(payload, roles, safe_c, safe_s).items()
    ):
        nuisance = crossfit_training_nuisance(
            evaluation,
            c_set,
            outcome_set,
            seed + 200 * offset,
            folds=5,
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
        improvements = metric_improvement(metrics)
        metric_levels = {}
        for interface, values in metrics.items():
            for metric in (
                "R_Loss",
                "DR_Qini",
                "Policy_Risk_Reduction",
                "Calibration_Correlation",
                "Calibration_Slope",
            ):
                metric_levels[f"{interface}_{metric}"] = values[metric]
        uncertainty = bootstrap_metric_improvement(
            evaluation,
            payload["tau"],
            nuisance,
            draws,
            seed + 200 * offset + 50,
        )
        rows.append(
            {
                "evaluator": name,
                "propensity_feature_count": len(c_set),
                "outcome_feature_count": len(outcome_set),
                **aipw_summary(evaluation, nuisance),
                **metric_levels,
                **improvements,
                **uncertainty,
            }
        )
    return pd.DataFrame(rows)


def _no_glucose_features(features: list[str]) -> list[str]:
    return [
        feature
        for feature in features
        if feature not in {"lab_glucose_last", "lab_glucose_last_missing"}
    ]


def glucose_missingness_audit(
    payload: dict,
    safe_c: list[str],
    safe_s: list[str],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = payload["train"]
    evaluation = payload["evaluation"]
    c_primary = list(safe_c)
    s_primary = list(safe_s)
    c_without_indicator = [
        feature for feature in c_primary if feature != "lab_glucose_last_missing"
    ]
    c_with_value = sorted(set(c_without_indicator) | {"lab_glucose_last"})
    interfaces = {
        "PrimaryMissingIndicator": AdjustmentInterface(
            "PrimaryMissingIndicator",
            tuple(c_primary),
            tuple(s_primary),
        ),
        "DropGlucoseMissingness": AdjustmentInterface(
            "DropGlucoseMissingness",
            tuple(c_without_indicator),
            tuple(s_primary),
        ),
        "ReplaceWithGlucoseValue": AdjustmentInterface(
            "ReplaceWithGlucoseValue",
            tuple(c_with_value),
            tuple(s_primary),
        ),
    }
    tau = {
        "PrimaryMissingIndicator": payload["tau"]["CCIA_CPlusS"],
        "AllVariables": payload["tau"]["AllVariables"],
        "DropGlucoseMissingness": fit_tau(
            payload["learner"],
            train,
            evaluation,
            interfaces["DropGlucoseMissingness"],
            seed + 10,
        ),
        "ReplaceWithGlucoseValue": fit_tau(
            payload["learner"],
            train,
            evaluation,
            interfaces["ReplaceWithGlucoseValue"],
            seed + 20,
        ),
    }
    evaluators = {
        "NoGlucoseRoleEvaluator": (
            c_without_indicator,
            sorted(set(c_without_indicator) | set(s_primary)),
        ),
        "AllBaselineWithoutGlucose": (
            _no_glucose_features(payload["feature_cols"]),
            _no_glucose_features(payload["feature_cols"]),
        ),
    }
    outcome = evaluation["Y"].to_numpy(dtype=int)
    treatment = evaluation["T"].to_numpy(dtype=int)
    rows = []
    bootstrap_rows = []
    for eval_offset, (eval_name, (c_set, outcome_set)) in enumerate(
        evaluators.items()
    ):
        nuisance = crossfit_training_nuisance(
            evaluation,
            c_set,
            outcome_set,
            seed + 100 * eval_offset,
            folds=5,
        )
        point_metrics = {
            name: evaluate_tau(outcome, treatment, values, nuisance)
            for name, values in tau.items()
        }
        all_metrics = point_metrics["AllVariables"]
        for name, metrics in point_metrics.items():
            if name == "AllVariables":
                c_set_label = ";".join(payload["feature_cols"])
                s_set_label = ""
            else:
                c_set_label = ";".join(interfaces[name].c_set)
                s_set_label = ";".join(interfaces[name].s_set)
            rows.append(
                {
                    "evaluator": eval_name,
                    "interface": name,
                    "c_set": c_set_label,
                    "s_set": s_set_label,
                    "R_Loss_Improvement_Vs_All": (
                        all_metrics["R_Loss"] - metrics["R_Loss"]
                    ),
                    "DR_Qini_Improvement_Vs_All": (
                        metrics["DR_Qini"] - all_metrics["DR_Qini"]
                    ),
                    "Policy_Improvement_Vs_All": (
                        metrics["Policy_Risk_Reduction"]
                        - all_metrics["Policy_Risk_Reduction"]
                    ),
                    **metrics,
                }
            )
        rng = np.random.default_rng(seed + 100 * eval_offset + 50)
        for draw in range(draws):
            index = rng.integers(0, len(evaluation), size=len(evaluation))
            draw_nuisance = {
                name: values[index] for name, values in nuisance.items()
            }
            draw_metrics = {
                name: evaluate_tau(
                    outcome[index],
                    treatment[index],
                    values[index],
                    draw_nuisance,
                )
                for name, values in tau.items()
            }
            primary = draw_metrics["PrimaryMissingIndicator"]
            all_variables = draw_metrics["AllVariables"]
            for name in (
                "PrimaryMissingIndicator",
                "DropGlucoseMissingness",
                "ReplaceWithGlucoseValue",
            ):
                bootstrap_rows.append(
                    {
                        "draw": draw,
                        "evaluator": eval_name,
                        "interface": name,
                        "R_Loss_Difference_Vs_Primary": (
                            primary["R_Loss"] - draw_metrics[name]["R_Loss"]
                        ),
                        "DR_Qini_Difference_Vs_Primary": (
                            draw_metrics[name]["DR_Qini"]
                            - primary["DR_Qini"]
                        ),
                        "Policy_Difference_Vs_Primary": (
                            draw_metrics[name]["Policy_Risk_Reduction"]
                            - primary["Policy_Risk_Reduction"]
                        ),
                        "R_Loss_Improvement_Vs_All": (
                            all_variables["R_Loss"]
                            - draw_metrics[name]["R_Loss"]
                        ),
                        "DR_Qini_Improvement_Vs_All": (
                            draw_metrics[name]["DR_Qini"]
                            - all_variables["DR_Qini"]
                        ),
                        "Policy_Improvement_Vs_All": (
                            draw_metrics[name]["Policy_Risk_Reduction"]
                            - all_variables["Policy_Risk_Reduction"]
                        ),
                    }
                )
    bootstrap = pd.DataFrame(bootstrap_rows)
    summary_rows = []
    for (evaluator, interface), group in bootstrap.groupby(
        ["evaluator", "interface"]
    ):
        row = {"evaluator": evaluator, "interface": interface, "draws": draws}
        for metric in (
            "R_Loss_Difference_Vs_Primary",
            "DR_Qini_Difference_Vs_Primary",
            "Policy_Difference_Vs_Primary",
            "R_Loss_Improvement_Vs_All",
            "DR_Qini_Improvement_Vs_All",
            "Policy_Improvement_Vs_All",
        ):
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
            row[f"{metric}_probability_positive"] = float(
                np.mean(values > 0)
            )
        summary_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def multiple_imputation_glucose_audit(
    payload: dict,
    safe_c: list[str],
    safe_s: list[str],
    imputations: int,
    seed: int,
) -> pd.DataFrame:
    continuous = [
        "lab_glucose_last",
        "trigger_value",
        "baseline_hb_min",
        "age",
        "lab_lactate_last",
        "lab_bun_last",
        "lab_creatinine_last",
        "lab_bicarbonate_last",
        "lab_platelets_last",
        "lab_inr_last",
        "urine_output_6h",
    ]
    continuous = [
        feature for feature in continuous if feature in payload["feature_cols"]
    ]
    c_without_indicator = [
        feature
        for feature in safe_c
        if feature != "lab_glucose_last_missing"
    ]
    s_set = list(safe_s)
    rows = []
    for imputation in range(imputations):
        imputer = IterativeImputer(
            max_iter=15,
            sample_posterior=True,
            random_state=seed + imputation,
            skip_complete=True,
        )
        train = payload["train"].copy()
        evaluation = payload["evaluation"].copy()
        train_imputation = train[continuous].copy()
        evaluation_imputation = evaluation[continuous].copy()
        for frame, imputation_frame in (
            (train, train_imputation),
            (evaluation, evaluation_imputation),
        ):
            for feature in continuous:
                missing_indicator = f"{feature}_missing"
                if missing_indicator in frame:
                    imputation_frame.loc[
                        frame[missing_indicator].eq(1),
                        feature,
                    ] = np.nan
            if "urine_output_6h" in continuous:
                imputation_frame.loc[
                    frame["urine_observed_6h"].eq(0),
                    "urine_output_6h",
                ] = np.nan
        imputer.fit(train_imputation)
        train_values = imputer.transform(train_imputation)
        evaluation_values = imputer.transform(evaluation_imputation)
        glucose_index = continuous.index("lab_glucose_last")
        train["lab_glucose_last_mi"] = train_values[:, glucose_index]
        evaluation["lab_glucose_last_mi"] = evaluation_values[:, glucose_index]
        c_set = sorted(set(c_without_indicator) | {"lab_glucose_last_mi"})
        interface = AdjustmentInterface(
            f"MultipleImputation_{imputation + 1}",
            tuple(c_set),
            tuple(s_set),
        )
        tau = fit_tau(
            payload["learner"],
            train,
            evaluation,
            interface,
            seed + 100 + imputation,
        )
        nuisance = crossfit_training_nuisance(
            evaluation,
            c_set,
            interface.outcome_set,
            seed + 200 + imputation,
            folds=5,
        )
        metrics = evaluate_tau(
            evaluation["Y"].to_numpy(dtype=int),
            evaluation["T"].to_numpy(dtype=int),
            tau,
            nuisance,
        )
        rows.append(
            {
                "imputation": imputation + 1,
                "c_set": ";".join(c_set),
                "glucose_imputed_train": int(
                    payload["train"]["lab_glucose_last_missing"].sum()
                ),
                "glucose_imputed_evaluation": int(
                    payload["evaluation"]["lab_glucose_last_missing"].sum()
                ),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def subgroup_definitions(evaluation: pd.DataFrame) -> dict[str, np.ndarray]:
    no_major_intervention = (
        evaluation["prior_vasopressor"].eq(0)
        & evaluation["pretrigger_operating_room"].eq(0)
        & evaluation["pretrigger_mechanical_support"].eq(0)
    ).to_numpy()
    no_advanced_support = (
        no_major_intervention
        & evaluation["pretrigger_invasive_respiratory_support"].eq(0).to_numpy()
        & evaluation["pretrigger_dialysis"].eq(0).to_numpy()
    )
    return {
        "FullTemporalTest": np.ones(len(evaluation), dtype=bool),
        "NoVasopressorORMechanicalSupport": no_major_intervention,
        "NoAdvancedSupport": no_advanced_support,
    }


def stable_proxy_subgroup_audit(
    payload: dict,
    safe_c: list[str],
    safe_s: list[str],
    draws: int,
    seed: int,
) -> pd.DataFrame:
    evaluation = payload["evaluation"]
    rows = []
    for offset, (name, mask) in enumerate(
        subgroup_definitions(evaluation).items()
    ):
        subset = evaluation.loc[mask].copy()
        if subset["T"].sum() < 25 or (1 - subset["T"]).sum() < 25:
            rows.append(
                {
                    "subgroup": name,
                    "n": len(subset),
                    "treated": int(subset["T"].sum()),
                    "status": "insufficient_treatment_arm_size",
                }
            )
            continue
        index = np.flatnonzero(mask)
        nuisance = crossfit_training_nuisance(
            subset,
            list(safe_c),
            sorted(set(safe_c) | set(safe_s)),
            seed + 100 * offset,
            folds=5,
        )
        subset_tau = {
            interface: values[index]
            for interface, values in payload["tau"].items()
        }
        metrics = {
            interface: evaluate_tau(
                subset["Y"].to_numpy(dtype=int),
                subset["T"].to_numpy(dtype=int),
                tau,
                nuisance,
            )
            for interface, tau in subset_tau.items()
        }
        point = metric_improvement(metrics)
        uncertainty = bootstrap_metric_improvement(
            subset,
            subset_tau,
            nuisance,
            draws,
            seed + 100 * offset + 50,
        )
        rows.append(
            {
                "subgroup": name,
                "n": len(subset),
                "treated": int(subset["T"].sum()),
                "treatment_rate": float(subset["T"].mean()),
                "mortality_rate": float(subset["Y"].mean()),
                "status": "complete",
                **point,
                **uncertainty,
            }
        )
    return pd.DataFrame(rows)


def safety_validated_roles(
    roles: dict,
    safe_c: list[str],
    safe_s: list[str],
) -> dict:
    validated = dict(roles)
    validated["c_proximal"] = list(safe_c)
    validated["s_precision"] = list(safe_s)
    validated["role_validation"] = {
        "criterion": (
            "development-only cross-fitted treatment log-loss gain above "
            "0.0005 or absolute C-weighted SMD above 0.10"
        ),
        "promoted_from_s_to_c": sorted(
            set(safe_c) - set(roles["c_proximal"])
        ),
        "retained_in_s": list(safe_s),
    }
    return validated


def repeated_split_safety_audit(
    payload: dict,
    roles: dict,
    safe_c: list[str],
    safe_s: list[str],
    repeats: int,
    test_size: float,
    bootstrap_draws: int,
    seed_start: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validated_roles = safety_validated_roles(roles, safe_c, safe_s)
    post_discovery = payload["frame"][
        payload["frame"]["era_index"].ge(3)
    ].copy()
    rows = []
    for offset in range(repeats):
        seed = seed_start + offset
        train, evaluation = train_test_split(
            post_discovery,
            test_size=test_size,
            random_state=seed,
            stratify=joint_strata(
                post_discovery["T"].to_numpy(dtype=int),
                post_discovery["Y"].to_numpy(dtype=int),
            ),
        )
        result, _, _ = run_comparison(
            train,
            evaluation,
            payload["feature_cols"],
            validated_roles,
            seed,
            "post_discovery_safety_validated",
            learners=[payload["learner"]],
        )
        rows.append(result)
        print(
            f"Safety-validated split {offset + 1}/{repeats} complete",
            flush=True,
        )
    combined = pd.concat(rows, ignore_index=True)
    summary = summarize_robustness(combined, bootstrap_draws)
    return combined, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory containing the immutable graph and locked learner.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "mimic_rbc_target_trial_v4_audit",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-missing", type=float, default=0.70)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--multiple-imputations", type=int, default=5)
    parser.add_argument("--robustness-repeats", type=int, default=20)
    parser.add_argument("--robustness-test-size", type=float, default=0.40)
    parser.add_argument("--robustness-seed-start", type=int, default=3100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = locked_analysis_inputs(
        args.database,
        args.source_results_dir,
        args.max_missing,
        args.seed,
    )
    roles = json.loads(
        (args.source_results_dir / "mimic_rbc_ccia_roles.json").read_text(
            encoding="utf-8"
        )
    )
    discovery_payload = dict(payload)
    discovery_payload["train"] = payload["train"][
        payload["train"]["era_index"].le(2)
    ].copy()
    safety_gate, safe_c, safe_s = treatment_neutrality_safety_gate(
        discovery_payload,
        args.seed + 5500,
    )
    safety_gate.to_csv(
        args.out_dir / "mimic_rbc_treatment_neutrality_safety_gate.csv",
        index=False,
    )

    balance_summary, balance_detail = full_surface_balance_audit(
        payload,
        roles,
        safe_c,
        safe_s,
        args.seed + 6000,
    )
    balance_summary.to_csv(
        args.out_dir / "mimic_rbc_full_surface_balance_summary.csv",
        index=False,
    )
    balance_detail.to_csv(
        args.out_dir / "mimic_rbc_full_surface_balance_detail.csv",
        index=False,
    )

    evaluator = independent_evaluator_audit(
        payload,
        roles,
        safe_c,
        safe_s,
        args.bootstrap_draws,
        args.seed + 7000,
    )
    evaluator.to_csv(
        args.out_dir / "mimic_rbc_independent_evaluator_audit.csv",
        index=False,
    )

    glucose, glucose_uncertainty = glucose_missingness_audit(
        payload,
        safe_c,
        safe_s,
        args.bootstrap_draws,
        args.seed + 8000,
    )
    glucose.to_csv(
        args.out_dir / "mimic_rbc_glucose_missingness_audit.csv",
        index=False,
    )
    glucose_uncertainty.to_csv(
        args.out_dir / "mimic_rbc_glucose_missingness_bootstrap_summary.csv",
        index=False,
    )

    multiple_imputation = multiple_imputation_glucose_audit(
        payload,
        safe_c,
        safe_s,
        args.multiple_imputations,
        args.seed + 9000,
    )
    multiple_imputation.to_csv(
        args.out_dir / "mimic_rbc_glucose_multiple_imputation.csv",
        index=False,
    )

    subgroups = stable_proxy_subgroup_audit(
        payload,
        safe_c,
        safe_s,
        args.bootstrap_draws,
        args.seed + 10000,
    )
    subgroups.to_csv(
        args.out_dir / "mimic_rbc_stable_proxy_subgroups.csv",
        index=False,
    )

    repeated_rows, repeated_summary = repeated_split_safety_audit(
        payload,
        roles,
        safe_c,
        safe_s,
        args.robustness_repeats,
        args.robustness_test_size,
        args.bootstrap_draws,
        args.robustness_seed_start,
    )
    repeated_rows.to_csv(
        args.out_dir / "mimic_rbc_safety_validated_repeated_splits.csv",
        index=False,
    )
    repeated_summary.to_csv(
        args.out_dir / "mimic_rbc_safety_validated_repeated_summary.csv",
        index=False,
    )
    write_json(
        safety_validated_roles(roles, safe_c, safe_s),
        args.out_dir / "mimic_rbc_safety_validated_roles.json",
    )

    write_json(
        {
            "analysis": "development-defined robustness audit",
            "role_validation_period": "2008-2016 graph-discovery period",
            "bootstrap_draws": args.bootstrap_draws,
            "multiple_imputations": args.multiple_imputations,
            "source_results_dir": args.source_results_dir.name,
            "safety_promoted_c_set": safe_c,
            "safety_retained_s_set": safe_s,
            "row_level_data_exported": False,
            "chartevents_used": False,
            "outputs": [
                "full_surface_balance_summary",
                "full_surface_balance_detail",
                "treatment_neutrality_safety_gate",
                "independent_evaluator_audit",
                "glucose_missingness_audit",
                "glucose_multiple_imputation",
                "stable_proxy_subgroups",
                "safety_validated_repeated_splits",
            ],
        },
        args.out_dir / "mimic_rbc_robustness_audit_manifest.json",
    )
    print("\nFull-surface balance", flush=True)
    print(balance_summary.to_string(index=False), flush=True)
    print("\nIndependent evaluators", flush=True)
    print(evaluator.to_string(index=False), flush=True)
    print("\nGlucose missingness", flush=True)
    print(glucose.to_string(index=False), flush=True)
    print("\nStable proxy subgroups", flush=True)
    print(subgroups.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
