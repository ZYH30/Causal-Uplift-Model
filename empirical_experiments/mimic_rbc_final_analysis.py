"""Locked common-support analysis for the chart-augmented MIMIC RBC study.

Propensity and learner choices use 2017--2019 validation data before the
2020--2022 temporal cohort is evaluated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import empirical_experiments.mimic_rbc_interpretation as interpretation
import empirical_experiments.mimic_rbc_target_trial as target
from empirical_experiments.mimic_rbc_chart_panel import (
    DEFAULT_OUT_DIR,
    PROFILE_CHART_FEATURES,
    build_or_load_frame,
    panel_features,
    read_json,
)
from empirical_experiments.mimic_rbc_propensity_selection import (
    STRATEGIES,
    diagnostics,
    model_for_strategy,
)


OVERLAP_DIRNAME = "overlap_restricted"
SUPPORT_VARIABLE = "hx_malignancy"
PRIMARY_INTERFACES = ("AllVariables", "RITE_CPlusS")
METRIC_DIRECTIONS = {
    "R_Loss": -1.0,
    "DR_Qini": 1.0,
    "Policy_Risk_Reduction": 1.0,
}


def write_json(payload: Any, path: Path) -> None:
    target.write_json(payload, path)


def treatment_only_crossfit_propensity(
    frame: pd.DataFrame,
    features: list[str],
    strategy: str,
    seed: int,
    folds: int = 5,
) -> np.ndarray:
    treatment = frame["T"].to_numpy(dtype=int)
    propensity = np.zeros(len(frame), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(frame, treatment)
    ):
        model = model_for_strategy(strategy, seed + fold)
        model.fit(frame.iloc[train_index][features], treatment[train_index])
        probabilities = model.predict_proba(frame.iloc[test_index][features])
        classes = list(model.classes_)
        propensity[test_index] = probabilities[:, classes.index(1)]
    return propensity


def configure_propensity(strategy: str, clip: float) -> None:
    """Install the validation-locked propensity rule for this process."""

    def fit_locked_propensity(
        x_train: pd.DataFrame,
        treatment_train: np.ndarray,
        x_eval: pd.DataFrame,
        seed: int,
    ) -> np.ndarray:
        classes = np.unique(treatment_train)
        if len(classes) < 2:
            return np.full(len(x_eval), float(classes[0]))
        model = model_for_strategy(strategy, seed)
        model.fit(x_train, treatment_train)
        probabilities = model.predict_proba(x_eval)
        model_classes = list(model.classes_)
        return probabilities[:, model_classes.index(1)]

    def locked_clip(_: int) -> float:
        return clip

    target.fit_propensity_or_constant = fit_locked_propensity
    target.propensity_clip_for_dimension = locked_clip
    # The continuous-outcome mechanism module imports these names directly.
    interpretation.fit_propensity_or_constant = fit_locked_propensity
    interpretation.propensity_clip_for_dimension = locked_clip


def filter_to_common_support(frame: pd.DataFrame) -> pd.DataFrame:
    values = pd.to_numeric(frame[SUPPORT_VARIABLE], errors="coerce").fillna(0)
    return frame.loc[values.eq(0)].copy()


def support_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    era_labels = {
        "discovery_2008_2016": frame["era_index"].le(2),
        "validation_2017_2019": frame["era_index"].eq(3),
        "temporal_2020_2022": frame["era_index"].eq(4),
        "overall": pd.Series(True, index=frame.index),
    }
    support = pd.to_numeric(frame[SUPPORT_VARIABLE], errors="coerce").fillna(0).eq(0)
    for era, era_mask in era_labels.items():
        for population, mask in (
            ("original", era_mask),
            ("common_support", era_mask & support),
        ):
            subset = frame.loc[mask]
            rows.append(
                {
                    "era": era,
                    "population": population,
                    "n": int(len(subset)),
                    "treated": int(subset["T"].sum()),
                    "controls": int((1 - subset["T"]).sum()),
                    "outcomes": int(subset["Y"].sum()),
                    "treatment_rate": float(subset["T"].mean()),
                    "outcome_rate": float(subset["Y"].mean()),
                }
            )
    return pd.DataFrame(rows)


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    frame, base_features = build_or_load_frame(args)
    panel_lock = read_json(args.out_dir / "selected_chartevents_panel.json")
    panel = str(panel_lock["selected_panel"])
    chart_features, _ = panel_features(base_features, panel)
    panel_dir = args.out_dir / "panels" / panel
    roles = read_json(panel_dir / "mimic_rbc_safety_validated_roles.json")
    overlap = filter_to_common_support(frame)

    chart_features = [
        feature for feature in chart_features if feature != SUPPORT_VARIABLE
    ]
    base_features_overlap = [
        feature for feature in base_features if feature != SUPPORT_VARIABLE
    ]
    roles_overlap = {
        key: [value for value in values if value != SUPPORT_VARIABLE]
        if isinstance(values, list)
        else values
        for key, values in roles.items()
    }
    c_set = [
        feature
        for feature in roles_overlap["c_proximal"]
        if feature in chart_features
    ]
    s_set = [
        feature
        for feature in roles_overlap["s_precision"]
        if feature in chart_features and feature not in c_set
    ]
    if not c_set or not s_set:
        raise ValueError("The overlap-restricted C or S set is empty")

    out_dir = args.out_dir / OVERLAP_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = support_audit(frame)
    audit.to_csv(out_dir / "common_support_cohort_audit.csv", index=False)
    write_json(
        {
            "rule": f"exclude records with {SUPPORT_VARIABLE} = 1",
            "rationale": (
                "No treated patient in the 2020--2022 era had a prior "
                "malignancy diagnosis, so effects in that stratum are not "
                "identified from the observed treatment support."
            ),
            "rule_inputs": "treatment and baseline covariates only; no outcomes",
            "selected_panel": panel,
            "model_feature_count_after_support": len(chart_features),
            "chart_feature_count": sum(
                feature.startswith("chart_") for feature in chart_features
            ),
            "structured_feature_count": sum(
                not feature.startswith("chart_") for feature in chart_features
            ),
            "c_set": c_set,
            "s_set": s_set,
        },
        out_dir / "common_support_protocol.json",
    )
    return {
        "frame": overlap,
        "original_frame": frame,
        "panel": panel,
        "panel_dir": panel_dir,
        "out_dir": out_dir,
        "chart_features": chart_features,
        "base_features": base_features_overlap,
        "roles": roles_overlap,
        "c_set": c_set,
        "s_set": s_set,
    }


def select_propensity(inputs: dict[str, Any], args: argparse.Namespace) -> dict:
    frame = inputs["frame"]
    features = inputs["chart_features"]
    c_set = inputs["c_set"]
    out_dir = inputs["out_dir"]
    discovery_raw = frame[frame["era_index"].le(2)].copy()
    validation_raw = frame[frame["era_index"].eq(3)].copy()
    _, validation, _ = target.impute_from_training(
        discovery_raw, validation_raw, features
    )

    rows = []
    partial_path = out_dir / "propensity_validation_repeats.partial.csv"
    for repeat in range(args.propensity_repeats):
        seed = args.seed + 20_000 + repeat * 101
        for strategy in STRATEGIES:
            raw = treatment_only_crossfit_propensity(
                validation, c_set, strategy, seed
            )
            for clip in args.propensity_clips:
                row = diagnostics(
                    validation,
                    c_set,
                    raw,
                    strategy,
                    clip,
                    "validation_2017_2019_common_support",
                )
                row.update({"repeat": repeat, "seed": seed})
                rows.append(row)
            pd.DataFrame(rows).to_csv(partial_path, index=False)
            print(
                f"Propensity repeat {repeat + 1}/{args.propensity_repeats}: "
                f"{strategy} complete",
                flush=True,
            )

    repeats = pd.DataFrame(rows)
    repeats.to_csv(out_dir / "propensity_validation_repeats.csv", index=False)
    summary = (
        repeats.groupby(["strategy", "clip"], as_index=False)
        .agg(
            repeats=("repeat", "nunique"),
            features_above_0_10_max=("features_above_0.10", "max"),
            features_above_0_10_mean=("features_above_0.10", "mean"),
            max_abs_smd_worst=("max_abs_smd", "max"),
            max_abs_smd_mean=("max_abs_smd", "mean"),
            mean_abs_smd_mean=("mean_abs_smd", "mean"),
            mean_abs_smd_sd=("mean_abs_smd", "std"),
            ess_ratio_min=("ess_ratio", "min"),
            ess_ratio_mean=("ess_ratio", "mean"),
            log_loss_mean=("log_loss", "mean"),
            auc_mean=("auc", "mean"),
        )
    )
    summary.to_csv(out_dir / "propensity_validation_summary.csv", index=False)
    eligible = summary[
        summary["ess_ratio_min"].ge(0.50)
        & summary["features_above_0_10_max"].eq(0)
        & summary["max_abs_smd_worst"].le(0.10)
    ].copy()
    if eligible.empty:
        eligible = summary[summary["ess_ratio_min"].ge(0.50)].copy()
    if eligible.empty:
        eligible = summary.copy()
    selected = eligible.sort_values(
        [
            "mean_abs_smd_mean",
            "max_abs_smd_mean",
            "ess_ratio_min",
            "log_loss_mean",
        ],
        ascending=[True, True, False, True],
    ).iloc[0]
    lock = {
        "selection_dataset": "MIMIC-IV 2017-2019 common-support cohort",
        "selection_uses_outcomes": False,
        "selection_rule": (
            "require every repeat to retain ESS/n >= 0.50 and every C-set "
            "variable to have absolute SMD below 0.10; among eligible models, "
            "minimize mean absolute SMD, then mean maximum SMD, and retain "
            "greater minimum ESS"
        ),
        "strategy": str(selected["strategy"]),
        "clip": float(selected["clip"]),
        "c_set": c_set,
        "validation_diagnostics": selected.to_dict(),
        "repeats": args.propensity_repeats,
    }
    write_json(lock, out_dir / "propensity_lock.json")
    return lock


def make_primary_interfaces(inputs: dict[str, Any]) -> list[target.AdjustmentInterface]:
    return [
        target.AdjustmentInterface(
            "AllVariables", tuple(inputs["chart_features"])
        ),
        target.AdjustmentInterface(
            "RITE_CPlusS", tuple(inputs["c_set"]), tuple(inputs["s_set"])
        ),
    ]


def evaluate_interfaces(
    train_raw: pd.DataFrame,
    evaluation_raw: pd.DataFrame,
    imputation_features: list[str],
    evaluator_interface: target.AdjustmentInterface,
    interfaces: list[target.AdjustmentInterface],
    learners: list[str],
    seed: int,
    split_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, dict[str, np.ndarray]]]:
    train, evaluation, medians = target.impute_from_training(
        train_raw, evaluation_raw, imputation_features
    )
    evaluator = target.crossfit_training_nuisance(
        evaluation,
        list(evaluator_interface.c_set),
        evaluator_interface.outcome_set,
        seed + 700,
        folds=5,
    )
    diagnostics_row, balance = target.overlap_balance_diagnostics(
        evaluation, list(evaluator_interface.c_set), evaluator["e"]
    )
    diagnostics_row.update(target.aipw_summary(evaluation, evaluator))
    rows = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for learner in learners:
        predictions[learner] = {}
        for interface in interfaces:
            started = time.time()
            tau = target.fit_tau(learner, train, evaluation, interface, seed)
            predictions[learner][interface.name] = tau
            metrics = target.evaluate_tau(
                evaluation["Y"].to_numpy(dtype=int),
                evaluation["T"].to_numpy(dtype=int),
                tau,
                evaluator,
            )
            rows.append(
                {
                    "Split": split_name,
                    "Seed": seed,
                    "Learner": learner,
                    "Interface": interface.name,
                    "Train_N": int(len(train)),
                    "Evaluation_N": int(len(evaluation)),
                    "Evaluation_Treated": int(evaluation["T"].sum()),
                    "C_Size": len(interface.c_set),
                    "S_Size": len(interface.s_set),
                    "C_Set": ";".join(interface.c_set),
                    "S_Set": ";".join(interface.s_set),
                    "Runtime_Sec": time.time() - started,
                    **metrics,
                }
            )
    payload = {
        "train": train,
        "evaluation": evaluation,
        "evaluator": evaluator,
        "imputation_medians": medians,
    }
    return pd.DataFrame(rows), balance, diagnostics_row, predictions | {
        "_payload": payload
    }


def repeated_validation_selection(
    inputs: dict[str, Any], args: argparse.Namespace, propensity_lock: dict
) -> dict:
    configure_propensity(propensity_lock["strategy"], propensity_lock["clip"])
    frame = inputs["frame"]
    discovery = frame[frame["era_index"].le(2)].copy()
    validation = frame[frame["era_index"].eq(3)].copy()
    interfaces = make_primary_interfaces(inputs)
    tlcd = interfaces[1]
    rows = []
    partial_path = inputs["out_dir"] / "learner_validation_repeats.partial.csv"
    strata = target.joint_strata(
        validation["T"].to_numpy(dtype=int),
        validation["Y"].to_numpy(dtype=int),
    )
    for repeat in range(args.learner_repeats):
        seed = args.seed + 30_000 + repeat
        val_train, val_eval = train_test_split(
            validation,
            test_size=args.validation_test_size,
            random_state=seed,
            stratify=strata,
        )
        train_raw = pd.concat([discovery, val_train], axis=0)
        result, _, _, _ = evaluate_interfaces(
            train_raw,
            val_eval,
            inputs["chart_features"],
            tlcd,
            interfaces,
            target.LEARNERS,
            seed,
            f"validation_repeat_{repeat:02d}",
        )
        result["Repeat"] = repeat
        rows.append(result)
        pd.concat(rows, ignore_index=True).to_csv(partial_path, index=False)
        print(
            f"Learner validation repeat {repeat + 1}/{args.learner_repeats} complete",
            flush=True,
        )
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows.to_csv(inputs["out_dir"] / "learner_validation_repeats.csv", index=False)
    summary = target.summarize_robustness(all_rows, args.bootstrap_draws)
    summary.to_csv(inputs["out_dir"] / "learner_validation_summary.csv", index=False)

    candidates = []
    metric_order = [
        "R_Loss_Improvement",
        "DR_Qini_Improvement",
        "Policy_Risk_Reduction_Improvement",
    ]
    rank_frame = summary.pivot(index="Learner", columns="Metric", values="Mean")
    rank_sum = sum(rank_frame[metric].rank(ascending=False) for metric in metric_order)
    for learner, group in summary.groupby("Learner"):
        indexed = group.set_index("Metric")
        candidates.append(
            {
                "Learner": learner,
                "ci_positive_count": int((group["CI_Low"] > 0).sum()),
                "positive_mean_count": int((group["Mean"] > 0).sum()),
                "mean_win_rate": float(group["Win_Rate"].mean()),
                "rank_sum": float(rank_sum.loc[learner]),
                **{
                    metric: float(indexed.loc[metric, "Mean"])
                    for metric in metric_order
                },
            }
        )
    candidates_frame = pd.DataFrame(candidates)
    selected = candidates_frame.sort_values(
        [
            "ci_positive_count",
            "positive_mean_count",
            "mean_win_rate",
            "rank_sum",
            "DR_Qini_Improvement",
        ],
        ascending=[False, False, False, True, False],
    ).iloc[0]
    candidates_frame.to_csv(
        inputs["out_dir"] / "learner_validation_candidates.csv", index=False
    )
    lock = {
        "selection_dataset": (
            "repeated held-out splits of MIMIC-IV 2017-2019 common-support "
            "cohort; 2008-2016 records were available only for model fitting"
        ),
        "selection_rule": (
            "maximize metrics with a positive repeated-split confidence bound, "
            "then positive mean gains, mean win rate, and equal-weight rank sum"
        ),
        "primary_learner": str(selected["Learner"]),
        "primary_interface": "RITE_CPlusS",
        "primary_comparator": "AllVariables",
        "validation_repeats": args.learner_repeats,
        "validation_test_size": args.validation_test_size,
        "selection_summary": selected.to_dict(),
        "all_candidates": candidates_frame.to_dict("records"),
        "temporal_test_used_for_selection": False,
        "propensity_lock": propensity_lock,
    }
    write_json(lock, inputs["out_dir"] / "learner_lock.json")
    return lock


def load_no_chart_roles(inputs: dict[str, Any], args: argparse.Namespace) -> dict:
    role_path = args.no_chart_roles
    if not role_path.exists():
        raise FileNotFoundError(f"Missing no-chart TLCD role artifact: {role_path}")
    roles = read_json(role_path)
    base = set(inputs["base_features"])
    c_set = [feature for feature in roles["c_proximal"] if feature in base]
    s_set = [
        feature
        for feature in roles["s_precision"]
        if feature in base and feature not in c_set
    ]
    return {"c_set": c_set, "s_set": s_set}


def paired_bootstrap(
    evaluation: pd.DataFrame,
    evaluator: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    pairs: list[tuple[str, str, str]],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcome = evaluation["Y"].to_numpy(dtype=int)
    treatment = evaluation["T"].to_numpy(dtype=int)
    full = {
        name: target.evaluate_tau(outcome, treatment, tau, evaluator)
        for name, tau in predictions.items()
    }
    rng = np.random.default_rng(seed)
    samples = []
    for draw in range(draws):
        index = rng.integers(0, len(evaluation), size=len(evaluation))
        nuisance = {key: value[index] for key, value in evaluator.items()}
        metrics = {
            name: target.evaluate_tau(
                outcome[index], treatment[index], tau[index], nuisance
            )
            for name, tau in predictions.items()
        }
        for label, intervention, comparator in pairs:
            samples.extend(
                [
                    {
                        "draw": draw,
                        "comparison": label,
                        "Metric": "R_Loss_Improvement",
                        "value": metrics[comparator]["R_Loss"]
                        - metrics[intervention]["R_Loss"],
                    },
                    {
                        "draw": draw,
                        "comparison": label,
                        "Metric": "DR_Qini_Improvement",
                        "value": metrics[intervention]["DR_Qini"]
                        - metrics[comparator]["DR_Qini"],
                    },
                    {
                        "draw": draw,
                        "comparison": label,
                        "Metric": "Policy_Risk_Reduction_Improvement",
                        "value": metrics[intervention]["Policy_Risk_Reduction"]
                        - metrics[comparator]["Policy_Risk_Reduction"],
                    },
                ]
            )
    sample_frame = pd.DataFrame(samples)
    rows = []
    for label, intervention, comparator in pairs:
        point = {
            "R_Loss_Improvement": full[comparator]["R_Loss"]
            - full[intervention]["R_Loss"],
            "DR_Qini_Improvement": full[intervention]["DR_Qini"]
            - full[comparator]["DR_Qini"],
            "Policy_Risk_Reduction_Improvement": full[intervention][
                "Policy_Risk_Reduction"
            ]
            - full[comparator]["Policy_Risk_Reduction"],
        }
        for metric, estimate in point.items():
            values = sample_frame.loc[
                sample_frame["comparison"].eq(label)
                & sample_frame["Metric"].eq(metric),
                "value",
            ].to_numpy(dtype=float)
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "comparison": label,
                    "intervention": intervention,
                    "comparator": comparator,
                    "Metric": metric,
                    "Point_Estimate": float(estimate),
                    "Bootstrap_Mean": float(values.mean()),
                    "CI_Low": float(low),
                    "CI_High": float(high),
                    "Probability_Positive": float(np.mean(values > 0)),
                    "Draws": draws,
                }
            )
    return sample_frame, pd.DataFrame(rows)


def clinical_chart_profiles(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluation = payload["evaluation"].copy()
    benefit = -payload["tau"]["RITE_CPlusS"]
    evaluation["benefit_quartile"] = (
        pd.qcut(pd.Series(benefit).rank(method="first"), 4, labels=False).to_numpy()
        + 1
    )
    features = [feature for feature in PROFILE_CHART_FEATURES if feature in evaluation]
    profile = evaluation.groupby("benefit_quartile", as_index=False).agg(
        n=("Y", "size"),
        **{f"{feature}_mean": (feature, "mean") for feature in features},
    )
    rows = []
    quartiles = profile["benefit_quartile"].to_numpy(dtype=float)
    for feature in features:
        values = profile[f"{feature}_mean"].to_numpy(dtype=float)
        scale = float(evaluation[feature].std())
        rows.append(
            {
                "feature": feature,
                "q1_mean": float(values[0]),
                "q4_mean": float(values[-1]),
                "q4_minus_q1": float(values[-1] - values[0]),
                "q4_minus_q1_sd": float((values[-1] - values[0]) / scale)
                if scale > 1e-12
                else 0.0,
                "spearman_r": float(pd.Series(values).corr(pd.Series(quartiles), method="spearman")),
                "adjacent_monotonicity_violations": int(
                    min(np.sum(np.diff(values) < 0), np.sum(np.diff(values) > 0))
                ),
            }
        )
    gradients = pd.DataFrame(rows)
    gradients["abs_q4_minus_q1_sd"] = gradients["q4_minus_q1_sd"].abs()
    return profile, gradients.sort_values("abs_q4_minus_q1_sd", ascending=False)


def temporal_and_interpretation(
    inputs: dict[str, Any],
    args: argparse.Namespace,
    propensity_lock: dict,
    learner_lock: dict,
) -> dict:
    configure_propensity(propensity_lock["strategy"], propensity_lock["clip"])
    frame = inputs["frame"]
    train_raw = frame[frame["era_index"].le(3)].copy()
    evaluation_raw = frame[frame["era_index"].eq(4)].copy()
    primary = make_primary_interfaces(inputs)
    no_chart_roles = load_no_chart_roles(inputs, args)
    no_chart = target.AdjustmentInterface(
        "NoChart_TLCD",
        tuple(no_chart_roles["c_set"]),
        tuple(no_chart_roles["s_set"]),
    )
    no_chart_all = target.AdjustmentInterface(
        "NoChart_AllVariables", tuple(inputs["base_features"])
    )
    interfaces = primary + [no_chart, no_chart_all]
    imputation_features = list(
        dict.fromkeys(inputs["chart_features"] + inputs["base_features"])
    )
    learner = learner_lock["primary_learner"]
    rows, balance, diagnostics_row, prediction_bundle = evaluate_interfaces(
        train_raw,
        evaluation_raw,
        imputation_features,
        primary[1],
        interfaces,
        [learner],
        args.seed + 1,
        "temporal_2020_2022_common_support",
    )
    payload_data = prediction_bundle.pop("_payload")
    predictions = prediction_bundle[learner]
    rows.to_csv(inputs["out_dir"] / "temporal_metrics.csv", index=False)
    balance.to_csv(inputs["out_dir"] / "temporal_balance.csv", index=False)
    write_json(diagnostics_row, inputs["out_dir"] / "temporal_diagnostics.json")

    pairs = [
        ("TLCD_chart_vs_all_chart", "RITE_CPlusS", "AllVariables"),
        ("TLCD_chart_vs_TLCD_no_chart", "RITE_CPlusS", "NoChart_TLCD"),
        (
            "TLCD_chart_vs_all_no_chart",
            "RITE_CPlusS",
            "NoChart_AllVariables",
        ),
    ]
    bootstrap_samples, bootstrap_summary = paired_bootstrap(
        payload_data["evaluation"],
        payload_data["evaluator"],
        predictions,
        pairs,
        args.bootstrap_draws,
        args.seed + 40_000,
    )
    bootstrap_samples.to_csv(
        inputs["out_dir"] / "temporal_paired_bootstrap_draws.csv", index=False
    )
    bootstrap_summary.to_csv(
        inputs["out_dir"] / "temporal_paired_bootstrap_summary.csv", index=False
    )

    payload = {
        "frame": frame,
        "train": payload_data["train"],
        "evaluation": payload_data["evaluation"],
        "feature_cols": inputs["chart_features"],
        "tlcd": primary[1],
        "all_variables": primary[0],
        "learner": learner,
        "analysis_seed": args.seed + 1,
        "evaluator": payload_data["evaluator"],
        "tau": {
            "AllVariables": predictions["AllVariables"],
            "RITE_CPlusS": predictions["RITE_CPlusS"],
        },
    }
    benefit, allocation = interpretation.benefit_quartile_profiles(payload)
    policy = interpretation.policy_budget_curve(payload)
    hb, hb_diagnostics = interpretation.hemoglobin_mechanism_analysis(payload)
    late = interpretation.late_transfusion_sensitivity(payload)
    chart, gradients = clinical_chart_profiles(payload)
    benefit.to_csv(inputs["out_dir"] / "temporal_benefit_profiles.csv", index=False)
    allocation.to_csv(inputs["out_dir"] / "temporal_allocation_profile.csv", index=False)
    policy.to_csv(inputs["out_dir"] / "temporal_policy_budget_curve.csv", index=False)
    hb.to_csv(inputs["out_dir"] / "temporal_hb_mechanism.csv", index=False)
    hb_diagnostics.to_csv(
        inputs["out_dir"] / "temporal_hb_observation_diagnostics.csv", index=False
    )
    late.to_csv(
        inputs["out_dir"] / "temporal_late_transfusion_sensitivity.csv", index=False
    )
    chart.to_csv(inputs["out_dir"] / "temporal_chart_profiles.csv", index=False)
    gradients.to_csv(
        inputs["out_dir"] / "temporal_chart_feature_gradients.csv", index=False
    )
    trends = []
    for name, values in (
        ("predicted_benefit", benefit["predicted_benefit_mean"].to_numpy()),
        ("dr_observed_benefit", benefit["dr_observed_benefit_mean"].to_numpy()),
        (
            "aipw_hb_change_difference",
            hb.loc[
                hb["group"].str.startswith("benefit_quartile"),
                "aipw_hb_change_difference",
            ].to_numpy(),
        ),
    ):
        trends.append(
            {
                "metric": name,
                "q4_minus_q1": float(values[-1] - values[0]),
                "spearman_r": float(
                    pd.Series(values).corr(
                        pd.Series(np.arange(1, 5)), method="spearman"
                    )
                ),
                "adjacent_monotonicity_violations": int(np.sum(np.diff(values) < 0)),
            }
        )
    pd.DataFrame(trends).to_csv(
        inputs["out_dir"] / "temporal_quartile_trends.csv", index=False
    )
    return {
        "learner": learner,
        "metrics": rows.to_dict("records"),
        "bootstrap": bootstrap_summary.to_dict("records"),
        "diagnostics": diagnostics_row,
    }


def post_discovery_robustness(
    inputs: dict[str, Any],
    args: argparse.Namespace,
    propensity_lock: dict,
    learner_lock: dict,
) -> pd.DataFrame:
    configure_propensity(propensity_lock["strategy"], propensity_lock["clip"])
    frame = inputs["frame"]
    discovery = frame[frame["era_index"].le(2)].copy()
    post_discovery = frame[frame["era_index"].ge(3)].copy()
    strata = target.joint_strata(
        post_discovery["T"].to_numpy(dtype=int),
        post_discovery["Y"].to_numpy(dtype=int),
    )
    interfaces = make_primary_interfaces(inputs)
    rows = []
    partial_path = inputs["out_dir"] / "post_discovery_repeated_splits.partial.csv"
    for repeat in range(args.robustness_repeats):
        seed = args.seed + 50_000 + repeat
        later_train, evaluation = train_test_split(
            post_discovery,
            test_size=args.robustness_test_size,
            random_state=seed,
            stratify=strata,
        )
        train_raw = pd.concat([discovery, later_train], axis=0)
        result, _, _, _ = evaluate_interfaces(
            train_raw,
            evaluation,
            inputs["chart_features"],
            interfaces[1],
            interfaces,
            [learner_lock["primary_learner"]],
            seed,
            f"post_discovery_repeat_{repeat:02d}",
        )
        rows.append(result)
        pd.concat(rows, ignore_index=True).to_csv(partial_path, index=False)
        print(
            f"Post-discovery repeat {repeat + 1}/{args.robustness_repeats} complete",
            flush=True,
        )
    all_rows = pd.concat(rows, ignore_index=True)
    all_rows.to_csv(inputs["out_dir"] / "post_discovery_repeated_splits.csv", index=False)
    summary = target.summarize_robustness(all_rows, args.bootstrap_draws)
    summary.to_csv(inputs["out_dir"] / "post_discovery_repeated_summary.csv", index=False)
    return summary


def load_or_select_propensity(
    inputs: dict[str, Any], args: argparse.Namespace
) -> dict:
    path = inputs["out_dir"] / "propensity_lock.json"
    if args.reuse_locks and path.exists():
        return read_json(path)
    return select_propensity(inputs, args)


def load_or_select_learner(
    inputs: dict[str, Any], args: argparse.Namespace, propensity_lock: dict
) -> dict:
    path = inputs["out_dir"] / "learner_lock.json"
    if args.reuse_locks and path.exists():
        return read_json(path)
    return repeated_validation_selection(inputs, args, propensity_lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["propensity", "learner", "temporal", "robustness", "all"],
        default="all",
    )
    parser.add_argument("--database", type=Path, default=target.DEFAULT_DATABASE)
    parser.add_argument(
        "--mimic-root", type=Path, default=target.DEFAULT_MIMIC_ROOT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--no-chart-roles",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "mimic_rbc_chart_augmented"
            / "baseline"
            / "no_chart_tlcd_roles.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-missing", type=float, default=0.70)
    parser.add_argument("--propensity-repeats", type=int, default=3)
    parser.add_argument(
        "--propensity-clips", type=float, nargs="+", default=[0.005, 0.01, 0.02]
    )
    parser.add_argument("--learner-repeats", type=int, default=10)
    parser.add_argument("--validation-test-size", type=float, default=0.40)
    parser.add_argument("--robustness-repeats", type=int, default=20)
    parser.add_argument("--robustness-test-size", type=float, default=0.40)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--reuse-locks", action="store_true")
    parser.add_argument("--rebuild-frame", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    inputs = load_inputs(args)
    propensity_lock = load_or_select_propensity(inputs, args)
    if args.mode == "propensity":
        return
    learner_lock = load_or_select_learner(inputs, args, propensity_lock)
    if args.mode == "learner":
        return
    temporal_result = None
    robustness_result = None
    if args.mode in {"temporal", "all"}:
        temporal_result = temporal_and_interpretation(
            inputs, args, propensity_lock, learner_lock
        )
    if args.mode in {"robustness", "all"}:
        robustness_result = post_discovery_robustness(
            inputs, args, propensity_lock, learner_lock
        ).to_dict("records")
    write_json(
        {
            "completed": True,
            "mode": args.mode,
            "runtime_sec": time.time() - started,
            "support_rule": f"{SUPPORT_VARIABLE} = 0",
            "propensity_lock": propensity_lock,
            "learner_lock": learner_lock,
            "temporal_result": temporal_result,
            "robustness_result": robustness_result,
        },
        inputs["out_dir"] / f"final_{args.mode}_manifest.json",
    )


if __name__ == "__main__":
    main()
