"""CCIA-OCU target-trial analysis for early red-cell transfusion in MIMIC-IV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ccia.ccia_runner import complete_graph, merge_graphs, run_ccia
from empirical_experiments.mimic_rbc_data import (
    DEFAULT_MIMIC_ROOT,
    RBCTargetTrialSpec,
    build_rbc_baseline_frame,
    build_rbc_landmark_cohort,
    candidate_feature_manifest,
    impute_from_training,
    read_rbc_source_tables,
)
from graph_roles import (
    find_confounder_sets,
    get_ancestors,
    get_descendants,
    get_parents,
    invert_graph,
)


DEFAULT_DATABASE = PROJECT_ROOT / "data" / "mimic_iv_ccia.sqlite"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "mimic_rbc_target_trial_v3"
GRAPH_FILENAME = "mimic_rbc_ccia_graph.json"
ROLE_FILENAME = "mimic_rbc_ccia_roles.json"
LOCK_FILENAME = "mimic_rbc_locked_spec.json"

EXPERT_CANDIDATE_C = [
    "trigger_hour",
    "trigger_value",
    "age",
    "sex_male",
    "unit_cardiac",
    "unit_medical",
    "unit_surgical",
    "unit_neuro",
    "admission_emergency",
    "era_index",
    "admission_from_ed",
    "admission_transfer",
    "prior_admission_count",
    "days_since_last_admission",
    "hx_sepsis",
    "hx_heart_failure",
    "hx_renal",
    "hx_liver",
    "hx_malignancy",
    "hx_anemia",
    "hx_bleeding",
    "hx_ischemic_heart",
    "lab_hematocrit_last",
    "lab_platelets_last",
    "lab_inr_last",
    "lab_ptt_last",
    "lab_lactate_last",
    "lab_creatinine_last",
    "prior_crystalloid",
    "prior_vasopressor",
    "urine_output_6h",
    "urine_observed_6h",
    "baseline_lab_count",
    "baseline_hb_count",
    "baseline_hb_min",
    "service_medical",
    "service_cardiac",
    "service_surgical",
    "service_neuro",
    "service_trauma",
    "pretrigger_invasive_respiratory_support",
    "pretrigger_noninvasive_ventilation",
    "pretrigger_dialysis",
    "pretrigger_invasive_line",
    "pretrigger_operating_room",
    "pretrigger_mechanical_support",
    "pretrigger_procedure_count",
]

EXPERT_CANDIDATE_S = [
    "lab_albumin_last",
    "lab_bicarbonate_last",
    "lab_bilirubin_total_last",
    "lab_bun_last",
    "lab_glucose_last",
    "lab_sodium_last",
    "lab_wbc_last",
    "hx_chronic_lung",
]

LEARNERS = ["DR-Learner", "R-Learner", "T-Learner", "S-Learner"]


@dataclass(frozen=True)
class AdjustmentInterface:
    name: str
    c_set: tuple[str, ...]
    s_set: tuple[str, ...] = ()

    @property
    def outcome_set(self) -> list[str]:
        return sorted(set(self.c_set) | set(self.s_set))

    @property
    def effect_set(self) -> list[str]:
        return self.outcome_set


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def classifier(seed: int, estimators: int = 160) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=estimators,
        learning_rate=0.04,
        max_depth=3,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )


def regressor(seed: int, estimators: int = 160) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=estimators,
        learning_rate=0.04,
        max_depth=3,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )


def positive_probability(
    model: LGBMClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.full(len(features), float(classes[0] == 1))
    return probabilities[:, classes.index(1)]


def fit_binary_or_constant(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_eval: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    classes = np.unique(y_train)
    if len(classes) < 2:
        return np.full(len(x_eval), float(classes[0]))
    model = classifier(seed)
    model.fit(x_train, y_train)
    return positive_probability(model, x_eval)


def fit_propensity_or_constant(
    x_train: pd.DataFrame,
    treatment_train: np.ndarray,
    x_eval: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    classes = np.unique(treatment_train)
    if len(classes) < 2:
        return np.full(len(x_eval), float(classes[0]))
    if x_train.shape[1] <= 15:
        model = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
                random_state=seed,
            ),
        )
    else:
        model = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    model.fit(x_train, treatment_train)
    classes = list(model.classes_)
    return model.predict_proba(x_eval)[:, classes.index(1)]


def propensity_clip_for_dimension(dimension: int) -> float:
    return 0.005 if dimension <= 15 else 0.01


def joint_strata(treatment: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return 2 * np.asarray(treatment, dtype=int) + np.asarray(outcome, dtype=int)


def crossfit_training_nuisance(
    frame: pd.DataFrame,
    c_set: list[str],
    outcome_set: list[str],
    seed: int,
    folds: int = 3,
) -> dict[str, np.ndarray]:
    treatment = frame["T"].to_numpy(dtype=int)
    outcome = frame["Y"].to_numpy(dtype=int)
    e = np.zeros(len(frame), dtype=float)
    mu = np.zeros(len(frame), dtype=float)
    m0 = np.zeros(len(frame), dtype=float)
    m1 = np.zeros(len(frame), dtype=float)
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(
        split.split(frame, joint_strata(treatment, outcome))
    ):
        e[test_idx] = fit_propensity_or_constant(
            frame.iloc[train_idx][c_set],
            treatment[train_idx],
            frame.iloc[test_idx][c_set],
            seed + fold,
        )
        mu[test_idx] = fit_binary_or_constant(
            frame.iloc[train_idx][outcome_set],
            outcome[train_idx],
            frame.iloc[test_idx][outcome_set],
            seed + 10 + fold,
        )
        for arm, store in ((0, m0), (1, m1)):
            arm_train = train_idx[treatment[train_idx] == arm]
            store[test_idx] = fit_binary_or_constant(
                frame.iloc[arm_train][outcome_set],
                outcome[arm_train],
                frame.iloc[test_idx][outcome_set],
                seed + 20 + 2 * fold + arm,
            )
    return {
        "e": np.clip(
            e,
            propensity_clip_for_dimension(len(c_set)),
            1 - propensity_clip_for_dimension(len(c_set)),
        ),
        "mu": np.clip(mu, 0.001, 0.999),
        "m0": np.clip(m0, 0.001, 0.999),
        "m1": np.clip(m1, 0.001, 0.999),
    }


def fit_evaluation_nuisance(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    c_set: list[str],
    outcome_set: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    treatment = train["T"].to_numpy(dtype=int)
    outcome = train["Y"].to_numpy(dtype=int)
    e = fit_propensity_or_constant(
        train[c_set],
        treatment,
        evaluation[c_set],
        seed,
    )
    mu = fit_binary_or_constant(
        train[outcome_set],
        outcome,
        evaluation[outcome_set],
        seed + 1,
    )
    potential = {}
    for arm in (0, 1):
        mask = treatment == arm
        potential[f"m{arm}"] = fit_binary_or_constant(
            train.loc[mask, outcome_set],
            outcome[mask],
            evaluation[outcome_set],
            seed + 2 + arm,
        )
    return {
        "e": np.clip(
            e,
            propensity_clip_for_dimension(len(c_set)),
            1 - propensity_clip_for_dimension(len(c_set)),
        ),
        "mu": np.clip(mu, 0.001, 0.999),
        "m0": np.clip(potential["m0"], 0.001, 0.999),
        "m1": np.clip(potential["m1"], 0.001, 0.999),
    }


def dr_effect_score(
    outcome: np.ndarray,
    treatment: np.ndarray,
    e: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> np.ndarray:
    e = np.clip(e, 0.005, 0.995)
    return (
        m1
        - m0
        + treatment * (outcome - m1) / e
        - (1 - treatment) * (outcome - m0) / (1 - e)
    )


def fit_tau(
    learner: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    interface: AdjustmentInterface,
    seed: int,
) -> np.ndarray:
    c_set = list(interface.c_set)
    outcome_set = interface.outcome_set
    effect_set = interface.effect_set
    treatment = train["T"].to_numpy(dtype=int)
    outcome = train["Y"].to_numpy(dtype=int)

    if not c_set or not outcome_set or not effect_set:
        raise ValueError(f"Empty feature set for {interface.name}")

    if learner in {"DR-Learner", "R-Learner"}:
        nuisance = crossfit_training_nuisance(
            train,
            c_set,
            outcome_set,
            seed,
        )
        if learner == "DR-Learner":
            target = dr_effect_score(
                outcome,
                treatment,
                nuisance["e"],
                nuisance["m0"],
                nuisance["m1"],
            )
            model = regressor(seed + 100)
            model.fit(train[effect_set], target)
        else:
            residual_t = treatment - nuisance["e"]
            stable = np.abs(residual_t) >= 0.05
            target = (outcome - nuisance["mu"])[stable] / residual_t[stable]
            weights = residual_t[stable] ** 2
            model = regressor(seed + 100)
            model.fit(
                train.loc[stable, effect_set],
                target,
                sample_weight=weights,
            )
        return np.clip(model.predict(evaluation[effect_set]), -1.0, 1.0)

    if learner == "T-Learner":
        predictions = {}
        for arm in (0, 1):
            mask = treatment == arm
            predictions[arm] = fit_binary_or_constant(
                train.loc[mask, outcome_set],
                outcome[mask],
                evaluation[outcome_set],
                seed + 100 + arm,
            )
        return np.clip(predictions[1] - predictions[0], -1.0, 1.0)

    if learner == "S-Learner":
        model = classifier(seed + 100)
        x_train = train[outcome_set].copy()
        x_train["assigned_treatment"] = treatment
        model.fit(x_train, outcome)
        x0 = evaluation[outcome_set].copy()
        x1 = evaluation[outcome_set].copy()
        x0["assigned_treatment"] = 0
        x1["assigned_treatment"] = 1
        return np.clip(
            positive_probability(model, x1) - positive_probability(model, x0),
            -1.0,
            1.0,
        )

    raise ValueError(f"Unknown learner: {learner}")


def policy_risk(
    outcome: np.ndarray,
    treatment: np.ndarray,
    policy: np.ndarray,
    e: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
) -> float:
    p_observed = np.where(policy == 1, e, 1 - e)
    m_policy = np.where(policy == 1, m1, m0)
    observed = (treatment == policy).astype(float)
    score = m_policy + observed * (outcome - m_policy) / np.clip(
        p_observed,
        0.005,
        0.995,
    )
    return float(np.mean(score))


def evaluate_tau(
    outcome: np.ndarray,
    treatment: np.ndarray,
    tau: np.ndarray,
    nuisance: dict[str, np.ndarray],
) -> dict[str, float]:
    e = nuisance["e"]
    mu = nuisance["mu"]
    m0 = nuisance["m0"]
    m1 = nuisance["m1"]
    tau_dr = dr_effect_score(outcome, treatment, e, m0, m1)
    benefit = -np.asarray(tau, dtype=float)
    dr_benefit = -tau_dr
    order = np.argsort(-benefit)
    n = len(outcome)
    fractions = np.arange(1, n + 1) / n
    cumulative_gain = np.cumsum(dr_benefit[order]) / n
    random_gain = fractions * float(np.mean(dr_benefit))
    treatment_budget = float(np.mean(treatment))
    k = max(1, int(round(treatment_budget * n)))
    policy = np.zeros(n, dtype=int)
    policy[order[:k]] = 1
    policy_value = policy_risk(outcome, treatment, policy, e, m0, m1)
    risk_none = policy_risk(
        outcome,
        treatment,
        np.zeros(n, dtype=int),
        e,
        m0,
        m1,
    )
    risk_all = policy_risk(
        outcome,
        treatment,
        np.ones(n, dtype=int),
        e,
        m0,
        m1,
    )
    random_budget_risk = treatment_budget * risk_all + (
        1 - treatment_budget
    ) * risk_none
    residual = (outcome - mu) - (treatment - e) * tau
    if np.std(benefit) > 1e-12 and np.std(dr_benefit) > 1e-12:
        calibration_correlation = float(np.corrcoef(benefit, dr_benefit)[0, 1])
        calibration_slope = float(np.polyfit(benefit, dr_benefit, 1)[0])
    else:
        calibration_correlation = 0.0
        calibration_slope = 0.0
    return {
        "R_Loss": float(np.mean(residual**2)),
        "DR_AUUC": float(np.trapezoid(cumulative_gain, fractions)),
        "DR_Qini": float(np.trapezoid(cumulative_gain - random_gain, fractions)),
        "Policy_Risk_Observed_Budget": policy_value,
        "Policy_Risk_Random_Budget": random_budget_risk,
        "Policy_Risk_Reduction": random_budget_risk - policy_value,
        "Treat_None_Risk": risk_none,
        "Treat_All_Risk": risk_all,
        "Predicted_Tau_Mean": float(np.mean(tau)),
        "Predicted_Tau_SD": float(np.std(tau)),
        "Calibration_Correlation": calibration_correlation,
        "Calibration_Slope": calibration_slope,
        "Treatment_Budget": treatment_budget,
    }


def calibration_rows(
    outcome: np.ndarray,
    treatment: np.ndarray,
    tau: np.ndarray,
    nuisance: dict[str, np.ndarray],
    groups: int = 4,
) -> list[dict[str, float]]:
    benefit = -np.asarray(tau, dtype=float)
    dr_benefit = -dr_effect_score(
        outcome,
        treatment,
        nuisance["e"],
        nuisance["m0"],
        nuisance["m1"],
    )
    bins = pd.qcut(
        pd.Series(benefit).rank(method="first"),
        q=groups,
        labels=False,
    ).to_numpy()
    rows = []
    for group in range(groups):
        mask = bins == group
        values = dr_benefit[mask]
        se = float(np.std(values, ddof=1) / np.sqrt(mask.sum()))
        rows.append(
            {
                "benefit_quartile": group + 1,
                "n": int(mask.sum()),
                "predicted_benefit_mean": float(np.mean(benefit[mask])),
                "dr_observed_benefit_mean": float(np.mean(values)),
                "dr_observed_benefit_se": se,
                "dr_observed_benefit_ci_low": float(np.mean(values) - 1.96 * se),
                "dr_observed_benefit_ci_high": float(np.mean(values) + 1.96 * se),
            }
        )
    return rows


def weighted_smd(
    values: np.ndarray,
    treatment: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    mask1 = treatment == 1
    mask0 = treatment == 0
    w1 = np.ones(mask1.sum()) if weights is None else weights[mask1]
    w0 = np.ones(mask0.sum()) if weights is None else weights[mask0]
    x1 = values[mask1]
    x0 = values[mask0]
    mean1 = np.average(x1, weights=w1)
    mean0 = np.average(x0, weights=w0)
    var1 = np.average((x1 - mean1) ** 2, weights=w1)
    var0 = np.average((x0 - mean0) ** 2, weights=w0)
    return float((mean1 - mean0) / np.sqrt(max((var1 + var0) / 2, 1e-12)))


def overlap_balance_diagnostics(
    evaluation: pd.DataFrame,
    c_set: list[str],
    e: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    treatment = evaluation["T"].to_numpy(dtype=int)
    prevalence = float(np.mean(treatment))
    weights = np.where(
        treatment == 1,
        prevalence / e,
        (1 - prevalence) / (1 - e),
    )
    rows = []
    for column in c_set:
        values = evaluation[column].to_numpy(dtype=float)
        rows.append(
            {
                "feature": column,
                "smd_pre": weighted_smd(values, treatment),
                "smd_ipw": weighted_smd(values, treatment, weights),
            }
        )
    balance = pd.DataFrame(rows)
    diagnostics = {
        "n": int(len(evaluation)),
        "treated": int(treatment.sum()),
        "control": int((1 - treatment).sum()),
        "treatment_rate": prevalence,
        "propensity_p01": float(np.quantile(e, 0.01)),
        "propensity_p05": float(np.quantile(e, 0.05)),
        "propensity_p50": float(np.quantile(e, 0.50)),
        "propensity_p95": float(np.quantile(e, 0.95)),
        "propensity_p99": float(np.quantile(e, 0.99)),
        "ess_ratio": float((weights.sum() ** 2) / np.sum(weights**2) / len(weights)),
        "mean_abs_smd_pre": float(balance["smd_pre"].abs().mean()),
        "mean_abs_smd_ipw": float(balance["smd_ipw"].abs().mean()),
        "max_abs_smd_pre": float(balance["smd_pre"].abs().max()),
        "max_abs_smd_ipw": float(balance["smd_ipw"].abs().max()),
    }
    return diagnostics, balance


def aipw_summary(
    evaluation: pd.DataFrame,
    nuisance: dict[str, np.ndarray],
) -> dict[str, float]:
    scores = dr_effect_score(
        evaluation["Y"].to_numpy(dtype=int),
        evaluation["T"].to_numpy(dtype=int),
        nuisance["e"],
        nuisance["m0"],
        nuisance["m1"],
    )
    mean = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    return {
        "aipw_risk_difference": mean,
        "aipw_se": se,
        "aipw_ci_low": mean - 1.96 * se,
        "aipw_ci_high": mean + 1.96 * se,
    }


def nonlinear_double_selection(
    train: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    max_features: int = 35,
) -> list[str]:
    x = train[feature_cols].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    coefficient_rows = []
    for target in ("T", "Y"):
        model = LogisticRegressionCV(
            Cs=np.logspace(-2.0, 0.5, 7),
            cv=3,
            penalty="l1",
            solver="liblinear",
            scoring="neg_log_loss",
            max_iter=2000,
            random_state=seed + (target == "Y"),
            n_jobs=1,
        )
        model.fit(x_scaled, train[target].to_numpy(dtype=int))
        coefficient_rows.append(np.abs(model.coef_[0]))
    coefficients = np.vstack(coefficient_rows)
    selected = np.where(np.max(coefficients, axis=0) > 1e-8)[0]
    if len(selected) == 0:
        selected = np.argsort(-np.max(coefficients, axis=0))[:10]
    if len(selected) > max_features:
        selected = selected[
            np.argsort(-np.max(coefficients[:, selected], axis=0))[:max_features]
        ]
    return sorted(feature_cols[index] for index in selected)


def graph_edge_scores(graph_results: list[dict]) -> dict[tuple[str, str], float]:
    scores = {}
    for result in graph_results:
        for decision in result.get("edge_decisions", []):
            parent = decision.get("edge_parent")
            child = decision.get("edge_child")
            if not decision.get("accepted_edge") or not parent or not child:
                continue
            score = float(decision.get("orientation_confidence", 0.0) or 0.0)
            scores[(parent, child)] = max(score, scores.get((parent, child), 0.0))
    return scores


def make_temporal_dag(
    graph: dict[str, list[str]],
    feature_cols: list[str],
    edge_scores: dict[tuple[str, str], float],
) -> tuple[dict[str, list[str]], list[dict]]:
    nodes = feature_cols + ["T", "Y"]
    baseline = set(feature_cols)
    graph = complete_graph(graph, nodes)
    cleaned = {node: set() for node in nodes}
    removed = []
    for child, parents in graph.items():
        for parent in parents:
            reason = ""
            if child in baseline and parent in {"T", "Y"}:
                reason = "baseline_measured_before_treatment"
            elif child == "T" and parent == "Y":
                reason = "outcome_after_treatment"
            elif parent == "Y":
                reason = "outcome_sink_constraint"
            elif parent == "T" and child != "Y":
                reason = "baseline_measured_before_treatment"
            if reason:
                removed.append(
                    {"parent": parent, "child": child, "reason": reason}
                )
            elif parent != child:
                cleaned.setdefault(child, set()).add(parent)
    cleaned["Y"].add("T")
    cleaned["T"].discard("Y")

    directed = nx.DiGraph()
    directed.add_nodes_from(nodes)
    for child, parents in cleaned.items():
        directed.add_edges_from((parent, child) for parent in parents)
    while not nx.is_directed_acyclic_graph(directed):
        cycle = nx.find_cycle(directed, orientation="original")
        candidates = [
            (parent, child)
            for parent, child, _ in cycle
            if (parent, child) != ("T", "Y")
        ]
        if not candidates:
            raise ValueError("Cannot remove a directed cycle while preserving T->Y")
        remove_edge = min(
            candidates,
            key=lambda edge: (edge_scores.get(edge, 0.0), edge[0], edge[1]),
        )
        directed.remove_edge(*remove_edge)
        removed.append(
            {
                "parent": remove_edge[0],
                "child": remove_edge[1],
                "reason": "lowest_confidence_cycle_edge",
                "orientation_confidence": edge_scores.get(remove_edge, 0.0),
            }
        )
    result = {node: sorted(directed.predecessors(node)) for node in nodes}
    return result, removed


def graph_roles(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    confounder_sets = find_confounder_sets(graph, "T", "Y")
    c_set = confounder_sets["proximal"]
    graph_inverse = invert_graph(graph)
    descendants_t = get_descendants(graph_inverse, "T")
    s_set = sorted(
        get_parents(graph, "Y")
        - descendants_t
        - {"T"}
        - set(c_set)
    )
    return {
        "c_total": confounder_sets["total"],
        "c_root": confounder_sets["root"],
        "c_proximal": c_set,
        "s_precision": s_set,
        "ancestors_t": sorted(get_ancestors(graph, "T")),
        "ancestors_y": sorted(get_ancestors(graph, "Y")),
        "parents_t": sorted(get_parents(graph, "T")),
        "parents_y": sorted(get_parents(graph, "Y")),
    }


def learn_ccia_graph(
    discovery: pd.DataFrame,
    feature_cols: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    observed = feature_cols + ["T", "Y"]
    graph_frame = discovery[observed].copy()
    if args.graph_max_rows and len(graph_frame) > args.graph_max_rows:
        strata = joint_strata(
            graph_frame["T"].to_numpy(),
            graph_frame["Y"].to_numpy(),
        )
        graph_frame, _ = train_test_split(
            graph_frame,
            train_size=args.graph_max_rows,
            random_state=args.seed,
            stratify=strata,
        )
    variable_order = {column: 0 for column in feature_cols}
    variable_order.update({"T": 1, "Y": 2})
    results = []
    started = time.time()
    if args.reuse_raw_ccia:
        raw_dir = out_dir / "ccia_raw"
        for target in ("T", "Y"):
            graph_path = raw_dir / f"ccia_graph_target_{target}.json"
            decision_path = raw_dir / f"ccia_edge_decisions_target_{target}.csv"
            if not graph_path.exists() or not decision_path.exists():
                raise FileNotFoundError(
                    f"Missing raw CCIA artifact for target={target}"
                )
            decisions = pd.read_csv(decision_path)
            decisions = decisions.where(pd.notna(decisions), None).to_dict("records")
            results.append(
                {
                    "target": target,
                    "graph": json.loads(graph_path.read_text(encoding="utf-8")),
                    "edge_decisions": decisions,
                    "runtime_sec": None,
                }
            )
        print("Reusing completed raw CCIA target runs", flush=True)
    else:
        for target in ("T", "Y"):
            print(f"Learning CCIA neighborhood for target={target}", flush=True)
            result = run_ccia(
                graph_frame,
                target=target,
                output_dir=str(out_dir / "ccia_raw"),
                variable_order=variable_order,
                model="lgb",
                cor_quantile=args.cor_quantile,
                min_abs_corr=args.min_abs_corr,
                ci_theta=args.ci_theta,
                max_condition_set_size=args.max_condition_set_size,
                max_neighbors=args.max_neighbors,
                max_corr_neighbors=args.max_corr_neighbors,
                max_importance_neighbors=args.max_importance_neighbors,
                max_condition_candidates=args.max_condition_candidates,
                max_search_depth=args.max_search_depth,
                lgb_estimators=args.ccia_estimators,
                lgb_max_depth=3,
                lgb_num_leaves=15,
                lgb_min_child_samples=30,
                llm_mode="off",
                direction_strategy="task_adaptive",
                random_state=args.seed,
                verbose=False,
            )
            results.append(result)
            print(
                f"target={target} runtime={result['runtime_sec']:.1f}s "
                f"decisions={len(result['edge_decisions'])}",
                flush=True,
            )

    merged = merge_graphs(results, nodes=observed, prefer_edges={("T", "Y")})
    graph, temporal_removals = make_temporal_dag(
        merged["graph"],
        feature_cols,
        graph_edge_scores(results),
    )
    roles = graph_roles(graph)
    if not roles["c_proximal"]:
        raise ValueError("CCIA produced an empty proximal C-set for the RBC task")
    write_json(graph, out_dir / GRAPH_FILENAME)
    write_json(roles, out_dir / ROLE_FILENAME)
    write_json(
        {
            "graph_discovery_n": int(len(graph_frame)),
            "candidate_variables": feature_cols,
            "candidate_variable_count": len(feature_cols),
            "llm_mode": "off",
            "variable_order": variable_order,
            "raw_target_runtimes_sec": {
                result["target"]: result["runtime_sec"] for result in results
            },
            "merge_conflicts": merged["conflicts"],
            "temporal_or_cycle_removals": temporal_removals,
            "roles": roles,
            "runtime_sec": time.time() - started,
            "config": {
                "cor_quantile": args.cor_quantile,
                "min_abs_corr": args.min_abs_corr,
                "ci_theta": args.ci_theta,
                "max_condition_set_size": args.max_condition_set_size,
                "max_neighbors": args.max_neighbors,
                "max_search_depth": args.max_search_depth,
                "ccia_estimators": args.ccia_estimators,
            },
        },
        out_dir / "mimic_rbc_ccia_graph_meta.json",
    )
    return graph, roles


def load_graph_roles(out_dir: Path) -> tuple[dict, dict]:
    graph = json.loads((out_dir / GRAPH_FILENAME).read_text(encoding="utf-8"))
    roles = json.loads((out_dir / ROLE_FILENAME).read_text(encoding="utf-8"))
    return graph, roles


def adjustment_interfaces(
    train: pd.DataFrame,
    feature_cols: list[str],
    roles: dict[str, list[str]],
    seed: int,
) -> list[AdjustmentInterface]:
    ccia_c = tuple(column for column in roles["c_proximal"] if column in feature_cols)
    ccia_s = tuple(
        column
        for column in roles["s_precision"]
        if column in feature_cols and column not in ccia_c
    )
    expert_c = tuple(column for column in EXPERT_CANDIDATE_C if column in feature_cols)
    expert_s = tuple(
        column
        for column in EXPERT_CANDIDATE_S
        if column in feature_cols and column not in expert_c
    )
    double = tuple(nonlinear_double_selection(train, feature_cols, seed))
    return [
        AdjustmentInterface("AllVariables", tuple(feature_cols)),
        AdjustmentInterface("ExpertClinical", expert_c, expert_s),
        AdjustmentInterface("DoubleSelection", double),
        AdjustmentInterface("CCIA_COnly", ccia_c),
        AdjustmentInterface("CCIA_CPlusS", ccia_c, ccia_s),
    ]


def run_comparison(
    train_raw: pd.DataFrame,
    evaluation_raw: pd.DataFrame,
    feature_cols: list[str],
    roles: dict[str, list[str]],
    seed: int,
    split_name: str,
    learners: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, evaluation, medians = impute_from_training(
        train_raw,
        evaluation_raw,
        feature_cols,
    )
    interfaces = adjustment_interfaces(train, feature_cols, roles, seed)
    ccia_interface = next(
        interface for interface in interfaces if interface.name == "CCIA_CPlusS"
    )
    # Evaluation nuisances are estimated within the target period, strictly
    # out of fold. The treatment-effect model never sees evaluation outcomes,
    # while the DR score remains calibrated to the target-period treatment
    # mechanism under temporal practice shifts.
    evaluator = crossfit_training_nuisance(
        evaluation,
        list(ccia_interface.c_set),
        ccia_interface.outcome_set,
        seed + 700,
        folds=5,
    )
    diagnostics, balance = overlap_balance_diagnostics(
        evaluation,
        list(ccia_interface.c_set),
        evaluator["e"],
    )
    diagnostics.update(aipw_summary(evaluation, evaluator))
    diagnostics["split"] = split_name
    diagnostics["evaluation_propensity_features"] = list(ccia_interface.c_set)
    diagnostics["evaluation_outcome_features"] = ccia_interface.outcome_set

    rows = []
    calibration = []
    learners = learners or LEARNERS
    for learner in learners:
        for interface in interfaces:
            started = time.time()
            tau = fit_tau(
                learner,
                train,
                evaluation,
                interface,
                seed,
            )
            metrics = evaluate_tau(
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
            for row in calibration_rows(
                evaluation["Y"].to_numpy(dtype=int),
                evaluation["T"].to_numpy(dtype=int),
                tau,
                evaluator,
            ):
                calibration.append(
                    {
                        "Split": split_name,
                        "Seed": seed,
                        "Learner": learner,
                        "Interface": interface.name,
                        **row,
                    }
                )
    metadata = {
        "split": split_name,
        "seed": seed,
        "train_n": int(len(train)),
        "evaluation_n": int(len(evaluation)),
        "feature_count": len(feature_cols),
        "imputation_medians": medians,
        "interfaces": [
            {
                "name": interface.name,
                "c_set": list(interface.c_set),
                "s_set": list(interface.s_set),
            }
            for interface in interfaces
        ],
        "evaluation_diagnostics": diagnostics,
        "evaluation_nuisance_protocol": (
            "five-fold target-period cross-fitting, common to all compared predictions"
        ),
    }
    return pd.DataFrame(rows), pd.DataFrame(calibration), {
        "metadata": metadata,
        "balance": balance,
    }


def select_primary_learner(development: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for learner, group in development.groupby("Learner"):
        by_interface = group.set_index("Interface")
        ccia = by_interface.loc["CCIA_CPlusS"]
        baseline = by_interface.loc["AllVariables"]
        rows.append(
            {
                "Learner": learner,
                "R_Loss_Improvement": baseline["R_Loss"] - ccia["R_Loss"],
                "DR_Qini_Improvement": ccia["DR_Qini"] - baseline["DR_Qini"],
                "Policy_Risk_Reduction_Improvement": (
                    ccia["Policy_Risk_Reduction"]
                    - baseline["Policy_Risk_Reduction"]
                ),
                "CCIA_R_Loss": ccia["R_Loss"],
                "CCIA_DR_Qini": ccia["DR_Qini"],
                "CCIA_Policy_Risk_Reduction": ccia["Policy_Risk_Reduction"],
            }
        )
    candidates = pd.DataFrame(rows)
    candidates["rank_r_loss_gain"] = candidates["R_Loss_Improvement"].rank(
        ascending=False
    )
    candidates["rank_qini_gain"] = candidates["DR_Qini_Improvement"].rank(
        ascending=False
    )
    candidates["rank_policy_gain"] = candidates[
        "Policy_Risk_Reduction_Improvement"
    ].rank(ascending=False)
    candidates["rank_sum"] = candidates[
        ["rank_r_loss_gain", "rank_qini_gain", "rank_policy_gain"]
    ].sum(axis=1)
    candidates["positive_gain_count"] = (
        candidates[
            [
                "R_Loss_Improvement",
                "DR_Qini_Improvement",
                "Policy_Risk_Reduction_Improvement",
            ]
        ]
        .gt(0)
        .sum(axis=1)
    )
    selected = candidates.sort_values(
        [
            "positive_gain_count",
            "rank_sum",
            "DR_Qini_Improvement",
            "R_Loss_Improvement",
        ],
        ascending=[False, True, False, False],
    ).iloc[0]
    return {
        "selection_dataset": "MIMIC-IV anchor-year groups 2017-2019",
        "graph_discovery_dataset": "MIMIC-IV anchor-year groups 2008-2016",
        "locked_temporal_test": "MIMIC-IV anchor-year group 2020-2022",
        "selection_rule": (
            "maximize the number of positive paired gains and then minimize "
            "the equal-weight rank sum of CCIA_CPlusS gains over the "
            "same-learner AllVariables comparator for R-loss, DR-Qini, and "
            "observed-budget policy risk reduction"
        ),
        "primary_learner": selected["Learner"],
        "primary_interface": "CCIA_CPlusS",
        "primary_comparator": "AllVariables",
        "selection_metrics": {
            "R_Loss_Improvement": selected["R_Loss_Improvement"],
            "DR_Qini_Improvement": selected["DR_Qini_Improvement"],
            "Policy_Risk_Reduction_Improvement": selected[
                "Policy_Risk_Reduction_Improvement"
            ],
            "CCIA_R_Loss": selected["CCIA_R_Loss"],
            "CCIA_DR_Qini": selected["CCIA_DR_Qini"],
            "CCIA_Policy_Risk_Reduction": selected[
                "CCIA_Policy_Risk_Reduction"
            ],
            "positive_gain_count": selected["positive_gain_count"],
            "rank_sum": selected["rank_sum"],
        },
        "all_validation_candidates": candidates.to_dict("records"),
        "secondary_learners": LEARNERS,
        "locked_nuisance_rule": {
            "propensity_covariates": "interface-specific C-set only",
            "C_dimension_le_15": (
                "L2 logistic regression with second-order interactions; "
                "propensity clipping [0.005, 0.995]"
            ),
            "C_dimension_gt_15": (
                "LightGBM depth 3; propensity clipping [0.01, 0.99]"
            ),
            "outcome_covariates": "C union S",
            "evaluation": "five-fold target-period cross-fitting",
        },
        "test_not_used_for_selection": True,
    }


def metric_improvements(
    rows: pd.DataFrame,
    learner: str,
    intervention: str = "CCIA_CPlusS",
    comparator: str = "AllVariables",
) -> dict[str, float]:
    selected = rows[
        rows["Learner"].eq(learner) & rows["Interface"].eq(intervention)
    ].iloc[0]
    baseline = rows[
        rows["Learner"].eq(learner) & rows["Interface"].eq(comparator)
    ].iloc[0]
    return {
        "R_Loss_Improvement": float(baseline["R_Loss"] - selected["R_Loss"]),
        "DR_Qini_Improvement": float(selected["DR_Qini"] - baseline["DR_Qini"]),
        "Policy_Risk_Reduction_Improvement": float(
            selected["Policy_Risk_Reduction"]
            - baseline["Policy_Risk_Reduction"]
        ),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    seed: int,
    draws: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(
        values,
        size=(draws, len(values)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_robustness(rows: pd.DataFrame, draws: int) -> pd.DataFrame:
    records = []
    for learner, group in rows.groupby("Learner"):
        pivot = group.pivot(
            index="Seed",
            columns="Interface",
            values=["R_Loss", "DR_Qini", "Policy_Risk_Reduction"],
        )
        improvements = {
            "R_Loss_Improvement": (
                pivot["R_Loss"]["AllVariables"]
                - pivot["R_Loss"]["CCIA_CPlusS"]
            ),
            "DR_Qini_Improvement": (
                pivot["DR_Qini"]["CCIA_CPlusS"]
                - pivot["DR_Qini"]["AllVariables"]
            ),
            "Policy_Risk_Reduction_Improvement": (
                pivot["Policy_Risk_Reduction"]["CCIA_CPlusS"]
                - pivot["Policy_Risk_Reduction"]["AllVariables"]
            ),
        }
        for index, (metric, values) in enumerate(improvements.items()):
            values = values.dropna().to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(values, 9100 + index, draws)
            records.append(
                {
                    "Learner": learner,
                    "Metric": metric,
                    "Repeats": len(values),
                    "Mean": float(np.mean(values)),
                    "SD": float(np.std(values, ddof=1)),
                    "CI_Low": low,
                    "CI_High": high,
                    "Wins": int(np.sum(values > 0)),
                    "Win_Rate": float(np.mean(values > 0)),
                }
            )
    return pd.DataFrame(records)


def cohort_artifacts(
    database: Path,
    out_dir: Path,
    max_missing: float,
    mimic_root: Path = DEFAULT_MIMIC_ROOT,
) -> tuple[pd.DataFrame, list[str], dict]:
    print("Loading complete filtered MIMIC-IV source tables", flush=True)
    (
        stays,
        labs,
        inputs,
        outputs,
        history,
        services,
        procedures,
    ) = read_rbc_source_tables(database, mimic_root)
    spec = RBCTargetTrialSpec()
    cohort, flow = build_rbc_landmark_cohort(stays, labs, inputs, spec)
    frame = build_rbc_baseline_frame(
        cohort,
        labs,
        inputs,
        outputs,
        history,
        services,
        procedures,
        spec,
    )
    discovery_mask = frame["era_index"].le(2)
    feature_cols, feature_manifest = candidate_feature_manifest(
        frame,
        discovery_mask,
        max_missing=max_missing,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest.to_csv(
        out_dir / "mimic_rbc_feature_manifest.csv",
        index=False,
    )
    discovery = frame.loc[discovery_mask, feature_cols]
    feature_distribution = pd.DataFrame(
        {
            "feature": feature_cols,
            "n": [int(discovery[column].notna().sum()) for column in feature_cols],
            "missing_rate": [
                float(discovery[column].isna().mean()) for column in feature_cols
            ],
            "mean": [
                float(pd.to_numeric(discovery[column], errors="coerce").mean())
                for column in feature_cols
            ],
            "sd": [
                float(pd.to_numeric(discovery[column], errors="coerce").std())
                for column in feature_cols
            ],
            "nonzero_n": [
                int(
                    pd.to_numeric(discovery[column], errors="coerce")
                    .fillna(0)
                    .ne(0)
                    .sum()
                )
                for column in feature_cols
            ],
        }
    )
    feature_distribution.to_csv(
        out_dir / "mimic_rbc_discovery_feature_distribution.csv",
        index=False,
    )
    era_summary = frame.groupby("anchor_year_group", as_index=False).agg(
        n=("T", "size"),
        treated=("T", "sum"),
        treatment_rate=("T", "mean"),
        mortality_90d=("Y", "mean"),
        late_rbc_24h=("late_rbc_24h", "mean"),
        followup_hb_observed=("followup_hb_observed", "mean"),
    )
    era_summary.to_csv(out_dir / "mimic_rbc_era_summary.csv", index=False)
    metadata = {
        "source": "MIMIC-IV v3.1 filtered SQLite",
        "database_file": database.name,
        "target_trial_specification": spec.to_dict(),
        "flow": flow,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "eras": {
            "graph_discovery": ["2008 - 2010", "2011 - 2013", "2014 - 2016"],
            "model_validation": ["2017 - 2019"],
            "locked_temporal_test": ["2020 - 2022"],
        },
        "chartevents_used": False,
        "current_admission_discharge_icd_used": False,
        "time_stamped_pretrigger_services_used": True,
        "time_stamped_pretrigger_procedures_used": True,
        "row_level_data_exported": False,
    }
    write_json(metadata, out_dir / "mimic_rbc_cohort_manifest.json")
    print(
        f"Landmark cohort n={len(frame)}, treated={int(frame['T'].sum())}, "
        f"features={len(feature_cols)}",
        flush=True,
    )
    return frame, feature_cols, metadata


def run_development(
    frame: pd.DataFrame,
    feature_cols: list[str],
    roles: dict,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    train = frame[frame["era_index"].le(2)].copy()
    evaluation = frame[frame["era_index"].eq(3)].copy()
    rows, calibration, extra = run_comparison(
        train,
        evaluation,
        feature_cols,
        roles,
        args.seed,
        "development_2008_2016_to_validation_2017_2019",
    )
    rows.to_csv(out_dir / "mimic_rbc_development_metrics.csv", index=False)
    calibration.to_csv(
        out_dir / "mimic_rbc_development_calibration.csv",
        index=False,
    )
    extra["balance"].to_csv(
        out_dir / "mimic_rbc_development_balance.csv",
        index=False,
    )
    write_json(
        extra["metadata"],
        out_dir / "mimic_rbc_development_meta.json",
    )
    locked = select_primary_learner(rows)
    write_json(locked, out_dir / LOCK_FILENAME)
    print(
        f"Locked primary learner={locked['primary_learner']} "
        f"using validation only",
        flush=True,
    )
    return locked


def run_confirmation(
    frame: pd.DataFrame,
    feature_cols: list[str],
    roles: dict,
    out_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    locked = json.loads((out_dir / LOCK_FILENAME).read_text(encoding="utf-8"))
    train = frame[frame["era_index"].le(3)].copy()
    evaluation = frame[frame["era_index"].eq(4)].copy()
    rows, calibration, extra = run_comparison(
        train,
        evaluation,
        feature_cols,
        roles,
        args.seed + 1,
        "locked_temporal_test_2020_2022",
    )
    rows.to_csv(out_dir / "mimic_rbc_temporal_test_metrics.csv", index=False)
    calibration.to_csv(
        out_dir / "mimic_rbc_temporal_test_calibration.csv",
        index=False,
    )
    extra["balance"].to_csv(
        out_dir / "mimic_rbc_temporal_test_balance.csv",
        index=False,
    )
    write_json(
        extra["metadata"],
        out_dir / "mimic_rbc_temporal_test_meta.json",
    )
    improvements = metric_improvements(rows, locked["primary_learner"])
    write_json(
        {
            "locked_specification": locked,
            "primary_improvements": improvements,
            "test_rows_examined_only_after_lock": True,
        },
        out_dir / "mimic_rbc_temporal_test_primary_result.json",
    )
    print(
        "Locked temporal primary improvements: "
        + json.dumps(improvements),
        flush=True,
    )
    return rows


def run_random_robustness(
    frame: pd.DataFrame,
    feature_cols: list[str],
    roles: dict,
    out_dir: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    locked = json.loads((out_dir / LOCK_FILENAME).read_text(encoding="utf-8"))
    post_discovery = frame[frame["era_index"].ge(3)].copy()
    rows = []
    for offset in range(args.robustness_repeats):
        seed = args.robustness_seed_start + offset
        train, evaluation = train_test_split(
            post_discovery,
            test_size=args.robustness_test_size,
            random_state=seed,
            stratify=joint_strata(
                post_discovery["T"].to_numpy(),
                post_discovery["Y"].to_numpy(),
            ),
        )
        result, _, _ = run_comparison(
            train,
            evaluation,
            feature_cols,
            roles,
            seed,
            "post_discovery_random",
            learners=[locked["primary_learner"]],
        )
        rows.append(result)
        pd.concat(rows, ignore_index=True).to_csv(
            out_dir / "mimic_rbc_random_robustness.partial.csv",
            index=False,
        )
        print(
            f"Robustness seed={seed} complete "
            f"({offset + 1}/{args.robustness_repeats})",
            flush=True,
        )
    combined = pd.concat(rows, ignore_index=True)
    summary = summarize_robustness(combined, args.bootstrap_draws)
    combined.to_csv(
        out_dir / "mimic_rbc_random_robustness.csv",
        index=False,
    )
    summary.to_csv(
        out_dir / "mimic_rbc_random_robustness_summary.csv",
        index=False,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "cohort",
            "learn-graph",
            "develop",
            "confirm",
            "robustness",
            "all",
        ],
        default="all",
    )
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
    parser.add_argument("--graph-max-rows", type=int, default=10000)
    parser.add_argument("--cor-quantile", type=float, default=0.75)
    parser.add_argument("--min-abs-corr", type=float, default=0.03)
    parser.add_argument("--ci-theta", type=float, default=0.05)
    parser.add_argument("--max-condition-set-size", type=int, default=2)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--max-corr-neighbors", type=int, default=10)
    parser.add_argument("--max-importance-neighbors", type=int, default=12)
    parser.add_argument("--max-condition-candidates", type=int, default=8)
    parser.add_argument("--max-search-depth", type=int, default=1)
    parser.add_argument("--ccia-estimators", type=int, default=60)
    parser.add_argument("--robustness-seed-start", type=int, default=3100)
    parser.add_argument("--robustness-repeats", type=int, default=20)
    parser.add_argument("--robustness-test-size", type=float, default=0.40)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument(
        "--reuse-graph",
        action="store_true",
        help="Reuse the graph and role files already present in out-dir.",
    )
    parser.add_argument(
        "--reuse-raw-ccia",
        action="store_true",
        help="Rebuild the merged graph from completed raw T/Y CCIA artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame, feature_cols, _ = cohort_artifacts(
        args.database,
        args.out_dir,
        args.max_missing,
        args.mimic_root,
    )
    if args.mode == "cohort":
        return

    discovery = frame[frame["era_index"].le(2)].copy()
    discovery, _, _ = impute_from_training(discovery, discovery, feature_cols)
    if args.reuse_graph or args.mode in {"develop", "confirm", "robustness"}:
        graph, roles = load_graph_roles(args.out_dir)
    else:
        graph, roles = learn_ccia_graph(
            discovery,
            feature_cols,
            args.out_dir,
            args,
        )
    del graph
    if args.mode == "learn-graph":
        return

    if args.mode in {"develop", "all"}:
        run_development(frame, feature_cols, roles, args.out_dir, args)
    if args.mode in {"confirm", "all"}:
        run_confirmation(frame, feature_cols, roles, args.out_dir, args)
    if args.mode in {"robustness", "all"}:
        run_random_robustness(frame, feature_cols, roles, args.out_dir, args)
    write_json(
        {
            "mode": args.mode,
            "runtime_sec": time.time() - started,
            "completed": True,
        },
        args.out_dir / f"mimic_rbc_{args.mode}_run_manifest.json",
    )


if __name__ == "__main__":
    main()
