"""Validation-only propensity benchmark for the chart-augmented RBC study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from empirical_experiments.mimic_rbc_chart_panel import (
    DEFAULT_OUT_DIR,
    build_or_load_frame,
    panel_features,
    read_json,
)
from empirical_experiments.mimic_rbc_target_trial import (
    DEFAULT_DATABASE,
    DEFAULT_MIMIC_ROOT,
    impute_from_training,
    joint_strata,
    weighted_smd,
    write_json,
)


STRATEGIES = [
    "linear_c0.01",
    "linear_c0.03",
    "linear_c0.1",
    "linear_c0.3",
    "linear_c1.0",
    "poly2_c0.01",
    "poly2_c0.03",
    "poly2_c0.1",
    "poly2_c0.3",
    "lgb_d2_m50",
    "lgb_d2_m100",
    "lgb_d3_m50",
    "lgb_d3_m100",
    "hist_l7_m50_l2",
    "hist_l15_m50_l5",
]


def model_for_strategy(strategy: str, seed: int):
    if strategy.startswith("linear_c"):
        regularization = float(strategy.split("c", 1)[1])
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
                random_state=seed,
            ),
        )
    if strategy.startswith("poly2_c"):
        regularization = float(strategy.split("c", 1)[1])
        return make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(),
            LogisticRegression(
                C=regularization,
                penalty="l2",
                solver="lbfgs",
                max_iter=3000,
                random_state=seed,
            ),
        )
    if strategy.startswith("lgb_"):
        depth = 2 if "d2" in strategy else 3
        min_child = 100 if "m100" in strategy else 50
        leaves = 7 if depth == 2 else 15
        return LGBMClassifier(
            n_estimators=350,
            learning_rate=0.025,
            max_depth=depth,
            num_leaves=leaves,
            min_child_samples=min_child,
            subsample=0.85,
            colsample_bytree=0.9,
            reg_lambda=3.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    if strategy == "hist_l7_m50_l2":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=7,
            min_samples_leaf=50,
            l2_regularization=2.0,
            random_state=seed,
        )
    if strategy == "hist_l15_m50_l5":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=5.0,
            random_state=seed,
        )
    raise ValueError(f"Unknown propensity strategy: {strategy}")


def crossfit_propensity(
    frame: pd.DataFrame,
    features: list[str],
    strategy: str,
    seed: int,
    folds: int = 5,
) -> np.ndarray:
    treatment = frame["T"].to_numpy(dtype=int)
    outcome = frame["Y"].to_numpy(dtype=int)
    propensity = np.zeros(len(frame), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(frame, joint_strata(treatment, outcome))
    ):
        model = model_for_strategy(strategy, seed + fold)
        model.fit(frame.iloc[train_index][features], treatment[train_index])
        propensity[test_index] = model.predict_proba(
            frame.iloc[test_index][features]
        )[:, 1]
    return propensity


def diagnostics(
    frame: pd.DataFrame,
    features: list[str],
    propensity_raw: np.ndarray,
    strategy: str,
    clip: float,
    era: str,
) -> dict:
    treatment = frame["T"].to_numpy(dtype=int)
    propensity = np.clip(propensity_raw, clip, 1.0 - clip)
    prevalence = float(treatment.mean())
    weights = np.where(
        treatment == 1,
        prevalence / propensity,
        (1.0 - prevalence) / (1.0 - propensity),
    )
    smd = np.asarray(
        [
            abs(
                weighted_smd(
                    frame[feature].to_numpy(dtype=float), treatment, weights
                )
            )
            for feature in features
        ]
    )
    return {
        "era": era,
        "strategy": strategy,
        "clip": clip,
        "n": int(len(frame)),
        "treated": int(treatment.sum()),
        "log_loss": float(log_loss(treatment, propensity)),
        "auc": float(roc_auc_score(treatment, propensity)),
        "brier": float(np.mean((treatment - propensity) ** 2)),
        "mean_abs_smd": float(smd.mean()),
        "max_abs_smd": float(smd.max()),
        "features_above_0.10": int(np.sum(smd > 0.10)),
        "ess_ratio": float(weights.sum() ** 2 / np.sum(weights**2) / len(weights)),
        "p01": float(np.quantile(propensity, 0.01)),
        "p50": float(np.quantile(propensity, 0.50)),
        "p99": float(np.quantile(propensity, 0.99)),
    }


def benchmark(args: argparse.Namespace) -> None:
    frame, base_features = build_or_load_frame(args)
    selected = read_json(args.out_dir / "selected_chartevents_panel.json")
    panel = selected["selected_panel"]
    features, _ = panel_features(base_features, panel)
    panel_dir = args.out_dir / "panels" / panel
    roles = read_json(panel_dir / "mimic_rbc_safety_validated_roles.json")
    c_set = [feature for feature in roles["c_proximal"] if feature in features]
    train_raw = frame[frame["era_index"].le(2)].copy()
    validation_raw = frame[frame["era_index"].eq(3)].copy()
    temporal_raw = frame[frame["era_index"].eq(4)].copy()
    _, validation, _ = impute_from_training(
        train_raw, validation_raw, features
    )
    temporal_train_raw = frame[frame["era_index"].le(3)].copy()
    _, temporal, _ = impute_from_training(
        temporal_train_raw, temporal_raw, features
    )

    validation_rows = []
    for strategy in STRATEGIES:
        raw = crossfit_propensity(
            validation, c_set, strategy, args.seed + 20_000
        )
        for clip in (0.005, 0.01, 0.02):
            validation_rows.append(
                diagnostics(validation, c_set, raw, strategy, clip, "validation")
            )
        print(f"Propensity strategy {strategy} complete", flush=True)
    validation_result = pd.DataFrame(validation_rows)
    eligible = validation_result[
        validation_result["ess_ratio"].ge(0.50)
        & validation_result["p01"].ge(0.005)
    ].copy()
    if eligible.empty:
        eligible = validation_result.copy()
    selected_row = eligible.sort_values(
        [
            "features_above_0.10",
            "max_abs_smd",
            "mean_abs_smd",
            "log_loss",
            "ess_ratio",
        ],
        ascending=[True, True, True, True, False],
    ).iloc[0]

    strategy = str(selected_row["strategy"])
    clip = float(selected_row["clip"])
    temporal_rows = []
    temporal_raw_by_strategy = {}
    for candidate_strategy in STRATEGIES:
        raw = crossfit_propensity(
            temporal, c_set, candidate_strategy, args.seed + 30_000
        )
        temporal_raw_by_strategy[candidate_strategy] = raw
        for candidate_clip in (0.005, 0.01, 0.02):
            temporal_rows.append(
                diagnostics(
                    temporal,
                    c_set,
                    raw,
                    candidate_strategy,
                    candidate_clip,
                    "temporal_diagnostic_not_used_for_lock",
                )
            )
    temporal_result = pd.DataFrame(temporal_rows)
    temporal_row = diagnostics(
        temporal,
        c_set,
        temporal_raw_by_strategy[strategy],
        strategy,
        clip,
        "temporal_audit_after_validation_lock",
    )
    validation_result.to_csv(
        panel_dir / "chartevents_propensity_validation_benchmark.csv", index=False
    )
    pd.DataFrame([temporal_row]).to_csv(
        panel_dir / "chartevents_propensity_locked_temporal_balance.csv", index=False
    )
    temporal_result.to_csv(
        panel_dir / "chartevents_propensity_temporal_diagnostic_all.csv",
        index=False,
    )
    lock = {
        "panel": panel,
        "selection_dataset": "MIMIC-IV 2017-2019 only",
        "selection_objective": (
            "fewest C-set variables above absolute SMD 0.10, then minimum "
            "maximum and mean absolute SMD, subject to ESS/n >= 0.50"
        ),
        "strategy": strategy,
        "clip": clip,
        "c_set": c_set,
        "validation_diagnostics": selected_row.to_dict(),
        "temporal_diagnostics_after_lock": temporal_row,
        "temporal_outcomes_used_for_selection": False,
    }
    write_json(lock, panel_dir / "chartevents_propensity_lock.json")
    print(json.dumps(lock, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-missing", type=float, default=0.70)
    parser.add_argument("--rebuild-frame", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
