"""Discovery-period validation of TLCD covariate roles for the RBC study."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from empirical_experiments.mimic_rbc_target_trial import (
    fit_propensity_or_constant,
    joint_strata,
    propensity_clip_for_dimension,
    weighted_smd,
)


def crossfit_propensity(
    frame: pd.DataFrame,
    features: list[str],
    seed: int,
    folds: int = 5,
) -> np.ndarray:
    treatment = frame["T"].to_numpy(dtype=int)
    outcome = frame["Y"].to_numpy(dtype=int)
    raw = np.zeros(len(frame), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(frame, joint_strata(treatment, outcome))
    ):
        raw[test_index] = fit_propensity_or_constant(
            frame.iloc[train_index][features],
            treatment[train_index],
            frame.iloc[test_index][features],
            seed + fold,
        )
    clip = propensity_clip_for_dimension(len(features))
    return np.clip(raw, clip, 1.0 - clip)


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


def treatment_neutrality_gate(
    payload: dict,
    seed: int,
    min_log_loss_gain: float = 0.0005,
    max_weighted_smd: float = 0.10,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Promote an apparent precision variable when it predicts treatment."""

    development = payload["train"]
    treatment = development["T"].to_numpy(dtype=int)
    c_set = list(payload["tlcd"].c_set)
    s_set = list(payload["tlcd"].s_set)
    baseline_e = crossfit_propensity(development, c_set, seed)
    baseline_loss = float(log_loss(treatment, baseline_e))
    baseline_auc = float(roc_auc_score(treatment, baseline_e))
    weights = stabilized_weights(treatment, baseline_e)

    rows: list[dict[str, object]] = []
    promoted: list[str] = []
    retained: list[str] = []
    for offset, feature in enumerate(s_set):
        augmented_e = crossfit_propensity(
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
    return pd.DataFrame(rows), safe_c, sorted(retained)


def validated_roles(
    roles: dict,
    safe_c: list[str],
    safe_s: list[str],
) -> dict:
    result = dict(roles)
    result["c_proximal"] = list(safe_c)
    result["s_precision"] = list(safe_s)
    result["role_validation"] = {
        "criterion": (
            "discovery-period cross-fitted treatment log-loss gain above "
            "0.0005 or absolute C-weighted SMD above 0.10"
        ),
        "promoted_from_s_to_c": sorted(
            set(safe_c) - set(roles["c_proximal"])
        ),
        "retained_in_s": list(safe_s),
    }
    return result
