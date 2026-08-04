"""Select a time-safe bedside feature panel and learn TLCD roles."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from empirical_experiments.mimic_rbc_interpretation import (
    benefit_quartile_profiles,
    hemoglobin_mechanism_analysis,
    late_transfusion_sensitivity,
    paired_patient_bootstrap,
    policy_budget_curve,
)
from empirical_experiments.mimic_rbc_role_validation import (
    treatment_neutrality_gate,
    validated_roles,
)
from empirical_experiments.mimic_rbc_target_trial import (
    AdjustmentInterface,
    DEFAULT_DATABASE,
    DEFAULT_MIMIC_ROOT,
    cohort_artifacts,
    crossfit_training_nuisance,
    fit_tau,
    impute_from_training,
    learn_tlcd_graph,
    run_confirmation,
    run_development,
    run_random_robustness,
    weighted_smd,
    write_json,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "generated" / "mimic-rbc-chart"
BASE_REFERENCE_DIRNAME = "base_reference_rebuild"
SELECTED_PANEL_FILENAME = "selected_chartevents_panel.json"

LAB_MISSING_DROPS = {
    "lab_albumin_last_missing",
    "lab_bicarbonate_last_missing",
    "lab_bun_last_missing",
    "lab_creatinine_last_missing",
    "lab_hematocrit_last_missing",
    "lab_inr_last_missing",
    "lab_lactate_last_missing",
    "lab_platelets_last_missing",
    "lab_potassium_last_missing",
    "lab_pt_last_missing",
    "lab_ptt_last_missing",
    "lab_sodium_last_missing",
    "lab_wbc_last_missing",
}
COMMON_BASE_DROPS = LAB_MISSING_DROPS | {"lab_pt_last"}

VITAL_EXTREMA = [
    "chart_heart_rate_last",
    "chart_heart_rate_max",
    "chart_sbp_last",
    "chart_sbp_min",
    "chart_map_last",
    "chart_map_min",
    "chart_resp_rate_last",
    "chart_resp_rate_max",
    "chart_spo2_last",
    "chart_spo2_min",
    "chart_temp_c_last",
    "chart_gcs_combined_last",
    "chart_rass_last",
    "chart_shock_index_last",
    "chart_weight_kg_last",
]

VITAL_BURDEN = [
    "chart_heart_rate_last",
    "chart_heart_rate_high_fraction",
    "chart_sbp_last",
    "chart_sbp_min",
    "chart_map_last",
    "chart_map_low_fraction",
    "chart_resp_rate_last",
    "chart_resp_rate_max",
    "chart_spo2_last",
    "chart_spo2_low_fraction",
    "chart_temp_c_last",
    "chart_gcs_combined_last",
    "chart_rass_last",
    "chart_shock_index_last",
    "chart_weight_kg_last",
]

OXYGEN_ADDITIONS = [
    "chart_o2_flow_last",
    "chart_fio2_recorded",
    "chart_ventilator_measurement_recorded",
]

PANEL_DEFINITIONS = {
    "vital_extrema": {
        "chart_features": VITAL_EXTREMA,
        "extra_base_drops": set(),
        "rationale": "Latest values and adverse extrema of core pre-trigger physiology.",
    },
    "vital_burden": {
        "chart_features": VITAL_BURDEN,
        "extra_base_drops": set(),
        "rationale": "Latest values plus pre-trigger tachycardia, hypotension, and hypoxemia burdens.",
    },
    "vital_oxygen": {
        "chart_features": VITAL_BURDEN + OXYGEN_ADDITIONS,
        "extra_base_drops": {
            "prior_furosemide",
            "prior_insulin",
            "pretrigger_noninvasive_ventilation",
        },
        "rationale": "Core physiologic burden augmented by oxygen and ventilator recording context.",
    },
}

PROFILE_CHART_FEATURES = [
    "chart_heart_rate_last",
    "chart_sbp_last",
    "chart_map_min",
    "chart_resp_rate_last",
    "chart_spo2_min",
    "chart_temp_c_last",
    "chart_gcs_combined_last",
    "chart_rass_last",
    "chart_shock_index_last",
    "chart_weight_kg_last",
    "chart_o2_flow_last",
    "chart_fio2_recorded",
    "chart_ventilator_measurement_recorded",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def graph_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        seed=args.seed,
        graph_max_rows=args.graph_max_rows,
        cor_quantile=args.cor_quantile,
        min_abs_corr=args.min_abs_corr,
        ci_theta=args.ci_theta,
        max_condition_set_size=args.max_condition_set_size,
        max_neighbors=args.max_neighbors,
        max_corr_neighbors=args.max_corr_neighbors,
        max_importance_neighbors=args.max_importance_neighbors,
        max_condition_candidates=args.max_condition_candidates,
        max_search_depth=args.max_search_depth,
        tlcd_estimators=args.tlcd_estimators,
        reuse_raw_tlcd=args.reuse_raw_tlcd,
        bootstrap_draws=args.bootstrap_draws,
        robustness_seed_start=args.robustness_seed_start,
        robustness_repeats=args.robustness_repeats,
        robustness_test_size=args.robustness_test_size,
    )


def add_chart_observation_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["chart_fio2_recorded"] = frame["chart_fio2_last"].notna().astype(int)
    frame["chart_o2_flow_recorded"] = frame["chart_o2_flow_last"].notna().astype(int)
    ventilator = [
        "chart_peep_last",
        "chart_tidal_volume_last",
        "chart_peak_pressure_last",
    ]
    frame["chart_ventilator_measurement_recorded"] = frame[
        ventilator
    ].notna().any(axis=1).astype(int)
    return frame


def build_or_load_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    private_dir = args.out_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    frame_path = private_dir / "mimic_rbc_chart_augmented_frame.pkl.gz"
    feature_path = private_dir / "base_feature_cols.json"
    if frame_path.exists() and feature_path.exists() and not args.rebuild_frame:
        return pd.read_pickle(frame_path), read_json(feature_path)["features"]

    base_dir = args.out_dir / BASE_REFERENCE_DIRNAME
    frame, base_features, metadata = cohort_artifacts(
        args.database,
        base_dir,
        args.max_missing,
        args.mimic_root,
    )
    chart_path = private_dir / "chartevents_features_wide.pkl.gz"
    if not chart_path.exists():
        raise FileNotFoundError(
            f"Missing {chart_path}; run mimic_chartevents_data.py first"
        )
    chart = pd.read_pickle(chart_path)
    frame = frame.merge(chart, on="stay_id", how="left", validate="one_to_one")
    frame = add_chart_observation_features(frame)
    if len(frame) != 15_568 or int(frame["T"].sum()) != 3_017:
        raise ValueError("Chart merge changed the frozen cohort")
    frame.to_pickle(frame_path)
    write_json(
        {
            "features": base_features,
            "base_metadata": metadata,
            "chart_source": "BigQuery aggregate cache",
            "row_level_data_exported": False,
        },
        feature_path,
    )
    return frame, base_features


def panel_features(base_features: list[str], panel: str) -> tuple[list[str], dict]:
    definition = PANEL_DEFINITIONS[panel]
    drops = COMMON_BASE_DROPS | set(definition["extra_base_drops"])
    base = [feature for feature in base_features if feature not in drops]
    chart = list(definition["chart_features"])
    features = base + chart
    if len(features) > 80:
        raise ValueError(f"Panel {panel} has {len(features)} features")
    if len(features) != len(set(features)):
        raise ValueError(f"Panel {panel} contains duplicate features")
    return features, {
        "panel": panel,
        "base_feature_count": len(base_features),
        "retained_base_count": len(base),
        "chart_feature_count": len(chart),
        "total_feature_count": len(features),
        "removed_base_features": sorted(drops & set(base_features)),
        "chart_features": chart,
        "features": features,
        "rationale": definition["rationale"],
        "selection_data": "prespecified clinical design; no temporal-test outcomes",
    }


def feature_audit(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    discovery = frame[frame["era_index"].le(2)].copy()
    treatment = discovery["T"].to_numpy(dtype=int)
    outcome = discovery["Y"].to_numpy(dtype=int)
    era_masks = {
        "discovery": frame["era_index"].le(2),
        "validation": frame["era_index"].eq(3),
        "temporal": frame["era_index"].eq(4),
    }
    rows = []
    for feature in sorted(set(features)):
        values = pd.to_numeric(discovery[feature], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        imputed = values.fillna(median).to_numpy(dtype=float)
        row = {
            "feature": feature,
            "discovery_coverage": float(values.notna().mean()),
            "discovery_unique": int(values.nunique(dropna=True)),
            "discovery_mean": float(values.mean()),
            "discovery_sd": float(values.std()),
            "discovery_abs_treatment_smd": abs(
                weighted_smd(imputed, treatment)
            ),
            "discovery_abs_outcome_smd": abs(weighted_smd(imputed, outcome)),
        }
        for era, mask in era_masks.items():
            era_values = pd.to_numeric(frame.loc[mask, feature], errors="coerce")
            row[f"{era}_coverage"] = float(era_values.notna().mean())
            row[f"{era}_mean"] = float(era_values.mean())
        scale = max(float(values.std()), 1e-8)
        row["validation_mean_shift_sd"] = (
            row["validation_mean"] - row["discovery_mean"]
        ) / scale
        row["temporal_mean_shift_sd"] = (
            row["temporal_mean"] - row["discovery_mean"]
        ) / scale
        rows.append(row)
    return pd.DataFrame(rows)


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frame, base_features = build_or_load_frame(args)
    panels: dict[str, list[str]] = {}
    chart_union: set[str] = set()
    panel_dir = args.out_dir / "panel_manifests"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for panel in PANEL_DEFINITIONS:
        features, manifest = panel_features(base_features, panel)
        missing = [feature for feature in features if feature not in frame]
        if missing:
            raise ValueError(f"Panel {panel} missing columns: {missing}")
        panels[panel] = features
        chart_union.update(manifest["chart_features"])
        write_json(manifest, panel_dir / f"{panel}.json")
    audit = feature_audit(frame, sorted(chart_union))
    audit.to_csv(args.out_dir / "chartevents_candidate_feature_audit.csv", index=False)
    write_json(
        {
            "cohort_n": int(len(frame)),
            "treated": int(frame["T"].sum()),
            "outcomes": int(frame["Y"].sum()),
            "panels": {panel: len(features) for panel, features in panels.items()},
            "temporal_test_outcomes_used_for_panel_design": False,
        },
        args.out_dir / "chartevents_prepare_manifest.json",
    )
    return frame, panels


def panel_result_dir(args: argparse.Namespace, panel: str) -> Path:
    return args.out_dir / "panels" / panel


def validate_roles(
    frame: pd.DataFrame,
    features: list[str],
    raw_roles: dict,
    panel_dir: Path,
    args: argparse.Namespace,
) -> dict:
    discovery_raw = frame[frame["era_index"].le(2)].copy()
    discovery, _, _ = impute_from_training(
        discovery_raw, discovery_raw, features
    )
    c_set = tuple(
        feature for feature in raw_roles["c_proximal"] if feature in features
    )
    s_set = tuple(
        feature
        for feature in raw_roles["s_precision"]
        if feature in features and feature not in c_set
    )
    payload = {
        "train": discovery,
        "tlcd": AdjustmentInterface("RITE_CPlusS", c_set, s_set),
    }
    gate, safe_c, safe_s = treatment_neutrality_gate(
        payload, args.seed + 5500
    )
    gate.to_csv(
        panel_dir / "mimic_rbc_treatment_neutrality_safety_gate.csv",
        index=False,
    )
    validated = validated_roles(raw_roles, safe_c, safe_s)
    write_json(validated, panel_dir / "mimic_rbc_safety_validated_roles.json")
    return validated


def develop_panel(
    frame: pd.DataFrame,
    features: list[str],
    panel: str,
    args: argparse.Namespace,
) -> dict:
    out_dir = panel_result_dir(args, panel)
    out_dir.mkdir(parents=True, exist_ok=True)
    gargs = graph_args(args)
    discovery_raw = frame[frame["era_index"].le(2)].copy()
    discovery, _, _ = impute_from_training(
        discovery_raw, discovery_raw, features
    )
    role_path = out_dir / "mimic_rbc_tlcd_roles.json"
    graph_path = out_dir / "mimic_rbc_tlcd_graph.json"
    if args.reuse_graph and role_path.exists() and graph_path.exists():
        raw_roles = read_json(role_path)
    else:
        _, raw_roles = learn_tlcd_graph(
            discovery, features, out_dir, gargs
        )
    roles = validate_roles(frame, features, raw_roles, out_dir, args)
    lock = run_development(frame, features, roles, out_dir, gargs)
    summary = {
        "panel": panel,
        "learner": lock["primary_learner"],
        **lock["selection_metrics"],
        "c_set": roles["c_proximal"],
        "s_set": roles["s_precision"],
    }
    write_json(summary, out_dir / "chartevents_validation_summary.json")
    return summary


def select_panel(summaries: list[dict], args: argparse.Namespace) -> dict:
    rows = pd.DataFrame(summaries)
    metrics = [
        "R_Loss_Improvement",
        "DR_Qini_Improvement",
        "Policy_Risk_Reduction_Improvement",
    ]
    rows["positive_gain_count_across_metrics"] = rows[metrics].gt(0).sum(axis=1)
    rank_columns = []
    for metric in metrics:
        rank = f"rank_{metric}"
        rows[rank] = rows[metric].rank(ascending=False)
        rank_columns.append(rank)
    rows["panel_rank_sum"] = rows[rank_columns].sum(axis=1)
    selected = rows.sort_values(
        [
            "positive_gain_count_across_metrics",
            "panel_rank_sum",
            "DR_Qini_Improvement",
            "R_Loss_Improvement",
        ],
        ascending=[False, True, False, False],
    ).iloc[0]
    payload = {
        "selection_dataset": "MIMIC-IV 2017-2019 only",
        "selection_rule": (
            "maximize positive validation gains, then minimize the equal-weight "
            "rank sum of R-loss, DR-Qini, and fixed-budget policy gains"
        ),
        "selected_panel": selected["panel"],
        "selected_learner": selected["learner"],
        "temporal_test_used_for_selection": False,
        "all_candidates": rows.to_dict("records"),
    }
    write_json(payload, args.out_dir / SELECTED_PANEL_FILENAME)
    rows.to_csv(args.out_dir / "chartevents_validation_panel_comparison.csv", index=False)
    return payload


def selected_panel(args: argparse.Namespace) -> str:
    return read_json(args.out_dir / SELECTED_PANEL_FILENAME)["selected_panel"]


def load_validated_roles(args: argparse.Namespace, panel: str) -> dict:
    return read_json(
        panel_result_dir(args, panel) / "mimic_rbc_safety_validated_roles.json"
    )


def analysis_payload(
    frame: pd.DataFrame,
    features: list[str],
    panel: str,
    args: argparse.Namespace,
) -> dict:
    out_dir = panel_result_dir(args, panel)
    roles = load_validated_roles(args, panel)
    lock = read_json(out_dir / "mimic_rbc_locked_spec.json")
    train_raw = frame[frame["era_index"].le(3)].copy()
    evaluation_raw = frame[frame["era_index"].eq(4)].copy()
    train, evaluation, _ = impute_from_training(
        train_raw, evaluation_raw, features
    )
    c_set = tuple(feature for feature in roles["c_proximal"] if feature in features)
    s_set = tuple(
        feature
        for feature in roles["s_precision"]
        if feature in features and feature not in c_set
    )
    tlcd = AdjustmentInterface("RITE_CPlusS", c_set, s_set)
    all_variables = AdjustmentInterface("AllVariables", tuple(features))
    analysis_seed = args.seed + 1
    evaluator = crossfit_training_nuisance(
        evaluation,
        list(tlcd.c_set),
        tlcd.outcome_set,
        analysis_seed + 700,
        folds=5,
    )
    learner = lock["primary_learner"]
    tau = {
        "AllVariables": fit_tau(
            learner, train, evaluation, all_variables, analysis_seed
        ),
        "RITE_CPlusS": fit_tau(
            learner, train, evaluation, tlcd, analysis_seed
        ),
    }
    return {
        "frame": frame,
        "train": train,
        "evaluation": evaluation,
        "feature_cols": features,
        "tlcd": tlcd,
        "all_variables": all_variables,
        "learner": learner,
        "analysis_seed": analysis_seed,
        "evaluator": evaluator,
        "tau": tau,
    }


def chart_profiles(payload: dict) -> pd.DataFrame:
    evaluation = payload["evaluation"].copy()
    benefit = -payload["tau"]["RITE_CPlusS"]
    evaluation["benefit_quartile"] = (
        pd.qcut(pd.Series(benefit).rank(method="first"), 4, labels=False).to_numpy()
        + 1
    )
    features = [feature for feature in PROFILE_CHART_FEATURES if feature in evaluation]
    return evaluation.groupby("benefit_quartile", as_index=False).agg(
        n=("Y", "size"),
        **{f"{feature}_mean": (feature, "mean") for feature in features},
    )


def trend_summary(profiles: pd.DataFrame, hb: pd.DataFrame) -> pd.DataFrame:
    quartiles = profiles["benefit_quartile"].to_numpy(dtype=float)
    records = []
    for metric in ("predicted_benefit_mean", "dr_observed_benefit_mean"):
        values = profiles[metric].to_numpy(dtype=float)
        records.append(
            {
                "metric": metric,
                "q4_minus_q1": float(values[-1] - values[0]),
                "spearman_r": float(spearmanr(quartiles, values).statistic),
                "adjacent_monotonicity_violations": int(np.sum(np.diff(values) < 0)),
            }
        )
    hb_values = hb[hb["group"].str.startswith("benefit_quartile")][
        "aipw_hb_change_difference"
    ].to_numpy(dtype=float)
    records.append(
        {
            "metric": "aipw_hb_change_difference",
            "q4_minus_q1": float(hb_values[-1] - hb_values[0]),
            "spearman_r": float(spearmanr(quartiles, hb_values).statistic),
            "adjacent_monotonicity_violations": int(np.sum(np.diff(hb_values) < 0)),
        }
    )
    return pd.DataFrame(records)


def interpret(
    frame: pd.DataFrame,
    features: list[str],
    panel: str,
    args: argparse.Namespace,
) -> None:
    out_dir = panel_result_dir(args, panel)
    payload = analysis_payload(frame, features, panel, args)
    samples, bootstrap = paired_patient_bootstrap(
        payload, args.bootstrap_draws, args.seed + 10_000
    )
    profiles, allocation = benefit_quartile_profiles(payload)
    chart = chart_profiles(payload)
    policy = policy_budget_curve(payload)
    hb, hb_diagnostics = hemoglobin_mechanism_analysis(payload)
    late = late_transfusion_sensitivity(payload)
    trends = trend_summary(profiles, hb)
    samples.to_csv(out_dir / "chartevents_temporal_bootstrap_draws.csv", index=False)
    bootstrap.to_csv(out_dir / "chartevents_temporal_bootstrap_summary.csv", index=False)
    profiles.to_csv(out_dir / "chartevents_temporal_benefit_profiles.csv", index=False)
    chart.to_csv(out_dir / "chartevents_temporal_chart_profiles.csv", index=False)
    allocation.to_csv(out_dir / "chartevents_temporal_allocation_profile.csv", index=False)
    policy.to_csv(out_dir / "chartevents_temporal_policy_budget_curve.csv", index=False)
    hb.to_csv(out_dir / "chartevents_temporal_hb_mechanism.csv", index=False)
    hb_diagnostics.to_csv(
        out_dir / "chartevents_temporal_hb_observation_diagnostics.csv", index=False
    )
    late.to_csv(out_dir / "chartevents_late_transfusion_sensitivity.csv", index=False)
    trends.to_csv(out_dir / "chartevents_quartile_trend_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["prepare", "develop", "confirm", "robustness", "interpret", "all"],
        default="all",
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--panels", default="all")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-missing", type=float, default=0.70)
    parser.add_argument("--graph-max-rows", type=int, default=10_000)
    parser.add_argument("--cor-quantile", type=float, default=0.75)
    parser.add_argument("--min-abs-corr", type=float, default=0.03)
    parser.add_argument("--ci-theta", type=float, default=0.05)
    parser.add_argument("--max-condition-set-size", type=int, default=2)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--max-corr-neighbors", type=int, default=10)
    parser.add_argument("--max-importance-neighbors", type=int, default=12)
    parser.add_argument("--max-condition-candidates", type=int, default=8)
    parser.add_argument("--max-search-depth", type=int, default=1)
    parser.add_argument("--tlcd-estimators", type=int, default=60)
    parser.add_argument("--robustness-seed-start", type=int, default=4100)
    parser.add_argument("--robustness-repeats", type=int, default=20)
    parser.add_argument("--robustness-test-size", type=float, default=0.40)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--reuse-graph", action="store_true")
    parser.add_argument("--reuse-raw-tlcd", action="store_true")
    parser.add_argument("--rebuild-frame", action="store_true")
    return parser.parse_args()


def requested_panels(args: argparse.Namespace) -> list[str]:
    if args.panels == "all":
        return list(PANEL_DEFINITIONS)
    panels = [value.strip() for value in args.panels.split(",") if value.strip()]
    unknown = sorted(set(panels) - set(PANEL_DEFINITIONS))
    if unknown:
        raise ValueError(f"Unknown panels: {unknown}")
    return panels


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    frame, panels = prepare(args)
    if args.mode == "prepare":
        return

    if args.mode in {"develop", "all"}:
        summaries = [
            develop_panel(frame, panels[panel], panel, args)
            for panel in requested_panels(args)
        ]
        select_panel(summaries, args)
    panel = selected_panel(args)
    features = panels[panel]
    roles = load_validated_roles(args, panel)
    gargs = graph_args(args)
    out_dir = panel_result_dir(args, panel)
    if args.mode in {"confirm", "all"}:
        run_confirmation(frame, features, roles, out_dir, gargs)
    if args.mode in {"robustness", "all"}:
        run_random_robustness(frame, features, roles, out_dir, gargs)
    if args.mode in {"interpret", "all"}:
        interpret(frame, features, panel, args)
    write_json(
        {
            "mode": args.mode,
            "selected_panel": panel,
            "runtime_sec": time.time() - started,
            "completed": True,
        },
        args.out_dir / f"chartevents_{args.mode}_run_manifest.json",
    )


if __name__ == "__main__":
    main()
