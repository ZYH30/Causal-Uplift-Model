import argparse
import copy
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tlcd.tlcd_runner import merge_graphs, run_tlcd, save_json
from tlcd.graph_metrics import complete_graph
from experiments.reproduce_synthetic import fit_predict_stage2
from experiments.run_tlcd_synthetic import (
    prune_deeper_edges_by_shallow_nonadjacency,
    rescue_outcome_edges_from_treatment_parents,
    rescue_post_treatment_children,
    rescue_precision_outcome_edges,
    rescue_treatment_parent_edges,
)
from models.class_transform import ClassTransformEstimator
from models.meta_learners import SLearner
from rite_framework import RITEPropensityStage


CYCLES = [(2011, "G"), (2013, "H"), (2015, "I"), (2017, "J")]
TABLES = ["DEMO", "RXQ_RX", "TRIGLY", "TCHOL", "HDL", "BPX", "BMX", "GHB", "DIQ", "BPQ", "MCQ", "SMQ", "PAQ"]
STATINS = ["atorvastatin", "simvastatin", "rosuvastatin", "pravastatin", "lovastatin", "fluvastatin", "pitavastatin"]


TASKS = {
    "antihypertensive_sbp": {
        "description": "Current antihypertensive medication use -> mean systolic blood pressure among adults told they had hypertension.",
        "outcome": "mean_sbp",
        "min_age": 20,
        "lower_is_better": True,
    },
    "chol_med_tchol": {
        "description": "Self-reported cholesterol-lowering medication use -> total cholesterol among adults told they had high cholesterol.",
        "outcome": "LBXTC",
        "min_age": 40,
        "lower_is_better": True,
    },
}


BASE_FEATURES = [
    "cycle_year",
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
    "DMDEDUC2",
    "DMDMARTL",
    "INDFMPIR",
    "RIDEXMON",
    "BMXBMI",
    "BMXWAIST",
    "BMXWT",
    "BMXHT",
    "BPXPLS",
    "mean_dbp",
    "LBXGH",
    "DIQ010",
    "DIQ050",
    "DIQ070",
    "BPQ020",
    "BPQ080",
    "MCQ160B",
    "MCQ160C",
    "MCQ160E",
    "MCQ160F",
    "MCQ220",
    "SMQ020",
    "SMQ040",
    "PAQ605",
    "PAQ620",
    "PAQ635",
    "PAQ650",
    "PAQ665",
    "PAD680",
]


def xpt_url(year: int, table: str, suffix: str) -> str:
    return f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{table}_{suffix}.XPT"


def download_table(cache_dir: Path, year: int, table: str, suffix: str) -> Path | None:
    out = cache_dir / str(year) / f"{table}_{suffix}.XPT"
    if out.exists() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = xpt_url(year, table, suffix)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if not data.startswith(b"HEADER RECORD"):
            return None
        out.write_bytes(data)
        return out
    except Exception:
        return None


def read_xpt(cache_dir: Path, year: int, table: str, suffix: str) -> pd.DataFrame | None:
    path = download_table(cache_dir, year, table, suffix)
    if path is None:
        return None
    df = pd.read_sas(path, format="xport")
    df["cycle_year"] = year
    return df


def decode_series(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: x.decode("latin1") if isinstance(x, bytes) else str(x))


def valid_code(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.mask(out.isin([7, 9, 77, 99, 777, 999]))
    return out


def mean_valid(df: pd.DataFrame, cols: list[str], low: float, high: float) -> pd.Series:
    vals = df[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    vals = vals.where((vals >= low) & (vals <= high))
    return vals.mean(axis=1)


def aggregate_rx(rx: pd.DataFrame) -> pd.DataFrame:
    if rx is None or rx.empty:
        return pd.DataFrame(columns=["SEQN", "statin_rx", "metformin_rx"])
    drug = decode_series(rx.get("RXDDRUG", pd.Series(index=rx.index, dtype=object))).str.lower()
    statin = drug.str.contains("|".join(STATINS), na=False)
    metformin = drug.str.contains("metformin", na=False)
    out = pd.DataFrame(
        {
            "SEQN": rx["SEQN"],
            "statin_rx_row": statin.astype(int),
            "metformin_rx_row": metformin.astype(int),
        }
    )
    out = out.groupby("SEQN", as_index=False).max()
    return out.rename(columns={"statin_rx_row": "statin_rx", "metformin_rx_row": "metformin_rx"})


def load_cycle(cache_dir: Path, year: int, suffix: str) -> pd.DataFrame:
    frames = {table: read_xpt(cache_dir, year, table, suffix) for table in TABLES}
    demo = frames["DEMO"]
    keep = [
        "SEQN",
        "cycle_year",
        "RIDSTATR",
        "RIAGENDR",
        "RIDAGEYR",
        "RIDRETH3",
        "RIDEXMON",
        "DMDEDUC2",
        "DMDMARTL",
        "INDFMPIR",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
    ]
    df = demo[[c for c in keep if c in demo.columns]].copy()
    rx = aggregate_rx(frames["RXQ_RX"])
    df = df.merge(rx, on="SEQN", how="left")
    df[["statin_rx", "metformin_rx"]] = df[["statin_rx", "metformin_rx"]].fillna(0).astype(int)

    merge_specs = {
        "TRIGLY": ["SEQN", "LBDLDL", "LBXTR"],
        "TCHOL": ["SEQN", "LBXTC", "LBDTCSI"],
        "HDL": ["SEQN", "LBDHDD", "LBDHDDSI"],
        "BMX": ["SEQN", "BMXBMI", "BMXWAIST", "BMXWT", "BMXHT"],
        "GHB": ["SEQN", "LBXGH"],
        "DIQ": ["SEQN", "DIQ010", "DIQ050", "DIQ070", "DID040"],
        "BPQ": ["SEQN", "BPQ020", "BPQ050A", "BPQ080", "BPQ100D"],
        "MCQ": ["SEQN", "MCQ160B", "MCQ160C", "MCQ160E", "MCQ160F", "MCQ220"],
        "SMQ": ["SEQN", "SMQ020", "SMQ040"],
        "PAQ": ["SEQN", "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665", "PAD680"],
    }
    for table, cols in merge_specs.items():
        data = frames.get(table)
        if data is None:
            continue
        df = df.merge(data[[c for c in cols if c in data.columns]], on="SEQN", how="left")

    bpx = frames.get("BPX")
    if bpx is not None:
        bpx_use = bpx[["SEQN"]].copy()
        bpx_use["mean_sbp"] = mean_valid(bpx, ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], 40, 300)
        bpx_use["mean_dbp"] = mean_valid(bpx, ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"], 20, 200)
        if "BPXPLS" in bpx.columns:
            bpx_use["BPXPLS"] = pd.to_numeric(bpx["BPXPLS"], errors="coerce")
        df = df.merge(bpx_use, on="SEQN", how="left")
    return df


def load_all_nhanes(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for year, suffix in CYCLES:
        rows.append(load_cycle(cache_dir, year, suffix))
    df = pd.concat(rows, ignore_index=True)
    for col in df.columns:
        if col == "SEQN":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    code_cols = [
        "RIAGENDR",
        "RIDRETH3",
        "RIDEXMON",
        "DMDEDUC2",
        "DMDMARTL",
        "DIQ010",
        "DIQ050",
        "DIQ070",
        "BPQ020",
        "BPQ050A",
        "BPQ080",
        "BPQ100D",
        "MCQ160B",
        "MCQ160C",
        "MCQ160E",
        "MCQ160F",
        "MCQ220",
        "SMQ020",
        "SMQ040",
        "PAQ605",
        "PAQ620",
        "PAQ635",
        "PAQ650",
        "PAQ665",
    ]
    for col in code_cols:
        if col in df.columns:
            df[col] = valid_code(df[col])
    return df


def task_frame(df: pd.DataFrame, task: str, max_missing: float = 0.35) -> tuple[pd.DataFrame, dict]:
    cfg = TASKS[task]
    data = df.copy()
    data = data[(data["RIDSTATR"] == 2) & (data["RIDAGEYR"] >= cfg["min_age"])]
    if task == "antihypertensive_sbp":
        data = data[data["BPQ020"] == 1]
        data["T"] = (data["BPQ050A"] == 1).astype(int)
        data["Y"] = data["mean_sbp"]
        # BPQ020 defines the cohort and is constant after filtering. Keeping it
        # in graph/nuisance inputs adds no information and can destabilize
        # variable screening.
        exclude = {"mean_sbp", "mean_dbp", "BPQ020", "BPQ050A"}
    elif task == "chol_med_tchol":
        data = data[(data["BPQ080"] == 1) & (data["BPQ100D"].isin([1, 2]))]
        data["T"] = (data["BPQ100D"] == 1).astype(int)
        data["Y"] = data["LBXTC"]
        exclude = {"LBDLDL", "LBXTR", "LBXTC", "LBDTCSI", "LBDHDD", "LBDHDDSI", "BPQ080", "BPQ100D", "statin_rx"}
    else:
        raise ValueError(task)

    features = [c for c in BASE_FEATURES if c in data.columns and c not in exclude]
    cols = ["SEQN", "T", "Y"] + features
    out = data[cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["T", "Y"])
    out = out[(out["T"].isin([0, 1])) & out["Y"].notna()]
    feature_missing = out[features].isna().mean()
    features = [c for c in features if feature_missing.get(c, 1.0) <= max_missing]
    out = out[["SEQN", "T", "Y"] + features].copy()

    for col in features:
        if out[col].isna().any():
            nunique = out[col].nunique(dropna=True)
            if nunique <= 20:
                out[col] = out[col].fillna(out[col].mode(dropna=True).iloc[0])
            else:
                out[col] = out[col].fillna(out[col].median())
    out = out.dropna().reset_index(drop=True)
    meta = {
        "task": task,
        "description": cfg["description"],
        "outcome": cfg["outcome"],
        "lower_is_better": cfg["lower_is_better"],
        "n": int(len(out)),
        "treated": int(out["T"].sum()),
        "control": int((1 - out["T"]).sum()),
        "treatment_rate": float(out["T"].mean()) if len(out) else np.nan,
        "n_features": len(features),
        "features": features,
    }
    return out, meta


def make_models(seed: int, estimators: int = 80):
    return {
        "S-Learner": SLearner(
            base_estimator=XGBRegressor(
                n_estimators=estimators,
                max_depth=3,
                learning_rate=0.05,
                verbosity=0,
                random_state=seed,
            )
        ),
        "ClassTransform": ClassTransformEstimator(
            base_estimator=XGBRegressor(
                n_estimators=estimators,
                max_depth=3,
                learning_rate=0.05,
                verbosity=0,
                random_state=seed,
            )
        ),
    }


def fit_nuisance(
    train: pd.DataFrame,
    test: pd.DataFrame,
    propensity_feature_cols: list[str],
    seed: int,
    outcome_feature_cols: list[str] | None = None,
    train_sample_weight: np.ndarray | None = None,
):
    """Fit evaluation nuisances with separate treatment/outcome covariates.

    The propensity model uses only the TLCD C-set when supplied by the RITE
    experiment. Outcome regressions may additionally use S-precision.
    """
    if not propensity_feature_cols:
        raise ValueError("propensity_feature_cols must contain at least one C-set variable")
    outcome_feature_cols = outcome_feature_cols or propensity_feature_cols
    x_e_train = train[propensity_feature_cols]
    x_e_test = test[propensity_feature_cols]
    x_m_train = train[outcome_feature_cols]
    x_m_test = test[outcome_feature_cols]
    e_model = LGBMClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, num_leaves=15, min_child_samples=30, random_state=seed, n_jobs=1, verbose=-1)
    fit_kwargs = {"sample_weight": train_sample_weight} if train_sample_weight is not None else {}
    e_model.fit(x_e_train, train["T"], **fit_kwargs)
    e_test = np.clip(e_model.predict_proba(x_e_test)[:, 1], 0.05, 0.95)

    mu_model = LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, num_leaves=15, min_child_samples=30, random_state=seed, n_jobs=1, verbose=-1)
    mu_model.fit(x_m_train, train["Y"], **fit_kwargs)
    mu_test = mu_model.predict(x_m_test)

    m0 = LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=3, num_leaves=15, min_child_samples=30, random_state=seed, n_jobs=1, verbose=-1)
    m1 = copy.deepcopy(m0)
    mask0 = train["T"].to_numpy() == 0
    mask1 = ~mask0
    weight0 = train_sample_weight[mask0] if train_sample_weight is not None else None
    weight1 = train_sample_weight[mask1] if train_sample_weight is not None else None
    kwargs0 = {"sample_weight": weight0} if weight0 is not None else {}
    kwargs1 = {"sample_weight": weight1} if weight1 is not None else {}
    m0.fit(train.loc[mask0, outcome_feature_cols], train.loc[mask0, "Y"], **kwargs0)
    m1.fit(train.loc[mask1, outcome_feature_cols], train.loc[mask1, "Y"], **kwargs1)
    m0_test = m0.predict(x_m_test)
    m1_test = m1.predict(x_m_test)
    return e_test, mu_test, m0_test, m1_test


def r_loss(y, t, tau, mu, e) -> float:
    residual = (y - mu) - (t - e) * tau
    return float(np.mean(residual**2))


def policy_outcome(y, t, policy, e, m0, m1) -> float:
    p_a = np.where(policy == 1, e, 1 - e)
    m_a = np.where(policy == 1, m1, m0)
    observed = (t == policy).astype(float)
    dr = m_a + observed * (y - m_a) / np.clip(p_a, 0.05, 0.95)
    return float(np.mean(dr))


def dr_effect_scores(y, t, e, m0, m1) -> np.ndarray:
    e = np.clip(e, 0.05, 0.95)
    return m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)


def uplift_ranking_metrics(
    y,
    t,
    tau_pred,
    e,
    m0,
    m1,
    lower_is_better: bool,
    observed_treat_rate: float,
) -> tuple[dict, pd.DataFrame]:
    """DR-AUUC/Qini and policy value curve for observational uplift ranking."""

    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    tau_pred = np.asarray(tau_pred, dtype=float)
    tau_dr = dr_effect_scores(y, t, e, m0, m1)
    pred_benefit = -tau_pred if lower_is_better else tau_pred
    dr_benefit = -tau_dr if lower_is_better else tau_dr

    order = np.argsort(-pred_benefit)
    sorted_benefit = dr_benefit[order]
    n = len(sorted_benefit)
    fractions = np.arange(1, n + 1) / n
    cumulative_gain = np.cumsum(sorted_benefit) / n
    random_gain = fractions * float(np.mean(dr_benefit))
    dr_auuc = float(np.trapezoid(cumulative_gain, fractions))
    dr_qini = float(np.trapezoid(cumulative_gain - random_gain, fractions))

    grid = sorted(set([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, observed_treat_rate, 0.6, 0.7, 0.8, 0.9, 1.0]))
    curve_rows = []
    for frac in grid:
        k = int(round(frac * n))
        policy = np.zeros(n, dtype=int)
        if k > 0:
            policy[order[:k]] = 1
        value = policy_outcome(y, t, policy, e, m0, m1)
        gain = float(np.sum(dr_benefit[order[:k]]) / n) if k > 0 else 0.0
        curve_rows.append(
            {
                "fraction": float(frac),
                "n_treated_policy": int(k),
                "treat_rate_policy": float(policy.mean()),
                "policy_outcome": value,
                "dr_cumulative_gain": gain,
            }
        )
    curve = pd.DataFrame(curve_rows)
    if lower_is_better:
        best_idx = curve["policy_outcome"].idxmin()
    else:
        best_idx = curve["policy_outcome"].idxmax()
    budget_idx = (curve["fraction"] - observed_treat_rate).abs().idxmin()
    metrics = {
        "DR_AUUC": dr_auuc,
        "DR_Qini": dr_qini,
        "DR_Gain_30pct": float(curve.loc[(curve["fraction"] - 0.3).abs().idxmin(), "dr_cumulative_gain"]),
        "DR_Gain_50pct": float(curve.loc[(curve["fraction"] - 0.5).abs().idxmin(), "dr_cumulative_gain"]),
        "PolicyOutcome_BestGrid": float(curve.loc[best_idx, "policy_outcome"]),
        "PolicyTreatRate_BestGrid": float(curve.loc[best_idx, "treat_rate_policy"]),
        "PolicyOutcome_ObservedBudget": float(curve.loc[budget_idx, "policy_outcome"]),
        "PolicyTreatRate_ObservedBudget": float(curve.loc[budget_idx, "treat_rate_policy"]),
    }
    return metrics, curve


def overlap_diagnostics(e: np.ndarray, t: np.ndarray) -> dict:
    e = np.asarray(e, dtype=float)
    t = np.asarray(t, dtype=int)
    p_t = float(t.mean())
    weights = np.where(t == 1, p_t / np.clip(e, 0.05, 0.95), (1 - p_t) / np.clip(1 - e, 0.05, 0.95))
    diagnostics = {
        "treatment_rate": p_t,
        "propensity_min": float(np.min(e)),
        "propensity_p01": float(np.quantile(e, 0.01)),
        "propensity_p05": float(np.quantile(e, 0.05)),
        "propensity_p50": float(np.quantile(e, 0.50)),
        "propensity_p95": float(np.quantile(e, 0.95)),
        "propensity_p99": float(np.quantile(e, 0.99)),
        "propensity_max": float(np.max(e)),
        "overlap_share_0p05_0p95": float(np.mean((e >= 0.05) & (e <= 0.95))),
        "ipw_ess": float((weights.sum() ** 2) / np.sum(weights**2)),
        "ipw_ess_ratio": float((weights.sum() ** 2) / np.sum(weights**2) / len(weights)),
        "ipw_weight_p95": float(np.quantile(weights, 0.95)),
        "ipw_weight_p99": float(np.quantile(weights, 0.99)),
        "ipw_weight_max": float(np.max(weights)),
    }
    for group in [0, 1]:
        group_e = e[t == group]
        if len(group_e) == 0:
            continue
        diagnostics[f"propensity_t{group}_p05"] = float(np.quantile(group_e, 0.05))
        diagnostics[f"propensity_t{group}_p50"] = float(np.quantile(group_e, 0.50))
        diagnostics[f"propensity_t{group}_p95"] = float(np.quantile(group_e, 0.95))
    return diagnostics


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights * values) / np.sum(weights))


def weighted_var(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    mean = weighted_mean(values, weights)
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.mean((values - mean) ** 2))
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(weights * (values - mean) ** 2) / np.sum(weights))


def balance_table(df: pd.DataFrame, covariates: list[str], weights: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    t = df["T"].to_numpy().astype(int)
    for col in covariates:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask0 = t == 0
        mask1 = t == 1
        w0 = weights[mask0] if weights is not None else None
        w1 = weights[mask1] if weights is not None else None
        m0 = weighted_mean(values[mask0], w0)
        m1 = weighted_mean(values[mask1], w1)
        v0 = weighted_var(values[mask0], w0)
        v1 = weighted_var(values[mask1], w1)
        pooled = np.sqrt(max((v0 + v1) / 2, 1e-12))
        rows.append(
            {
                "covariate": col,
                "mean_t0": m0,
                "mean_t1": m1,
                "smd": float((m1 - m0) / pooled),
                "abs_smd": float(abs((m1 - m0) / pooled)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False)


def aipw_scores_crossfit(
    df: pd.DataFrame,
    propensity_feature_cols: list[str],
    seed: int,
    outcome_feature_cols: list[str] | None = None,
    sample_weight_col: str | None = None,
) -> np.ndarray:
    outcome_feature_cols = outcome_feature_cols or propensity_feature_cols
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    scores = np.zeros(len(df))
    for train_idx, test_idx in kf.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        e, _, m0, m1 = fit_nuisance(
            train,
            test,
            propensity_feature_cols,
            seed,
            outcome_feature_cols=outcome_feature_cols,
            train_sample_weight=(
                train[sample_weight_col].to_numpy(dtype=float) if sample_weight_col else None
            ),
        )
        y_test = test["Y"].to_numpy()
        t_test = test["T"].to_numpy()
        scores[test_idx] = m1 - m0 + t_test * (y_test - m1) / e - (1 - t_test) * (y_test - m0) / (1 - e)
    return scores


def aipw_ate_crossfit(
    df: pd.DataFrame,
    propensity_feature_cols: list[str],
    seed: int,
    outcome_feature_cols: list[str] | None = None,
) -> dict:
    scores = aipw_scores_crossfit(
        df,
        propensity_feature_cols,
        seed,
        outcome_feature_cols=outcome_feature_cols,
    )
    ate = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    return {"aipw_ate": ate, "aipw_se": se, "aipw_ci_low": ate - 1.96 * se, "aipw_ci_high": ate + 1.96 * se}


def screen_tasks(args):
    data = load_all_nhanes(Path(args.cache_dir))
    rows = []
    task_metas = {}
    for task in TASKS:
        df_task, meta = task_frame(data, task, max_missing=args.max_missing)
        feature_cols = [c for c in df_task.columns if c not in {"SEQN", "T", "Y"}]
        raw_diff = float(df_task.loc[df_task["T"] == 1, "Y"].mean() - df_task.loc[df_task["T"] == 0, "Y"].mean())
        ate = aipw_ate_crossfit(df_task, feature_cols, args.seed) if meta["treated"] >= 50 and meta["control"] >= 50 else {}
        row = {**meta, "raw_diff_treated_minus_control": raw_diff, **ate}
        rows.append(row)
        task_metas[task] = meta
        out_path = Path(args.out_dir) / f"nhanes_{task}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_task.to_csv(out_path, index=False)
    result = pd.DataFrame(rows)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(args.out_dir) / "nhanes_task_screening.csv", index=False)
    save_json(task_metas, str(Path(args.out_dir) / "nhanes_task_metadata.json"))
    print(result[["task", "n", "treated", "control", "treatment_rate", "n_features", "raw_diff_treated_minus_control", "aipw_ate", "aipw_se"]].to_string(index=False))
    return result


def run_tlcd_graph(train: pd.DataFrame, feature_cols: list[str], args) -> tuple[dict, list[dict]]:
    observed_cols = ["T", "Y"] + feature_cols
    graph_results = []
    for target in ["Y", "T"]:
        print(f"\n=== Empirical TLCD target={target} ===", flush=True)
        result = run_tlcd(
            train[observed_cols],
            target=target,
            output_dir=args.out_dir,
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
            lgb_estimators=args.tlcd_lgb_estimators,
            lgb_max_depth=3,
            lgb_num_leaves=15,
            lgb_min_child_samples=30,
            llm_mode="off",
            direction_strategy="task_adaptive",
            random_state=args.seed,
            variable_order=None,
            verbose=False,
        )
        graph_results.append(result)
        print(f"target={target}, edges={sum(len(v) for v in result['graph'].values())}, runtime={result['runtime_sec']:.1f}s", flush=True)
    merged = merge_graphs(graph_results, nodes=observed_cols, prefer_edges={("T", "Y")})
    learned_graph = merged["graph"]

    rescue_args = SimpleNamespace(
        seed=args.seed,
        ci_theta=args.ci_theta,
        post_treatment_rescue_lgb_estimators=args.tlcd_lgb_estimators,
        rescue_lgb_estimators=args.rescue_lgb_estimators,
        tlcd_lgb_max_depth=3,
        tlcd_lgb_num_leaves=15,
        tlcd_lgb_min_child_samples=30,
        rescue_min_importance_ratio=0.05,
        rescue_max_rank=12,
        precision_rescue_min_importance_ratio=0.10,
        precision_rescue_max_rank=8,
        precision_rescue_max_treatment_assoc=0.05,
        treatment_rescue_min_importance_ratio=0.30,
        treatment_rescue_max_rank=8,
        treatment_rescue_min_assoc=0.03,
    )
    learned_graph, pruned_edges = prune_deeper_edges_by_shallow_nonadjacency(learned_graph, graph_results, rescue_args)
    learned_graph, treatment_rescue = rescue_treatment_parent_edges(train, learned_graph, observed_cols, "T", "Y", rescue_args)
    learned_graph, outcome_rescue = rescue_outcome_edges_from_treatment_parents(train, learned_graph, observed_cols, "T", "Y", rescue_args)
    learned_graph, precision_rescue = rescue_precision_outcome_edges(train, learned_graph, observed_cols, "T", "Y", rescue_args)
    learned_graph, post_rescue = rescue_post_treatment_children(train, learned_graph, observed_cols, "T", "Y", rescue_args)
    learned_graph, sanitized_edges = sanitize_uplift_graph(learned_graph, observed_cols, post_rescue)
    meta = {
        **merged,
        "pruned_edges": pruned_edges,
        "treatment_rescue_edges": treatment_rescue,
        "outcome_rescue_edges": outcome_rescue,
        "precision_rescue_edges": precision_rescue,
        "post_treatment_rescue_edges": post_rescue,
        "sanitized_edges": sanitized_edges,
    }
    return complete_graph(learned_graph, observed_cols), graph_results, meta


def sanitize_uplift_graph(graph: dict, observed_cols: list[str], post_rescue_edges: list[dict]) -> tuple[dict, list[dict]]:
    """Apply minimal task-level constraints before RITE feature extraction.

    The statistical direction voter is intentionally run without temporal order.
    For downstream RITE, however, Y must not be an upstream cause of baseline
    covariates/T, and T must not cause ordinary covariates. Post-treatment
    variables detected statistically are exempted.
    """

    post_children = {edge["child"] for edge in post_rescue_edges if edge.get("child") not in {"T", "Y"}}
    sanitized = {}
    removed = []
    for child in observed_cols:
        clean_parents = set()
        for parent in graph.get(child, []):
            if parent == child:
                removed.append({"parent": parent, "child": child, "reason": "self_edge"})
                continue
            if child == "T" and parent in {"T", "Y"}:
                removed.append({"parent": parent, "child": child, "reason": "outcome_or_self_parent_of_treatment"})
                continue
            if child == "Y":
                if parent == "Y":
                    removed.append({"parent": parent, "child": child, "reason": "self_edge"})
                    continue
                clean_parents.add(parent)
                continue
            if child not in post_children and parent in {"T", "Y"}:
                removed.append({"parent": parent, "child": child, "reason": "task_temporal_sanitizer"})
                continue
            clean_parents.add(parent)
        sanitized[child] = sorted(clean_parents)
    return sanitized, removed


def compute_stage1(df_train, graph, args):
    deconfounder = RITEPropensityStage(
        treatment_var="T",
        target_var="Y",
        ancestors_graph=graph,
        method="IPW",
        conf_type=getattr(args, "conf_type", "proximal"),
        is_precise=True,
    )
    df_stg1, weights, p_scores, c_set, s_prec, p_model = deconfounder.fit_transform(
        df_train,
        n_trials=0,
        random_state=args.seed,
    )
    return {"df": df_stg1, "weights": weights, "p_scores": p_scores, "c_set": c_set, "s_prec": s_prec, "p_model": p_model}


def run_task(args):
    started = time.time()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_data = load_all_nhanes(Path(args.cache_dir))
    df_task, meta = task_frame(all_data, args.task, max_missing=args.max_missing)
    if args.max_rows and len(df_task) > args.max_rows:
        df_task = df_task.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
    feature_cols = [c for c in df_task.columns if c not in {"SEQN", "T", "Y"}]
    train, test = train_test_split(df_task, test_size=args.test_size, random_state=args.seed, stratify=df_task["T"])

    raw_diff = float(df_task.loc[df_task["T"] == 1, "Y"].mean() - df_task.loc[df_task["T"] == 0, "Y"].mean())
    ate_all_baseline = aipw_ate_crossfit(df_task, feature_cols, args.seed)
    if args.reuse_graph_json:
        learned_graph = json.loads(Path(args.reuse_graph_json).read_text())
        graph_results = []
        graph_meta = {"source": args.reuse_graph_json, "mode": "reuse_graph_json"}
    else:
        learned_graph, graph_results, graph_meta = run_tlcd_graph(train, feature_cols, args)
    save_json(learned_graph, str(Path(args.out_dir) / f"nhanes_{args.task}_tlcd_graph.json"))
    save_json(graph_meta, str(Path(args.out_dir) / f"nhanes_{args.task}_tlcd_graph_meta.json"))

    all_decisions = []
    for result in graph_results:
        for decision in result["edge_decisions"]:
            row = dict(decision)
            row["target_run"] = result["target"]
            row["ci_tests"] = json.dumps(row.get("ci_tests", []), ensure_ascii=False)
            all_decisions.append(row)
    pd.DataFrame(all_decisions).to_csv(Path(args.out_dir) / f"nhanes_{args.task}_tlcd_edge_decisions.csv", index=False)

    stage1 = compute_stage1(train, learned_graph, args)
    c_set = sorted(stage1["c_set"])
    s_prec = sorted(stage1["s_prec"])
    outcome_c_plus_s = sorted(set(c_set) | set(s_prec))
    aipw_c_only = aipw_ate_crossfit(df_task, c_set, args.seed, outcome_feature_cols=c_set)
    aipw_c_plus_s = aipw_ate_crossfit(
        df_task,
        c_set,
        args.seed,
        outcome_feature_cols=outcome_c_plus_s,
    )
    pd.DataFrame(
        [
            {"Estimand": "AIPW_AllBaseline", "PropensityFeatures": ";".join(feature_cols), "OutcomeFeatures": ";".join(feature_cols), **ate_all_baseline},
            {"Estimand": "AIPW_RITE_COnly", "PropensityFeatures": ";".join(c_set), "OutcomeFeatures": ";".join(c_set), **aipw_c_only},
            {"Estimand": "AIPW_RITE_CPlusS", "PropensityFeatures": ";".join(c_set), "OutcomeFeatures": ";".join(outcome_c_plus_s), **aipw_c_plus_s},
        ]
    ).to_csv(Path(args.out_dir) / f"nhanes_{args.task}_aipw_estimands.csv", index=False)

    # DR uplift evaluation follows the same identification contract as RITE:
    # e(C) for treatment assignment and m(C,S) for outcome prediction.
    e_test, mu_test, m0_test, m1_test = fit_nuisance(
        train,
        test,
        c_set,
        args.seed,
        outcome_feature_cols=outcome_c_plus_s,
    )
    overlap = overlap_diagnostics(stage1["p_scores"], train["T"].to_numpy())
    balance_pre = balance_table(train, feature_cols)
    balance_post = balance_table(train, feature_cols, stage1["weights"])
    balance_c_pre = balance_table(train, c_set)
    balance_c_post = balance_table(train, c_set, stage1["weights"])
    balance_pre.to_csv(Path(args.out_dir) / f"nhanes_{args.task}_balance_pre.csv", index=False)
    balance_post.to_csv(Path(args.out_dir) / f"nhanes_{args.task}_balance_ipw.csv", index=False)
    balance_c_pre.to_csv(Path(args.out_dir) / f"nhanes_{args.task}_balance_c_pre.csv", index=False)
    balance_c_post.to_csv(Path(args.out_dir) / f"nhanes_{args.task}_balance_c_ipw.csv", index=False)

    rows = []
    curve_rows = []
    policies = {
        "Treat_All": np.ones(len(test), dtype=int),
        "Treat_None": np.zeros(len(test), dtype=int),
    }
    for policy_name, policy in policies.items():
        rows.append(
            {
                "Model": policy_name,
                "Method": "PolicyReference",
                "R_Loss": np.nan,
                "PolicyOutcome": policy_outcome(test["Y"].to_numpy(), test["T"].to_numpy(), policy, e_test, m0_test, m1_test),
                "TauMean": np.nan,
                "TauStd": np.nan,
            }
        )

    for model_name, base_model in make_models(args.seed, args.uplift_estimators).items():
        model = copy.deepcopy(base_model)
        model.fit(train[feature_cols].values, train["Y"].values, train["T"].values)
        tau = model.predict(test[feature_cols].values)
        policy = (tau < 0).astype(int) if TASKS[args.task]["lower_is_better"] else (tau > 0).astype(int)
        ranking_metrics, curve = uplift_ranking_metrics(
            test["Y"].to_numpy(),
            test["T"].to_numpy(),
            tau,
            e_test,
            m0_test,
            m1_test,
            lower_is_better=TASKS[args.task]["lower_is_better"],
            observed_treat_rate=float(test["T"].mean()),
        )
        curve["Model"] = model_name
        curve["Method"] = "Naive_AllFeatures"
        curve_rows.append(curve)
        rows.append(
            {
                "Model": model_name,
                "Method": "Naive_AllFeatures",
                "R_Loss": r_loss(test["Y"].to_numpy(), test["T"].to_numpy(), tau, mu_test, e_test),
                "PolicyOutcome": policy_outcome(test["Y"].to_numpy(), test["T"].to_numpy(), policy, e_test, m0_test, m1_test),
                "TauMean": float(np.mean(tau)),
                "TauStd": float(np.std(tau)),
                "TreatRatePolicy": float(policy.mean()),
                **ranking_metrics,
            }
        )

        for use_precision, method in [(False, "RITE_COnly"), (True, "RITE_CPlusS")]:
            try:
                tau = fit_predict_stage2(base_model, test, "T", "Y", stage1, use_precision=use_precision, args=args)
                policy = (tau < 0).astype(int) if TASKS[args.task]["lower_is_better"] else (tau > 0).astype(int)
                ranking_metrics, curve = uplift_ranking_metrics(
                    test["Y"].to_numpy(),
                    test["T"].to_numpy(),
                    tau,
                    e_test,
                    m0_test,
                    m1_test,
                    lower_is_better=TASKS[args.task]["lower_is_better"],
                    observed_treat_rate=float(test["T"].mean()),
                )
                curve["Model"] = model_name
                curve["Method"] = method
                curve_rows.append(curve)
                rows.append(
                    {
                        "Model": model_name,
                        "Method": method,
                        "R_Loss": r_loss(test["Y"].to_numpy(), test["T"].to_numpy(), tau, mu_test, e_test),
                        "PolicyOutcome": policy_outcome(test["Y"].to_numpy(), test["T"].to_numpy(), policy, e_test, m0_test, m1_test),
                        "TauMean": float(np.mean(tau)),
                        "TauStd": float(np.std(tau)),
                        "TreatRatePolicy": float(policy.mean()),
                        **ranking_metrics,
                    }
                )
            except Exception as exc:
                rows.append({"Model": model_name, "Method": method, "Error": str(exc)})

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(Path(args.out_dir) / f"nhanes_{args.task}_uplift_eval.csv", index=False)
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(
            Path(args.out_dir) / f"nhanes_{args.task}_policy_value_curve.csv",
            index=False,
        )
    run_meta = {
        **meta,
        "raw_diff_treated_minus_control": raw_diff,
        # Main AIPW row is theory-aligned: e(C), m(C,S).
        **aipw_c_plus_s,
        "aipw_estimands": {
            "AIPW_AllBaseline": ate_all_baseline,
            "AIPW_RITE_COnly": aipw_c_only,
            "AIPW_RITE_CPlusS": aipw_c_plus_s,
        },
        "aipw_main_estimand": "AIPW_RITE_CPlusS",
        "evaluation_propensity_features": c_set,
        "evaluation_outcome_features": outcome_c_plus_s,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "stage1_c_set": stage1.get("c_set"),
        "stage1_s_prec": stage1.get("s_prec"),
        "overlap": overlap,
        "balance_pre_max_abs_smd": float(balance_pre["abs_smd"].max()) if not balance_pre.empty else np.nan,
        "balance_ipw_max_abs_smd": float(balance_post["abs_smd"].max()) if not balance_post.empty else np.nan,
        "balance_pre_mean_abs_smd": float(balance_pre["abs_smd"].mean()) if not balance_pre.empty else np.nan,
        "balance_ipw_mean_abs_smd": float(balance_post["abs_smd"].mean()) if not balance_post.empty else np.nan,
        "balance_c_pre_max_abs_smd": float(balance_c_pre["abs_smd"].max()) if not balance_c_pre.empty else np.nan,
        "balance_c_ipw_max_abs_smd": float(balance_c_post["abs_smd"].max()) if not balance_c_post.empty else np.nan,
        "balance_c_pre_mean_abs_smd": float(balance_c_pre["abs_smd"].mean()) if not balance_c_pre.empty else np.nan,
        "balance_c_ipw_mean_abs_smd": float(balance_c_post["abs_smd"].mean()) if not balance_c_post.empty else np.nan,
        "runtime_sec": time.time() - started,
    }
    save_json(run_meta, str(Path(args.out_dir) / f"nhanes_{args.task}_run_meta.json"))
    print("\n=== Task meta ===")
    print(json.dumps(run_meta, indent=2, ensure_ascii=False, default=str))
    print("\n=== Uplift evaluation ===")
    print(eval_df.to_string(index=False))
    return eval_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["screen", "run-task"], default="screen")
    parser.add_argument("--task", choices=list(TASKS), default="antihypertensive_sbp")
    parser.add_argument("--cache-dir", default="data/nhanes_raw")
    parser.add_argument("--out-dir", default="results/nhanes_empirical")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--max-missing", type=float, default=0.35)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--cor-quantile", type=float, default=0.75)
    parser.add_argument("--min-abs-corr", type=float, default=0.03)
    parser.add_argument("--ci-theta", type=float, default=0.05)
    parser.add_argument("--max-condition-set-size", type=int, default=2)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--max-corr-neighbors", type=int, default=10)
    parser.add_argument("--max-importance-neighbors", type=int, default=12)
    parser.add_argument("--max-condition-candidates", type=int, default=10)
    parser.add_argument("--max-search-depth", type=int, default=1)
    parser.add_argument("--tlcd-lgb-estimators", type=int, default=80)
    parser.add_argument("--rescue-lgb-estimators", type=int, default=200)
    parser.add_argument("--uplift-estimators", type=int, default=80)
    parser.add_argument("--n-trials", type=int, default=0)
    parser.add_argument("--reuse-graph-json", default="")
    parser.add_argument("--conf-type", choices=["proximal", "total", "root"], default="proximal")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.mode == "screen":
        screen_tasks(parsed)
    else:
        run_task(parsed)
