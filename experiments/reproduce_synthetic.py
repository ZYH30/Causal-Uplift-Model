import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_generation import generate_complex_environment
from models.class_transform import ClassTransformEstimator
from models.meta_learners import SLearner, TLearner, XLearner
from models.tree import UniversalUpliftForest
from rite_framework import RITEPropensityStage, RITETreatmentEffectStage


def build_models(args):
    return {
        "S-Learner": SLearner(
            base_estimator=XGBRegressor(
                n_estimators=args.estimators,
                max_depth=args.max_depth,
                verbosity=0,
                random_state=args.seed,
            )
        ),
        "T-Learner": TLearner(
            estimator_t=XGBRegressor(
                n_estimators=args.estimators,
                max_depth=args.max_depth,
                verbosity=0,
                random_state=args.seed,
            )
        ),
        "X-Learner": XLearner(
            outcome_learner=XGBRegressor(
                n_estimators=args.estimators,
                max_depth=args.max_depth,
                verbosity=0,
                random_state=args.seed,
            ),
            effect_learner=XGBRegressor(
                n_estimators=args.estimators,
                max_depth=args.max_depth,
                verbosity=0,
                random_state=args.seed,
            ),
        ),
        "ClassTransform": ClassTransformEstimator(
            base_estimator=XGBRegressor(
                n_estimators=args.estimators,
                max_depth=3,
                learning_rate=0.05,
                verbosity=0,
                random_state=args.seed,
            )
        ),
        "UpliftForest": UniversalUpliftForest(
            n_estimators=args.forest_estimators,
            max_depth=args.forest_depth,
            min_samples_treatment=args.min_samples_treatment,
            n_jobs=args.n_jobs,
            random_state=args.seed,
        ),
    }


def effect_metrics(y_true_tau, pred_tau):
    rho, _ = spearmanr(y_true_tau, pred_tau)
    return {
        "MAE": mean_absolute_error(y_true_tau, pred_tau),
        "MSE": mean_squared_error(y_true_tau, pred_tau),
        "R2": r2_score(y_true_tau, pred_tau),
        "Spearman": rho,
        "Bias": float(np.mean(pred_tau - y_true_tau)),
    }


def compute_stage1(df_train, graph, treat_col, target_col, conf_type, args):
    deconfounder = RITEPropensityStage(
        treatment_var=treat_col,
        target_var=target_col,
        ancestors_graph=graph,
        method="IPW",
        conf_type=conf_type,
        is_precise=True,
    )
    df_stg1, weights, p_scores, c_set, s_prec, p_model = deconfounder.fit_transform(
        df_train,
        n_trials=args.n_trials,
        random_state=args.seed,
    )
    return {
        "df": df_stg1,
        "weights": weights,
        "p_scores": p_scores,
        "c_set": c_set,
        "s_prec": s_prec,
        "p_model": p_model,
    }


def fit_predict_stage2(base_model, df_test, treat_col, target_col, stage1, use_precision, args):
    trainer = RITETreatmentEffectStage(
        copy.deepcopy(base_model),
        stage1["p_model"],
        stage1["c_set"],
        stage1["s_prec"] if use_precision else [],
        random_state=args.seed,
    )
    trainer.fit(
        stage1["df"],
        treat_col,
        target_col,
        stage1["p_scores"],
        stage1["weights"],
        verbose=False,
    )
    return trainer.predict(df_test, use_ps=base_model.__class__.__name__ == "XLearner")


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    df_full, graph, _ = generate_complex_environment(n=args.n, seed=args.seed)
    treat_col, target_col = "T", "Y"
    feature_cols = [c for c in df_full.columns if c not in [treat_col, target_col, "true_tau"]]

    df_train, df_test = train_test_split(
        df_full,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df_full[treat_col],
    )

    rows = []
    models = build_models(args)
    selected = set(args.models) if args.models else None

    print("\n=== RITE propensity stage: total adjustment set ===", flush=True)
    stage1_total = compute_stage1(df_train, graph, treat_col, target_col, "total", args)
    print("\n=== RITE propensity stage: proximal adjustment set ===", flush=True)
    stage1_proximal = compute_stage1(df_train, graph, treat_col, target_col, "proximal", args)

    for model_name, base_model in models.items():
        if selected and model_name not in selected:
            continue

        print(f"\n=== {model_name} ===", flush=True)

        model = copy.deepcopy(base_model)
        model.fit(
            df_train[feature_cols].values,
            df_train[target_col].values,
            df_train[treat_col].values,
        )
        pred = model.predict(df_test[feature_cols].values)
        rows.append({"Model": model_name, "Method": "Naive", **effect_metrics(df_test["true_tau"], pred)})

        pred = fit_predict_stage2(
            base_model,
            df_test,
            treat_col,
            target_col,
            stage1_total,
            use_precision=False,
            args=args,
        )
        rows.append({"Model": model_name, "Method": "Trad_IPW_NoPrec", **effect_metrics(df_test["true_tau"], pred)})

        pred = fit_predict_stage2(
            base_model,
            df_test,
            treat_col,
            target_col,
            stage1_total,
            use_precision=True,
            args=args,
        )
        rows.append({"Model": model_name, "Method": "Trad_IPW_WithPrec", **effect_metrics(df_test["true_tau"], pred)})

        pred = fit_predict_stage2(
            base_model,
            df_test,
            treat_col,
            target_col,
            stage1_proximal,
            use_precision=True,
            args=args,
        )
        rows.append({"Model": model_name, "Method": "RITE_Ours", **effect_metrics(df_test["true_tau"], pred)})

        partial = pd.DataFrame(rows)
        partial.to_csv(os.path.join(args.out_dir, args.out_csv), index=False)
        print(partial[partial["Model"] == model_name].to_string(index=False), flush=True)

    result = pd.DataFrame(rows)
    result["RuntimeSec"] = time.time() - started
    result.to_csv(os.path.join(args.out_dir, args.out_csv), index=False)

    meta = {
        "n": args.n,
        "seed": args.seed,
        "test_size": args.test_size,
        "n_trials": args.n_trials,
        "estimators": args.estimators,
        "forest_estimators": args.forest_estimators,
        "runtime_sec": time.time() - started,
        "train_shape": list(df_train.shape),
        "test_shape": list(df_test.shape),
        "treatment_rate_train": float(df_train[treat_col].mean()),
        "treatment_rate_test": float(df_test[treat_col].mean()),
    }
    with open(os.path.join(args.out_dir, args.out_json), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Final MAE/R2 table ===")
    print(result.pivot(index="Model", columns="Method", values=["MAE", "R2"]).to_string())
    print(f"\nSaved: {os.path.join(args.out_dir, args.out_csv)}")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=32)
    parser.add_argument("--estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--forest-estimators", type=int, default=100)
    parser.add_argument("--forest-depth", type=int, default=5)
    parser.add_argument("--min-samples-treatment", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--out-csv", default="synthetic_reproduction_mae_r2.csv")
    parser.add_argument("--out-json", default="synthetic_reproduction_meta.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
