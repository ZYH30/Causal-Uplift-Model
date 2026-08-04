"""Deterministic LightGBM propensity fitting used by the RITE first stage."""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class PropensityModel:
    booster: lgb.Booster
    feature_names: list[str]

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return np.asarray(self.booster.predict(features), dtype=float)


def _train_booster(
    features: pd.DataFrame,
    treatment: np.ndarray,
    params: dict,
    seed: int,
) -> tuple[lgb.Booster, float]:
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        treatment,
        test_size=0.2,
        random_state=42,
    )
    train_data = lgb.Dataset(x_train, label=y_train)
    valid_data = lgb.Dataset(x_valid, label=y_valid)
    full_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        **params,
    }
    booster = lgb.train(
        full_params,
        train_data,
        valid_sets=[valid_data],
        num_boost_round=256,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    validation_probability = np.clip(booster.predict(x_valid), 1e-6, 1.0 - 1e-6)
    return booster, float(log_loss(y_valid, validation_probability))


def fit_propensity(
    features: pd.DataFrame,
    treatment: np.ndarray,
    *,
    n_trials: int = 0,
    seed: int = 2026,
) -> tuple[PropensityModel, dict]:
    if features.shape[1] == 0:
        raise ValueError("The propensity model requires at least one C-set variable.")
    treatment = np.asarray(treatment, dtype=int)
    if set(np.unique(treatment)) != {0, 1}:
        raise ValueError("Treatment must contain both binary levels 0 and 1.")

    fixed = {
        "learning_rate": 0.05,
        "max_depth": 30,
        "num_leaves": 256,
        "min_data_in_leaf": 256,
        "min_data_in_bin": 256,
        "subsample": 0.8,
    }
    if n_trials > 0:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.4),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "num_leaves": trial.suggest_int("num_leaves", 40, 200),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 200),
                "min_data_in_bin": trial.suggest_int("min_data_in_bin", 50, 200),
                "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            }
            _, loss = _train_booster(features, treatment, params, seed)
            return loss

        study.optimize(objective, n_trials=n_trials)
        best_params = dict(study.best_params)
    else:
        best_params = fixed

    booster, _ = _train_booster(features, treatment, best_params, seed)
    model = PropensityModel(booster=booster, feature_names=list(features.columns))
    return model, best_params
