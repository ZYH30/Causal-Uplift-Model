"""Role-separated deconfounding and uplift estimation."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from graph_roles import find_confounder_sets, precision_variables
from propensity import PropensityModel, fit_propensity


class FirstStageDeconfounder:
    """Estimate `e(C)` and stabilized IPW from graph-derived causal roles."""

    def __init__(
        self,
        treatment_var: str = "T",
        target_var: str = "Y",
        ancestors_graph: dict[str, list[str]] | None = None,
        method: str = "IPW",
        conf_type: str = "proximal",
        is_precise: bool = True,
        propensity_clip: float = 0.05,
    ) -> None:
        if method != "IPW":
            raise ValueError("The released paper pipeline supports method='IPW'.")
        if conf_type not in {"total", "proximal", "root"}:
            raise ValueError("conf_type must be 'total', 'proximal', or 'root'.")
        if not 0.0 < propensity_clip < 0.5:
            raise ValueError("propensity_clip must lie in (0, 0.5).")
        self.method = method
        self.t_var = treatment_var
        self.y_var = target_var
        self.graph = ancestors_graph
        self.conf_type = conf_type
        self.is_precise = is_precise
        self.propensity_clip = propensity_clip
        self.c_opt: list[str] = []
        self.s_prec: list[str] = []
        self.propensity_model: PropensityModel | None = None
        self.best_params: dict[str, Any] | None = None

    def _extract_causal_features(self) -> None:
        if self.graph is None:
            raise ValueError("A learned or oracle ancestral graph is required.")
        confounders = find_confounder_sets(self.graph, self.t_var, self.y_var)
        self.c_opt = list(confounders[self.conf_type])
        self.s_prec = (
            sorted(precision_variables(self.graph, self.t_var, self.y_var, self.c_opt))
            if self.is_precise
            else []
        )
        if not self.c_opt:
            raise ValueError("The selected graph does not yield a nonempty adjustment set.")

    def fit_transform(
        self,
        data: pd.DataFrame,
        n_trials: int = 0,
        random_state: int = 2026,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str], PropensityModel]:
        self._extract_causal_features()
        required = self.c_opt + self.s_prec + [self.t_var, self.y_var]
        missing = sorted(set(required) - set(data.columns))
        if missing:
            raise ValueError(f"Input data are missing required columns: {missing}")

        treatment = data[self.t_var].to_numpy(dtype=int)
        self.propensity_model, self.best_params = fit_propensity(
            data[self.c_opt],
            treatment,
            n_trials=n_trials,
            seed=random_state,
        )
        propensity = np.clip(
            self.propensity_model.predict(data[self.c_opt]),
            self.propensity_clip,
            1.0 - self.propensity_clip,
        )

        marginal = float(np.mean(treatment))
        weights = np.where(
            treatment == 1,
            marginal / propensity,
            (1.0 - marginal) / (1.0 - propensity),
        )
        for level in (0, 1):
            mask = treatment == level
            weights[mask] *= mask.sum() / weights[mask].sum()

        return (
            data,
            weights,
            propensity,
            list(self.c_opt),
            list(self.s_prec),
            self.propensity_model,
        )


class SecondStageUpliftTrainer:
    """Fit an uplift learner on `C` or `C union S` using first-stage weights."""

    def __init__(
        self,
        base_uplift_model,
        propensity_model: PropensityModel,
        c_opt: list[str],
        s_prec: list[str],
        random_state: int = 2026,
    ) -> None:
        self.base_model = copy.deepcopy(base_uplift_model)
        self.propensity_model = propensity_model
        self.c_opt = sorted(set(c_opt))
        self.s_prec = sorted(set(s_prec))
        self.x_fit_cols_ = sorted(set(self.c_opt) | set(self.s_prec))
        self.random_state = random_state

    def fit(
        self,
        data: pd.DataFrame,
        t_var: str,
        y_var: str,
        ps_score: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        verbose: bool = False,
    ) -> "SecondStageUpliftTrainer":
        del verbose
        missing = sorted(set(self.x_fit_cols_ + [t_var, y_var]) - set(data.columns))
        if missing:
            raise ValueError(f"Training data are missing required columns: {missing}")
        self.base_model.fit(
            data[self.x_fit_cols_].to_numpy(),
            data[y_var].to_numpy(),
            data[t_var].to_numpy(),
            sample_weight=weights,
            p_scores=ps_score,
        )
        return self

    def predict(self, data: pd.DataFrame, use_ps: bool = False) -> np.ndarray:
        missing = sorted(set(self.x_fit_cols_) - set(data.columns))
        if missing:
            raise ValueError(f"Prediction data are missing required columns: {missing}")
        features = data[self.x_fit_cols_].to_numpy()
        if not use_ps:
            return np.asarray(self.base_model.predict(features), dtype=float)
        propensity = np.clip(
            self.propensity_model.predict(data[self.c_opt]),
            0.05,
            0.95,
        )
        return np.asarray(self.base_model.predict(features, p_scores=propensity), dtype=float)
