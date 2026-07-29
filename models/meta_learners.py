"""Weighted S-, T-, and X-learner implementations."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import XGBClassifier, XGBRegressor

from .base import BaseUpliftEstimator


def _fit(estimator, X, y, sample_weight=None):
    if sample_weight is None:
        return estimator.fit(X, y)
    try:
        return estimator.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        return estimator.fit(X, y)


class SLearner(BaseUpliftEstimator):
    def __init__(self, base_estimator=None):
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                verbosity=0,
            )
        )

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        del p_scores
        X, y = check_X_y(X, y)
        T = check_array(T, ensure_2d=False)
        self.estimator_ = clone(self.base_estimator)
        _fit(self.estimator_, np.column_stack([X, T]), y, sample_weight)
        return self

    def predict(self, X, p_scores=None):
        del p_scores
        check_is_fitted(self, "estimator_")
        X = check_array(X)
        mu0 = self.estimator_.predict(np.column_stack([X, np.zeros(len(X))]))
        mu1 = self.estimator_.predict(np.column_stack([X, np.ones(len(X))]))
        return mu1 - mu0


class TLearner(BaseUpliftEstimator):
    def __init__(self, estimator_t=None, estimator_c=None):
        self.estimator_t = (
            estimator_t
            if estimator_t is not None
            else XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                verbosity=0,
            )
        )
        self.estimator_c = estimator_c

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        del p_scores
        X, y = check_X_y(X, y)
        T = check_array(T, ensure_2d=False)
        treated = T == 1
        control = T == 0
        if not treated.any() or not control.any():
            raise ValueError("Both treatment arms are required.")
        self.model_t_ = clone(self.estimator_t)
        control_estimator = self.estimator_c if self.estimator_c is not None else self.estimator_t
        self.model_c_ = clone(control_estimator)
        w1 = sample_weight[treated] if sample_weight is not None else None
        w0 = sample_weight[control] if sample_weight is not None else None
        _fit(self.model_t_, X[treated], y[treated], w1)
        _fit(self.model_c_, X[control], y[control], w0)
        return self

    def predict(self, X, p_scores=None):
        del p_scores
        check_is_fitted(self, ["model_t_", "model_c_"])
        X = check_array(X)
        return self.model_t_.predict(X) - self.model_c_.predict(X)


class XLearner(BaseUpliftEstimator):
    def __init__(self, outcome_learner=None, effect_learner=None, propensity_learner=None):
        self.outcome_learner = (
            outcome_learner
            if outcome_learner is not None
            else XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                verbosity=0,
            )
        )
        self.effect_learner = (
            effect_learner
            if effect_learner is not None
            else XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                verbosity=0,
            )
        )
        self.propensity_learner = (
            propensity_learner
            if propensity_learner is not None
            else XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                eval_metric="logloss",
                verbosity=0,
            )
        )

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        X, y = check_X_y(X, y)
        T = check_array(T, ensure_2d=False)
        treated = T == 1
        control = T == 0
        if not treated.any() or not control.any():
            raise ValueError("Both treatment arms are required.")
        w1 = sample_weight[treated] if sample_weight is not None else None
        w0 = sample_weight[control] if sample_weight is not None else None

        self.model_mu_t_ = clone(self.outcome_learner)
        self.model_mu_c_ = clone(self.outcome_learner)
        _fit(self.model_mu_t_, X[treated], y[treated], w1)
        _fit(self.model_mu_c_, X[control], y[control], w0)

        d1 = y[treated] - self.model_mu_c_.predict(X[treated])
        d0 = self.model_mu_t_.predict(X[control]) - y[control]
        self.model_tau_t_ = clone(self.effect_learner)
        self.model_tau_c_ = clone(self.effect_learner)
        _fit(self.model_tau_t_, X[treated], d1, w1)
        _fit(self.model_tau_c_, X[control], d0, w0)

        self.model_e_ = None
        if p_scores is None:
            self.model_e_ = clone(self.propensity_learner)
            _fit(self.model_e_, X, T, sample_weight)
        return self

    def predict(self, X, p_scores=None):
        check_is_fitted(self, ["model_tau_t_", "model_tau_c_"])
        X = check_array(X)
        tau1 = self.model_tau_t_.predict(X)
        tau0 = self.model_tau_c_.predict(X)
        if p_scores is not None:
            propensity = np.asarray(p_scores, dtype=float)
        elif self.model_e_ is not None:
            propensity = self.model_e_.predict_proba(X)[:, 1]
        else:
            raise ValueError("X-Learner prediction requires external propensity scores.")
        return propensity * tau0 + (1.0 - propensity) * tau1
