"""Centered transformed-outcome learner for observational treatment assignment."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from xgboost import XGBClassifier, XGBRegressor

from .base import BaseUpliftEstimator


class ClassTransformEstimator(BaseUpliftEstimator):
    def __init__(
        self,
        base_estimator=None,
        propensity_estimator=None,
        clip_propensity: float = 0.05,
        center_outcome: bool = True,
        outcome_estimator=None,
    ):
        self.base_estimator = (
            base_estimator
            if base_estimator is not None
            else XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                verbosity=0,
            )
        )
        self.propensity_estimator = (
            propensity_estimator
            if propensity_estimator is not None
            else XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                eval_metric="logloss",
                verbosity=0,
            )
        )
        self.clip_propensity = clip_propensity
        self.center_outcome = center_outcome
        self.outcome_estimator = (
            outcome_estimator
            if outcome_estimator is not None
            else XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                verbosity=0,
            )
        )

    @staticmethod
    def _fit_optional_weight(model, X, y, sample_weight):
        if sample_weight is None:
            return model.fit(X, y)
        try:
            return model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            return model.fit(X, y)

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        X, y = check_X_y(X, y)
        T = check_array(T, ensure_2d=False)
        if p_scores is None:
            self.propensity_model_ = clone(self.propensity_estimator)
            self._fit_optional_weight(self.propensity_model_, X, T, sample_weight)
            propensity = self.propensity_model_.predict_proba(X)[:, 1]
        else:
            self.propensity_model_ = None
            propensity = np.asarray(p_scores, dtype=float)
        propensity = np.clip(
            propensity,
            self.clip_propensity,
            1.0 - self.clip_propensity,
        )

        if self.center_outcome:
            self.outcome_model_ = clone(self.outcome_estimator)
            self._fit_optional_weight(self.outcome_model_, X, y, sample_weight)
            centered_outcome = y - self.outcome_model_.predict(X)
        else:
            self.outcome_model_ = None
            centered_outcome = y

        multiplier = (T - propensity) / (propensity * (1.0 - propensity))
        transformed_outcome = centered_outcome * multiplier
        self.estimator_ = clone(self.base_estimator)

        # The transformed outcome already contains inverse-propensity scaling.
        # Applying the OCU IPW weights again would create double weighting.
        self.estimator_.fit(X, transformed_outcome)
        return self

    def predict(self, X, p_scores=None):
        del p_scores
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(check_array(X))
