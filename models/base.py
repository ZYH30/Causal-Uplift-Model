"""Common estimator protocol for conditional treatment-effect learners."""

from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin


class BaseUpliftEstimator(BaseEstimator, RegressorMixin):
    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        raise NotImplementedError

    def predict(self, X, p_scores=None):
        raise NotImplementedError
