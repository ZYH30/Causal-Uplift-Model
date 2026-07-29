"""Deterministic weighted uplift tree and forest."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed

from .base import BaseUpliftEstimator


class _Node:
    def __init__(self, value: dict[int, float], depth: int):
        self.value = value
        self.depth = depth
        self.feature_idx: int | None = None
        self.threshold: float | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_idx is None


class _UpliftTree:
    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        min_samples_treatment: int,
        max_features: str | float,
        random_state: int,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y, T, sample_weight):
        self.treatment_classes_ = sorted(level for level in np.unique(T) if level != 0)
        if not self.treatment_classes_:
            raise ValueError("At least one treated observation is required.")
        self.rng_ = np.random.default_rng(self.random_state)
        self.root_ = self._grow(X, y, T, sample_weight, depth=0)
        return self

    def _cate(self, y, T, weights) -> dict[int, float]:
        control = T == 0
        if not control.any():
            return {level: 0.0 for level in self.treatment_classes_}
        mu0 = np.average(y[control], weights=weights[control])
        result = {}
        for level in self.treatment_classes_:
            mask = T == level
            result[level] = (
                float(np.average(y[mask], weights=weights[mask]) - mu0)
                if mask.any()
                else 0.0
            )
        return result

    def _feature_indices(self, n_features: int) -> np.ndarray:
        if self.max_features == "sqrt":
            count = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            count = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, float):
            count = max(1, int(self.max_features * n_features))
        else:
            count = n_features
        return self.rng_.choice(n_features, min(count, n_features), replace=False)

    def _best_split(self, X, y, T, weights):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        classes = [0] + self.treatment_classes_
        class_index = {level: index for index, level in enumerate(classes)}
        n_samples, n_features = X.shape
        weighted_y = np.zeros((n_samples, len(classes)))
        weighted_n = np.zeros((n_samples, len(classes)))
        for row, level in enumerate(T):
            if level in class_index:
                col = class_index[level]
                weighted_y[row, col] = weights[row] * y[row]
                weighted_n[row, col] = weights[row]
        total_y = weighted_y.sum(axis=0)
        total_n = weighted_n.sum(axis=0)

        for feature in self._feature_indices(n_features):
            order = np.argsort(X[:, feature])
            values = X[order, feature]
            left_y = np.cumsum(weighted_y[order], axis=0)
            left_n = np.cumsum(weighted_n[order], axis=0)
            right_y = total_y - left_y
            right_n = total_n - left_n
            valid = (
                (left_n >= self.min_samples_treatment).all(axis=1)
                & (right_n >= self.min_samples_treatment).all(axis=1)
                & (left_n.sum(axis=1) >= self.min_samples_leaf)
                & (right_n.sum(axis=1) >= self.min_samples_leaf)
            )
            valid &= np.r_[np.diff(values) != 0, False]
            if not valid.any():
                continue
            mu_left = left_y[valid] / (left_n[valid] + 1e-9)
            mu_right = right_y[valid] / (right_n[valid] + 1e-9)
            tau_left = mu_left[:, 1:] - mu_left[:, :1]
            tau_right = mu_right[:, 1:] - mu_right[:, :1]
            gain = (
                np.sum(tau_left**2, axis=1) * left_n[valid].sum(axis=1)
                + np.sum(tau_right**2, axis=1) * right_n[valid].sum(axis=1)
            )
            local = int(np.argmax(gain))
            if gain[local] > best_gain:
                valid_rows = np.flatnonzero(valid)
                best_gain = float(gain[local])
                best_feature = int(feature)
                best_threshold = float(values[valid_rows[local]])
        return best_feature, best_threshold

    def _grow(self, X, y, T, weights, depth: int) -> _Node:
        node = _Node(self._cate(y, T, weights), depth)
        if depth >= self.max_depth or weights.sum() < self.min_samples_leaf:
            return node
        feature, threshold = self._best_split(X, y, T, weights)
        if feature is None:
            return node
        left = X[:, feature] <= threshold
        if left.all() or (~left).all():
            return node
        node.feature_idx = feature
        node.threshold = threshold
        node.left = self._grow(X[left], y[left], T[left], weights[left], depth + 1)
        node.right = self._grow(X[~left], y[~left], T[~left], weights[~left], depth + 1)
        return node

    def _predict_row(self, row: np.ndarray) -> np.ndarray:
        node = self.root_
        while not node.is_leaf:
            node = node.left if row[node.feature_idx] <= node.threshold else node.right
        return np.array([node.value[level] for level in self.treatment_classes_])

    def predict(self, X):
        values = np.vstack([self._predict_row(row) for row in X])
        return values[:, 0] if values.shape[1] == 1 else values


class UniversalUpliftForest(BaseUpliftEstimator):
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_leaf: int = 10,
        min_samples_treatment: int = 5,
        max_features: str | float = "sqrt",
        n_jobs: int = -1,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        X = np.asarray(X)
        y = np.asarray(y)
        T = np.asarray(T)
        if sample_weight is None:
            if p_scores is None:
                sample_weight = np.ones(len(y), dtype=float)
            else:
                p_scores = np.clip(np.asarray(p_scores), 0.01, 0.99)
                sample_weight = np.where(T == 1, 1.0 / p_scores, 1.0 / (1.0 - p_scores))
        sample_weight = np.asarray(sample_weight, dtype=float)
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31 - 1, size=self.n_estimators)
        bootstrap_indices = [
            np.random.default_rng(seed).choice(len(y), len(y), replace=True)
            for seed in seeds
        ]

        def fit_tree(seed, indices):
            tree = _UpliftTree(
                self.max_depth,
                self.min_samples_leaf,
                self.min_samples_treatment,
                self.max_features,
                int(seed),
            )
            return tree.fit(
                X[indices],
                y[indices],
                T[indices],
                sample_weight[indices],
            )

        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_tree)(seed, indices)
            for seed, indices in zip(seeds, bootstrap_indices)
        )
        return self

    def predict(self, X, p_scores=None):
        del p_scores
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(np.asarray(X)) for tree in self.trees_
        )
        return np.mean(np.asarray(predictions), axis=0)
