from __future__ import annotations

import unittest

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from tlcd.graph_metrics import evaluate_graph_recovery
from data_generation import generate_complex_environment
from graph_roles import causal_role_sets
from models.class_transform import ClassTransformEstimator
from rite_framework import RITEPropensityStage


class CorePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.graph, _ = generate_complex_environment(n=600, seed=2026)

    def test_oracle_graph_yields_expected_role_sets(self):
        roles = causal_role_sets(self.graph)
        self.assertEqual(
            roles["C_Proximal"],
            {"C_prox_1", "C_prox_2", "C_prox_3"},
        )
        self.assertEqual(roles["S_Prec"], {"P_1", "P_2"})
        self.assertNotIn("Z_1", roles["C_Proximal"])
        self.assertNotIn("Collid_1", roles["C_Proximal"])

    def test_first_stage_propensity_uses_only_selected_c_set(self):
        stage = RITEPropensityStage(
            treatment_var="T",
            target_var="Y",
            ancestors_graph=self.graph,
            conf_type="proximal",
        )
        _, weights, propensity, c_set, s_set, model = stage.fit_transform(
            self.data,
            n_trials=0,
            random_state=2026,
        )
        self.assertEqual(model.feature_names, c_set)
        self.assertEqual(set(s_set), {"P_1", "P_2"})
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(((propensity >= 0.05) & (propensity <= 0.95)).all())

    def test_perfect_graph_has_perfect_recovery_metrics(self):
        metrics = evaluate_graph_recovery(self.graph, self.graph, nodes=self.graph)
        self.assertEqual(metrics["directed"]["f1"], 1.0)
        self.assertEqual(metrics["skeleton"]["f1"], 1.0)
        self.assertEqual(metrics["shd"], 0)
        self.assertEqual(metrics["rite_sets"]["C_Proximal"]["f1"], 1.0)

    def test_centered_class_transform_returns_finite_cate(self):
        features = self.data[["C_prox_1", "C_prox_2", "C_prox_3", "P_1", "P_2"]]
        estimator = ClassTransformEstimator(
            base_estimator=RandomForestRegressor(
                n_estimators=10,
                max_depth=3,
                random_state=2026,
            ),
            propensity_estimator=LogisticRegression(max_iter=500),
            outcome_estimator=RandomForestRegressor(
                n_estimators=10,
                max_depth=3,
                random_state=2026,
            ),
        )
        estimator.fit(
            features,
            self.data["Y"],
            self.data["T"],
        )
        prediction = estimator.predict(features.iloc[:20])
        self.assertEqual(prediction.shape, (20,))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
