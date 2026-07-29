import json
import unittest
from pathlib import Path

import pandas as pd

from ccia.llm_causal.neurosymbolic_merge import _violates_task_order
from ccia.llm_causal.semantic_adjudication import directed_edges, is_acyclic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
REQUIRED_RESULTS = RESULTS / "nhanes_llm_fixed_confirmatory_v1/fixed_confirmatory_per_seed.csv"


@unittest.skipUnless(REQUIRED_RESULTS.exists(), "paper result artifacts are not present")
class ExperimentIntegrityTests(unittest.TestCase):
    def test_locked_confirmation_has_all_prespecified_seeds(self):
        rows = pd.read_csv(
            RESULTS / "nhanes_llm_fixed_confirmatory_v1/fixed_confirmatory_per_seed.csv"
        )
        expected = set(range(2040, 2060))
        self.assertEqual(len(rows), 40)
        for _, group in rows.groupby("Task"):
            self.assertEqual(set(group["Seed"]), expected)
            self.assertEqual(len(group), 20)
        self.assertFalse(
            rows[
                [
                    "R_Loss_Improvement",
                    "DR_Qini_Improvement",
                    "ObservedBudgetOutcome_Improvement",
                ]
            ].isna().any().any()
        )

    def test_conservative_graphs_are_dags_without_impossible_directions(self):
        graph_dir = RESULTS / "llm_adjudication"
        paths = [
            graph_dir / "nhanes_antihypertensive_sbp_graph_llm_conservative.json",
            graph_dir / "nhanes_chol_med_tchol_graph_llm_conservative.json",
        ]
        for path in paths:
            graph = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(is_acyclic(graph))
            violations = [
                edge
                for edge in directed_edges(graph)
                if edge != ("T", "Y") and _violates_task_order(*edge)
            ]
            self.assertEqual(violations, [])
            self.assertIn(("T", "Y"), directed_edges(graph))

    def test_api_manifest_matches_decision_archive(self):
        result_dir = RESULTS / "llm_adjudication"
        manifest = json.loads(
            (result_dir / "llm_adjudication_manifest.json").read_text(encoding="utf-8")
        )
        decisions = pd.read_csv(result_dir / "llm_edge_decisions.csv")
        self.assertEqual(len(decisions), manifest["pair_count"])
        self.assertEqual(decisions["pair_id"].nunique(), manifest["pair_count"])
        self.assertEqual(
            decisions["consensus_judgement"].value_counts().to_dict(),
            manifest["consensus_counts"],
        )
        usage = manifest["actual_usage"]
        self.assertLessEqual(usage["http_responses"], manifest["budget"]["max_http_responses"])
        self.assertLessEqual(usage["total_tokens"], manifest["budget"]["max_total_tokens"])

    def test_survey_sensitivity_has_finite_design_estimates(self):
        rows = pd.read_csv(
            RESULTS
            / "nhanes_survey_aipw_sensitivity_v1/nhanes_survey_weighted_aipw_sensitivity.csv"
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue((rows["survey_design_df"] > 0).all())
        self.assertTrue((rows["survey_aipw_ci_high"] < 0).all())


if __name__ == "__main__":
    unittest.main()
