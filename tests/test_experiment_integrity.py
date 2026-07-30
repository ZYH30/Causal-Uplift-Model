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
    def test_multiseed_graph_recovery_matches_locked_protocol(self):
        rows = pd.read_csv(
            RESULTS
            / "ccia_synthetic_multiseed_dag_gated_v2"
            / "ccia_multiseed_per_seed.csv"
        )
        self.assertEqual(set(rows["seed"]), set(range(2026, 2031)))
        self.assertEqual(len(rows), 5)
        self.assertTrue(rows["pred_T_to_Y_path"].astype(bool).all())
        self.assertTrue((rows["C_Proximal_recall"] == 1.0).all())
        self.assertTrue((rows["S_Prec_recall"] == 1.0).all())
        self.assertAlmostEqual(rows["directed_f1"].mean(), 0.8549448064, places=8)

    def test_mimic_aggregate_results_are_complete_and_deidentified(self):
        result_dir = RESULTS / "mimic_rbc_target_trial_v3"
        manifest = json.loads(
            (result_dir / "mimic_rbc_cohort_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["flow"]["landmark_eligible"], 15568)
        self.assertEqual(manifest["feature_count"], 79)
        self.assertEqual(manifest["database_file"], "mimic_iv_ccia.sqlite")

        forbidden_columns = {"subject_id", "hadm_id", "stay_id"}
        result_dirs = [
            result_dir,
            RESULTS / "mimic_rbc_target_trial_v5_product_audit",
        ]
        for directory in result_dirs:
            for path in directory.glob("*.csv"):
                columns = set(pd.read_csv(path, nrows=0).columns)
                self.assertFalse(
                    columns & forbidden_columns,
                    f"Patient identifiers found in {path.name}",
                )

    def test_mimic_legacy_total_set_sensitivity_is_intact(self):
        result_dir = RESULTS / "mimic_rbc_target_trial_v3"
        sensitivity = pd.read_csv(
            result_dir / "mimic_rbc_temporal_c_total_sensitivity.csv"
        ).iloc[0]
        self.assertLess(
            sensitivity["mean_abs_smd_ipw"],
            sensitivity["mean_abs_smd_pre"],
        )
        self.assertGreater(sensitivity["ess_ratio"], 0.90)

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

    def test_systematic_benchmark_is_complete(self):
        result_dir = RESULTS / "systematic_simulation_v1"
        graph_path = result_dir / "systematic_graph_results.csv"
        cate_path = result_dir / "systematic_cate_results.csv"
        if not graph_path.exists() or not cate_path.exists():
            self.skipTest("systematic benchmark artifacts are not present")
        graph = pd.read_csv(graph_path)
        cate = pd.read_csv(cate_path)
        self.assertEqual(len(graph), 800)
        self.assertEqual(len(cate), 1280)
        self.assertTrue(graph["Status"].eq("complete").all())
        self.assertEqual(graph["Scenario"].nunique(), 4)
        self.assertEqual(graph["N"].nunique(), 4)
        self.assertEqual(graph["Seed"].nunique(), 10)
        self.assertEqual(graph["Method"].nunique(), 5)
        self.assertEqual(cate["Interface"].nunique(), 8)

    def test_component_ablation_covers_prespecified_variants(self):
        path = (
            RESULTS
            / "ccia_component_ablation_v1/ccia_component_ablation.csv"
        )
        if not path.exists():
            self.skipTest("component-ablation artifact is not present")
        rows = pd.read_csv(path)
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows["Seed"].nunique(), 10)
        self.assertEqual(
            set(rows["Variant"]),
            {
                "Complete_CCIA",
                "PairwiseANM_FullRepair",
                "TaskAdaptive_NoDAGGate",
                "TaskAdaptive_NoRepair",
            },
        )
        means = rows.groupby("Variant").mean(numeric_only=True)
        self.assertGreater(
            means.loc["Complete_CCIA", "Directed_F1"],
            means.loc["PairwiseANM_FullRepair", "Directed_F1"],
        )
        self.assertGreater(
            means.loc["Complete_CCIA", "C_Proximal_F1"],
            means.loc["TaskAdaptive_NoRepair", "C_Proximal_F1"],
        )

    def test_mimic_safety_audit_matches_final_interface(self):
        result_dir = RESULTS / "mimic_rbc_target_trial_v5_product_audit"
        roles_path = result_dir / "mimic_rbc_safety_validated_roles.json"
        repeated_path = (
            result_dir / "mimic_rbc_safety_validated_repeated_summary.csv"
        )
        if not roles_path.exists() or not repeated_path.exists():
            self.skipTest("MIMIC safety-audit artifacts are not present")
        roles = json.loads(roles_path.read_text(encoding="utf-8"))
        repeated = pd.read_csv(repeated_path)
        self.assertEqual(len(roles["c_proximal"]), 7)
        self.assertEqual(len(roles["s_precision"]), 1)
        r_loss = repeated.loc[
            repeated["Metric"].eq("R_Loss_Improvement")
        ].iloc[0]
        self.assertEqual(int(r_loss["Repeats"]), 20)
        self.assertEqual(int(r_loss["Wins"]), 19)
        self.assertGreater(float(r_loss["CI_Low"]), 0.0)


if __name__ == "__main__":
    unittest.main()
