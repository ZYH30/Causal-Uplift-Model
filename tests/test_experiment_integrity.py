import json
import unittest
from pathlib import Path

import pandas as pd

from tlcd.llm_causal.neurosymbolic_merge import _violates_task_order
from tlcd.llm_causal.semantic_adjudication import directed_edges, is_acyclic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
REQUIRED_RESULTS = RESULTS / "nhanes_llm_fixed_confirmatory_v1/fixed_confirmatory_per_seed.csv"


@unittest.skipUnless(REQUIRED_RESULTS.exists(), "paper result artifacts are not present")
class ExperimentIntegrityTests(unittest.TestCase):
    def test_multiseed_graph_recovery_matches_locked_protocol(self):
        rows = pd.read_csv(
            RESULTS
            / "tlcd_synthetic_multiseed_dag_gated_v2"
            / "tlcd_multiseed_per_seed.csv"
        )
        self.assertEqual(set(rows["seed"]), set(range(2026, 2031)))
        self.assertEqual(len(rows), 5)
        self.assertTrue(rows["pred_T_to_Y_path"].astype(bool).all())
        self.assertTrue((rows["C_Proximal_recall"] == 1.0).all())
        self.assertTrue((rows["S_Prec_recall"] == 1.0).all())
        self.assertAlmostEqual(rows["directed_f1"].mean(), 0.8549448064, places=8)

    def test_mimic_aggregate_results_are_complete_and_deidentified(self):
        result_dir = RESULTS / "mimic_rbc_chart_augmented"
        manifest = json.loads(
            (result_dir / "chartevents_extraction_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["cohort_n"], 15568)
        self.assertEqual(manifest["stays_with_chart_features"], 15509)
        self.assertFalse(manifest["patient_level_raw_events_exported"])

        variables = pd.read_csv(result_dir / "variable_manifest.csv")
        self.assertEqual(len(variables), 80)
        self.assertEqual(variables["variable"].str.startswith("chart_").sum(), 18)

        cohort = pd.read_csv(result_dir / "final/common_support_cohort_audit.csv")
        final = cohort.loc[
            cohort["era"].eq("overall")
            & cohort["population"].eq("common_support")
        ].iloc[0]
        self.assertEqual(int(final["n"]), 13990)
        self.assertEqual(int(final["treated"]), 2819)
        self.assertEqual(int(final["outcomes"]), 2649)

        forbidden_columns = {"subject_id", "hadm_id", "stay_id"}
        for path in result_dir.rglob("*.csv"):
            columns = set(pd.read_csv(path, nrows=0).columns)
            self.assertFalse(
                columns & forbidden_columns,
                f"Patient identifiers found in {path.name}",
            )

    def test_mimic_temporal_balance_and_effect_metrics_match_paper(self):
        result_dir = RESULTS / "mimic_rbc_chart_augmented/final"
        diagnostics = json.loads(
            (result_dir / "temporal_diagnostics.json").read_text(encoding="utf-8")
        )
        self.assertLess(
            diagnostics["mean_abs_smd_ipw"],
            diagnostics["mean_abs_smd_pre"],
        )
        self.assertGreater(diagnostics["ess_ratio"], 0.70)

        temporal = pd.read_csv(result_dir / "temporal_paired_bootstrap_summary.csv")
        primary = temporal.loc[temporal["comparison"].eq("TLCD_chart_vs_all_chart")]
        self.assertEqual(len(primary), 3)
        self.assertTrue(primary["Point_Estimate"].gt(0).all())

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
            / "tlcd_component_ablation_v1/tlcd_component_ablation.csv"
        )
        if not path.exists():
            self.skipTest("component-ablation artifact is not present")
        rows = pd.read_csv(path)
        self.assertEqual(len(rows), 40)
        self.assertEqual(rows["Seed"].nunique(), 10)
        self.assertEqual(
            set(rows["Variant"]),
            {
                "Complete_TLCD",
                "PairwiseANM_FullRepair",
                "TaskAdaptive_NoDAGGate",
                "TaskAdaptive_NoRepair",
            },
        )
        means = rows.groupby("Variant").mean(numeric_only=True)
        self.assertGreater(
            means.loc["Complete_TLCD", "Directed_F1"],
            means.loc["PairwiseANM_FullRepair", "Directed_F1"],
        )
        self.assertGreater(
            means.loc["Complete_TLCD", "C_Proximal_F1"],
            means.loc["TaskAdaptive_NoRepair", "C_Proximal_F1"],
        )

    def test_mimic_final_role_interface_and_repeated_splits(self):
        result_dir = RESULTS / "mimic_rbc_chart_augmented"
        protocol = json.loads(
            (result_dir / "final/common_support_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        repeated = pd.read_csv(
            result_dir / "final/post_discovery_repeated_summary.csv"
        )
        self.assertEqual(len(protocol["c_set"]), 11)
        self.assertEqual(len(protocol["s_set"]), 3)
        self.assertEqual(set(repeated["Wins"]), {12, 15})
        self.assertTrue(repeated["CI_Low"].gt(0).all())


if __name__ == "__main__":
    unittest.main()
