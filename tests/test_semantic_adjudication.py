from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ccia.llm_causal.semantic_adjudication import (
    build_edge_queue,
    build_messages,
    is_acyclic,
    materialize_node_descriptions,
)
from ccia.llm_causal.neurosymbolic_merge import merge_semantic_graph, needs_tiebreak
from experiments.run_nhanes_llm_adjudication import normalize_response_decisions


class SemanticAdjudicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = {
            "C": [],
            "S": [],
            "T": ["C"],
            "Y": ["C", "S", "T"],
        }

    def test_queue_always_contains_fixed_treatment_outcome_pair(self) -> None:
        queue, summary = build_edge_queue("toy", self.graph, {}, [])
        treatment_outcome = [row for row in queue if row["required_t_y"]]
        self.assertEqual(len(treatment_outcome), 1)
        self.assertEqual(treatment_outcome[0]["priority"], "P0")
        self.assertTrue(summary["graph_is_acyclic"])

    def test_counterbalanced_prompt_reverses_x_y_position(self) -> None:
        queue, _ = build_edge_queue("toy", self.graph, {}, [])
        record = next(row for row in queue if row["required_t_y"])
        descriptions = {node: {"label": node} for node in self.graph}
        template = {
            "system": "system",
            "judgement_labels": {"X_TO_Y": "x to y"},
            "output_schema": {"pair_id": "string", "judgement": "X_TO_Y"},
            "batch_output": {"decisions": []},
        }
        first = build_messages(template, "toy task", "context", [record], descriptions, 0)
        second = build_messages(template, "toy task", "context", [record], descriptions, 1)
        first_pair = json.loads(first[1]["content"])["pairs"][0]
        second_pair = json.loads(second[1]["content"])["pairs"][0]
        self.assertEqual((first_pair["X"], first_pair["Y"]), ("T", "Y"))
        self.assertEqual((second_pair["X"], second_pair["Y"]), ("Y", "T"))
        self.assertEqual(first_pair["current_statistical_direction"], "X_TO_Y")
        self.assertEqual(second_pair["current_statistical_direction"], "Y_TO_X")

    def test_materialized_task_descriptions_apply_overrides(self) -> None:
        run_meta = {"features": ["C"]}
        catalog = {"C": {"label": "confounder", "eligible_for_c_set": True}}
        config = {
            "treatment": {"label": "treatment"},
            "outcome": {"label": "outcome"},
            "task_specific_overrides": {"C": {"eligible_for_c_set": False}},
        }
        descriptions = materialize_node_descriptions(run_meta, catalog, config)
        self.assertFalse(descriptions["C"]["eligible_for_c_set"])
        self.assertEqual(descriptions["T"]["label"], "treatment")
        self.assertEqual(descriptions["Y"]["label"], "outcome")

    def test_cycle_detection(self) -> None:
        self.assertTrue(is_acyclic(self.graph))
        cyclic = {"A": ["B"], "B": ["A"], "T": [], "Y": []}
        self.assertFalse(is_acyclic(cyclic))

    def test_swapped_model_output_is_canonicalized(self) -> None:
        queue, _ = build_edge_queue("toy", self.graph, {}, [])
        record = next(row for row in queue if row["required_t_y"])
        descriptions = {node: {"label": node} for node in self.graph}
        template = {
            "system": "Return strict JSON.",
            "judgement_labels": {"X_TO_Y": "x to y", "Y_TO_X": "y to x"},
            "output_schema": {
                "pair_id": "string",
                "judgement": "X_TO_Y | Y_TO_X | NO_DIRECT_EDGE | UNRESOLVED",
                "confidence": "number",
                "reason": "string",
                "constraints_applied": [],
            },
        }
        messages = build_messages(template, "toy", "context", [record], descriptions, vote_index=1)
        response = {
            "choices": [{"message": {"content": json.dumps({
                "decisions": [{
                    "pair_id": record["pair_id"],
                    "judgement": "Y_TO_X",
                    "confidence": 0.9,
                    "reason": "The intervention precedes the outcome.",
                    "constraints_applied": ["intervention order"],
                }]
            })}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        votes = normalize_response_decisions(
            response,
            messages,
            [record],
            provider="test",
            model="test-model",
            call_id="test-call",
            vote_index=1,
            round_name="test",
        )
        self.assertEqual(votes[0]["canonical_judgement"], "A_TO_B")
        self.assertEqual((votes[0]["proposed_parent"], votes[0]["proposed_child"]), ("T", "Y"))

    def test_merge_reorients_impossible_age_edge_and_preserves_t_to_y(self) -> None:
        graph = {
            "BMXHT": [],
            "RIDAGEYR": ["BMXHT"],
            "T": ["RIDAGEYR"],
            "Y": ["RIDAGEYR"],
        }
        decision = {
            "task": "toy",
            "pair_id": "toy::BMXHT::RIDAGEYR",
            "node_a": "BMXHT",
            "node_b": "RIDAGEYR",
            "priority": "P0",
            "current_parent": "BMXHT",
            "current_child": "RIDAGEYR",
            "consensus_judgement": "B_TO_A",
            "proposed_parent": "RIDAGEYR",
            "proposed_child": "BMXHT",
            "semantic_confidence": 0.9,
        }
        merged, _, summary = merge_semantic_graph(graph, [decision])
        self.assertIn("RIDAGEYR", merged["BMXHT"])
        self.assertNotIn("BMXHT", merged["RIDAGEYR"])
        self.assertIn("T", merged["Y"])
        self.assertTrue(summary["graph_is_acyclic"])

    def test_tiebreak_rule(self) -> None:
        agreeing = [
            {"canonical_judgement": "A_TO_B", "confidence": 0.8},
            {"canonical_judgement": "A_TO_B", "confidence": 0.7},
        ]
        disagreeing = [
            {"canonical_judgement": "A_TO_B", "confidence": 0.8},
            {"canonical_judgement": "B_TO_A", "confidence": 0.8},
        ]
        self.assertFalse(needs_tiebreak(agreeing))
        self.assertTrue(needs_tiebreak(disagreeing))


if __name__ == "__main__":
    unittest.main()
