"""Build deterministic semantic-fusion graph variants from saved LLM votes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlcd.llm_causal.neurosymbolic_merge import merge_semantic_graph
from tlcd.llm_causal.semantic_adjudication import load_json, write_json


RESULT_DIR = PROJECT_ROOT / "results" / "llm_adjudication"
TASKS = {
    "antihypertensive_sbp": {
        "source": PROJECT_ROOT / "results" / "nhanes_antihypertensive_sbp_v3_total_conf" / "nhanes_antihypertensive_sbp_tlcd_graph.json",
        "stem": "nhanes_antihypertensive_sbp_graph_llm",
    },
    "chol_med_tchol": {
        "source": PROJECT_ROOT / "results" / "nhanes_chol_med_tchol_v1_dr_metrics" / "nhanes_chol_med_tchol_tlcd_graph.json",
        "stem": "nhanes_chol_med_tchol_graph_llm",
    },
}
VARIANTS = {
    "aggressive": {
        "min_direction_confidence": 0.45,
        "min_remove_confidence": 0.45,
        "min_add_confidence": 0.60,
    },
    "conservative": {
        "min_direction_confidence": 0.60,
        "min_remove_confidence": 0.75,
        "min_add_confidence": 0.75,
    },
    "hard_only": {
        "min_direction_confidence": 1.01,
        "min_remove_confidence": 1.01,
        "min_add_confidence": 1.01,
    },
}


def main() -> None:
    decisions = load_json(RESULT_DIR / "llm_edge_decisions.json")
    summaries = {}
    for task, spec in TASKS.items():
        source = load_json(spec["source"])
        task_decisions = [decision for decision in decisions if decision["task"] == task]
        summaries[task] = {}
        for variant, thresholds in VARIANTS.items():
            graph, actions, summary = merge_semantic_graph(source, task_decisions, **thresholds)
            graph_path = RESULT_DIR / f"{spec['stem']}_{variant}.json"
            write_json(graph_path, graph)
            write_json(
                graph_path.with_name(graph_path.stem + "_meta.json"),
                {
                    "task": task,
                    "variant": variant,
                    "source_graph": str(spec["source"].relative_to(PROJECT_ROOT)),
                    "actions": actions,
                    **summary,
                },
            )
            summaries[task][variant] = summary
    write_json(RESULT_DIR / "nhanes_llm_graph_variant_summary.json", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
