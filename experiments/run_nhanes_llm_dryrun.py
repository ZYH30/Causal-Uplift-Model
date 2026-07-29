"""Prepare NHANES semantic-adjudication queues and token budgets without API calls."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ccia.llm_causal.semantic_adjudication import (
    build_edge_queue,
    build_messages,
    estimate_call_plan,
    load_edge_decisions,
    load_json,
    materialize_node_descriptions,
    sha256_file,
    write_json,
    write_queue_csv,
)


RESOURCE_DIR = PROJECT_ROOT / "ccia" / "llm_causal" / "nhanes"
DEFAULT_TASKS = ["antihypertensive_sbp", "chol_med_tchol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "llm_adjudication_dryrun"),
    )
    parser.add_argument("--votes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.35)
    parser.add_argument("--max-output-tokens-per-edge", type=int, default=180)
    parser.add_argument("--reasoning-token-reserve-per-edge", type=int, default=250)
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def scoped_resources(resources: list[dict], priorities: set[str]) -> list[dict]:
    scoped = []
    for resource in resources:
        item = copy.copy(resource)
        item["queue"] = [record for record in resource["queue"] if record["priority"] in priorities]
        scoped.append(item)
    return scoped


def build_report(
    task_summaries: list[dict],
    budgets: dict,
    output_dir: Path,
) -> str:
    try:
        display_dir = output_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        display_dir = output_dir
    lines = [
        "# NHANES LLM semantic adjudication dry-run",
        "",
        "This dry-run reads no API key and makes no network or model call.",
        "",
        "## Edge queue",
        "",
        "| Task | Graph nodes | Graph edges | Selected pairs | P0 | P1 | P2 | DAG |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for summary in task_summaries:
        counts = summary["priority_counts"]
        lines.append(
            f"| {summary['task']} | {summary['node_count']} | {summary['edge_count']} | "
            f"{summary['selected_pair_count']} | {counts['P0']} | {counts['P1']} | "
            f"{counts['P2']} | {'yes' if summary['graph_is_acyclic'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "P0 covers fixed treatment-outcome or set-critical/temporally impossible directions. "
            "P1 covers the remaining T/Y-touching, C/S-sensitive, rescue, and sanitizer-audit pairs. "
            "P2 covers lower-impact uncertain ancestral directions.",
            "",
            "## Token scenarios",
            "",
            "| Scope | Strategy | Edge pairs | API calls | Input central | Input upper | Output upper | Total upper |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scope, plans in budgets["scenarios"].items():
        for strategy, plan in plans.items():
            lines.append(
                f"| {scope} | {strategy} | {plan['edge_pairs']} | {plan['api_calls']} | "
                f"{plan['estimated_input_tokens']['central']} | {plan['estimated_input_tokens']['upper']} | "
                f"{plan['configured_output_token_upper_bound']} | {plan['estimated_total_token_upper_bound']} |"
            )

    lines.extend(
        [
            "",
            "The recommended execution scope is `P0+P1` with counterbalanced three-vote batches. "
            "Pairwise estimates are retained as a transparent baseline. Token counts are a conservative "
            "character-based range because the proprietary Kimi/DeepSeek tokenizer is not available locally; "
            "the exact serialized prompts, character counts, and assumptions are saved in the budget JSON.",
            "",
            "## Outputs",
            "",
            f"- `{display_dir / 'nhanes_llm_edge_queue.csv'}`",
            f"- `{display_dir / 'nhanes_llm_edge_queue.json'}`",
            f"- `{display_dir / 'nhanes_llm_token_budget.json'}`",
            f"- `{display_dir / 'nhanes_llm_prompt_samples.json'}`",
            f"- `{display_dir / 'nhanes_llm_dryrun_manifest.json'}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.votes <= 0 or args.votes % 2 == 0:
        raise ValueError("--votes must be a positive odd number")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = RESOURCE_DIR / "nhanes_variable_catalog.json"
    config_path = RESOURCE_DIR / "task_semantic_config.json"
    prompt_path = RESOURCE_DIR / "prompt_nhanes_causal_adjudication.json"
    catalog = load_json(catalog_path)
    task_configs = load_json(config_path)
    prompt_template = load_json(prompt_path)

    all_queue = []
    task_summaries = []
    task_resources = []
    input_paths = [catalog_path, config_path, prompt_path]

    for task in args.tasks:
        if task not in task_configs:
            raise KeyError(f"Unknown semantic task {task!r}")
        config = task_configs[task]
        graph_path = resolve_project_path(config["graph_path"])
        graph_meta_path = resolve_project_path(config["graph_meta_path"])
        decisions_path = resolve_project_path(config["edge_decisions_path"])
        run_meta_path = resolve_project_path(config["run_meta_path"])
        domain_path = resolve_project_path(config["domain_context_path"])
        descriptions_path = resolve_project_path(config["nodes_description_path"])
        input_paths.extend([graph_path, graph_meta_path, decisions_path, run_meta_path, domain_path])

        graph = load_json(graph_path)
        graph_meta = load_json(graph_meta_path)
        run_meta = load_json(run_meta_path)
        node_descriptions = materialize_node_descriptions(run_meta, catalog, config)
        graph_nodes = set(graph) | {parent for parents in graph.values() for parent in parents}
        missing_descriptions = sorted(graph_nodes - set(node_descriptions))
        if missing_descriptions:
            raise KeyError(f"{task}: graph nodes missing semantic descriptions: {missing_descriptions}")
        write_json(descriptions_path, node_descriptions)

        queue, summary = build_edge_queue(
            task,
            graph,
            graph_meta,
            load_edge_decisions(decisions_path),
            low_confidence_threshold=args.low_confidence_threshold,
        )
        all_queue.extend(queue)
        task_summaries.append(summary)
        task_resources.append(
            {
                "task": task,
                "task_display_name": config["display_name"],
                "domain_context": domain_path.read_text(encoding="utf-8"),
                "node_descriptions": node_descriptions,
                "queue": queue,
            }
        )

    all_queue.sort(key=lambda row: (row["priority"], row["task"], row["pair_id"]))
    write_queue_csv(output_dir / "nhanes_llm_edge_queue.csv", all_queue)
    write_json(output_dir / "nhanes_llm_edge_queue.json", all_queue)

    scopes = {
        "P0": scoped_resources(task_resources, {"P0"}),
        "P0+P1": scoped_resources(task_resources, {"P0", "P1"}),
        "P0+P1+P2": scoped_resources(task_resources, {"P0", "P1", "P2"}),
    }
    scenarios = {}
    for scope, resources in scopes.items():
        scenarios[scope] = {
            "pairwise": estimate_call_plan(
                resources,
                prompt_template,
                args.votes,
                batch_size=1,
                max_output_tokens_per_edge=args.max_output_tokens_per_edge,
                reasoning_token_reserve_per_edge=args.reasoning_token_reserve_per_edge,
            ),
            f"batched_{args.batch_size}": estimate_call_plan(
                resources,
                prompt_template,
                args.votes,
                batch_size=args.batch_size,
                max_output_tokens_per_edge=args.max_output_tokens_per_edge,
                reasoning_token_reserve_per_edge=args.reasoning_token_reserve_per_edge,
            ),
        }

    budget = {
        "dry_run_only": True,
        "api_key_read": False,
        "api_calls_made": 0,
        "model_not_yet_selected": True,
        "token_estimation_method": {
            "description": "Exact serialized prompts with a multilingual character-based tokenizer envelope.",
            "lower": "ASCII/4.5 + non-ASCII/2.0 + chat overhead",
            "central": "ASCII/4.0 + non-ASCII/1.5 + chat overhead",
            "upper": "ASCII/3.0 + non-ASCII/1.0 + chat overhead",
            "caveat": "Actual provider tokenization and reasoning-token accounting may differ.",
        },
        "parameters": {
            "votes": args.votes,
            "recommended_batch_size": args.batch_size,
            "max_output_tokens_per_edge": args.max_output_tokens_per_edge,
            "reasoning_token_reserve_per_edge": args.reasoning_token_reserve_per_edge,
            "low_confidence_threshold": args.low_confidence_threshold,
        },
        "task_queue_summaries": task_summaries,
        "recommended_scope": "P0+P1",
        "recommended_strategy": f"batched_{args.batch_size}",
        "scenarios": scenarios,
    }
    write_json(output_dir / "nhanes_llm_token_budget.json", budget)

    prompt_samples = {}
    for resource in task_resources:
        p0 = [record for record in resource["queue"] if record["priority"] == "P0"]
        sample_queue = p0 or resource["queue"]
        prompt_samples[resource["task"]] = {
            "counterbalanced_pair_vote_1": build_messages(
                prompt_template,
                resource["task_display_name"],
                resource["domain_context"],
                sample_queue[:1],
                resource["node_descriptions"],
                vote_index=0,
            ),
            "counterbalanced_pair_vote_2": build_messages(
                prompt_template,
                resource["task_display_name"],
                resource["domain_context"],
                sample_queue[:1],
                resource["node_descriptions"],
                vote_index=1,
            ),
            "batched_vote_1": build_messages(
                prompt_template,
                resource["task_display_name"],
                resource["domain_context"],
                sample_queue[: args.batch_size],
                resource["node_descriptions"],
                vote_index=0,
            ),
        }
    write_json(output_dir / "nhanes_llm_prompt_samples.json", prompt_samples)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([os.path.basename(sys.executable), *sys.argv]),
        "project_root": str(PROJECT_ROOT),
        "dry_run_only": True,
        "api_key_read": False,
        "api_calls_made": 0,
        "inputs": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in sorted(set(input_paths))
        },
        "outputs": [
            "nhanes_llm_edge_queue.csv",
            "nhanes_llm_edge_queue.json",
            "nhanes_llm_token_budget.json",
            "nhanes_llm_prompt_samples.json",
            "nhanes_llm_dryrun_report.md",
        ],
    }
    write_json(output_dir / "nhanes_llm_dryrun_manifest.json", manifest)
    (output_dir / "nhanes_llm_dryrun_report.md").write_text(
        build_report(task_summaries, budget, output_dir), encoding="utf-8"
    )

    recommended = scenarios["P0+P1"][f"batched_{args.batch_size}"]
    print(json.dumps({
        "dry_run_only": True,
        "selected_pairs": len(all_queue),
        "recommended_scope": "P0+P1",
        "recommended_api_calls": recommended["api_calls"],
        "recommended_input_tokens_central": recommended["estimated_input_tokens"]["central"],
        "recommended_total_tokens_upper": recommended["estimated_total_token_upper_bound"],
        "out_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
