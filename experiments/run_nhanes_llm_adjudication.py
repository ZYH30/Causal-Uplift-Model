"""Run adaptive Kimi/DeepSeek semantic adjudication and merge NHANES TLCD graphs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlcd.llm_causal.api_client import (
    ApiCallError,
    CompatibleChatClient,
    provider_config,
    response_content,
    response_reasoning_content,
    response_usage,
)
from tlcd.llm_causal.neurosymbolic_merge import (
    aggregate_votes,
    graph_diff_rows,
    merge_semantic_graph,
    needs_tiebreak,
    set_diff_rows,
)
from tlcd.llm_causal.semantic_adjudication import (
    build_messages,
    chunks,
    load_json,
    sha256_file,
    write_json,
)


RESOURCE_DIR = PROJECT_ROOT / "tlcd" / "llm_causal" / "nhanes"
DEFAULT_QUEUE = PROJECT_ROOT / "results" / "llm_adjudication_dryrun" / "nhanes_llm_edge_queue.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "llm_adjudication"
VALID_JUDGEMENTS = {"X_TO_Y", "Y_TO_X", "NO_DIRECT_EDGE", "UNRESOLVED"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", default=str(DEFAULT_QUEUE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--priorities", nargs="+", default=["P0", "P1"])
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--kimi-max-completion-tokens", type=int, default=2200)
    parser.add_argument("--deepseek-max-completion-tokens", type=int, default=6000)
    parser.add_argument("--max-http-responses", type=int, default=45)
    parser.add_argument("--max-total-tokens", type=int, default=356_961)
    parser.add_argument("--agreement-confidence", type=float, default=0.60)
    parser.add_argument("--direction-confidence", type=float, default=0.60)
    parser.add_argument("--remove-confidence", type=float, default=0.75)
    parser.add_argument("--add-confidence", type=float, default=0.75)
    parser.add_argument("--kimi-thinking", action="store_true")
    parser.add_argument("--deepseek-no-thinking", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model output must be a JSON object")
    return value


def prompt_pairs(messages: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    payload = json.loads(messages[1]["content"])
    return {pair["pair_id"]: pair for pair in payload["pairs"]}


def normalize_response_decisions(
    response: dict[str, Any],
    messages: list[dict[str, str]],
    records: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    call_id: str,
    vote_index: int,
    round_name: str,
) -> list[dict[str, Any]]:
    parsed = parse_json_object(response_content(response))
    decisions = parsed.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Model JSON must contain a decisions array")

    expected = {record["pair_id"]: record for record in records}
    pair_context = prompt_pairs(messages)
    by_id: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Each decision must be a JSON object")
        pair_id = str(decision.get("pair_id", ""))
        if pair_id in by_id:
            raise ValueError(f"Duplicate pair_id in response: {pair_id}")
        by_id[pair_id] = decision
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))
        extra = sorted(set(by_id) - set(expected))
        raise ValueError(f"Response pair mismatch; missing={missing}, extra={extra}")

    usage = response_usage(response)
    normalized = []
    for pair_id, record in expected.items():
        decision = by_id[pair_id]
        judgement = str(decision.get("judgement", "")).upper().strip()
        if judgement not in VALID_JUDGEMENTS:
            raise ValueError(f"Invalid judgement {judgement!r} for {pair_id}")
        try:
            confidence = float(decision.get("confidence"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid confidence for {pair_id}") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence outside [0,1] for {pair_id}")
        reason = str(decision.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"Missing reason for {pair_id}")
        constraints = decision.get("constraints_applied", [])
        if not isinstance(constraints, list):
            constraints = [str(constraints)]

        x = pair_context[pair_id]["X"]
        y = pair_context[pair_id]["Y"]
        proposed_parent = proposed_child = ""
        if judgement == "X_TO_Y":
            proposed_parent, proposed_child = x, y
        elif judgement == "Y_TO_X":
            proposed_parent, proposed_child = y, x

        if judgement in {"NO_DIRECT_EDGE", "UNRESOLVED"}:
            canonical = judgement
        elif (proposed_parent, proposed_child) == (record["node_a"], record["node_b"]):
            canonical = "A_TO_B"
        elif (proposed_parent, proposed_child) == (record["node_b"], record["node_a"]):
            canonical = "B_TO_A"
        else:
            raise ValueError(f"Could not canonicalize direction for {pair_id}")

        normalized.append(
            {
                "task": record["task"],
                "pair_id": pair_id,
                "node_a": record["node_a"],
                "node_b": record["node_b"],
                "provider": provider,
                "model": model,
                "call_id": call_id,
                "round": round_name,
                "vote_index": vote_index,
                "prompt_X": x,
                "prompt_Y": y,
                "raw_judgement": judgement,
                "canonical_judgement": canonical,
                "proposed_parent": proposed_parent,
                "proposed_child": proposed_child,
                "confidence": confidence,
                "reason": reason,
                "constraints_applied": [str(value) for value in constraints],
                "call_usage": usage,
            }
        )
    return normalized


class BudgetTracker:
    def __init__(self, raw_dir: Path, max_http_responses: int, max_total_tokens: int) -> None:
        self.raw_dir = raw_dir
        self.max_http_responses = max_http_responses
        self.max_total_tokens = max_total_tokens
        self.http_responses = 0
        self.total_tokens = 0
        for path in raw_dir.glob("**/*.json") if raw_dir.exists() else []:
            try:
                record = load_json(path)
            except Exception:
                continue
            if isinstance(record, dict) and isinstance(record.get("response"), dict):
                self.http_responses += 1
                self.total_tokens += response_usage(record["response"]).get("total_tokens", 0)

    def before_call(self) -> None:
        if self.http_responses >= self.max_http_responses:
            raise RuntimeError(f"HTTP response cap reached: {self.http_responses}/{self.max_http_responses}")
        if self.total_tokens >= self.max_total_tokens:
            raise RuntimeError(f"Token cap reached: {self.total_tokens}/{self.max_total_tokens}")

    def record(self, response: dict[str, Any]) -> None:
        self.http_responses += 1
        self.total_tokens += response_usage(response).get("total_tokens", 0)
        if self.total_tokens > self.max_total_tokens:
            raise RuntimeError(f"Token cap exceeded: {self.total_tokens}/{self.max_total_tokens}")


def archive_invalid(path: Path) -> None:
    if not path.exists():
        return
    index = 1
    while True:
        archived = path.with_name(f"{path.stem}.invalid{index}{path.suffix}")
        if not archived.exists():
            path.replace(archived)
            return
        index += 1


def execute_call(
    *,
    client: CompatibleChatClient,
    task_resource: dict[str, Any],
    records: list[dict[str, Any]],
    vote_index: int,
    round_name: str,
    batch_number: int,
    output_dir: Path,
    tracker: BudgetTracker,
    max_completion_tokens: int,
    thinking_enabled: bool,
) -> list[dict[str, Any]]:
    provider = client.config.name
    call_id = f"{round_name}_{provider}_{task_resource['task']}_b{batch_number:02d}"
    raw_path = output_dir / "raw" / provider / f"{call_id}.json"
    request_path = output_dir / "requests" / provider / f"{call_id}.json"
    if raw_path.exists():
        cached = load_json(raw_path)
        if cached.get("validation_status") == "valid":
            return cached["normalized_votes"]
        archive_invalid(raw_path)

    messages = build_messages(
        task_resource["prompt_template"],
        task_resource["task_display_name"],
        task_resource["domain_context"],
        records,
        task_resource["node_descriptions"],
        vote_index,
    )
    request_hash = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    atomic_write_json(
        request_path,
        {
            "call_id": call_id,
            "provider": provider,
            "model": client.config.model,
            "round": round_name,
            "vote_index": vote_index,
            "request_hash": request_hash,
            "pair_ids": [record["pair_id"] for record in records],
            "thinking_enabled": thinking_enabled,
            "max_completion_tokens": max_completion_tokens,
            "messages": messages,
        },
    )

    tracker.before_call()
    try:
        response, transport_attempts = client.create(
            messages,
            max_completion_tokens=max_completion_tokens,
            thinking_enabled=thinking_enabled,
            prompt_cache_key=f"tlcd-nhanes-{task_resource['task']}",
        )
    except ApiCallError as exc:
        error_record = {
            "timestamp_utc": utc_now(),
            "call_id": call_id,
            "provider": provider,
            "model": client.config.model,
            "request_hash": request_hash,
            "error": str(exc),
            "status_code": exc.status_code,
            "response_body": exc.response_body,
        }
        atomic_write_json(output_dir / "errors" / f"{call_id}.json", error_record)
        append_jsonl(output_dir / "call_ledger.jsonl", {**error_record, "event": "api_error"})
        raise

    tracker.record(response)
    base_record = {
        "timestamp_utc": utc_now(),
        "call_id": call_id,
        "provider": provider,
        "model": client.config.model,
        "round": round_name,
        "vote_index": vote_index,
        "request_hash": request_hash,
        "pair_ids": [record["pair_id"] for record in records],
        "transport_attempts": transport_attempts,
        "usage": response_usage(response),
        "has_reasoning_content": response_reasoning_content(response) is not None,
        "response": response,
    }
    try:
        votes = normalize_response_decisions(
            response,
            messages,
            records,
            provider=provider,
            model=client.config.model,
            call_id=call_id,
            vote_index=vote_index,
            round_name=round_name,
        )
    except Exception as exc:
        invalid = {**base_record, "validation_status": "invalid", "validation_error": str(exc)}
        atomic_write_json(raw_path, invalid)
        append_jsonl(
            output_dir / "call_ledger.jsonl",
            {key: value for key, value in invalid.items() if key != "response"} | {"event": "invalid_response"},
        )
        raise

    valid = {**base_record, "validation_status": "valid", "normalized_votes": votes}
    atomic_write_json(raw_path, valid)
    append_jsonl(
        output_dir / "call_ledger.jsonl",
        {key: value for key, value in valid.items() if key not in {"response", "normalized_votes"}}
        | {"event": "valid_response"},
    )
    print(
        f"[{tracker.http_responses}/{tracker.max_http_responses}] {call_id}: "
        f"pairs={len(records)}, tokens={response_usage(response).get('total_tokens', 0)}, "
        f"cumulative={tracker.total_tokens}",
        flush=True,
    )
    return votes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def flatten_decision(decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": decision["task"],
        "pair_id": decision["pair_id"],
        "node_a": decision["node_a"],
        "node_b": decision["node_b"],
        "priority": decision["priority"],
        "current_parent": decision["current_parent"],
        "current_child": decision["current_child"],
        "consensus_judgement": decision["consensus_judgement"],
        "proposed_parent": decision["proposed_parent"],
        "proposed_child": decision["proposed_child"],
        "semantic_confidence": decision["semantic_confidence"],
        "agreement_ratio": decision["agreement_ratio"],
        "vote_count": decision["vote_count"],
        "vote_counts": decision["vote_counts"],
        "providers": decision["providers"],
        "models": decision["models"],
        "reasons": decision["reasons"],
        "constraints_applied": decision["constraints_applied"],
        "selection_reasons": decision["selection_reasons"],
        "provenance": decision["provenance"],
    }


def load_task_resources(task_configs: dict[str, Any], prompt_template: dict[str, Any]) -> dict[str, dict[str, Any]]:
    resources = {}
    for task, config in task_configs.items():
        domain_path = PROJECT_ROOT / config["domain_context_path"]
        descriptions_path = PROJECT_ROOT / config["nodes_description_path"]
        resources[task] = {
            "task": task,
            "task_display_name": config["display_name"],
            "domain_context": domain_path.read_text(encoding="utf-8"),
            "node_descriptions": load_json(descriptions_path),
            "prompt_template": prompt_template,
            "graph_path": PROJECT_ROOT / config["graph_path"],
        }
    return resources


def usage_summary(output_dir: Path) -> dict[str, Any]:
    calls = []
    for path in sorted((output_dir / "raw").glob("**/*.json")):
        record = load_json(path)
        if not isinstance(record.get("response"), dict):
            continue
        usage = response_usage(record["response"])
        calls.append(
            {
                "call_id": record.get("call_id"),
                "provider": record.get("provider"),
                "model": record.get("model"),
                "validation_status": record.get("validation_status"),
                **usage,
            }
        )
    by_provider = {}
    for provider in sorted({call["provider"] for call in calls}):
        selected = [call for call in calls if call["provider"] == provider]
        by_provider[provider] = {
            "calls": len(selected),
            "prompt_tokens": sum(call.get("prompt_tokens", 0) for call in selected),
            "completion_tokens": sum(call.get("completion_tokens", 0) for call in selected),
            "total_tokens": sum(call.get("total_tokens", 0) for call in selected),
        }
    return {
        "http_responses": len(calls),
        "valid_calls": sum(call["validation_status"] == "valid" for call in calls),
        "total_prompt_tokens": sum(call.get("prompt_tokens", 0) for call in calls),
        "total_completion_tokens": sum(call.get("completion_tokens", 0) for call in calls),
        "total_tokens": sum(call.get("total_tokens", 0) for call in calls),
        "by_provider": by_provider,
        "calls": calls,
    }


def main() -> None:
    args = parse_args()
    if os.environ.get("ALLOW_PAID_API") != "1":
        raise RuntimeError(
            "Paid API calls are locked. Run the dry-run, review its budget, "
            "and set ALLOW_PAID_API=1 explicitly."
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dotenv_values(WORKSPACE_ROOT / ".env")
    kimi_key = str(env.get("Kimi_Key") or "").strip()
    deepseek_key = str(env.get("deepseek_Key") or "").strip()
    if not kimi_key or not deepseek_key:
        raise RuntimeError(".env must define non-empty Kimi_Key and deepseek_Key")

    queue_path = Path(args.queue_json).resolve()
    queue = [
        record
        for record in load_json(queue_path)
        if record["priority"] in set(args.priorities)
    ]
    queue.sort(key=lambda record: (record["task"], record["priority"], record["pair_id"]))
    task_configs = load_json(RESOURCE_DIR / "task_semantic_config.json")
    prompt_template = load_json(RESOURCE_DIR / "prompt_nhanes_causal_adjudication.json")
    resources = load_task_resources(task_configs, prompt_template)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in queue:
        by_task[record["task"]].append(record)

    tracker = BudgetTracker(output_dir / "raw", args.max_http_responses, args.max_total_tokens)
    votes: list[dict[str, Any]] = []
    provider_thinking = {
        "kimi": bool(args.kimi_thinking),
        "deepseek": not args.deepseek_no_thinking,
    }
    provider_max_tokens = {
        "kimi": args.kimi_max_completion_tokens,
        "deepseek": args.deepseek_max_completion_tokens,
    }

    clients = {
        "kimi": CompatibleChatClient(provider_config("kimi", kimi_key)),
        "deepseek": CompatibleChatClient(provider_config("deepseek", deepseek_key)),
    }
    try:
        # Round 1: one vote from each independent provider, with reversed X/Y order.
        for task in sorted(by_task):
            task_batches = list(chunks(by_task[task], args.batch_size))
            for batch_number, batch in enumerate(task_batches, start=1):
                votes.extend(
                    execute_call(
                        client=clients["kimi"],
                        task_resource=resources[task],
                        records=batch,
                        vote_index=0,
                        round_name="round1",
                        batch_number=batch_number,
                        output_dir=output_dir,
                        tracker=tracker,
                        max_completion_tokens=provider_max_tokens["kimi"],
                        thinking_enabled=provider_thinking["kimi"],
                    )
                )
                votes.extend(
                    execute_call(
                        client=clients["deepseek"],
                        task_resource=resources[task],
                        records=batch,
                        vote_index=1,
                        round_name="round1",
                        batch_number=batch_number,
                        output_dir=output_dir,
                        tracker=tracker,
                        max_completion_tokens=provider_max_tokens["deepseek"],
                        thinking_enabled=provider_thinking["deepseek"],
                    )
                )

        first_round_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for vote in votes:
            first_round_by_pair[vote["pair_id"]].append(vote)
        tie_records = [
            record
            for record in queue
            if needs_tiebreak(first_round_by_pair[record["pair_id"]], args.agreement_confidence)
        ]
        print(
            f"Round 1 complete: pairs={len(queue)}, tie-break pairs={len(tie_records)}",
            flush=True,
        )

        # Round 2: Kimi tie-break only for disagreement, unresolved, or low confidence.
        tie_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in tie_records:
            tie_by_task[record["task"]].append(record)
        for task in sorted(tie_by_task):
            for batch_number, batch in enumerate(chunks(tie_by_task[task], args.batch_size), start=1):
                votes.extend(
                    execute_call(
                        client=clients["kimi"],
                        task_resource=resources[task],
                        records=batch,
                        vote_index=2,
                        round_name="tiebreak",
                        batch_number=batch_number,
                        output_dir=output_dir,
                        tracker=tracker,
                        max_completion_tokens=provider_max_tokens["kimi"],
                        thinking_enabled=provider_thinking["kimi"],
                    )
                )
    finally:
        for client in clients.values():
            client.close()

    votes.sort(key=lambda vote: (vote["task"], vote["pair_id"], vote["vote_index"], vote["provider"]))
    (output_dir / "llm_votes_raw.jsonl").write_text(
        "".join(json.dumps(vote, ensure_ascii=False) + "\n" for vote in votes),
        encoding="utf-8",
    )

    decisions = aggregate_votes(votes, queue)
    write_json(output_dir / "llm_edge_decisions.json", decisions)
    write_csv(output_dir / "llm_edge_decisions.csv", [flatten_decision(decision) for decision in decisions])

    graph_diff = []
    cset_diff = []
    merge_summaries = {}
    for task in sorted(by_task):
        source_graph = load_json(resources[task]["graph_path"])
        task_decisions = [decision for decision in decisions if decision["task"] == task]
        merged_graph, actions, summary = merge_semantic_graph(
            source_graph,
            task_decisions,
            min_direction_confidence=args.direction_confidence,
            min_remove_confidence=args.remove_confidence,
            min_add_confidence=args.add_confidence,
        )
        graph_name = (
            "nhanes_antihypertensive_sbp_graph_llm.json"
            if task == "antihypertensive_sbp"
            else "nhanes_chol_med_tchol_graph_llm.json"
        )
        write_json(output_dir / graph_name, merged_graph)
        write_json(
            output_dir / graph_name.replace(".json", "_meta.json"),
            {"task": task, "source_graph": str(resources[task]["graph_path"]), "actions": actions, **summary},
        )
        graph_diff.extend(graph_diff_rows(source_graph, merged_graph, task))
        cset_diff.extend(set_diff_rows(summary["sets_before"], summary["sets_after"], task))
        merge_summaries[task] = summary

    write_csv(output_dir / "stat_only_vs_llm_graph_diff.csv", graph_diff)
    write_csv(output_dir / "stat_only_vs_llm_cset_diff.csv", cset_diff)
    usage = usage_summary(output_dir)
    write_json(output_dir / "llm_usage_summary.json", usage)

    consensus_counts = Counter(decision["consensus_judgement"] for decision in decisions)
    manifest = {
        "completed_at_utc": utc_now(),
        "providers": {"kimi": "kimi-k2.6", "deepseek": "deepseek-v4-flash"},
        "thinking_enabled": provider_thinking,
        "priorities": args.priorities,
        "pair_count": len(queue),
        "tie_break_pair_count": len(tie_records),
        "consensus_counts": dict(consensus_counts),
        "budget": {
            "max_http_responses": args.max_http_responses,
            "max_total_tokens": args.max_total_tokens,
        },
        "actual_usage": usage,
        "merge_summaries": merge_summaries,
        "input_hashes": {
            str(queue_path.relative_to(PROJECT_ROOT)): sha256_file(queue_path),
            "tlcd/llm_causal/nhanes/task_semantic_config.json": sha256_file(RESOURCE_DIR / "task_semantic_config.json"),
            "tlcd/llm_causal/nhanes/prompt_nhanes_causal_adjudication.json": sha256_file(
                RESOURCE_DIR / "prompt_nhanes_causal_adjudication.json"
            ),
        },
    }
    write_json(output_dir / "llm_adjudication_manifest.json", manifest)
    print(
        json.dumps(
            {
                "pair_count": len(queue),
                "tie_break_pair_count": len(tie_records),
                "consensus_counts": dict(consensus_counts),
                "actual_calls": usage["http_responses"],
                "actual_total_tokens": usage["total_tokens"],
                "merge_summaries": merge_summaries,
                "out_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
