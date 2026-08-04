"""Utilities for reproducible post-hoc semantic auditing of TLCD graphs.

This module contains no API client. It prepares auditable edge queues and
OpenAI-compatible message payloads without reading credentials or making
network requests.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
from contextlib import redirect_stdout
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

from tlcd.graph_metrics import confounder_and_precision_sets


TASK_NODES = {"T", "Y"}
HARD_EXOGENOUS = {"cycle_year", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "RIDEXMON"}
PREDETERMINED_ADULT = {"BMXHT"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_graph(graph: dict[str, list[str]], nodes: Iterable[str] = ()) -> dict[str, list[str]]:
    all_nodes = set(nodes) | set(graph)
    for parents in graph.values():
        all_nodes.update(parents)
    return {
        node: sorted(set(graph.get(node, [])) - {node})
        for node in sorted(all_nodes)
    }


def directed_edges(graph: dict[str, list[str]]) -> set[tuple[str, str]]:
    return {
        (parent, child)
        for child, parents in graph.items()
        for parent in parents
        if parent != child
    }


def graph_children(graph: dict[str, list[str]]) -> dict[str, set[str]]:
    children = {node: set() for node in graph}
    for parent, child in directed_edges(graph):
        children.setdefault(parent, set()).add(child)
        children.setdefault(child, set())
    return children


def ancestors(graph: dict[str, list[str]], node: str) -> set[str]:
    found: set[str] = set()
    queue = deque(graph.get(node, []))
    while queue:
        current = queue.popleft()
        if current in found or current == node:
            continue
        found.add(current)
        queue.extend(graph.get(current, []))
    return found


def is_acyclic(graph: dict[str, list[str]]) -> bool:
    children = graph_children(graph)
    indegree = {node: len(set(graph.get(node, []))) for node in children}
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        parent = queue.popleft()
        visited += 1
        for child in children.get(parent, set()):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited == len(indegree)


def remove_edge(graph: dict[str, list[str]], parent: str, child: str) -> dict[str, list[str]]:
    modified = copy.deepcopy(graph)
    modified.setdefault(child, [])
    modified[child] = sorted(set(modified[child]) - {parent})
    return complete_graph(modified)


def add_edge(graph: dict[str, list[str]], parent: str, child: str) -> dict[str, list[str]]:
    modified = copy.deepcopy(graph)
    modified.setdefault(parent, [])
    modified.setdefault(child, [])
    modified[child] = sorted(set(modified[child]) | {parent})
    return complete_graph(modified)


def reverse_edge(graph: dict[str, list[str]], parent: str, child: str) -> dict[str, list[str]]:
    return add_edge(remove_edge(graph, parent, child), child, parent)


def causal_sets(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    # The legacy proximal-set optimizer prints progress unconditionally.
    # Suppress it here so dry-runs remain machine-readable and reproducible.
    with redirect_stdout(io.StringIO()):
        sets = confounder_and_precision_sets(graph, "T", "Y")
    return {name: sorted(values) for name, values in sets.items()}


def set_delta(base: dict[str, list[str]], alternative: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    delta: dict[str, dict[str, list[str]]] = {}
    for name in sorted(set(base) | set(alternative)):
        before = set(base.get(name, []))
        after = set(alternative.get(name, []))
        added = sorted(after - before)
        removed = sorted(before - after)
        if added or removed:
            delta[name] = {"added": added, "removed": removed}
    return delta


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def parse_float(value: str | float | None) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def load_edge_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def index_statistical_decisions(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        a = str(row.get("node", "")).strip()
        b = str(row.get("candidate", "")).strip()
        if a and b and a != b:
            grouped[pair_key(a, b)].append(row)

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for key, pair_rows in grouped.items():
        accepted = [row for row in pair_rows if parse_bool(row.get("accepted_edge"))]
        orientations = [
            f"{row.get('edge_parent')}->{row.get('edge_child')}"
            for row in accepted
            if row.get("edge_parent") and row.get("edge_child")
        ]
        confidences = [
            value
            for row in accepted
            if (value := parse_float(row.get("orientation_confidence"))) is not None
        ]
        corr_values = [
            value
            for row in pair_rows
            if (value := parse_float(row.get("abs_corr"))) is not None
        ]
        ci_values = [
            value
            for row in pair_rows
            if (value := parse_float(row.get("best_ci_p_value"))) is not None
        ]
        indexed[key] = {
            "candidate_decision_count": len(pair_rows),
            "accepted_decision_count": len(accepted),
            "accepted_orientations": sorted(set(orientations)),
            "orientation_conflict": len(set(orientations)) > 1,
            "confidence_min": min(confidences) if confidences else None,
            "confidence_mean": sum(confidences) / len(confidences) if confidences else None,
            "confidence_max": max(confidences) if confidences else None,
            "max_abs_corr": max(corr_values) if corr_values else None,
            "max_ci_p_value": max(ci_values) if ci_values else None,
            "methods": sorted({str(row.get("direction_method")) for row in accepted if row.get("direction_method")}),
        }
    return indexed


def graph_provenance(meta: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    index: dict[tuple[str, str], set[str]] = defaultdict(set)
    fields = {
        "treatment_rescue_edges": "treatment_rescue",
        "outcome_rescue_edges": "outcome_rescue",
        "precision_rescue_edges": "precision_rescue",
        "post_treatment_rescue_edges": "post_treatment_rescue",
        "sanitized_edges": "sanitized",
        "pruned_edges": "pruned",
    }
    for field, label in fields.items():
        for edge in meta.get(field, []) or []:
            parent = str(edge.get("parent", "")).strip()
            child = str(edge.get("child", "")).strip()
            if parent and child and parent != child:
                index[pair_key(parent, child)].add(label)
    return index


def immutable_direction_violation(parent: str, child: str) -> bool:
    if child in HARD_EXOGENOUS and parent not in HARD_EXOGENOUS:
        return True
    if child in PREDETERMINED_ADULT and parent in TASK_NODES:
        return True
    if parent in TASK_NODES and child in HARD_EXOGENOUS | PREDETERMINED_ADULT:
        return True
    return False


def _compact_delta(delta: dict[str, dict[str, list[str]]]) -> str:
    parts = []
    for name, values in sorted(delta.items()):
        added = "+" + ",".join(values["added"]) if values["added"] else ""
        removed = "-" + ",".join(values["removed"]) if values["removed"] else ""
        parts.append(f"{name}:{added}{removed}")
    return ";".join(parts)


def build_edge_queue(
    task: str,
    graph: dict[str, list[str]],
    graph_meta: dict[str, Any],
    decision_rows: list[dict[str, Any]],
    low_confidence_threshold: float = 0.35,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = complete_graph(graph)
    base_sets = causal_sets(graph)
    final_edges = directed_edges(graph)
    final_by_pair = {pair_key(parent, child): (parent, child) for parent, child in final_edges}
    statistical = index_statistical_decisions(decision_rows)
    provenance = graph_provenance(graph_meta)

    candidate_pairs = set(final_by_pair) | set(provenance) | {pair_key("T", "Y")}
    relevant_ancestors = ancestors(graph, "T") | ancestors(graph, "Y") | TASK_NODES
    queue: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []

    for unordered in sorted(candidate_pairs):
        final_orientation = final_by_pair.get(unordered)
        if final_orientation:
            parent, child = final_orientation
            current_state = "directed_edge"
        else:
            parent, child = unordered
            current_state = "no_final_edge"

        pair_provenance = sorted(provenance.get(unordered, set()))
        stat = statistical.get(unordered, {})
        direct_task_edge = bool(set(unordered) & TASK_NODES)
        required_t_y = set(unordered) == TASK_NODES
        immutable_violation = bool(final_orientation and immutable_direction_violation(parent, child))
        stat_confidence = stat.get("confidence_mean")
        low_confidence = stat_confidence is None or stat_confidence < low_confidence_threshold
        statistical_conflict = bool(stat.get("orientation_conflict", False))
        ancestor_relevant = bool(set(unordered) & relevant_ancestors)

        remove_delta: dict[str, dict[str, list[str]]] = {}
        reverse_delta: dict[str, dict[str, list[str]]] = {}
        add_forward_delta: dict[str, dict[str, list[str]]] = {}
        add_reverse_delta: dict[str, dict[str, list[str]]] = {}
        reverse_creates_cycle = False

        if final_orientation:
            removed = remove_edge(graph, parent, child)
            remove_delta = set_delta(base_sets, causal_sets(removed))
            reversed_graph = reverse_edge(graph, parent, child)
            reverse_creates_cycle = not is_acyclic(reversed_graph)
            if not reverse_creates_cycle:
                reverse_delta = set_delta(base_sets, causal_sets(reversed_graph))
        else:
            forward_graph = add_edge(graph, parent, child)
            if is_acyclic(forward_graph):
                add_forward_delta = set_delta(base_sets, causal_sets(forward_graph))
            reverse_graph = add_edge(graph, child, parent)
            if is_acyclic(reverse_graph):
                add_reverse_delta = set_delta(base_sets, causal_sets(reverse_graph))

        set_sensitive = bool(remove_delta or reverse_delta or add_forward_delta or add_reverse_delta)
        semantic_sensitive = immutable_violation or "sanitized" in pair_provenance
        rescue_edge = any(label.endswith("rescue") for label in pair_provenance)
        uncertain_relevant = (low_confidence or statistical_conflict) and ancestor_relevant

        selection_reasons: list[str] = []
        if required_t_y:
            selection_reasons.append("fixed_treatment_outcome_query")
        if direct_task_edge:
            selection_reasons.append("touches_treatment_or_outcome")
        if set_sensitive:
            selection_reasons.append("changes_c_or_s_set_under_perturbation")
        if immutable_violation:
            selection_reasons.append("violates_predetermined_variable_order")
        if rescue_edge:
            selection_reasons.append("algorithmic_rescue_edge")
        if "sanitized" in pair_provenance:
            selection_reasons.append("task_sanitizer_audit")
        if statistical_conflict:
            selection_reasons.append("conflicting_statistical_orientations")
        if uncertain_relevant:
            selection_reasons.append("low_or_missing_statistical_confidence")

        selected = bool(
            required_t_y
            or direct_task_edge
            or set_sensitive
            or semantic_sensitive
            or rescue_edge
            or uncertain_relevant
        )

        if required_t_y or immutable_violation or (direct_task_edge and set_sensitive):
            priority = "P0"
        elif direct_task_edge or set_sensitive or rescue_edge or "sanitized" in pair_provenance:
            priority = "P1"
        else:
            priority = "P2"

        pair_id = f"{task}::{unordered[0]}::{unordered[1]}"
        record = {
            "task": task,
            "pair_id": pair_id,
            "node_a": unordered[0],
            "node_b": unordered[1],
            "current_parent": parent if final_orientation else "",
            "current_child": child if final_orientation else "",
            "current_state": current_state,
            "priority": priority,
            "selected": selected,
            "selection_reasons": selection_reasons,
            "direct_task_edge": direct_task_edge,
            "required_t_y": required_t_y,
            "ancestor_relevant": ancestor_relevant,
            "set_sensitive": set_sensitive,
            "immutable_direction_violation": immutable_violation,
            "low_statistical_confidence": low_confidence,
            "statistical_conflict": statistical_conflict,
            "statistical_evidence": stat,
            "provenance": pair_provenance,
            "remove_set_delta": remove_delta,
            "reverse_set_delta": reverse_delta,
            "add_forward_set_delta": add_forward_delta,
            "add_reverse_set_delta": add_reverse_delta,
            "reverse_creates_cycle": reverse_creates_cycle,
        }
        all_candidates.append(record)
        if selected:
            queue.append(record)

    queue.sort(key=lambda row: (row["priority"], row["pair_id"]))
    summary = {
        "task": task,
        "node_count": len(graph),
        "edge_count": len(final_edges),
        "graph_is_acyclic": is_acyclic(graph),
        "base_sets": base_sets,
        "candidate_pair_count": len(all_candidates),
        "selected_pair_count": len(queue),
        "priority_counts": {
            priority: sum(row["priority"] == priority for row in queue)
            for priority in ["P0", "P1", "P2"]
        },
        "selection_reason_counts": {
            reason: sum(reason in row["selection_reasons"] for row in queue)
            for reason in sorted({reason for row in queue for reason in row["selection_reasons"]})
        },
    }
    return queue, summary


def flatten_queue_record(record: dict[str, Any]) -> dict[str, Any]:
    stat = record.get("statistical_evidence", {})
    return {
        "task": record["task"],
        "pair_id": record["pair_id"],
        "node_a": record["node_a"],
        "node_b": record["node_b"],
        "current_parent": record["current_parent"],
        "current_child": record["current_child"],
        "current_state": record["current_state"],
        "priority": record["priority"],
        "selection_reasons": ";".join(record["selection_reasons"]),
        "direct_task_edge": record["direct_task_edge"],
        "required_t_y": record["required_t_y"],
        "ancestor_relevant": record["ancestor_relevant"],
        "set_sensitive": record["set_sensitive"],
        "immutable_direction_violation": record["immutable_direction_violation"],
        "low_statistical_confidence": record["low_statistical_confidence"],
        "statistical_conflict": record["statistical_conflict"],
        "statistical_confidence_mean": stat.get("confidence_mean"),
        "statistical_confidence_min": stat.get("confidence_min"),
        "statistical_confidence_max": stat.get("confidence_max"),
        "accepted_statistical_orientations": ";".join(stat.get("accepted_orientations", [])),
        "statistical_methods": ";".join(stat.get("methods", [])),
        "max_abs_corr": stat.get("max_abs_corr"),
        "max_ci_p_value": stat.get("max_ci_p_value"),
        "provenance": ";".join(record["provenance"]),
        "remove_set_delta": _compact_delta(record["remove_set_delta"]),
        "reverse_set_delta": _compact_delta(record["reverse_set_delta"]),
        "add_forward_set_delta": _compact_delta(record["add_forward_set_delta"]),
        "add_reverse_set_delta": _compact_delta(record["add_reverse_set_delta"]),
        "reverse_creates_cycle": record["reverse_creates_cycle"],
    }


def write_queue_csv(path: Path, queue: list[dict[str, Any]]) -> None:
    rows = [flatten_queue_record(record) for record in queue]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def materialize_node_descriptions(
    run_meta: dict[str, Any],
    catalog: dict[str, Any],
    task_config: dict[str, Any],
) -> dict[str, Any]:
    descriptions: dict[str, Any] = {}
    overrides = task_config.get("task_specific_overrides", {})
    for node in run_meta.get("features", []):
        if node not in catalog:
            raise KeyError(f"No NHANES semantic description for feature {node!r}")
        description = copy.deepcopy(catalog[node])
        description.update(copy.deepcopy(overrides.get(node, {})))
        descriptions[node] = description
    descriptions["T"] = copy.deepcopy(task_config["treatment"])
    descriptions["Y"] = copy.deepcopy(task_config["outcome"])
    return descriptions


def _pair_payload(record: dict[str, Any], node_descriptions: dict[str, Any], vote_index: int) -> dict[str, Any]:
    if record["current_state"] == "directed_edge":
        x, y = record["current_parent"], record["current_child"]
    else:
        x, y = record["node_a"], record["node_b"]
    if vote_index % 2 == 1:
        x, y = y, x

    if record["current_state"] != "directed_edge":
        current_direction = "NO_FINAL_EDGE"
    elif record["current_parent"] == x and record["current_child"] == y:
        current_direction = "X_TO_Y"
    else:
        current_direction = "Y_TO_X"

    return {
        "pair_id": record["pair_id"],
        "X": x,
        "Y": y,
        "X_description": node_descriptions[x],
        "Y_description": node_descriptions[y],
        "current_statistical_direction": current_direction,
        "audit_priority": record["priority"],
        "selection_reasons": record["selection_reasons"],
        "statistical_evidence": record["statistical_evidence"],
        "graph_perturbation_impact": {
            "removing_current_edge": record["remove_set_delta"],
            "reversing_current_edge": record["reverse_set_delta"],
            "adding_X_to_Y_or_original_forward": record["add_forward_set_delta"],
            "adding_Y_to_X_or_original_reverse": record["add_reverse_set_delta"],
            "reverse_creates_cycle": record["reverse_creates_cycle"],
        },
        "instruction": "Judge the plausible direct causal relation. Do not defer merely because the statistical graph has a direction; use UNRESOLVED only for genuine semantic or temporal non-identifiability.",
    }


def build_messages(
    prompt_template: dict[str, Any],
    task_name: str,
    domain_context: str,
    records: list[dict[str, Any]],
    node_descriptions: dict[str, Any],
    vote_index: int,
) -> list[dict[str, str]]:
    payload = {
        "task": task_name,
        "domain_context": domain_context,
        "vote_index": vote_index + 1,
        "position_bias_control": "X/Y order is counterbalanced across votes; judge the named variables, not list position.",
        "judgement_labels": prompt_template["judgement_labels"],
        "pairs": [_pair_payload(record, node_descriptions, vote_index) for record in records],
        "output_requirements": {
            "strict_json_only": True,
            "schema": {"decisions": [prompt_template["output_schema"]]},
            "one_decision_per_pair": True,
            "copy_pair_id_exactly": True,
        },
    }
    return [
        {"role": "system", "content": prompt_template["system"]},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def estimate_text_tokens(text: str) -> dict[str, int]:
    ascii_count = sum(ord(char) < 128 for char in text)
    non_ascii_count = len(text) - ascii_count
    return {
        "lower": math.ceil(ascii_count / 4.5 + non_ascii_count / 2.0),
        "central": math.ceil(ascii_count / 4.0 + non_ascii_count / 1.5),
        "upper": math.ceil(ascii_count / 3.0 + non_ascii_count),
    }


def estimate_message_tokens(messages: list[dict[str, str]]) -> dict[str, int]:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    estimate = estimate_text_tokens(serialized)
    overhead = 12 + 4 * len(messages)
    return {name: value + overhead for name, value in estimate.items()}


def chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def estimate_call_plan(
    task_resources: list[dict[str, Any]],
    prompt_template: dict[str, Any],
    votes: int,
    batch_size: int,
    max_output_tokens_per_edge: int,
    reasoning_token_reserve_per_edge: int = 0,
) -> dict[str, Any]:
    task_budgets = []
    total_input = {"lower": 0, "central": 0, "upper": 0}
    total_edges = 0
    total_calls = 0
    for resource in task_resources:
        task_input = {"lower": 0, "central": 0, "upper": 0}
        calls = 0
        for vote_index in range(votes):
            for batch in chunks(resource["queue"], batch_size):
                estimate = estimate_message_tokens(
                    build_messages(
                        prompt_template,
                        resource["task_display_name"],
                        resource["domain_context"],
                        batch,
                        resource["node_descriptions"],
                        vote_index,
                    )
                )
                for name in task_input:
                    task_input[name] += estimate[name]
                calls += 1
        edge_count = len(resource["queue"])
        visible_output_upper = edge_count * votes * max_output_tokens_per_edge
        reasoning_reserve = edge_count * votes * reasoning_token_reserve_per_edge
        output_upper = visible_output_upper + reasoning_reserve
        task_budgets.append(
            {
                "task": resource["task"],
                "edge_pairs": edge_count,
                "votes_per_edge": votes,
                "batch_size": batch_size,
                "api_calls": calls,
                "estimated_input_tokens": task_input,
                "configured_visible_output_token_upper_bound": visible_output_upper,
                "reasoning_token_reserve": reasoning_reserve,
                "configured_output_token_upper_bound": output_upper,
                "estimated_total_token_upper_bound": task_input["upper"] + output_upper,
            }
        )
        for name in total_input:
            total_input[name] += task_input[name]
        total_edges += edge_count
        total_calls += calls

    total_visible_output_upper = total_edges * votes * max_output_tokens_per_edge
    total_reasoning_reserve = total_edges * votes * reasoning_token_reserve_per_edge
    total_output_upper = total_visible_output_upper + total_reasoning_reserve
    return {
        "batch_size": batch_size,
        "votes_per_edge": votes,
        "edge_pairs": total_edges,
        "api_calls": total_calls,
        "estimated_input_tokens": total_input,
        "configured_visible_output_token_upper_bound": total_visible_output_upper,
        "reasoning_token_reserve": total_reasoning_reserve,
        "configured_output_token_upper_bound": total_output_upper,
        "estimated_total_token_upper_bound": total_input["upper"] + total_output_upper,
        "tasks": task_budgets,
    }
