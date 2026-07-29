"""Voting and DAG-consistent graph merge for statistical and semantic CCIA evidence."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from typing import Any

from ccia.llm_causal.semantic_adjudication import (
    add_edge,
    causal_sets,
    complete_graph,
    directed_edges,
    immutable_direction_violation,
    is_acyclic,
    pair_key,
    remove_edge,
)


DIRECTION_LABELS = {"A_TO_B", "B_TO_A"}


def needs_tiebreak(votes: list[dict[str, Any]], min_confidence: float = 0.60) -> bool:
    if len(votes) < 2:
        return True
    labels = [vote["canonical_judgement"] for vote in votes]
    if labels[0] != labels[1] or labels[0] == "UNRESOLVED":
        return True
    return min(float(vote["confidence"]) for vote in votes[:2]) < min_confidence


def aggregate_votes(
    votes: list[dict[str, Any]],
    queue: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for vote in votes:
        by_pair[vote["pair_id"]].append(vote)

    decisions = []
    for record in queue:
        pair_votes = sorted(by_pair.get(record["pair_id"], []), key=lambda vote: vote["vote_index"])
        counts = Counter(vote["canonical_judgement"] for vote in pair_votes)
        winner = "UNRESOLVED"
        winning_count = 0
        if counts:
            candidate, count = counts.most_common(1)[0]
            if count >= 2 or (len(pair_votes) == 1 and candidate != "UNRESOLVED"):
                winner = candidate
                winning_count = count
        winning_votes = [vote for vote in pair_votes if vote["canonical_judgement"] == winner]
        mean_confidence = (
            sum(float(vote["confidence"]) for vote in winning_votes) / len(winning_votes)
            if winning_votes
            else 0.0
        )
        agreement_ratio = winning_count / len(pair_votes) if pair_votes else 0.0
        semantic_confidence = mean_confidence * agreement_ratio

        if winner == "A_TO_B":
            proposed_parent, proposed_child = record["node_a"], record["node_b"]
        elif winner == "B_TO_A":
            proposed_parent, proposed_child = record["node_b"], record["node_a"]
        else:
            proposed_parent = proposed_child = ""

        decisions.append(
            {
                "task": record["task"],
                "pair_id": record["pair_id"],
                "node_a": record["node_a"],
                "node_b": record["node_b"],
                "priority": record["priority"],
                "current_parent": record["current_parent"],
                "current_child": record["current_child"],
                "consensus_judgement": winner,
                "proposed_parent": proposed_parent,
                "proposed_child": proposed_child,
                "semantic_confidence": semantic_confidence,
                "mean_winning_confidence": mean_confidence,
                "agreement_ratio": agreement_ratio,
                "vote_count": len(pair_votes),
                "vote_counts": dict(counts),
                "providers": [vote["provider"] for vote in pair_votes],
                "models": [vote["model"] for vote in pair_votes],
                "reasons": [vote["reason"] for vote in winning_votes],
                "constraints_applied": sorted(
                    {
                        constraint
                        for vote in winning_votes
                        for constraint in vote.get("constraints_applied", [])
                    }
                ),
                "selection_reasons": record["selection_reasons"],
                "statistical_evidence": record["statistical_evidence"],
                "provenance": record["provenance"],
            }
        )
    return decisions


def _remove_pair(graph: dict[str, list[str]], a: str, b: str) -> dict[str, list[str]]:
    graph = remove_edge(graph, a, b)
    return remove_edge(graph, b, a)


def _violates_task_order(parent: str, child: str) -> bool:
    if immutable_direction_violation(parent, child):
        return True
    if parent == "Y" and child != "Y":
        return True
    if parent == "T" and child not in {"Y"}:
        return True
    if parent == "Y" and child == "T":
        return True
    return False


def merge_semantic_graph(
    graph: dict[str, list[str]],
    decisions: list[dict[str, Any]],
    *,
    min_direction_confidence: float = 0.60,
    min_remove_confidence: float = 0.75,
    min_add_confidence: float = 0.75,
) -> tuple[dict[str, list[str]], list[dict[str, Any]], dict[str, Any]]:
    original = complete_graph(graph)
    merged = copy.deepcopy(original)
    actions: list[dict[str, Any]] = []

    ordered = sorted(
        decisions,
        key=lambda decision: (
            decision["priority"],
            -float(decision["semantic_confidence"]),
            decision["pair_id"],
        ),
    )
    for decision in ordered:
        a, b = decision["node_a"], decision["node_b"]
        label = decision["consensus_judgement"]
        confidence = float(decision["semantic_confidence"])
        original_has_edge = (a, b) in directed_edges(merged) or (b, a) in directed_edges(merged)
        action = "retained_statistical"
        note = ""

        if {a, b} == {"T", "Y"}:
            merged = _remove_pair(merged, a, b)
            merged = add_edge(merged, "T", "Y")
            action = "fixed_treatment_outcome"
            note = "intervention_query_constraint"
        elif label == "UNRESOLVED":
            action = "retained_statistical"
            note = "semantic_unresolved"
        elif label == "NO_DIRECT_EDGE":
            if confidence >= min_remove_confidence:
                merged = _remove_pair(merged, a, b)
                action = "removed_by_semantic_consensus"
            else:
                action = "retained_statistical"
                note = "semantic removal confidence below conservative threshold"
        elif label in DIRECTION_LABELS:
            parent = decision["proposed_parent"]
            child = decision["proposed_child"]
            if confidence < min_direction_confidence:
                action = "retained_statistical"
                note = "semantic direction confidence below threshold"
            elif _violates_task_order(parent, child):
                merged = _remove_pair(merged, a, b)
                action = "removed_invalid_semantic_direction"
                note = f"blocked {parent}->{child} by task-order constraint"
            elif not original_has_edge and confidence < min_add_confidence:
                action = "kept_absent_low_add_confidence"
                note = "semantic evidence insufficient to add a missing edge"
            else:
                candidate = add_edge(_remove_pair(merged, a, b), parent, child)
                if is_acyclic(candidate):
                    merged = candidate
                    action = "oriented_by_semantic_consensus"
                else:
                    merged = _remove_pair(merged, a, b)
                    action = "removed_cycle_blocked_orientation"
                    note = f"proposed {parent}->{child} would create a directed cycle"

        actions.append(
            {
                "task": decision["task"],
                "pair_id": decision["pair_id"],
                "node_a": a,
                "node_b": b,
                "consensus_judgement": label,
                "semantic_confidence": confidence,
                "action": action,
                "note": note,
            }
        )

    # These are design-level temporal constraints, not uncertain medical
    # priors. Remove any residual impossible direction that was retained only
    # because semantic votes were unresolved.
    for parent, child in sorted(directed_edges(merged)):
        if (parent, child) == ("T", "Y"):
            continue
        if _violates_task_order(parent, child):
            merged = remove_edge(merged, parent, child)
            actions.append(
                {
                    "task": decisions[0]["task"] if decisions else "",
                    "pair_id": f"hard_constraint::{parent}::{child}",
                    "node_a": parent,
                    "node_b": child,
                    "consensus_judgement": "HARD_CONSTRAINT",
                    "semantic_confidence": 1.0,
                    "action": "removed_by_hard_task_order",
                    "note": f"removed impossible residual direction {parent}->{child}",
                }
            )

    # Enforce the intervention query after all pairwise operations.
    merged = _remove_pair(merged, "T", "Y")
    merged = add_edge(merged, "T", "Y")
    if not is_acyclic(merged):
        raise ValueError("Semantic merge produced a cyclic graph after enforcing T->Y")

    before_edges = directed_edges(original)
    after_edges = directed_edges(merged)
    before_sets = causal_sets(original)
    after_sets = causal_sets(merged)
    summary = {
        "graph_is_acyclic": is_acyclic(merged),
        "edge_count_before": len(before_edges),
        "edge_count_after": len(after_edges),
        "edges_added": sorted([list(edge) for edge in after_edges - before_edges]),
        "edges_removed": sorted([list(edge) for edge in before_edges - after_edges]),
        "sets_before": before_sets,
        "sets_after": after_sets,
        "action_counts": dict(Counter(action["action"] for action in actions)),
        "fusion_thresholds": {
            "min_direction_confidence": min_direction_confidence,
            "min_remove_confidence": min_remove_confidence,
            "min_add_confidence": min_add_confidence,
        },
    }
    return merged, actions, summary


def graph_diff_rows(
    before: dict[str, list[str]],
    after: dict[str, list[str]],
    task: str,
) -> list[dict[str, Any]]:
    before_edges = directed_edges(before)
    after_edges = directed_edges(after)
    rows = []
    for parent, child in sorted(before_edges | after_edges):
        rows.append(
            {
                "task": task,
                "parent": parent,
                "child": child,
                "in_statistical_graph": (parent, child) in before_edges,
                "in_semantic_graph": (parent, child) in after_edges,
                "change": (
                    "retained"
                    if (parent, child) in before_edges & after_edges
                    else "added"
                    if (parent, child) in after_edges
                    else "removed"
                ),
            }
        )
    return rows


def set_diff_rows(before: dict[str, list[str]], after: dict[str, list[str]], task: str) -> list[dict[str, Any]]:
    rows = []
    for set_name in sorted(set(before) | set(after)):
        before_values = set(before.get(set_name, []))
        after_values = set(after.get(set_name, []))
        for variable in sorted(before_values | after_values):
            rows.append(
                {
                    "task": task,
                    "set_name": set_name,
                    "variable": variable,
                    "in_statistical_set": variable in before_values,
                    "in_semantic_set": variable in after_values,
                    "change": (
                        "retained"
                        if variable in before_values & after_values
                        else "added"
                        if variable in after_values
                        else "removed"
                    ),
                }
            )
    return rows
