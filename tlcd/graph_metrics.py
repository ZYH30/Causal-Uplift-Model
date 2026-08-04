"""Graph recovery metrics for TLCD synthetic experiments."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from graph_roles import find_confounder_sets, get_descendants, get_parents, invert_graph


def complete_graph(graph: dict[str, list[str]], nodes: set[str] | list[str] | None = None) -> dict[str, list[str]]:
    all_nodes = set([] if nodes is None else list(nodes))
    all_nodes.update(graph.keys())
    for parents in graph.values():
        all_nodes.update(parents)
    return {node: sorted(set(graph.get(node, [])) - {node}) for node in sorted(all_nodes)}


def directed_edges(graph: dict[str, list[str]]) -> set[tuple[str, str]]:
    return {
        (parent, child)
        for child, parents in graph.items()
        for parent in parents
        if parent != child
    }


def skeleton_edges(graph: dict[str, list[str]]) -> set[frozenset[str]]:
    return {frozenset(edge) for edge in directed_edges(graph)}


def prf(true_set: set[Any], pred_set: set[Any]) -> dict[str, float]:
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    jaccard = tp / len(true_set | pred_set) if true_set | pred_set else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "true_size": float(len(true_set)),
        "pred_size": float(len(pred_set)),
    }


def structural_hamming_distance(
    true_graph: dict[str, list[str]],
    pred_graph: dict[str, list[str]],
    nodes: set[str] | list[str] | None = None,
) -> int:
    nodes_set = set(nodes or [])
    nodes_set.update(true_graph.keys())
    nodes_set.update(pred_graph.keys())
    true_edges = directed_edges(true_graph)
    pred_edges = directed_edges(pred_graph)
    shd = 0
    nodes_list = sorted(nodes_set)
    for i, a in enumerate(nodes_list):
        for b in nodes_list[i + 1 :]:
            true_ab = (a, b) in true_edges
            true_ba = (b, a) in true_edges
            pred_ab = (a, b) in pred_edges
            pred_ba = (b, a) in pred_edges
            true_state = "ab" if true_ab else "ba" if true_ba else "none"
            pred_state = "ab" if pred_ab else "ba" if pred_ba else "none"
            if true_state != pred_state:
                shd += 1
    return shd


def ancestors(graph: dict[str, list[str]], node: str) -> set[str]:
    out = set()
    queue = deque(graph.get(node, []))
    visited = set(graph.get(node, []))
    while queue:
        current = queue.popleft()
        if current == node:
            continue
        out.add(current)
        for parent in graph.get(current, []):
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)
    return out


def has_directed_path(graph: dict[str, list[str]], source: str, target: str) -> bool:
    graph_adj = invert_graph(graph)
    queue = deque(graph_adj.get(source, []))
    visited = set(graph_adj.get(source, []))
    while queue:
        current = queue.popleft()
        if current == target:
            return True
        for child in graph_adj.get(current, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return False


def precision_variables(graph: dict[str, list[str]], t: str, y: str, c_set: set[str]) -> set[str]:
    graph_adj = invert_graph(graph)
    parents_y = get_parents(graph, y)
    descendants_t = get_descendants(graph_adj, t)
    return set(parents_y) - set(descendants_t) - {t} - set(c_set)


def confounder_and_precision_sets(graph: dict[str, list[str]], t: str, y: str) -> dict[str, set[str]]:
    conf = find_confounder_sets(graph, t, y)
    total = set(conf.get("total", []))
    proximal = set(conf.get("proximal", []))
    root = set(conf.get("root", []))
    return {
        "C_Total": total,
        "C_Proximal": proximal,
        "C_Root": root,
        "S_Prec": precision_variables(graph, t, y, proximal),
    }


def flatten_metric(prefix: str, metric: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metric.items()}


def evaluate_graph_recovery(
    true_graph: dict[str, list[str]],
    pred_graph: dict[str, list[str]],
    t: str = "T",
    y: str = "Y",
    nodes: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    nodes_set = set([] if nodes is None else list(nodes))
    nodes_set.update(true_graph.keys())
    nodes_set.update(pred_graph.keys())
    true_graph = complete_graph(true_graph, nodes_set)
    pred_graph = complete_graph(pred_graph, nodes_set)

    true_directed = directed_edges(true_graph)
    pred_directed = directed_edges(pred_graph)
    true_skeleton = skeleton_edges(true_graph)
    pred_skeleton = skeleton_edges(pred_graph)

    metrics: dict[str, Any] = {
        "directed": prf(true_directed, pred_directed),
        "skeleton": prf(true_skeleton, pred_skeleton),
        "shd": structural_hamming_distance(true_graph, pred_graph, nodes_set),
        "true_edge_count": len(true_directed),
        "pred_edge_count": len(pred_directed),
    }

    local_sets = {
        "Parents_Y": (set(true_graph.get(y, [])), set(pred_graph.get(y, []))),
        "Parents_T": (set(true_graph.get(t, [])), set(pred_graph.get(t, []))),
        "Ancestors_Y": (ancestors(true_graph, y), ancestors(pred_graph, y)),
        "Ancestors_T": (ancestors(true_graph, t), ancestors(pred_graph, t)),
    }
    metrics["local"] = {name: prf(true_set, pred_set) for name, (true_set, pred_set) in local_sets.items()}

    try:
        true_sets = confounder_and_precision_sets(true_graph, t, y)
        pred_sets = confounder_and_precision_sets(pred_graph, t, y)
        metrics["rite_sets"] = {
            name: prf(true_sets[name], pred_sets[name])
            for name in ["C_Total", "C_Proximal", "C_Root", "S_Prec"]
        }
        metrics["true_sets"] = {name: sorted(values) for name, values in true_sets.items()}
        metrics["pred_sets"] = {name: sorted(values) for name, values in pred_sets.items()}
    except Exception as exc:
        metrics["rite_sets_error"] = str(exc)

    metrics["path_checks"] = {
        "true_T_to_Y_edge": (t, y) in true_directed,
        "pred_T_to_Y_edge": (t, y) in pred_directed,
        "true_T_to_Y_path": has_directed_path(true_graph, t, y),
        "pred_T_to_Y_path": has_directed_path(pred_graph, t, y),
    }

    flat: dict[str, float] = {
        "SHD": float(metrics["shd"]),
        "true_edge_count": float(metrics["true_edge_count"]),
        "pred_edge_count": float(metrics["pred_edge_count"]),
    }
    flat.update(flatten_metric("directed", metrics["directed"]))
    flat.update(flatten_metric("skeleton", metrics["skeleton"]))
    for name, metric in metrics.get("local", {}).items():
        flat.update(flatten_metric(name, metric))
    for name, metric in metrics.get("rite_sets", {}).items():
        flat.update(flatten_metric(name, metric))
    for name, value in metrics["path_checks"].items():
        flat[name] = float(value)
    metrics["flat"] = flat
    return metrics
