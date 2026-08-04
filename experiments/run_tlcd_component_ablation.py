"""Component ablation for the task-adaptive TLCD graph learner."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlcd.tlcd_runner import (
    complete_graph,
    merge_graphs,
    project_graph_to_dag,
    run_tlcd,
)
from tlcd.graph_metrics import (
    directed_edges,
    evaluate_graph_recovery,
    prf,
    skeleton_edges,
    structural_hamming_distance,
)
from experiments.run_tlcd_synthetic import (
    detect_post_treatment_variables,
    outcome_importance_without_post_treatment,
    prune_deeper_edges_by_shallow_nonadjacency,
    rescue_outcome_edges_from_treatment_parents,
    rescue_post_treatment_children,
    rescue_precision_outcome_edges,
    rescue_treatment_parent_edges,
    treatment_importance_without_post_outcome,
)
from experiments.run_systematic_simulation import tlcd_config, generate_environment


def raw_tlcd(
    frame: pd.DataFrame,
    observed: list[str],
    seed: int,
    direction_strategy: str,
) -> tuple[dict[str, list[str]], list[dict]]:
    results = []
    for target in ("Y", "T"):
        results.append(
            run_tlcd(
                frame[observed],
                target=target,
                model="lgb",
                cor_quantile=0.75,
                min_abs_corr=0.03,
                ci_theta=0.05,
                direct_theta=0.10,
                anm_margin=0.03,
                max_condition_set_size=2,
                max_neighbors=16,
                max_corr_neighbors=10,
                max_importance_neighbors=12,
                max_condition_candidates=10,
                max_search_depth=1,
                min_samples=50,
                lgb_estimators=60,
                lgb_max_depth=3,
                lgb_num_leaves=15,
                lgb_min_child_samples=30,
                llm_mode="off",
                direction_strategy=direction_strategy,
                random_state=seed,
                verbose=False,
            )
        )
    merged = merge_graphs(
        results,
        nodes=observed,
        prefer_edges={("T", "Y")},
    )
    graph, _ = prune_deeper_edges_by_shallow_nonadjacency(
        merged["graph"],
        results,
        tlcd_config(seed),
    )
    return graph, results


def repair_graph(
    frame: pd.DataFrame,
    observed: list[str],
    graph: dict[str, list[str]],
    seed: int,
) -> tuple[dict[str, list[str]], set[tuple[str, str]]]:
    args = tlcd_config(seed)
    post_scores = detect_post_treatment_variables(
        frame,
        observed,
        "T",
        "Y",
        args,
    )
    outcome_importance, _ = outcome_importance_without_post_treatment(
        frame,
        observed,
        "T",
        "Y",
        args,
        post_scores=post_scores,
    )
    treatment_importance, _ = treatment_importance_without_post_outcome(
        frame,
        observed,
        "T",
        "Y",
        args,
        post_scores=post_scores,
    )
    graph, treatment_edges = rescue_treatment_parent_edges(
        frame,
        graph,
        observed,
        "T",
        "Y",
        args,
        post_scores=post_scores,
        treatment_importance=treatment_importance,
    )
    graph, outcome_edges = rescue_outcome_edges_from_treatment_parents(
        frame,
        graph,
        observed,
        "T",
        "Y",
        args,
        post_scores=post_scores,
        outcome_importance=outcome_importance,
    )
    graph, precision_edges = rescue_precision_outcome_edges(
        frame,
        graph,
        observed,
        "T",
        "Y",
        args,
        post_scores=post_scores,
        outcome_importance=outcome_importance,
    )
    graph, post_edges = rescue_post_treatment_children(
        frame,
        graph,
        observed,
        "T",
        "Y",
        args,
        post_scores=post_scores,
    )
    protected = {("T", "Y")}
    for edges in (
        treatment_edges,
        outcome_edges,
        precision_edges,
        post_edges,
    ):
        protected.update((edge["parent"], edge["child"]) for edge in edges)
    return graph, protected


def is_dag(graph: dict[str, list[str]], observed: list[str]) -> bool:
    directed = nx.DiGraph()
    directed.add_nodes_from(observed)
    directed.add_edges_from(directed_edges(graph))
    return nx.is_directed_acyclic_graph(directed)


def metric_row(
    true_graph: dict[str, list[str]],
    graph: dict[str, list[str]],
    observed: list[str],
    role_metrics: bool,
) -> dict[str, float]:
    true_graph = complete_graph(true_graph, observed)
    graph = complete_graph(graph, observed)
    direct = prf(directed_edges(true_graph), directed_edges(graph))
    skeleton = prf(skeleton_edges(true_graph), skeleton_edges(graph))
    row = {
        "Directed_Precision": direct["precision"],
        "Directed_Recall": direct["recall"],
        "Directed_F1": direct["f1"],
        "Skeleton_F1": skeleton["f1"],
        "SHD": structural_hamming_distance(
            true_graph,
            graph,
            observed,
        ),
        "DAG_Valid": is_dag(graph, observed),
        "Predicted_Edges": len(directed_edges(graph)),
    }
    if role_metrics:
        with contextlib.redirect_stdout(io.StringIO()):
            full = evaluate_graph_recovery(
                true_graph,
                graph,
                t="T",
                y="Y",
                nodes=observed,
            )["flat"]
        row.update(
            {
                "Parents_T_F1": full["Parents_T_f1"],
                "Parents_Y_F1": full["Parents_Y_f1"],
                "C_Proximal_F1": full.get("C_Proximal_f1", np.nan),
                "S_Precision_F1": full.get("S_Prec_f1", np.nan),
            }
        )
    else:
        row.update(
            {
                "Parents_T_F1": np.nan,
                "Parents_Y_F1": np.nan,
                "C_Proximal_F1": np.nan,
                "S_Precision_F1": np.nan,
            }
        )
    return row


def run_seed(
    seed: int,
    n: int,
    scenario: str,
) -> list[dict]:
    frame, true_graph, _ = generate_environment(n, seed, scenario)
    observed = [column for column in frame if column != "true_tau"]
    rows = []

    started = time.perf_counter()
    task_raw, _ = raw_tlcd(
        frame,
        observed,
        seed,
        "task_adaptive",
    )
    raw_runtime = time.perf_counter() - started
    no_repair = project_graph_to_dag(
        task_raw,
        nodes=observed,
        protected_edges={("T", "Y")},
    )["graph"]
    rows.append(
        {
            "Seed": seed,
            "Variant": "TaskAdaptive_NoRepair",
            "Runtime_Sec": raw_runtime,
            **metric_row(true_graph, no_repair, observed, True),
        }
    )

    started = time.perf_counter()
    repaired, protected = repair_graph(
        frame,
        observed,
        task_raw,
        seed,
    )
    repair_runtime = time.perf_counter() - started
    rows.append(
        {
            "Seed": seed,
            "Variant": "TaskAdaptive_NoDAGGate",
            "Runtime_Sec": raw_runtime + repair_runtime,
            **metric_row(
                true_graph,
                repaired,
                observed,
                is_dag(repaired, observed),
            ),
        }
    )
    complete = project_graph_to_dag(
        repaired,
        nodes=observed,
        protected_edges=protected,
    )["graph"]
    rows.append(
        {
            "Seed": seed,
            "Variant": "Complete_TLCD",
            "Runtime_Sec": raw_runtime + repair_runtime,
            **metric_row(true_graph, complete, observed, True),
        }
    )

    started = time.perf_counter()
    anm_raw, _ = raw_tlcd(
        frame,
        observed,
        seed,
        "anm_only",
    )
    anm_repaired, anm_protected = repair_graph(
        frame,
        observed,
        anm_raw,
        seed,
    )
    anm = project_graph_to_dag(
        anm_repaired,
        nodes=observed,
        protected_edges=anm_protected,
    )["graph"]
    rows.append(
        {
            "Seed": seed,
            "Variant": "PairwiseANM_FullRepair",
            "Runtime_Sec": time.perf_counter() - started,
            **metric_row(true_graph, anm, observed, True),
        }
    )
    for row in rows:
        row.update({"N": n, "Scenario": scenario})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--scenario", default="nonlinear_reference")
    parser.add_argument(
        "--seeds",
        default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/tlcd_component_ablation_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_seed, seed, args.n, args.scenario): seed
            for seed in seeds
        }
        for index, future in enumerate(as_completed(futures), start=1):
            seed = futures[future]
            rows.extend(future.result())
            pd.DataFrame(rows).to_csv(
                args.out_dir / "tlcd_component_ablation.csv",
                index=False,
            )
            print(
                f"Ablation seed {seed} complete ({index}/{len(seeds)})",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    metrics = [
        "Directed_Precision",
        "Directed_Recall",
        "Directed_F1",
        "Skeleton_F1",
        "SHD",
        "Parents_T_F1",
        "Parents_Y_F1",
        "C_Proximal_F1",
        "S_Precision_F1",
        "Runtime_Sec",
    ]
    summary = (
        frame.groupby("Variant")[metrics]
        .agg(["mean", "std"])
        .stack(level=0, future_stack=True)
        .reset_index()
        .rename(columns={"level_1": "Metric", "mean": "Mean", "std": "SD"})
    )
    summary.to_csv(
        args.out_dir / "tlcd_component_ablation_summary.csv",
        index=False,
    )
    manifest = {
        "scenario": args.scenario,
        "n": args.n,
        "seeds": seeds,
        "llm_mode": "off",
        "pre_treatment_whitelist": False,
        "variants": sorted(frame["Variant"].unique()),
    }
    (args.out_dir / "tlcd_component_ablation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
