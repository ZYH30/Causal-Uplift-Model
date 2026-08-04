"""Systematic graph and treatment-effect benchmark for TLCD-RITE."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tlcd.tlcd_runner import merge_graphs, project_graph_to_dag, run_tlcd
from tlcd.graph_metrics import (
    complete_graph,
    confounder_and_precision_sets,
    directed_edges,
    evaluate_graph_recovery,
    prf,
    skeleton_edges,
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


SCENARIOS = (
    "nonlinear_reference",
    "linear_additive",
    "weak_edges",
    "noisy_missing",
)
METHODS = ("TLCD", "PC", "FCI", "GES", "DirectLiNGAM")


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = float(values.std())
    if scale <= 1e-12:
        return values - values.mean()
    return (values - values.mean()) / scale


def generate_environment(
    n: int,
    seed: int,
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, list[str]], np.ndarray]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    roots = rng.normal(size=(n, 10))
    root_names = [f"C_root_{index}" for index in range(10)]
    data = {
        name: roots[:, index].copy()
        for index, name in enumerate(root_names)
    }
    z1 = rng.normal(size=n)
    z2 = rng.uniform(-2.0, 2.0, size=n)
    p1 = rng.normal(size=n)
    p2 = rng.chisquare(df=2, size=n) - 2.0
    noise1 = rng.normal(size=n)
    noise2 = rng.normal(size=n)

    if scenario == "linear_additive":
        c1 = _standardize(
            roots[:, 0]
            + 0.8 * roots[:, 1]
            + 0.5 * roots[:, 2]
            + 0.4 * roots[:, 3]
            + rng.normal(0.0, 0.15, n)
        )
        c2 = _standardize(
            0.5 * roots[:, 3]
            + 0.8 * roots[:, 4]
            + 0.6 * roots[:, 5]
            - 0.5 * roots[:, 6]
            + rng.normal(0.0, 0.15, n)
        )
        c3 = _standardize(
            0.4 * roots[:, 6]
            + roots[:, 7]
            + 0.8 * roots[:, 8]
            + 0.4 * roots[:, 9]
            + rng.normal(0.0, 0.15, n)
        )
        treatment_signal = (
            c1 - 0.8 * c2 + 0.6 * c3 + 1.2 * z1 + 0.6 * z2
        )
        treatment_scale = 2.0
        y0 = (
            0.6 * c1
            + 0.5 * c2
            + 0.3 * c3
            + 2.0 * p1
            + p2
            + rng.normal(0.0, 0.5, n)
        )
        true_tau = 1.0 + 0.5 * c1 + 0.3 * c2 + 0.3 * p1
    else:
        proximal_noise = 0.8 if scenario == "weak_edges" else 0.1
        c1 = _standardize(
            1.2 * roots[:, 0]
            + 0.8 * np.tanh(roots[:, 1])
            + 0.5 * roots[:, 2] * roots[:, 3]
            + rng.normal(0.0, proximal_noise, n)
        )
        c2 = _standardize(
            0.5 * roots[:, 3] ** 2
            + np.sin(2.0 * roots[:, 4])
            + 0.7 * np.abs(roots[:, 5])
            - 0.5 * roots[:, 6]
            + rng.normal(0.0, proximal_noise, n)
        )
        c3 = _standardize(
            roots[:, 7]
            + 0.8 * np.where(roots[:, 8] > 0.0, 1.0, -1.0)
            + 0.4 * np.exp(roots[:, 9] / 2.0)
            + 0.3 * roots[:, 6]
            + rng.normal(0.0, proximal_noise, n)
        )
        confounding_scale = 0.35 if scenario == "weak_edges" else 1.0
        treatment_signal = (
            confounding_scale * (c1 - 0.8 * c2 + 0.6 * c3)
            + 0.4 * confounding_scale * c1 * c2
            + 1.5 * z1
            + 0.8 * np.sin(2.0 * z2)
        )
        treatment_scale = 2.5
        y0 = (
            confounding_scale
            * (0.5 * c1 + 0.5 * c2**2 + 0.3 * np.abs(c3))
            + 2.5 * p1
            + 1.5 * np.log1p(np.abs(p2))
            + rng.normal(0.0, 0.5, n)
        )
        true_tau = (
            1.0
            + 1.5 * (c1 > 0.5).astype(float)
            + 0.5 * c2
            + 0.3 * p1
        )

    treatment_signal = _standardize(treatment_signal) * treatment_scale
    propensity = 1.0 / (1.0 + np.exp(-treatment_signal))
    treatment = rng.binomial(1, propensity)
    outcome = y0 + treatment * true_tau
    collider1 = 0.6 * treatment + 0.4 * outcome + rng.normal(0.0, 0.1, n)
    collider2 = np.sin(treatment * outcome) + rng.normal(0.0, 0.1, n)

    data.update(
        {
            "C_prox_1": c1,
            "C_prox_2": c2,
            "C_prox_3": c3,
            "Z_1": z1,
            "Z_2": z2,
            "P_1": p1,
            "P_2": p2,
            "Noise_1": noise1,
            "Noise_2": noise2,
            "Collid_1": collider1,
            "Collid_2": collider2,
            "T": treatment,
            "Y": outcome,
            "true_tau": true_tau,
        }
    )
    frame = pd.DataFrame(data)

    if scenario == "noisy_missing":
        measured = [
            column
            for column in frame
            if column not in {"T", "Y", "true_tau"}
        ]
        for column in measured:
            frame[column] += rng.normal(
                0.0,
                0.20 * max(float(frame[column].std()), 1e-6),
                n,
            )
            missing = rng.random(n) < 0.10
            frame.loc[missing, column] = np.nan
            frame[column] = frame[column].fillna(frame[column].median())

    graph: dict[str, list[str]] = {name: [] for name in root_names}
    graph.update(
        {
            "Z_1": [],
            "Z_2": [],
            "P_1": [],
            "P_2": [],
            "Noise_1": [],
            "Noise_2": [],
            "C_prox_1": [
                "C_root_0",
                "C_root_1",
                "C_root_2",
                "C_root_3",
            ],
            "C_prox_2": [
                "C_root_3",
                "C_root_4",
                "C_root_5",
                "C_root_6",
            ],
            "C_prox_3": [
                "C_root_6",
                "C_root_7",
                "C_root_8",
                "C_root_9",
            ],
            "T": ["C_prox_1", "C_prox_2", "C_prox_3", "Z_1", "Z_2"],
            "Y": ["C_prox_1", "C_prox_2", "C_prox_3", "P_1", "P_2", "T"],
            "Collid_1": ["T", "Y"],
            "Collid_2": ["T", "Y"],
        }
    )
    return frame, graph, propensity


def tlcd_config(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        seed=seed,
        ci_theta=0.05,
        tlcd_lgb_max_depth=3,
        tlcd_lgb_num_leaves=15,
        tlcd_lgb_min_child_samples=30,
        rescue_lgb_estimators=160,
        post_treatment_rescue_lgb_estimators=60,
        rescue_min_importance_ratio=0.05,
        rescue_max_rank=12,
        precision_rescue_min_importance_ratio=0.10,
        precision_rescue_max_rank=8,
        precision_rescue_max_treatment_assoc=0.05,
        treatment_rescue_min_importance_ratio=0.30,
        treatment_rescue_max_rank=8,
        treatment_rescue_min_assoc=0.05,
    )


def learn_tlcd_graph(
    frame: pd.DataFrame,
    observed: list[str],
    seed: int,
) -> dict[str, list[str]]:
    args = tlcd_config(seed)
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
                direction_strategy="task_adaptive",
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
        args,
    )
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
    return project_graph_to_dag(
        graph,
        nodes=observed,
        protected_edges=protected,
    )["graph"]


def matrix_graph(
    matrix: np.ndarray,
    columns: list[str],
) -> tuple[dict[str, list[str]], set[frozenset[str]]]:
    graph = {column: [] for column in columns}
    skeleton: set[frozenset[str]] = set()
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            endpoint_left = int(matrix[left, right])
            endpoint_right = int(matrix[right, left])
            if endpoint_left == 0 and endpoint_right == 0:
                continue
            skeleton.add(frozenset((columns[left], columns[right])))
            if endpoint_left == -1 and endpoint_right == 1:
                graph[columns[right]].append(columns[left])
            elif endpoint_left == 1 and endpoint_right == -1:
                graph[columns[left]].append(columns[right])
    return complete_graph(graph, columns), skeleton


def learn_baseline_graph(
    method: str,
    frame: pd.DataFrame,
    columns: list[str],
    seed: int,
) -> tuple[dict[str, list[str]], set[frozenset[str]]]:
    values = frame[columns].to_numpy(dtype=float)
    values = (values - values.mean(axis=0)) / np.maximum(
        values.std(axis=0),
        1e-8,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        if method == "PC":
            from causallearn.search.ConstraintBased.PC import pc

            matrix = pc(
                values,
                alpha=0.05,
                show_progress=False,
            ).G.graph
            return matrix_graph(np.asarray(matrix), columns)
        if method == "FCI":
            from causallearn.search.ConstraintBased.FCI import fci

            graph, _ = fci(
                values,
                alpha=0.05,
                show_progress=False,
            )
            return matrix_graph(np.asarray(graph.graph), columns)
        if method == "GES":
            from causallearn.search.ScoreBased.GES import ges

            graph = ges(values)["G"]
            return matrix_graph(np.asarray(graph.graph), columns)
        if method == "DirectLiNGAM":
            from causallearn.search.FCMBased.lingam import DirectLiNGAM

            model = DirectLiNGAM(random_state=seed).fit(values)
            adjacency = np.asarray(model.adjacency_matrix_)
            graph = {column: [] for column in columns}
            for child, parent in zip(*np.where(np.abs(adjacency) >= 0.05)):
                if child != parent:
                    graph[columns[child]].append(columns[parent])
            projected = project_graph_to_dag(
                graph,
                nodes=columns,
            )["graph"]
            return projected, skeleton_edges(projected)
    raise ValueError(f"Unknown baseline: {method}")


def graph_metric_row(
    true_graph: dict[str, list[str]],
    predicted_graph: dict[str, list[str]],
    predicted_skeleton: set[frozenset[str]],
    observed: list[str],
) -> dict[str, float]:
    predicted_graph = project_graph_to_dag(
        predicted_graph,
        nodes=observed,
        protected_edges={("T", "Y")}
        if "T" in predicted_graph.get("Y", [])
        else None,
    )["graph"]
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = evaluate_graph_recovery(
            true_graph,
            predicted_graph,
            t="T",
            y="Y",
            nodes=observed,
        )
    skeleton = prf(skeleton_edges(true_graph), predicted_skeleton)
    flat = metrics["flat"]
    return {
        "Directed_Precision": flat["directed_precision"],
        "Directed_Recall": flat["directed_recall"],
        "Directed_F1": flat["directed_f1"],
        "Skeleton_Precision": skeleton["precision"],
        "Skeleton_Recall": skeleton["recall"],
        "Skeleton_F1": skeleton["f1"],
        "SHD_DirectedProjection": flat["SHD"],
        "Parents_T_F1": flat["Parents_T_f1"],
        "Parents_Y_F1": flat["Parents_Y_f1"],
        "C_Total_F1": flat.get("C_Total_f1", np.nan),
        "C_Proximal_F1": flat.get("C_Proximal_f1", np.nan),
        "S_Precision_F1": flat.get("S_Prec_f1", np.nan),
        "Predicted_Directed_Edges": len(directed_edges(predicted_graph)),
        "Predicted_Adjacencies": len(predicted_skeleton),
    }


def role_sets(
    graph: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sets = confounder_and_precision_sets(graph, "T", "Y")
        return sorted(sets["C_Proximal"]), sorted(sets["S_Prec"])
    except Exception:
        return [], []


def fit_role_learner(
    train: pd.DataFrame,
    test: pd.DataFrame,
    c_set: list[str],
    s_set: list[str],
    seed: int,
) -> np.ndarray | None:
    c_set = [column for column in c_set if column in train]
    outcome_set = sorted(
        {
            column
            for column in c_set + s_set
            if column in train and column not in {"T", "Y", "true_tau"}
        }
    )
    if not c_set or not outcome_set:
        return None
    propensity = LGBMClassifier(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )
    propensity.fit(train[c_set], train["T"])
    e = np.clip(
        propensity.predict_proba(train[c_set])[:, 1],
        0.02,
        0.98,
    )
    treatment = train["T"].to_numpy(dtype=int)
    prevalence = float(treatment.mean())
    weights = np.where(
        treatment == 1,
        prevalence / e,
        (1.0 - prevalence) / (1.0 - e),
    )
    outcome_model = LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=seed + 1,
        n_jobs=1,
        verbose=-1,
    )
    x_train = train[outcome_set].copy()
    x_train["assigned_treatment"] = treatment
    outcome_model.fit(x_train, train["Y"], sample_weight=weights)
    x0 = test[outcome_set].copy()
    x1 = test[outcome_set].copy()
    x0["assigned_treatment"] = 0
    x1["assigned_treatment"] = 1
    return outcome_model.predict(x1) - outcome_model.predict(x0)


def cate_metrics(
    true_tau: np.ndarray,
    predicted_tau: np.ndarray | None,
    budget: float = 0.30,
) -> dict[str, float]:
    if predicted_tau is None or not np.all(np.isfinite(predicted_tau)):
        return {
            "CATE_MAE": np.nan,
            "CATE_R2": np.nan,
            "Policy_Regret": np.nan,
        }
    true_tau = np.asarray(true_tau, dtype=float)
    predicted_tau = np.asarray(predicted_tau, dtype=float)
    k = max(1, int(round(budget * len(true_tau))))
    oracle = np.argpartition(-true_tau, k - 1)[:k]
    learned = np.argpartition(-predicted_tau, k - 1)[:k]
    oracle_value = float(np.sum(true_tau[oracle]) / len(true_tau))
    learned_value = float(np.sum(true_tau[learned]) / len(true_tau))
    return {
        "CATE_MAE": float(np.mean(np.abs(true_tau - predicted_tau))),
        "CATE_R2": float(r2_score(true_tau, predicted_tau)),
        "Policy_Regret": oracle_value - learned_value,
    }


def run_cell(
    scenario: str,
    n: int,
    seed: int,
    methods: tuple[str, ...],
) -> tuple[list[dict], list[dict]]:
    train, true_graph, propensity = generate_environment(
        n,
        seed,
        scenario,
    )
    test, _, _ = generate_environment(
        max(2000, min(5000, n)),
        seed + 100_000,
        scenario,
    )
    observed = [
        column
        for column in train
        if column != "true_tau"
    ]
    true_graph = complete_graph(true_graph, observed)
    graph_rows = []
    cate_rows = []
    learned_graphs = {}
    for method in methods:
        started = time.perf_counter()
        try:
            if method == "TLCD":
                graph = learn_tlcd_graph(train, observed, seed)
                skeleton = skeleton_edges(graph)
            else:
                graph, skeleton = learn_baseline_graph(
                    method,
                    train,
                    observed,
                    seed,
                )
            runtime = time.perf_counter() - started
            learned_graphs[method] = graph
            graph_rows.append(
                {
                    "Scenario": scenario,
                    "N": n,
                    "Seed": seed,
                    "Method": method,
                    "Runtime_Sec": runtime,
                    "Treatment_Prevalence": float(train["T"].mean()),
                    "Propensity_Extreme_Fraction": float(
                        np.mean((propensity < 0.05) | (propensity > 0.95))
                    ),
                    "Status": "complete",
                    **graph_metric_row(
                        true_graph,
                        graph,
                        skeleton,
                        observed,
                    ),
                }
            )
        except Exception as exc:
            graph_rows.append(
                {
                    "Scenario": scenario,
                    "N": n,
                    "Seed": seed,
                    "Method": method,
                    "Runtime_Sec": time.perf_counter() - started,
                    "Status": "failed",
                    "Error": f"{type(exc).__name__}: {exc}",
                }
            )

    with contextlib.redirect_stdout(io.StringIO()):
        true_sets = confounder_and_precision_sets(true_graph, "T", "Y")
    fixed_interfaces = {
        "OracleRoles": (
            sorted(true_sets["C_Proximal"]),
            sorted(true_sets["S_Prec"]),
        ),
        "AllTruePretreatment": (
            [
                column
                for column in observed
                if column not in {"T", "Y", "Collid_1", "Collid_2"}
            ],
            [],
        ),
        "AllRecorded": (
            [column for column in observed if column not in {"T", "Y"}],
            [],
        ),
    }
    interfaces = dict(fixed_interfaces)
    for method, graph in learned_graphs.items():
        interfaces[f"{method}Roles"] = role_sets(graph)
    for offset, (interface, (c_set, s_set)) in enumerate(
        interfaces.items()
    ):
        started = time.perf_counter()
        prediction = fit_role_learner(
            train,
            test,
            c_set,
            s_set,
            seed + 1000 + offset,
        )
        cate_rows.append(
            {
                "Scenario": scenario,
                "N": n,
                "Seed": seed,
                "Interface": interface,
                "C_Size": len(c_set),
                "S_Size": len(s_set),
                "Runtime_Sec": time.perf_counter() - started,
                **cate_metrics(
                    test["true_tau"].to_numpy(dtype=float),
                    prediction,
                ),
            }
        )
    return graph_rows, cate_rows


def summarize(
    frame: pd.DataFrame,
    groups: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(groups, keys))
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    **base,
                    "Metric": metric,
                    "Mean": float(values.mean()),
                    "SD": float(values.std(ddof=1)),
                    "Median": float(values.median()),
                    "Min": float(values.min()),
                    "Max": float(values.max()),
                    "Runs": len(values),
                }
            )
    return pd.DataFrame(rows)


def parse_csv_values(raw: str, cast=str) -> tuple:
    return tuple(cast(value.strip()) for value in raw.split(",") if value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
    )
    parser.add_argument("--sample-sizes", default="500,1000,5000,10000")
    parser.add_argument(
        "--seeds",
        default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/systematic_simulation_v1"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = parse_csv_values(args.scenarios)
    sample_sizes = parse_csv_values(args.sample_sizes, int)
    seeds = parse_csv_values(args.seeds, int)
    methods = parse_csv_values(args.methods)
    unknown_scenarios = set(scenarios) - set(SCENARIOS)
    unknown_methods = set(methods) - set(METHODS)
    if unknown_scenarios or unknown_methods:
        raise ValueError(
            f"Unknown scenarios={unknown_scenarios}, methods={unknown_methods}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = args.out_dir / "systematic_graph_results.csv"
    cate_path = args.out_dir / "systematic_cate_results.csv"
    existing_graph = (
        pd.read_csv(graph_path)
        if graph_path.exists() and not args.force
        else pd.DataFrame()
    )
    existing_cate = (
        pd.read_csv(cate_path)
        if cate_path.exists() and not args.force
        else pd.DataFrame()
    )
    completed = set()
    if not existing_graph.empty:
        status = existing_graph.groupby(["Scenario", "N", "Seed"])[
            "Method"
        ].nunique()
        completed = {
            tuple(index)
            for index, count in status.items()
            if count >= len(methods)
        }
    cells = [
        (scenario, n, seed, methods)
        for scenario in scenarios
        for n in sample_sizes
        for seed in seeds
        if (scenario, n, seed) not in completed
    ]
    graph_rows = existing_graph.to_dict("records")
    cate_rows = existing_cate.to_dict("records")
    if cells:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_cell, *cell): cell[:3] for cell in cells
            }
            for index, future in enumerate(as_completed(futures), start=1):
                cell = futures[future]
                try:
                    graph_result, cate_result = future.result()
                    graph_rows.extend(graph_result)
                    cate_rows.extend(cate_result)
                    status = "complete"
                except Exception as exc:
                    status = f"failed: {type(exc).__name__}: {exc}"
                pd.DataFrame(graph_rows).to_csv(graph_path, index=False)
                pd.DataFrame(cate_rows).to_csv(cate_path, index=False)
                print(
                    f"Cell {index}/{len(cells)} {cell}: {status}",
                    flush=True,
                )

    graph_frame = pd.DataFrame(graph_rows)
    cate_frame = pd.DataFrame(cate_rows)
    graph_summary = summarize(
        graph_frame[graph_frame["Status"].eq("complete")],
        ["Scenario", "N", "Method"],
        [
            "Directed_F1",
            "Skeleton_F1",
            "SHD_DirectedProjection",
            "Parents_T_F1",
            "Parents_Y_F1",
            "C_Proximal_F1",
            "S_Precision_F1",
            "Runtime_Sec",
        ],
    )
    cate_summary = summarize(
        cate_frame,
        ["Scenario", "N", "Interface"],
        ["CATE_MAE", "CATE_R2", "Policy_Regret", "Runtime_Sec"],
    )
    graph_summary.to_csv(
        args.out_dir / "systematic_graph_summary.csv",
        index=False,
    )
    cate_summary.to_csv(
        args.out_dir / "systematic_cate_summary.csv",
        index=False,
    )
    manifest = {
        "scenarios": list(scenarios),
        "sample_sizes": list(sample_sizes),
        "seeds": list(seeds),
        "methods": list(methods),
        "workers": args.workers,
        "causal_learn_version": "0.1.4.8",
        "llm_mode": "off",
        "pre_treatment_whitelist": False,
        "partial_graph_role_projection": "definite directed edges only",
        "direct_lingam_threshold": 0.05,
        "policy_budget": 0.30,
    }
    (args.out_dir / "systematic_simulation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"Completed graph rows={len(graph_frame)}, "
        f"CATE rows={len(cate_frame)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
