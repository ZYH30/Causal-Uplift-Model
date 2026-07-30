import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ccia.ccia_runner import merge_graphs, project_graph_to_dag, run_ccia, save_json
from ccia.direction_methods import infer_var_types, post_treatment_score
from ccia.graph_metrics import complete_graph, evaluate_graph_recovery
from experiments.reproduce_synthetic import effect_metrics, fit_predict_stage2
from data_generation import generate_complex_environment
from models.class_transform import ClassTransformEstimator
from models.meta_learners import SLearner, TLearner, XLearner
from ocu_framework import FirstStageDeconfounder


def build_uplift_models(args):
    return {
        "S-Learner": SLearner(
            base_estimator=XGBRegressor(
                n_estimators=args.uplift_estimators,
                max_depth=args.uplift_max_depth,
                learning_rate=0.05,
                verbosity=0,
                random_state=args.seed,
            )
        ),
        "T-Learner": TLearner(
            estimator_t=XGBRegressor(
                n_estimators=args.uplift_estimators,
                max_depth=args.uplift_max_depth,
                learning_rate=0.05,
                verbosity=0,
                random_state=args.seed,
            )
        ),
        "X-Learner": XLearner(
            outcome_learner=XGBRegressor(
                n_estimators=args.uplift_estimators,
                max_depth=args.uplift_max_depth,
                learning_rate=0.05,
                verbosity=0,
                random_state=args.seed,
            ),
            effect_learner=XGBRegressor(
                n_estimators=args.uplift_estimators,
                max_depth=args.uplift_max_depth,
                learning_rate=0.05,
                verbosity=0,
                random_state=args.seed,
            ),
        ),
        "ClassTransform": ClassTransformEstimator(
            base_estimator=XGBRegressor(
                n_estimators=args.uplift_estimators,
                max_depth=3,
                learning_rate=0.05,
                verbosity=0,
                random_state=args.seed,
            )
        ),
    }


def compute_stage1(df_train, graph, treat_col, target_col, conf_type, args):
    deconfounder = FirstStageDeconfounder(
        treatment_var=treat_col,
        target_var=target_col,
        ancestors_graph=graph,
        method="IPW",
        conf_type=conf_type,
        is_precise=True,
    )
    df_stg1, weights, p_scores, c_set, s_prec, p_model = deconfounder.fit_transform(
        df_train,
        n_trials=args.uplift_n_trials,
        random_state=args.seed,
    )
    return {
        "df": df_stg1,
        "weights": weights,
        "p_scores": p_scores,
        "c_set": c_set,
        "s_prec": s_prec,
        "p_model": p_model,
    }


def run_downstream_uplift(df_train, df_test, oracle_graph, learned_graph, args):
    treat_col, target_col = "T", "Y"
    feature_cols = [c for c in df_train.columns if c not in [treat_col, target_col, "true_tau"]]
    selected = set(args.uplift_models)
    models = {
        name: model
        for name, model in build_uplift_models(args).items()
        if name in selected
    }
    rows = []

    oracle_stage = compute_stage1(df_train, oracle_graph, treat_col, target_col, "proximal", args)
    learned_stage = None
    learned_error = None
    try:
        learned_stage = compute_stage1(df_train, learned_graph, treat_col, target_col, "proximal", args)
    except Exception as exc:
        learned_error = str(exc)

    for model_name, base_model in models.items():
        model = copy.deepcopy(base_model)
        model.fit(
            df_train[feature_cols].values,
            df_train[target_col].values,
            df_train[treat_col].values,
        )
        pred = model.predict(df_test[feature_cols].values)
        rows.append(
            {
                "Model": model_name,
                "Setting": "Naive",
                "Graph": "None",
                **effect_metrics(df_test["true_tau"], pred),
            }
        )

        pred = fit_predict_stage2(
            base_model,
            df_test,
            treat_col,
            target_col,
            oracle_stage,
            use_precision=True,
            args=args,
        )
        rows.append(
            {
                "Model": model_name,
                "Setting": "OCU_Oracle",
                "Graph": "Oracle",
                **effect_metrics(df_test["true_tau"], pred),
            }
        )

        if learned_stage is not None:
            pred = fit_predict_stage2(
                base_model,
                df_test,
                treat_col,
                target_col,
                learned_stage,
                use_precision=True,
                args=args,
            )
            rows.append(
                {
                    "Model": model_name,
                    "Setting": "OCU_CCIA",
                    "Graph": "CCIA learned",
                    **effect_metrics(df_test["true_tau"], pred),
                }
            )
        else:
            rows.append(
                {
                    "Model": model_name,
                    "Setting": "OCU_CCIA",
                    "Graph": "CCIA learned",
                    "Error": learned_error,
                }
            )

    return pd.DataFrame(rows)


def rescue_outcome_edges_from_treatment_parents(
    df_train,
    learned_graph,
    observed_cols,
    treat_col,
    target_col,
    args,
    post_scores=None,
    outcome_importance=None,
):
    if post_scores is None:
        post_scores = detect_post_treatment_variables(
            df_train,
            observed_cols,
            treat_col,
            target_col,
            args,
        )
    post_variables = {name for name, score in post_scores.items() if score["is_post"]}
    if outcome_importance is None:
        outcome_importance, _ = outcome_importance_without_post_treatment(
            df_train,
            observed_cols,
            treat_col,
            target_col,
            args,
            post_scores=post_scores,
        )
    importance = outcome_importance
    max_importance = float(importance.max()) if len(importance) else 0.0
    rescue_edges = []
    if max_importance <= 0:
        return learned_graph, rescue_edges

    ranked = importance.sort_values(ascending=False)
    rank_map = {name: rank + 1 for rank, name in enumerate(ranked.index)}
    y_parents = set(learned_graph.get(target_col, []))

    for parent in learned_graph.get(treat_col, []):
        if parent == target_col or parent in y_parents:
            continue
        ratio = float(importance.get(parent, 0.0) / max_importance)
        rank = rank_map.get(parent, 10**9)
        if ratio >= args.rescue_min_importance_ratio and rank <= args.rescue_max_rank:
            y_parents.add(parent)
            rescue_edges.append(
                {
                    "parent": parent,
                    "child": target_col,
                    "importance": float(importance.get(parent, 0.0)),
                    "importance_ratio": ratio,
                    "importance_rank": rank,
                    "post_variables_excluded": sorted(post_variables),
                    "reason": "treatment_parent_with_outcome_importance",
                }
            )

    learned_graph = {node: list(parents) for node, parents in learned_graph.items()}
    learned_graph[target_col] = sorted(y_parents)
    return learned_graph, rescue_edges


def _decision_best_ci_p(decision):
    value = decision.get("best_ci_p_value")
    if value is not None and not pd.isna(value):
        return float(value)
    tests = decision.get("ci_tests") or []
    if isinstance(tests, str):
        try:
            tests = json.loads(tests)
        except json.JSONDecodeError:
            tests = []
    p_values = [item.get("p_value") for item in tests if item.get("p_value") is not None]
    return max(p_values) if p_values else np.nan


def detect_post_treatment_variables(df_train, observed_cols, treat_col, target_col, args):
    var_types = infer_var_types(df_train[observed_cols])
    scores = {}
    for variable in observed_cols:
        if variable in {treat_col, target_col}:
            continue
        scores[variable] = post_treatment_score(
            df_train[observed_cols],
            variable,
            treatment=treat_col,
            outcome=target_col,
            var_types=var_types,
            random_state=args.seed,
            n_estimators=args.post_treatment_rescue_lgb_estimators,
        )
    return scores


def outcome_importance_without_post_treatment(
    df_train,
    observed_cols,
    treat_col,
    target_col,
    args,
    post_scores=None,
):
    if post_scores is None:
        post_scores = detect_post_treatment_variables(
            df_train,
            observed_cols,
            treat_col,
            target_col,
            args,
        )
    post_variables = {name for name, score in post_scores.items() if score["is_post"]}
    feature_cols = [c for c in observed_cols if c != target_col and c not in post_variables]
    model = LGBMRegressor(
        n_estimators=args.rescue_lgb_estimators,
        learning_rate=0.05,
        max_depth=args.ccia_lgb_max_depth,
        num_leaves=args.ccia_lgb_num_leaves,
        min_child_samples=args.ccia_lgb_min_child_samples,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=args.seed,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(df_train[feature_cols], df_train[target_col])
    importance = pd.Series(model.feature_importances_, index=feature_cols, dtype=float)
    return importance.sort_values(ascending=False), post_scores


def treatment_importance_without_post_outcome(
    df_train,
    observed_cols,
    treat_col,
    target_col,
    args,
    post_scores=None,
):
    if post_scores is None:
        post_scores = detect_post_treatment_variables(
            df_train,
            observed_cols,
            treat_col,
            target_col,
            args,
        )
    post_variables = {name for name, score in post_scores.items() if score["is_post"]}
    feature_cols = [c for c in observed_cols if c not in {treat_col, target_col} and c not in post_variables]
    model = LGBMClassifier(
        n_estimators=args.rescue_lgb_estimators,
        learning_rate=0.05,
        max_depth=args.ccia_lgb_max_depth,
        num_leaves=args.ccia_lgb_num_leaves,
        min_child_samples=args.ccia_lgb_min_child_samples,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=args.seed,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(df_train[feature_cols], df_train[treat_col])
    importance = pd.Series(model.feature_importances_, index=feature_cols, dtype=float)
    return importance.sort_values(ascending=False), post_scores


def prune_deeper_edges_by_shallow_nonadjacency(learned_graph, graph_results, args):
    """Use shallower target-neighborhood CI evidence to suppress deeper false positives."""

    shallow_nonadjacency = {}
    accepted_depth = {}
    for result in graph_results:
        for decision in result.get("edge_decisions", []):
            node = decision.get("node")
            candidate = decision.get("candidate")
            if not node or not candidate:
                continue
            depth = int(decision.get("node_depth", 999))
            pair = frozenset((node, candidate))
            if decision.get("accepted_edge") and decision.get("edge_parent") and decision.get("edge_child"):
                edge = (decision["edge_parent"], decision["edge_child"])
                accepted_depth[edge] = min(depth, accepted_depth.get(edge, 999))
                continue
            if decision.get("ci_removed") and decision.get("final_relation") == "non_adjacent":
                best_p = _decision_best_ci_p(decision)
                if np.isfinite(best_p) and best_p >= args.ci_theta:
                    current = shallow_nonadjacency.get(pair)
                    if current is None or depth < current["depth"] or (depth == current["depth"] and best_p > current["best_p"]):
                        shallow_nonadjacency[pair] = {
                            "depth": depth,
                            "best_p": best_p,
                            "node": node,
                            "candidate": candidate,
                        }

    pruned_edges = []
    pruned_graph = {node: set(parents) for node, parents in learned_graph.items()}
    for child, parents in list(pruned_graph.items()):
        for parent in list(parents):
            edge = (parent, child)
            pair = frozenset(edge)
            negative = shallow_nonadjacency.get(pair)
            if negative is None:
                continue
            edge_depth = accepted_depth.get(edge, 999)
            if negative["depth"] < edge_depth:
                parents.remove(parent)
                pruned_edges.append(
                    {
                        "parent": parent,
                        "child": child,
                        "accepted_depth": edge_depth,
                        "negative_depth": negative["depth"],
                        "negative_best_p": negative["best_p"],
                        "negative_node": negative["node"],
                        "negative_candidate": negative["candidate"],
                        "reason": "shallower_nonadjacency",
                    }
                )

    return {node: sorted(parents) for node, parents in pruned_graph.items()}, pruned_edges


def rescue_post_treatment_children(
    df_train,
    learned_graph,
    observed_cols,
    treat_col,
    target_col,
    args,
    post_scores=None,
):
    """Add T/Y -> variable edges for statistically detected post-treatment variables."""

    if post_scores is None:
        post_scores = detect_post_treatment_variables(
            df_train,
            observed_cols,
            treat_col,
            target_col,
            args,
        )
    rescued_edges = []
    learned_graph = {node: set(parents) for node, parents in learned_graph.items()}
    for variable, score in post_scores.items():
        if not score["is_post"]:
            continue
        learned_graph.setdefault(variable, set())
        for parent in [treat_col, target_col]:
            if parent in observed_cols and parent not in learned_graph[variable]:
                learned_graph[variable].add(parent)
                rescued_edges.append(
                    {
                        "parent": parent,
                        "child": variable,
                        "score": score,
                        "reason": "post_treatment_detector",
                    }
                )

    return {node: sorted(parents) for node, parents in learned_graph.items()}, rescued_edges


def rescue_precision_outcome_edges(
    df_train,
    learned_graph,
    observed_cols,
    treat_col,
    target_col,
    args,
    post_scores=None,
    outcome_importance=None,
):
    """Add precision-variable parents of Y using outcome importance and low treatment association."""

    if outcome_importance is None:
        outcome_importance, post_scores = outcome_importance_without_post_treatment(
            df_train,
            observed_cols,
            treat_col,
            target_col,
            args,
            post_scores=post_scores,
        )
    importance = outcome_importance
    max_importance = float(importance.max()) if len(importance) else 0.0
    if max_importance <= 0:
        return learned_graph, []

    post_variables = {name for name, score in post_scores.items() if score["is_post"]}
    rank_map = {name: rank + 1 for rank, name in enumerate(importance.index)}
    treatment_parents = set(learned_graph.get(treat_col, []))
    y_parents = set(learned_graph.get(target_col, []))
    learned_graph = {node: set(parents) for node, parents in learned_graph.items()}
    rescue_edges = []

    for variable, value in importance.items():
        if variable in {treat_col, target_col} or variable in post_variables:
            continue
        if variable in y_parents or variable in treatment_parents:
            continue
        rank = rank_map[variable]
        ratio = float(value / max_importance)
        rho, _ = spearmanr(df_train[variable], df_train[treat_col])
        treatment_assoc = abs(float(rho)) if not np.isnan(rho) else 0.0
        if (
            rank <= args.precision_rescue_max_rank
            and ratio >= args.precision_rescue_min_importance_ratio
            and treatment_assoc <= args.precision_rescue_max_treatment_assoc
        ):
            learned_graph.setdefault(target_col, set()).add(variable)
            y_parents.add(variable)
            rescue_edges.append(
                {
                    "parent": variable,
                    "child": target_col,
                    "importance": float(value),
                    "importance_ratio": ratio,
                    "importance_rank": rank,
                    "treatment_assoc": treatment_assoc,
                    "post_variables_excluded": sorted(post_variables),
                    "reason": "precision_outcome_importance_low_treatment_assoc",
                }
            )

    return {node: sorted(parents) for node, parents in learned_graph.items()}, rescue_edges


def rescue_treatment_parent_edges(
    df_train,
    learned_graph,
    observed_cols,
    treat_col,
    target_col,
    args,
    post_scores=None,
    treatment_importance=None,
):
    """Add treatment parents using propensity importance after excluding post-outcome leakage."""

    if treatment_importance is None:
        treatment_importance, post_scores = treatment_importance_without_post_outcome(
            df_train,
            observed_cols,
            treat_col,
            target_col,
            args,
            post_scores=post_scores,
        )
    importance = treatment_importance
    max_importance = float(importance.max()) if len(importance) else 0.0
    if max_importance <= 0:
        return learned_graph, []

    post_variables = {name for name, score in post_scores.items() if score["is_post"]}
    t_parents = set(learned_graph.get(treat_col, []))
    rank_map = {name: rank + 1 for rank, name in enumerate(importance.index)}
    learned_graph = {node: set(parents) for node, parents in learned_graph.items()}
    rescue_edges = []

    for variable, value in importance.items():
        if variable in {treat_col, target_col} or variable in post_variables or variable in t_parents:
            continue
        ratio = float(value / max_importance)
        rank = rank_map[variable]
        rho, _ = spearmanr(df_train[variable], df_train[treat_col])
        treatment_assoc = abs(float(rho)) if not np.isnan(rho) else 0.0
        if (
            rank <= args.treatment_rescue_max_rank
            and ratio >= args.treatment_rescue_min_importance_ratio
            and treatment_assoc >= args.treatment_rescue_min_assoc
        ):
            learned_graph.setdefault(treat_col, set()).add(variable)
            t_parents.add(variable)
            rescue_edges.append(
                {
                    "parent": variable,
                    "child": treat_col,
                    "importance": float(value),
                    "importance_ratio": ratio,
                    "importance_rank": rank,
                    "treatment_assoc": treatment_assoc,
                    "post_variables_excluded": sorted(post_variables),
                    "outcome_excluded": target_col,
                    "reason": "treatment_importance_without_post_outcome",
                }
            )

    return {node: sorted(parents) for node, parents in learned_graph.items()}, rescue_edges


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()
    df_full, true_graph, _ = generate_complex_environment(n=args.n, seed=args.seed)
    treat_col, target_col = "T", "Y"
    excluded_cols = {"true_tau"}
    if not args.include_colliders:
        excluded_cols.update({c for c in df_full.columns if c.startswith("Collid_")})
    observed_cols = [c for c in df_full.columns if c not in excluded_cols]
    variable_order = None
    if args.use_uplift_temporal_order:
        variable_order = {c: 0 for c in observed_cols}
        variable_order[treat_col] = 1
        variable_order[target_col] = 2

    df_train, df_test = train_test_split(
        df_full,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df_full[treat_col],
    )

    graph_results = []
    for target in args.targets:
        print(f"\n=== CCIA target={target} ===", flush=True)
        result = run_ccia(
            df_train[observed_cols],
            target=target,
            output_dir=args.out_dir,
            model=args.ccia_model,
            cor_theta=args.cor_theta,
            cor_quantile=args.cor_quantile,
            min_abs_corr=args.min_abs_corr,
            ci_theta=args.ci_theta,
            direct_theta=args.direct_theta,
            anm_margin=args.anm_margin,
            max_condition_set_size=args.max_condition_set_size,
            max_neighbors=args.max_neighbors,
            max_corr_neighbors=args.max_corr_neighbors,
            max_importance_neighbors=args.max_importance_neighbors,
            max_condition_candidates=args.max_condition_candidates,
            max_search_depth=args.max_search_depth,
            min_samples=args.min_samples,
            use_optuna=args.ccia_use_optuna,
            lgb_estimators=args.ccia_lgb_estimators,
            lgb_max_depth=args.ccia_lgb_max_depth,
            lgb_num_leaves=args.ccia_lgb_num_leaves,
            lgb_min_child_samples=args.ccia_lgb_min_child_samples,
            llm_mode="off",
            direction_strategy=args.direction_strategy,
            random_state=args.seed,
            variable_order=variable_order,
            verbose=args.verbose,
        )
        graph_results.append(result)
        print(
            f"target={target}, edges={sum(len(v) for v in result['graph'].values())}, "
            f"runtime={result['runtime_sec']:.1f}s",
            flush=True,
        )

    merged = merge_graphs(graph_results, nodes=observed_cols, prefer_edges={(treat_col, target_col)})
    learned_graph = merged["graph"]
    pruned_edges = []
    if args.prune_deeper_edges_by_shallow_nonadjacency:
        learned_graph, pruned_edges = prune_deeper_edges_by_shallow_nonadjacency(
            learned_graph,
            graph_results,
            args,
        )
        merged["pruned_edges"] = pruned_edges

    treatment_rescue_edges = []
    post_scores = detect_post_treatment_variables(
        df_train,
        observed_cols,
        treat_col,
        target_col,
        args,
    )
    outcome_importance, _ = outcome_importance_without_post_treatment(
        df_train,
        observed_cols,
        treat_col,
        target_col,
        args,
        post_scores=post_scores,
    )
    treatment_importance, _ = treatment_importance_without_post_outcome(
        df_train,
        observed_cols,
        treat_col,
        target_col,
        args,
        post_scores=post_scores,
    )
    if args.rescue_treatment_parent_edges:
        learned_graph, treatment_rescue_edges = rescue_treatment_parent_edges(
            df_train=df_train,
            learned_graph=learned_graph,
            observed_cols=observed_cols,
            treat_col=treat_col,
            target_col=target_col,
            args=args,
            post_scores=post_scores,
            treatment_importance=treatment_importance,
        )
        merged["treatment_rescue_edges"] = treatment_rescue_edges

    rescue_edges = []
    if args.rescue_treatment_parent_outcome_edges:
        learned_graph, rescue_edges = rescue_outcome_edges_from_treatment_parents(
            df_train=df_train,
            learned_graph=learned_graph,
            observed_cols=observed_cols,
            treat_col=treat_col,
            target_col=target_col,
            args=args,
            post_scores=post_scores,
            outcome_importance=outcome_importance,
        )
        merged["rescue_edges"] = rescue_edges
    precision_rescue_edges = []
    if args.rescue_precision_outcome_edges:
        learned_graph, precision_rescue_edges = rescue_precision_outcome_edges(
            df_train=df_train,
            learned_graph=learned_graph,
            observed_cols=observed_cols,
            treat_col=treat_col,
            target_col=target_col,
            args=args,
            post_scores=post_scores,
            outcome_importance=outcome_importance,
        )
        merged["precision_rescue_edges"] = precision_rescue_edges
    post_treatment_rescue_edges = []
    if args.rescue_post_treatment_children:
        learned_graph, post_treatment_rescue_edges = rescue_post_treatment_children(
            df_train=df_train,
            learned_graph=learned_graph,
            observed_cols=observed_cols,
            treat_col=treat_col,
            target_col=target_col,
            args=args,
            post_scores=post_scores,
        )
        merged["post_treatment_rescue_edges"] = post_treatment_rescue_edges

    protected_edges = {(treat_col, target_col)}
    for repair_group in (
        treatment_rescue_edges,
        rescue_edges,
        precision_rescue_edges,
        post_treatment_rescue_edges,
    ):
        protected_edges.update(
            (edge["parent"], edge["child"])
            for edge in repair_group
        )
    projection = project_graph_to_dag(
        learned_graph,
        nodes=observed_cols,
        protected_edges=protected_edges,
    )
    learned_graph = projection["graph"]
    merged["post_repair_cycle_rejections"] = projection["rejected_edges"]

    oracle_graph = complete_graph(true_graph, observed_cols)
    save_json(learned_graph, os.path.join(args.out_dir, "ccia_graph_union.json"))
    save_json(merged, os.path.join(args.out_dir, "ccia_graph_union_meta.json"))
    save_json(oracle_graph, os.path.join(args.out_dir, "oracle_graph.json"))

    all_decisions = []
    for result in graph_results:
        for decision in result["edge_decisions"]:
            row = dict(decision)
            row["target_run"] = result["target"]
            row["ci_tests"] = json.dumps(row.get("ci_tests", []), ensure_ascii=False)
            all_decisions.append(row)
    decisions_df = pd.DataFrame(all_decisions)
    decisions_df.to_csv(os.path.join(args.out_dir, "ccia_edge_decisions_all.csv"), index=False)

    metrics = evaluate_graph_recovery(
        oracle_graph,
        learned_graph,
        t=treat_col,
        y=target_col,
        nodes=observed_cols,
    )
    metrics["runtime_sec"] = time.time() - started
    metrics["config"] = vars(args)
    save_json(metrics, os.path.join(args.out_dir, "ccia_graph_metrics.json"))
    pd.DataFrame([metrics["flat"]]).to_csv(
        os.path.join(args.out_dir, "ccia_graph_metrics_flat.csv"),
        index=False,
    )

    print("\n=== Graph recovery summary ===", flush=True)
    print(pd.DataFrame([metrics["flat"]]).T.to_string(header=False), flush=True)
    print("\nTrue OCU sets:", metrics.get("true_sets"), flush=True)
    print("Pred OCU sets:", metrics.get("pred_sets"), flush=True)
    if pruned_edges:
        print("Pruned edges:", pruned_edges, flush=True)
    if treatment_rescue_edges:
        print("Treatment rescue edges:", treatment_rescue_edges, flush=True)
    if rescue_edges:
        print("Rescue edges:", rescue_edges, flush=True)
    if precision_rescue_edges:
        print("Precision rescue edges:", precision_rescue_edges, flush=True)
    if post_treatment_rescue_edges:
        print("Post-treatment rescue edges:", post_treatment_rescue_edges, flush=True)

    if not args.graph_only:
        print("\n=== Downstream uplift with learned graph ===", flush=True)
        uplift_df = run_downstream_uplift(df_train, df_test, oracle_graph, learned_graph, args)
        uplift_df.to_csv(os.path.join(args.out_dir, "ccia_downstream_uplift.csv"), index=False)
        print(uplift_df.to_string(index=False), flush=True)

    return metrics


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-size", type=float, default=0.4)
    parser.add_argument("--targets", nargs="+", default=["Y", "T"])
    parser.add_argument("--out-dir", default="results/ccia_synthetic")
    parser.add_argument("--include-colliders", action="store_true", default=False)
    parser.add_argument("--use-uplift-temporal-order", action="store_true", default=True)
    parser.add_argument("--no-uplift-temporal-order", dest="use_uplift_temporal_order", action="store_false")

    parser.add_argument("--ccia-model", choices=["lgb", "kan"], default="lgb")
    parser.add_argument("--ccia-use-optuna", action="store_true", default=False)
    parser.add_argument("--cor-theta", type=float, default=None)
    parser.add_argument("--cor-quantile", type=float, default=0.75)
    parser.add_argument("--min-abs-corr", type=float, default=0.03)
    parser.add_argument("--ci-theta", type=float, default=0.05)
    parser.add_argument("--direct-theta", type=float, default=0.10)
    parser.add_argument("--anm-margin", type=float, default=0.03)
    parser.add_argument("--max-condition-set-size", type=int, default=2)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--max-corr-neighbors", type=int, default=8)
    parser.add_argument("--max-importance-neighbors", type=int, default=8)
    parser.add_argument("--max-condition-candidates", type=int, default=8)
    parser.add_argument("--max-search-depth", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--ccia-lgb-estimators", type=int, default=80)
    parser.add_argument("--ccia-lgb-max-depth", type=int, default=3)
    parser.add_argument("--ccia-lgb-num-leaves", type=int, default=15)
    parser.add_argument("--ccia-lgb-min-child-samples", type=int, default=30)
    parser.add_argument("--direction-strategy", choices=["anm_only", "default_parent", "task_adaptive"], default="anm_only")
    parser.add_argument("--rescue-treatment-parent-outcome-edges", action="store_true", default=True)
    parser.add_argument("--no-rescue-treatment-parent-outcome-edges", dest="rescue_treatment_parent_outcome_edges", action="store_false")
    parser.add_argument("--rescue-min-importance-ratio", type=float, default=0.05)
    parser.add_argument("--rescue-max-rank", type=int, default=12)
    parser.add_argument("--rescue-lgb-estimators", type=int, default=200)
    parser.add_argument("--prune-deeper-edges-by-shallow-nonadjacency", action="store_true", default=True)
    parser.add_argument("--no-prune-deeper-edges-by-shallow-nonadjacency", dest="prune_deeper_edges_by_shallow_nonadjacency", action="store_false")
    parser.add_argument("--rescue-treatment-parent-edges", action="store_true", default=True)
    parser.add_argument("--no-rescue-treatment-parent-edges", dest="rescue_treatment_parent_edges", action="store_false")
    parser.add_argument("--treatment-rescue-min-importance-ratio", type=float, default=0.30)
    parser.add_argument("--treatment-rescue-max-rank", type=int, default=8)
    parser.add_argument("--treatment-rescue-min-assoc", type=float, default=0.05)
    parser.add_argument("--rescue-precision-outcome-edges", action="store_true", default=True)
    parser.add_argument("--no-rescue-precision-outcome-edges", dest="rescue_precision_outcome_edges", action="store_false")
    parser.add_argument("--precision-rescue-min-importance-ratio", type=float, default=0.10)
    parser.add_argument("--precision-rescue-max-rank", type=int, default=8)
    parser.add_argument("--precision-rescue-max-treatment-assoc", type=float, default=0.05)
    parser.add_argument("--rescue-post-treatment-children", action="store_true", default=True)
    parser.add_argument("--no-rescue-post-treatment-children", dest="rescue_post_treatment_children", action="store_false")
    parser.add_argument("--post-treatment-rescue-lgb-estimators", type=int, default=80)

    parser.add_argument("--graph-only", action="store_true", default=False)
    parser.add_argument("--uplift-models", nargs="+", default=["S-Learner", "ClassTransform"])
    parser.add_argument("--uplift-estimators", type=int, default=80)
    parser.add_argument("--uplift-max-depth", type=int, default=3)
    parser.add_argument("--uplift-n-trials", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
