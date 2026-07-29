"""Nonlinear observational benchmark with known DAG and individual effects."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_complex_environment(
    n: int = 10_000,
    seed: int = 2026,
) -> tuple[pd.DataFrame, dict[str, list[str]], np.ndarray]:
    """Generate the paper's high-dimensional confounding-bottleneck design.

    Ten distal common causes map nonlinearly into three graph-proximal
    confounders. Treatment-only instruments, outcome-only precision variables,
    noise variables, and post-treatment colliders are observed alongside `T`
    and `Y`.
    """

    np.random.seed(seed)

    def noise(scale: float = 0.1) -> np.ndarray:
        return np.random.normal(0.0, scale, n)

    roots = np.random.normal(0.0, 1.0, size=(n, 10))
    root_names = [f"C_root_{index}" for index in range(10)]
    values = {name: roots[:, index] for index, name in enumerate(root_names)}

    z1 = np.random.normal(0.0, 1.0, n)
    z2 = np.random.uniform(-2.0, 2.0, n)
    p1 = np.random.normal(0.0, 1.0, n)
    p2 = np.random.chisquare(df=2, size=n) - 2.0
    noise1 = np.random.normal(0.0, 1.0, n)
    noise2 = np.random.normal(0.0, 1.0, n)

    raw_c1 = (
        1.2 * roots[:, 0]
        + 0.8 * np.tanh(roots[:, 1])
        + 0.5 * roots[:, 2] * roots[:, 3]
        + noise()
    )
    c1 = (raw_c1 - raw_c1.mean()) / raw_c1.std()

    raw_c2 = (
        0.5 * roots[:, 3] ** 2
        + np.sin(2.0 * roots[:, 4])
        + 0.7 * np.abs(roots[:, 5])
        - 0.5 * roots[:, 6]
        + noise()
    )
    c2 = (raw_c2 - raw_c2.mean()) / raw_c2.std()

    raw_c3 = (
        roots[:, 7]
        + 0.8 * np.where(roots[:, 8] > 0.0, 1.0, -1.0)
        + 0.4 * np.exp(roots[:, 9] / 2.0)
        + 0.3 * roots[:, 6]
        + noise()
    )
    c3 = (raw_c3 - raw_c3.mean()) / raw_c3.std()

    treatment_logit = (
        c1
        - 0.8 * c2
        + 0.6 * c3
        + 0.4 * c1 * c2
        + 1.5 * z1
        + 0.8 * np.sin(2.0 * z2)
    )
    treatment_logit = (
        (treatment_logit - treatment_logit.mean()) / treatment_logit.std() * 2.5
    )
    propensity = 1.0 / (1.0 + np.exp(-treatment_logit))
    treatment = np.random.binomial(1, propensity)

    y0 = (
        0.5 * c1
        + 0.5 * c2**2
        + 0.3 * np.abs(c3)
        + 2.5 * p1
        + 1.5 * np.log1p(np.abs(p2))
        + noise(scale=0.5)
    )
    true_tau = 1.0 + 1.5 * (c1 > 0.5).astype(float) + 0.5 * c2 + 0.3 * p1
    outcome = np.where(treatment == 1, y0 + true_tau, y0)

    collider1 = 0.6 * treatment + 0.4 * outcome + noise()
    collider2 = np.sin(treatment * outcome) + noise()

    values.update(
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

    graph: dict[str, list[str]] = {name: [] for name in root_names}
    graph.update({name: [] for name in ["Z_1", "Z_2", "P_1", "P_2", "Noise_1", "Noise_2"]})
    graph["C_prox_1"] = ["C_root_0", "C_root_1", "C_root_2", "C_root_3"]
    graph["C_prox_2"] = ["C_root_3", "C_root_4", "C_root_5", "C_root_6"]
    graph["C_prox_3"] = ["C_root_6", "C_root_7", "C_root_8", "C_root_9"]
    graph["T"] = ["C_prox_1", "C_prox_2", "C_prox_3", "Z_1", "Z_2"]
    graph["Y"] = ["C_prox_1", "C_prox_2", "C_prox_3", "P_1", "P_2", "T"]
    graph["Collid_1"] = ["T", "Y"]
    graph["Collid_2"] = ["T", "Y"]

    return pd.DataFrame(values), graph, propensity
