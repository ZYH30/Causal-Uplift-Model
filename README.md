# CCIA-OCU

Reference implementation for **task-local neuro-symbolic causal discovery and
role-separated observational uplift estimation**.

The pipeline learns a local ancestral graph around treatment `T` and outcome
`Y`, compresses that graph into a graph-proximal adjustment set `C` and
outcome-precision set `S`, and then estimates heterogeneous treatment effects
with separate nuisance roles:

```text
observational variables
  -> bounded statistical CCIA discovery
  -> optional semantic adjudication with DAG gates
  -> graph-proximal C-set and S-precision extraction
  -> propensity e(C) and outcome models m_t(C, S)
  -> CATE ranking and treatment-policy evaluation
```

## Paper-Facing Evidence

The repository contains the immutable compact artifacts used by the
manuscript.

| Experiment | Main result |
|---|---|
| Synthetic CATE, `n=10,000` | CCIA-OCU achieves the lowest MAE for all five uplift learners. |
| Mixed-type direction benchmark | Task-adaptive orientation reaches `1.000` mean accuracy over five seeds versus `0.585` for pairwise ANM. |
| Synthetic local graph recovery | Directed F1 `0.885`, SHD `6`, exact recovery of `Pa(T)`, `Pa(Y)`, and `S`; graph-proximal `C` F1 `0.857`. |
| Antihypertensive medication to SBP | AIPW contrast `-3.49` mmHg; mean absolute C-set SMD decreases from `0.383` to `0.079`. |
| Cholesterol medication to total cholesterol | AIPW contrast `-43.82` mg/dL; IPW effective-sample-size ratio `0.960`. |
| Locked real-world confirmation | CCIA-OCU improves R-loss in `15/20` and `19/20` unseen splits for the two tasks; the cholesterol task also improves DR-Qini. |

Real-world AIPW, DR-Qini, and policy values are identified under measured
confounding assumptions. They are diagnostics and estimands, not observed
individual treatment-effect labels.

## Installation

Python 3.10 or later is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

The paper experiments use LightGBM. The optional KAN residual backend has
separate dependencies:

```bash
.venv/bin/pip install -r requirements-kan.txt
```

## Reproduction

Run commands from the repository root:

```bash
make synthetic
make direction
make ccia-synthetic
make nhanes-confirmatory
make nhanes-survey
make figures
```

NHANES scripts download public 2011-2018 XPT tables from CDC and cache them
under `data/nhanes_raw/`. Raw data are never committed.

## Semantic Audit

Prepare the edge queue and token budget without network access:

```bash
make llm-dryrun
```

Rebuilding graph variants from archived semantic decisions is also free:

```bash
make llm-variants
```

Fresh Kimi and DeepSeek calls are paid. Put credentials in an untracked
`.env`, review the dry-run budget, and explicitly unlock execution:

```bash
ALLOW_PAID_API=1 make llm-api
```

The API client never writes keys to requests, responses, ledgers, or result
artifacts.

## Repository Layout

- `ccia/`: bounded local discovery, mixed-type direction methods, graph
  recovery metrics, and semantic fusion.
- `models/`: S-, T-, X-, class-transformation, and uplift-forest learners.
- `graph_roles.py`: graph-proximal adjustment and precision-variable
  extraction.
- `ocu_framework.py`: role-separated propensity weighting and uplift fitting.
- `experiments/`: synthetic and semantic-audit entry points.
- `empirical_experiments/`: NHANES processing, validation, locked confirmation,
  survey sensitivity, and figure generation.
- `results/`: compact immutable artifacts reported in the paper.
- `figures/`: regenerated publication figures.

Detailed commands and provenance are documented in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## License

MIT. See [`LICENSE`](LICENSE).
The optional KAN backend retains its upstream attribution in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
