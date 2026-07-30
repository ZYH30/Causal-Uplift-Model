# CCIA-OCU

Reference implementation for learning personalized treatment benefit from
observational clinical data when the required adjustment variables are not
known in advance.

```text
clinical treatment question and target trial
  -> task-local statistical causal discovery
  -> optional constrained semantic adjudication
  -> graph-proximal adjustment set C and precision set S
  -> propensity e(C) and outcome models m_t(C,S)
  -> individualized benefit ranking and policy evaluation
```

## Method

- `ccia/ccia_runner.py`: bounded local ancestor search, generalized covariance
  conditional-independence tests, priority-ordered DAG projection, and complete
  edge-decision provenance.
- `ccia/direction_methods.py`: mixed-type randomized quantile residuals,
  additive-noise evidence, and task-adaptive direction rules.
- `ccia/llm_causal/`: clinical variable dictionaries, multi-model semantic
  adjudication, and DAG-constrained graph fusion.
- `graph_roles.py`: scalable extraction of total, graph-proximal, and root
  adjustment sets plus outcome-precision variables.
- `ocu_framework.py`: role-separated propensity weighting and heterogeneous
  treatment-effect learning.

The propensity model uses only `C`. Outcome and treatment-effect models may use
`C union S`.

## Paper-Facing Evidence

| Experiment | Main result |
|---|---|
| Synthetic CATE, `n=10,000` | OCU reduces mean CATE MAE by 9.1% across five uplift learners. |
| Mixed-type direction, five seeds | Task-adaptive direction accuracy is `1.000` over 135 decisions versus `0.585` for pairwise ANM. |
| Local graph recovery, five seeds | Directed F1 is `0.855 +/- 0.060`; proximal-set recall is `1.000` in every sample. |
| Learned-graph transfer | Learned graphs retain 100.0% and 97.2% of the oracle-to-naive MAE gain for S-learning and class transformation. |
| MIMIC-IV RBC target trial | `N=15,568`; graph discovery, learner selection, and temporal testing use non-overlapping eras. |
| MIMIC-IV repeated splits | Mean R-loss, DR-Qini, and policy-risk gains over the same all-variable learner have intervals excluding zero. |
| NHANES medication studies | Locked R-loss gains occur in both tasks; the cholesterol task also has a significant DR-Qini gain. |

Real-data AIPW, DR-Qini, and policy values are estimands under measured
confounding assumptions. They are not observed individual counterfactual
labels or clinical treatment recommendations.

## Installation

Python 3.10 or later is required.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

The paper experiments use LightGBM. The optional KAN residual backend has
separate dependencies:

```bash
.venv/bin/pip install -r requirements-kan.txt
```

## Reproduction

```bash
make synthetic
make direction
make ccia-synthetic
make nhanes-confirmatory
make nhanes-survey
make figures
```

MIMIC-IV v3.1 requires independent PhysioNet credentialing and acceptance of
its data-use agreement:

```bash
make mimic-db MIMIC_ROOT=/path/to/mimiciv/3.1
make mimic MIMIC_ROOT=/path/to/mimiciv/3.1
make mimic-interpretation MIMIC_ROOT=/path/to/mimiciv/3.1
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the target-trial
protocol, immutable artifact provenance, and expected outputs.

## Semantic Audit

Prepare the edge queue and token budget without network access:

```bash
make llm-dryrun
```

Fresh Kimi and DeepSeek calls are paid and remain locked by default. Put
credentials in an untracked `.env`, inspect the dry-run budget, and explicitly
unlock execution:

```bash
ALLOW_PAID_API=1 make llm-api
```

The API client never writes keys to requests, responses, ledgers, or results.

## Data Safety

- MIMIC-IV source files, SQLite databases, and patient-level extracts are not
  distributed.
- NHANES public XPT files are downloaded from CDC and cached under
  `data/nhanes_raw/`.
- Committed MIMIC artifacts are aggregate CSV/JSON outputs only.
- `.env`, raw data, generated results, and local build caches are excluded from
  version control.

## Repository Layout

- `ccia/`: local discovery, direction methods, graph metrics, and semantic
  fusion.
- `models/`: S-, T-, X-, class-transformation, and uplift-forest learners.
- `experiments/`: synthetic graph recovery, CATE, direction, and semantic-audit
  entry points.
- `empirical_experiments/`: MIMIC-IV and NHANES target-study pipelines.
- `results/`: compact immutable aggregate artifacts reported in the paper.
- `figures/`: publication figures regenerated from committed artifacts.
- `tests/`: algorithm, temporal-feature, semantic, and artifact-integrity
  checks.

## License

MIT. See [`LICENSE`](LICENSE). The optional KAN backend retains upstream
attribution in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
