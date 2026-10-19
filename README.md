# TLCD-RITE

Reference implementation of **task-local causal discovery (TLCD)** and
**role-informed treatment-effect learning (RITE)** for personalized treatment
effect estimation from observational clinical data.

TLCD learns treatment- and outcome-local structure and assigns measured
variables to an adjustment set `C`, an outcome-precision set `S`, or exclusion.
RITE preserves this role interface across estimation: the propensity model uses
only `C`, while outcome and treatment-effect models may use `C + S`.

```text
clinical treatment question and target-trial protocol
  -> TLCD local graph and optional semantic review
  -> role interface: adjustment C, precision S, exclusion
  -> RITE nuisance and heterogeneous-effect models
  -> benefit strata and capacity-constrained treatment ranking
```

## Main Components

- `tlcd/tlcd_runner.py`: bounded local search, conditional-independence tests,
  mixed-type direction evidence, endpoint checks, and acyclic projection.
- `tlcd/direction_methods.py`: randomized quantile residuals, additive-noise
  evidence, and task-adaptive direction rules.
- `tlcd/llm_causal/`: optional multi-model semantic review with constrained,
  provenance-preserving graph fusion.
- `graph_roles.py`: adjustment- and precision-set extraction from a local DAG.
- `rite_framework.py`: propensity weighting with `C` and treatment-effect
  learning with `C + S`.
- `empirical_experiments/`: synthetic, MIMIC-IV, and NHANES pipelines.
- `figure_generation/`: production programs and aggregate inputs for final
  Figures 3--6 and their 19 panels, with a standalone plotting environment.

## Paper Evidence

| Study | Paper-facing result |
|---|---|
| Known-graph simulation | Across 160 datasets, TLCD improves directed and role-set recovery over GES in nonlinear and measurement-perturbed mechanisms; learned roles approach oracle-role CATE performance. |
| Direction benchmark | The task-adaptive rule resolves the endpoint direction failures seen with pairwise ANM under the study mechanisms. |
| MIMIC-IV early RBC initiation | The common-support cohort contains 13,990 stays. TLCD learns 11 adjustment and 3 precision variables from 80 candidates, including 18 pre-trigger bedside features. |
| MIMIC-IV post-discovery splits | Relative to the same R-learner using all 79 analysis variables, RITE improves mean R-loss by `3.48e-4`, DR-Qini by `0.0103`, and policy risk reduction by `1.59` percentage points over 20 splits. |
| MIMIC-IV temporal cohort | All three point differences remain positive. Mean absolute SMD falls from `0.386` to `0.111`, with an ESS ratio of `0.773`. |
| NHANES medication studies | RITE lowers held-out R-loss in both tasks and improves DR-Qini in the cholesterol-medication task. |

Real-data estimands rely on consistency, positivity, and exchangeability given
the measured adjustment variables. They are not observed individual
counterfactual labels or clinical treatment recommendations.

## Installation

Python 3.10 or later is required.

```bash
make env
make test
```

The optional KAN residual backend has separate dependencies:

```bash
.venv/bin/pip install -r requirements-kan.txt
```

## Reproduction

The nonrestricted experiments run from the repository root:

```bash
make synthetic
make direction
make tlcd-synthetic
make systematic
make tlcd-ablation
make nhanes-confirmatory
make nhanes-survey
make figures
```

To generate the manuscript figures from the included results, install only the
plotting dependencies and follow [`figure_generation/README.md`](figure_generation/README.md).
The figure entry point exports PDF, SVG, and 900-dpi PNG files, with optional
individual-panel exports.

### MIMIC-IV

MIMIC-IV v3.1 requires independent PhysioNet credentialing and acceptance of
its data-use agreement. Structured source files are read locally; pre-trigger
`chartevents` aggregates are queried from the authorized PhysioNet BigQuery
dataset. Patient-level data are never committed.

```bash
make mimic-db MIMIC_ROOT=/path/to/mimiciv/3.1
make mimic-chart-data GCP_PROJECT=your-authorized-gcp-project
make mimic-chart-discovery MIMIC_ROOT=/path/to/mimiciv/3.1
make mimic-chart-analysis MIMIC_ROOT=/path/to/mimiciv/3.1
```

The complete sequence is also available as:

```bash
make mimic-chart \
  MIMIC_ROOT=/path/to/mimiciv/3.1 \
  GCP_PROJECT=your-authorized-gcp-project
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the cohort
protocol, temporal split, output contracts, and data-access requirements.

## Semantic Review

Prepare the edge queue and token budget without network access:

```bash
make llm-dryrun
```

Fresh Kimi and DeepSeek calls are paid and locked by default. Store credentials
in an untracked `.env`, review the dry-run budget, and explicitly unlock calls:

```bash
ALLOW_PAID_API=1 make llm-api
```

The API client does not write credentials to requests, responses, ledgers, or
result artifacts.

## Released Artifacts

- `results/mimic_rbc_chart_augmented/`: aggregate chart extraction, selected
  TLCD graph and roles, validation locks, repeated splits, temporal evaluation,
  benefit phenotypes, and policy curves.
- `results/systematic_simulation_v1/`: systematic graph and downstream CATE
  benchmark.
- `results/tlcd_component_ablation_v1/`: TLCD component ablation.
- `results/llm_adjudication/`: normalized semantic-review decisions and API
  usage records.
- `results/nhanes_llm_fixed_confirmatory_v1/`: locked NHANES comparisons.

Raw MIMIC-IV data, patient-level caches, NHANES downloads, `.env`, and generated
intermediate results are excluded from version control.

## License

MIT. The optional KAN backend retains upstream attribution in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
