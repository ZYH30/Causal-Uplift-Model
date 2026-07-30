# Reproducibility Guide

This guide reproduces the paper-facing OCU and CCIA experiments. Run all
commands from the repository root.

## Environment

```bash
make env
make test
```

The main statistical experiments use LightGBM. KAN is an optional nonlinear
residual model and requires:

```bash
.venv/bin/pip install -r requirements-kan.txt
```

## Synthetic OCU

```bash
make synthetic
```

Expected output:

```text
results/generated/synthetic/synthetic_reproduction_mae_r2.csv
```

The experiment uses `n=10000`, seed 2026, and reports CATE MAE, R2,
Spearman correlation, and bias for five uplift learners.

## Statistical CCIA

Run the five-seed direction benchmark and five-seed complete graph-recovery
and learned-graph transfer pipeline:

```bash
make direction
make ccia-synthetic
```

The graph experiment intentionally includes collider variables and disables
uplift temporal ordering. It therefore evaluates the data-driven statistical
direction mechanism rather than a pre-treatment filter. The final graph uses a
priority-ordered DAG gate: endpoint-level and higher-confidence directions are
admitted first, and every cycle-closing edge is rejected with provenance.

Expected aggregate outputs:

```text
results/generated/ccia-synthetic/ccia_multiseed_per_seed.csv
results/generated/ccia-synthetic/ccia_multiseed_summary.csv
```

Run the paper's systematic graph, role, and downstream-effect benchmark plus
the ten-seed component ablation:

```bash
make systematic
make ccia-ablation
```

The systematic benchmark evaluates four structural mechanisms, four sample
sizes, ten seeds, and CCIA, PC, FCI, GES, and DirectLiNGAM on the same 23
observed variables. No method receives a pre-treatment whitelist, and both
post-treatment colliders remain in the observed surface. Outputs are written
to `results/generated/systematic/` and
`results/generated/ccia-ablation/`.

## MIMIC-IV Target-Trial Study

MIMIC-IV v3.1 must be obtained independently through PhysioNet credentialing
and its data-use agreement. Source CSV files and patient-level extracts are
never included in the repository.

Build the filtered SQLite database used by the paper:

```bash
make mimic-db MIMIC_ROOT=/path/to/mimiciv/3.1
```

The paper environment uses `--skip-chartevents` because the local
`chartevents.csv.gz` failed the official MIMIC-IV SHA256 and gzip-completeness
checks. Users with a complete manifest-matching file may omit that flag. The
final analysis uses complete laboratory, input, output, service, procedure,
admission, and patient sources.

Reproduce cohort construction, statistical CCIA, validation-only learner
selection, temporal evaluation, and 20 repeated splits:

```bash
make mimic MIMIC_ROOT=/path/to/mimiciv/3.1
```

Reproduce patient-bootstrap uncertainty, benefit phenotypes, the hemoglobin
mechanism endpoint, policy-budget curves, late-transfusion sensitivity, and
the conservative `C_total` propensity sensitivity:

```bash
make mimic-interpretation MIMIC_ROOT=/path/to/mimiciv/3.1
```

Run the discovery-only treatment-neutrality gate, full-surface balance audit,
independent evaluator, glucose-missingness analyses, stable-proxy subgroup,
and 20 repeated post-discovery splits:

```bash
make mimic-audit
```

The generated output directory is:

```text
results/generated/mimic-rbc-v3/
```

The analysis never exports row-level MIMIC-IV data. Release archives contain
only aggregate CSV/JSON results and source-integrity manifests.

## NHANES Data

The scripts download public NHANES XPT files from CDC into
`data/nhanes_raw/`. Raw data are excluded from version control. The two paper
tasks use the 2011-2018 cycles.

## LLM Semantic Audit

Always run the no-cost dry-run first:

```bash
make llm-dryrun
```

It creates the edge queue and token budget without reading API keys. Rebuilding
the graph variants from an archived decision file is also free:

```bash
make llm-variants
```

Fresh semantic adjudication is a paid network operation. Place `Kimi_Key` and
`deepseek_Key` in the workspace-level `.env`, review the budget, and explicitly
unlock the target:

```bash
ALLOW_PAID_API=1 make llm-api
```

The paper run used 39 HTTP responses and 192,664 total tokens. API keys are
never written to request, response, ledger, or result files.

## Locked NHANES Confirmation

```bash
make nhanes-confirmatory
make nhanes-survey
```

The primary confirmatory protocol fixes the conservative LLM graph, proximal
C-set, OCU C-only variant, base learner, and seeds 2040-2059. Propensity
nuisance models use only `C`; outcome nuisance models use `C union S`.

The survey sensitivity uses `WTMEC2YR/4` for four two-year cycles and Taylor
linearization with cycle-specific masked strata and PSU variables.

## Figures

```bash
make figures
```

This target regenerates PDF and PNG figures from committed aggregate result
files. It does not refit models or rerun paid APIs.

The public code repository does not include the submission manuscript.

## Result Provenance

- `results/llm_adjudication/`: raw normalized votes, decision archive, API usage,
  and deterministic graph variants.
- `results/nhanes_llm_fixed_confirmatory_v1/`: 20 locked train-test splits.
- `results/nhanes_survey_aipw_sensitivity_v1/`: complex-survey AIPW sensitivity.
- `results/mimic_rbc_target_trial_v3/`: locked longitudinal RBC study,
  repeated-split comparisons, clinical interpretation, and sensitivities.
- `results/mimic_rbc_target_trial_v5_product_audit/`: safety-validated roles,
  independent evaluation, full-surface balance, missingness audits, and stable
  proxy sensitivities.
- `results/ccia_synthetic_multiseed_dag_gated_v2/`: five-sample graph
  recovery and learned-versus-oracle OCU transfer.
- `results/systematic_simulation_v1/`: four-mechanism, four-size, ten-seed
  graph and downstream-effect benchmark.
- `results/ccia_component_ablation_v1/`: ten-seed CCIA component ablation.
- `results/paper_tables/`: manuscript-facing aggregate tables.
- `figures/`: manuscript-facing figures regenerated from committed artifacts.

`results/generated/` is ignored because intermediate files can be large. The
repository includes the compact final CSV/JSON artifacts required by
`tests/test_experiment_integrity.py`.
