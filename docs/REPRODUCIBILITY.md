# Reproducibility Guide

All commands run from the repository root. Compact result artifacts reported
in the manuscript are committed under `results/`; new runs write to
`results/generated/`.

## Environment

```bash
make env
make test
```

The released experiments use LightGBM for CCIA residual fitting. KAN is an
optional backend and is not required for any reported number.

## Synthetic Experiments

```bash
make synthetic
make direction
make ccia-synthetic
```

`make synthetic` evaluates CATE MAE, MSE, R2, Spearman correlation, and bias
for S-, T-, X-, centered class-transformation, and uplift-forest learners.

The direction benchmark uses seeds 2026-2030 and evaluates true adjacent
directions under the mixed-type nonlinear data-generating process.

The graph-recovery command includes the post-treatment colliders and does not
supply pre-treatment ordering. It evaluates directed/skeleton precision,
recall and F1, structural Hamming distance, endpoint ancestor and parent
recovery, causal-role recovery, and downstream oracle-versus-learned-graph
CATE performance.

## NHANES Data

The empirical scripts fetch 2011-2018 public NHANES XPT tables from CDC into
`data/nhanes_raw/`. The two locked tasks are:

1. Current prescribed antihypertensive medication to mean systolic blood
   pressure among adults reporting diagnosed hypertension.
2. Current prescribed cholesterol-lowering medication to total cholesterol
   among adults reporting diagnosed high cholesterol.

Raw data and participant-level analysis frames are excluded from version
control.

## Semantic Adjudication

Always run the no-cost audit first:

```bash
make llm-dryrun
```

This materializes the impact-ranked edge queue, counterbalanced messages, call
count, and upper token budget without reading credentials or using a network.

Archived decisions can be fused without API calls:

```bash
make llm-variants
```

Fresh calls require both `Kimi_Key` and `deepseek_Key` in `.env`, plus the
explicit execution gate:

```bash
ALLOW_PAID_API=1 make llm-api
```

## Locked Real-World Confirmation

```bash
make nhanes-confirmatory
make nhanes-survey
```

The confirmatory protocol fixes graph fusion, graph-proximal C-set, base
learner, OCU variant, and seeds before running seeds 2040-2059. Evaluation
propensity models use `e(C)` only. Outcome nuisance models use `m_t(C union S)`.

The survey sensitivity uses `WTMEC2YR / 4` for four two-year cycles and Taylor
linearization over cycle-specific masked strata and primary sampling units.

## Figures

```bash
make figures
```

The figure script reads only committed immutable artifacts and writes vector
PDF plus high-resolution PNG files. It does not fit a model or invoke an API.

## Artifact Provenance

- `results/synthetic/`: synthetic CATE estimates.
- `results/direction_benchmark_multiseed_all_v1/`: five-seed orientation audit.
- `results/ccia_synthetic_n10000_task_adaptive_allvars_v5_treatment_rescue/`:
  graph and downstream evidence.
- `results/llm_adjudication/`: aggregate votes, usage summary, and deterministic
  graph variants.
- `results/nhanes_llm_fixed_confirmatory_v1/`: locked 20-split confirmation.
- `results/nhanes_survey_aipw_sensitivity_v1/`: complex-survey sensitivity.
- `results/paper_tables/`: manuscript-facing CSV tables.
