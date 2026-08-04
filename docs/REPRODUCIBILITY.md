# Reproducibility Guide

Run commands from the repository root. Committed result files are aggregate,
non-identifying artifacts; `results/generated/` is reserved for local reruns.

## Environment

```bash
make env
make test
```

The main experiments use LightGBM. KAN is an optional residual backend:

```bash
.venv/bin/pip install -r requirements-kan.txt
```

## Synthetic Experiments

```bash
make synthetic
make direction
make tlcd-synthetic
make systematic
make tlcd-ablation
```

The systematic study crosses four structural mechanisms, four sample sizes,
ten seeds, and five graph learners on the same 23 observed variables. The
surface includes post-treatment colliders and does not provide a pretreatment
whitelist. TLCD component ablations use ten seeds. The default standalone
synthetic CATE run uses `n=10,000`.

## MIMIC-IV RBC Study

### Data access

1. Obtain MIMIC-IV v3.1 access through PhysioNet and accept its data-use
   agreement.
2. Download the structured `hosp/` and `icu/` CSV files.
3. Authorize Application Default Credentials for a Google Cloud project that
   can query `physionet-data.mimiciv_3_1_icu.chartevents`.
4. Set `MIMIC_ROOT` to the directory containing `hosp/` and `icu/`, and pass
   the authorized billing project as `GCP_PROJECT`.

The pipeline stores structured tables in a filtered local SQLite database.
`chartevents` is queried separately through BigQuery after uploading only the
study stay IDs and trigger times to a temporary table in the user's project.
The temporary table expires within 24 hours and is deleted after extraction.

### Target-trial protocol

- Population: adults at their first ICU stay with a first hemoglobin value of
  7--9 g/dL during ICU hours 0--24.
- Exclusions: RBC transfusion before the trigger, or death/ICU discharge before
  the six-hour landmark.
- Strategies: initiate RBC transfusion within six hours of the trigger versus
  do not initiate by six hours.
- Time zero: the six-hour landmark.
- Outcome: 90-day all-cause mortality.
- Bedside window: the half-open interval `[trigger - 24 h, trigger)`; urine
  output uses the preceding six hours.
- Common support: exclude prior malignancy because the 2020--2022 cohort has no
  treated patient in that baseline stratum.

The resulting common-support cohort contains 13,990 stays: 9,018 in discovery
(2008--2016), 2,865 in validation (2017--2019), and 2,107 in temporal evaluation
(2020--2022).

### Pipeline

Build the structured SQLite database without loading the large local
`chartevents` table:

```bash
make mimic-db MIMIC_ROOT=/path/to/mimiciv/3.1
```

Extract time-safe chart aggregates through BigQuery:

```bash
make mimic-chart-data GCP_PROJECT=your-authorized-gcp-project
```

Compare the three prespecified bedside panels in the validation era, learn the
80-variable TLCD graph for the selected panel, and validate the `C`/`S` roles:

```bash
make mimic-chart-discovery MIMIC_ROOT=/path/to/mimiciv/3.1
```

Lock the propensity and heterogeneous-effect learners using validation data,
then run the 20 post-discovery splits and non-overlapping temporal analysis:

```bash
make mimic-chart-analysis MIMIC_ROOT=/path/to/mimiciv/3.1
```

Generated files are written to `results/generated/mimic-rbc-chart/`. The
compact paper-facing archive is `results/mimic_rbc_chart_augmented/` and
contains:

- the parameterized BigQuery extraction and integrity manifest;
- the 80-variable manifest, selected TLCD graph, and role sets;
- common-support, propensity, and learner locks;
- split-level and aggregate R-loss, DR-Qini, and policy comparisons;
- temporal balance, bootstrap summaries, benefit strata, hemoglobin response,
  bedside profiles, and treatment-budget curves.

No committed CSV or JSON contains `subject_id`, `hadm_id`, or `stay_id`.

## NHANES Studies

```bash
make nhanes-confirmatory
make nhanes-survey
```

The scripts download public 2011--2018 NHANES XPT files into the ignored
`data/nhanes_raw/` cache. The confirmatory protocol fixes graph variants,
roles, learners, and seeds 2040--2059. Survey sensitivity uses
`WTMEC2YR / 4` and Taylor linearization with cycle-specific strata and PSU
identifiers.

## Semantic Review

Always prepare the edge queue and token budget first:

```bash
make llm-dryrun
make llm-variants
```

Fresh calls require untracked credentials and an explicit paid-call gate:

```bash
ALLOW_PAID_API=1 make llm-api
```

The archived run contains 39 HTTP responses, 38 valid calls, and 192,664 total
tokens. Deterministic graph fusion can be reproduced from the archived decision
file without contacting an API.

## Figures

```bash
make figures
```

This rebuilds MIMIC figure inputs from committed aggregate results and renders
Figures 2--5 as PDF, SVG, and 600-dpi PNG. It does not refit models, access
patient-level data, or invoke paid APIs.
