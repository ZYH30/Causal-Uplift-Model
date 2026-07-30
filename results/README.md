# Result Artifacts

This directory contains compact, immutable aggregate artifacts reported in the
paper. It contains no MIMIC-IV patient-level rows and no API credentials.

- `synthetic/`: fixed-seed CATE benchmark.
- `direction_benchmark_multiseed_all_v1/`: mixed-type orientation benchmark.
- `ccia_synthetic_multiseed_dag_gated_v2/`: five-seed graph recovery and
  learned-graph downstream transfer.
- `systematic_simulation_v1/`: four-mechanism, four-size, ten-seed graph and
  downstream-effect benchmark with five graph learners.
- `ccia_component_ablation_v1/`: ten-seed task-adaptive direction and endpoint
  repair ablation.
- `mimic_rbc_target_trial_v3/`: aggregate target-trial metrics, balance,
  bootstrap summaries, clinical profiles, and sensitivity analyses.
- `mimic_rbc_target_trial_v5_product_audit/`: aggregate safety-role,
  independent-evaluator, balance, missingness, and stable-proxy audits.
- `nhanes_*`: aggregate medication-study and survey-sensitivity results.
- `llm_adjudication/`: normalized semantic decisions, graph variants, and
  aggregate token usage.
- `paper_tables/`: manuscript-facing aggregate tables.

New experiment outputs are written below `results/generated/`, which is
ignored by Git.
