# Chart-Augmented MIMIC-IV Results

This directory contains only aggregate, non-identifying artifacts from the
paper's final MIMIC-IV analysis.

- `variable_manifest.csv`: 80 graph candidates and final interface roles.
- `panel/`: selected bedside panel, TLCD graph, raw roles, and discovery-period
  treatment-neutrality validation.
- `baseline/`: locked no-chart TLCD roles used only for the matched temporal
  contribution comparison.
- `final/`: common-support protocol, propensity and learner locks, 20
  post-discovery splits, temporal balance and paired-bootstrap comparisons,
  benefit phenotypes, hemoglobin response, and treatment-budget curves.
- `chartevents_aggregate_query.sql`: parameterized form of the BigQuery query.
- `chartevents_extraction_manifest.json`: source, timing window, item IDs,
  range checks, bytes processed, and aggregate row counts.

The analysis code writes patient-level caches only under the ignored
`results/generated/.../private/` directory. No released artifact contains a
MIMIC patient, admission, or ICU-stay identifier.
