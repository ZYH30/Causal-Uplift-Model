# Clinical context: antihypertensive medication and systolic blood pressure

## Study question

The local causal query is the individualized effect of current prescribed antihypertensive medication use (`T`) on mean systolic blood pressure (`Y`) among adults who reported that a health professional had diagnosed hypertension (`BPQ020 = 1`). `T=1` denotes current use of prescribed medicine for hypertension; `T=0` denotes no current use within this diagnosed cohort. `Y` is the mean of available NHANES systolic blood-pressure readings in mmHg. Lower `Y` is treated as the clinically favorable direction.

Medication status is reported before or around the NHANES examination, whereas blood pressure is measured during the mobile examination-center visit after seated rest. The intervention query fixes `T -> Y`: the graph audit must not reverse this edge merely because current blood pressure is associated with treatment. Prior blood pressure severity, duration of hypertension, comorbidity, and healthcare contact can influence both current treatment and current blood pressure.

## Confounding by indication

Treatment is not randomly assigned. Older or clinically higher-risk participants are more likely to receive medication. The treated group can therefore have a worse unadjusted outcome even when medication is beneficial. In the current cohort, the raw treated-minus-control SBP difference is positive, while the cross-fitted AIPW estimate is negative. This pattern is compatible with confounding by indication and motivates recovery of a pre-treatment C-set for propensity modeling.

Variables that precede both treatment assignment and outcome may be confounders. Variables that predict the outcome but do not affect treatment can be S-precision variables. A current or post-treatment variable must not be placed in the propensity C-set solely because it predicts treatment. Disease-history variables usually precede the current medication report, but same-visit biomarkers and behaviors can have ambiguous timing relative to long-term medication use.

## Temporal and structural constraints

- Chronological age, recorded sex, race/ethnicity category, survey cycle, and examination season cannot be caused by current medication or current SBP.
- Adult standing height is effectively predetermined for this intervention query.
- Historical diagnoses and prior cardiovascular events should not be represented as effects of current treatment status.
- Current medication may affect current SBP and pulse, but same-visit association alone is insufficient to orient other biomarker edges.
- Weight and height determine BMI by formula; avoid treating BMI as a direct cause of its components.
- Participation in vigorous work activity logically precedes its conditional frequency measure.
- If an edge is better explained by shared disease severity, healthcare access, socioeconomic status, or another measured variable, prefer `NO_DIRECT_EDGE`.
- If the cross-sectional design leaves both directions clinically plausible, prefer `UNRESOLVED`.

## Role in OCU

The semantic audit is local and task-oriented. Direction changes are most important when they alter ancestors or parents of `T` and `Y`, the proximal C-set used in `e(C)=P(T=1|C)`, or S-precision variables used only in outcome/CATE modeling. Semantic evidence complements the statistical CCIA score; it does not replace statistical evidence or create a ground-truth clinical DAG.

## Official data sources

- NHANES Blood Pressure and Cholesterol questionnaire: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPQ_J.htm
- NHANES Blood Pressure examination: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPX_J.htm
- NHANES Demographics: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm
- NHANES Body Measures: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BMX_J.htm
- NHANES analytic guidance: https://wwwn.cdc.gov/nchs/nhanes/QualityAnalysesGuidelines.aspx
