# Clinical context: cholesterol-lowering medication and total cholesterol

## Study question

The local causal query is the individualized effect of current cholesterol-lowering medication use (`T`) on serum total cholesterol (`Y`) among adults aged 40 years or older who reported that a health professional had told them their cholesterol was high (`BPQ080 = 1`) and who had an observed response to the medication-advice question. `T=1` denotes currently following advice to take prescribed cholesterol-lowering medicine; `T=0` denotes not currently following that advice. `Y` is directly measured total cholesterol (`LBXTC`) in mg/dL. Lower `Y` is treated as the clinically favorable direction.

Current treatment use precedes the laboratory measurement for the target intervention. The query therefore fixes `T -> Y`. A high or low cholesterol value at the examination must not be used to reverse the intervention edge, although prior lipid severity is an important unmeasured or imperfectly measured cause of both medication use and current cholesterol.

## Confounding by indication and prevention intensity

Lipid-lowering medication is selected using prior cholesterol, age, diabetes, cardiovascular history, and global atherosclerotic risk. Participants with heart attack, coronary disease, stroke, diabetes, or older age may be treated more intensively. These factors can also predict current lipid levels and adherence. The graph should therefore distinguish pre-treatment common causes from outcome-only precision variables and from variables that may already be affected by long-term treatment.

The observed medication signal in this cohort is strong: both the raw treated-minus-control difference and cross-fitted AIPW estimate indicate lower total cholesterol among treated participants. This supports the clinical relevance of the task but does not make the observational treatment assignment random.

## Temporal and structural constraints

- Chronological age, recorded sex, race/ethnicity category, survey cycle, and examination season cannot be caused by medication or current cholesterol.
- Adult standing height is effectively predetermined for this intervention query.
- Prior heart attack, coronary disease, stroke, hypertension diagnosis, and diabetes diagnosis should not be represented as consequences of current medication status.
- Current insulin use is a co-treatment and diabetes-severity marker; its relation to cardiovascular variables can be confounded.
- Total cholesterol is the terminal outcome in the local query and cannot cause predetermined demographics or historical diagnoses.
- Weight and height determine BMI by formula; avoid orienting BMI as a direct cause of its components.
- Same-visit pulse, diastolic pressure, anthropometry, HbA1c, and cholesterol have ambiguous temporal relations with long-term medication use; use `UNRESOLVED` when semantics cannot identify direction.
- Prefer `NO_DIRECT_EDGE` for disease co-occurrence that is likely explained by shared cardiometabolic risk rather than a direct mechanism.

## Role in OCU

The semantic audit focuses on edges whose orientation can change ancestors or parents of `T` and `Y`, the proximal C-set used in the propensity score, or S-precision variables used in outcome/CATE modeling. Semantic decisions are combined with statistical CCIA evidence through explicit voting and confidence rules. They are not treated as a substitute for data or as a known clinical ground-truth DAG.

## Official data sources

- NHANES Blood Pressure and Cholesterol questionnaire: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BPQ_J.htm
- NHANES Total Cholesterol laboratory file: https://wwwn.cdc.gov/nchs/data/nhanes/public/2017/datafiles/TCHOL_J.htm
- NHANES Demographics: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.htm
- NHANES Diabetes questionnaire: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DIQ_J.htm
- NHANES analytic guidance: https://wwwn.cdc.gov/nchs/nhanes/QualityAnalysesGuidelines.aspx
