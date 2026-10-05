# Controlled SMOTENC Boolean comparison

Treating Booleans as categorical is supported and preserves binary synthetic values. Numeric treatment allows fractional synthetic values. These results describe this dataset, split and seed.

## Method

- Same notebook 2 stratified 80/20 split, random_state=42: 16,000 training and 4,000 test rows (68 test fraud cases). Transaction ID excluded.
- Five identical stratified shuffled CV folds within the training portion, random_state=42.
- Continuous numeric features standardized before sampling; original categorical features ordinal encoded; Boolean features converted to 0/1.
- Only SMOTENC categorical_features changes: original categories plus Booleans, versus original categories alone. SMOTENC uses random_state=42 and default full balancing.
- Identical downstream processing: standardize numeric columns, one-hot encode original categories, and pass Boolean columns through as single numeric columns in both treatments.
- Same model settings as notebook 2. SVC probability calibration is disabled because only predicted labels are required. Random forest uses 300 trees and one job.
- Default prediction thresholds, no tuning. Each CV score is the arithmetic mean of five fold F1 scores.
- These are controlled experimental pipelines, rather than the original notebook pipelines unchanged. Both scale before sampling and use single-column Booleans to isolate sampler behavior.
- Numeric synthetic Boolean means represent average interpolated values, not the proportion of True values.

## Test F1

| Model | Boolean categorical | Boolean numeric | Numeric minus categorical |
|---|---:|---:|---:|
| Logistic Regression | 0.1414 | 0.2088 | +0.0673 |
| SVM | 0.1429 | 0.2452 | +0.1023 |
| KNN | 0.1014 | 0.1146 | +0.0133 |
| Random Forest | 0.0708 | 0.0988 | +0.0280 |
| HistGradientBoosting | 0.1468 | 0.1667 | +0.0199 |

## Mean cross-validation F1

| Model | Boolean categorical | Boolean numeric | Numeric minus categorical |
|---|---:|---:|---:|
| Logistic Regression | 0.1491 | 0.2098 | +0.0607 |
| SVM | 0.1994 | 0.2637 | +0.0643 |
| KNN | 0.1016 | 0.1106 | +0.0090 |
| Random Forest | 0.1324 | 0.1292 | -0.0031 |
| HistGradientBoosting | 0.2358 | 0.2282 | -0.0076 |

Numeric treatment improves test F1 for every model in this split. In CV it improves logistic regression, SVM and KNN, while the tree models slightly favor categorical treatment. One split and seed do not establish statistical significance or a universal winner.

## Reproduce

Run `python analysis/compare_smotenc_booleans.py` from the repository root with pandas, scikit-learn, imbalanced-learn, and threadpoolctl installed. This run used `/Users/tommy/.venv/bin/python`, scikit-learn 1.7.2, imbalanced-learn 0.14.2, pandas 2.3.3.

Raw fold scores, summary scores and synthetic Boolean means are saved in the CSV files beside the script. The notebooks were not modified by this analysis.
