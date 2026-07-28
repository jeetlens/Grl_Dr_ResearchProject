# Results

This file summarizes the saved experimental outputs without retraining or
modifying metrics. Raw files under `outputs/reports/` are the source of truth.

## Dataset Validation

Validated CSV files:

| Split | File | Usable Rows |
|---|---|---:|
| Train | `data/both_train.csv` | 13,727 |
| Validation | `data/both_val.csv` | 1,488 |
| Test | `data/both_test.csv` | 1,488 |

Classes:

- ADHD
- Anxiety
- Bipolar
- Depression
- PTSD
- None

Validation and test contain 248 examples per class. Training contains 2,465
ADHD, 2,422 Anxiety, 2,407 Bipolar, 2,450 Depression, 2,001 PTSD, and 1,982
None examples.

## Validation Metrics

Best single-seed validation metrics from the saved training histories:

| Model | Best Validation Macro F1 | Best Epoch |
|---|---:|---:|
| DistilRoBERTa | 84.97% | 2 |
| DistilRoBERTa + GRL | 84.63% | 3 |

Three-seed validation averages from
`outputs/reports/multiseed/final_summary.csv`:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| DistilRoBERTa | 84.59 +/- 0.27 | 84.74 +/- 0.29 | 84.74 +/- 0.29 |
| DistilRoBERTa + GRL | 84.36 +/- 0.21 | 84.50 +/- 0.18 | 84.50 +/- 0.18 |

## Single-Seed Test Metrics

From `outputs/reports/comparison.csv`:

| Model | Accuracy | Precision | Recall | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| DistilRoBERTa | 84.01% | 84.46% | 84.01% | 84.02% | 84.02% | 97.78% |
| DistilRoBERTa + GRL | 84.61% | 84.74% | 84.61% | 84.61% | 84.61% | 97.87% |

In this single seed, the GRL model performs better than the baseline.

## Three-Seed Test Metrics

From `outputs/reports/multiseed/paper_ready_test_results.csv`:

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| DistilRoBERTa | 84.81 +/- 0.71 | 84.83 +/- 0.71 | 84.83 +/- 0.71 | 97.81 +/- 0.03 |
| DistilRoBERTa + GRL | 84.59 +/- 0.84 | 84.61 +/- 0.81 | 84.61 +/- 0.81 | 97.79 +/- 0.16 |

Across three test seeds, the baseline has a small average advantage of about
0.22 percentage points in macro F1. No statistical significance test is present
in the repository, so this should be reported as a descriptive difference only.

## Per-Class Single-Seed Test Report

Baseline from `outputs/reports/baseline_classification_report.csv`:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ADHD | 81.15% | 85.08% | 83.07% | 248 |
| Anxiety | 78.57% | 84.27% | 81.32% | 248 |
| Bipolar | 83.57% | 71.77% | 77.22% | 248 |
| Depression | 75.00% | 85.89% | 80.08% | 248 |
| PTSD | 90.91% | 80.65% | 85.47% | 248 |
| None | 97.55% | 96.37% | 96.96% | 248 |

GRL from `outputs/reports/grl_classification_report.csv`:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ADHD | 86.42% | 84.68% | 85.54% | 248 |
| Anxiety | 80.80% | 81.45% | 81.12% | 248 |
| Bipolar | 80.60% | 75.40% | 77.92% | 248 |
| Depression | 76.90% | 85.89% | 81.14% | 248 |
| PTSD | 87.76% | 83.87% | 85.77% | 248 |
| None | 95.98% | 96.37% | 96.18% | 248 |

## Interpretation

Both models are close. The GRL branch improves the saved single-seed test
comparison, but the baseline has a small average advantage across three seeds.
Because the domain labels are text-length proxy domains, the result does not
demonstrate true cross-domain adaptation. A stronger conclusion would require
explicit source/domain metadata and significance testing.
