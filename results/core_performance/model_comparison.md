# Model Comparison: Baseline vs GRL

## Overall Metrics

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| DistilRoBERTa (Baseline) | 0.8401 | 0.8446 | 0.8401 | 0.8402 | 0.8402 | 0.9778 |
| DistilRoBERTa + GRL | 0.8461 | 0.8474 | 0.8461 | 0.8461 | 0.8461 | 0.9787 |

## Per-Class Metrics (Baseline)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| adhd | 0.8115 | 0.8508 | 0.8307 | 248 |
| anxiety | 0.7857 | 0.8427 | 0.8132 | 248 |
| bipolar | 0.8357 | 0.7177 | 0.7722 | 248 |
| depression | 0.7500 | 0.8589 | 0.8008 | 248 |
| none | 0.9755 | 0.9637 | 0.9696 | 248 |
| ptsd | 0.9091 | 0.8065 | 0.8547 | 248 |

## Per-Class Metrics (GRL)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| adhd | 0.8642 | 0.8468 | 0.8554 | 248 |
| anxiety | 0.8080 | 0.8145 | 0.8112 | 248 |
| bipolar | 0.8060 | 0.7540 | 0.7792 | 248 |
| depression | 0.7690 | 0.8589 | 0.8114 | 248 |
| none | 0.9598 | 0.9637 | 0.9618 | 248 |
| ptsd | 0.8776 | 0.8387 | 0.8577 | 248 |
