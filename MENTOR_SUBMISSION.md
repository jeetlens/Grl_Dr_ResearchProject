# Mentor Submission

## Project Summary

This project studies multiclass mental health text classification using
DistilRoBERTa. It compares a supervised DistilRoBERTa baseline with a
DistilRoBERTa model extended with a Gradient Reversal Layer (GRL) and an
auxiliary domain classifier.

The purpose is to evaluate whether proxy-domain adversarial training improves
classification robustness while preserving predictive performance.

## Reference Paper

Murarka, A., Radhakrishnan, B., and Ravichandran, S. (2021).
Classification of mental illnesses on social media using RoBERTa.
Proceedings of the 12th International Workshop on Health Text Mining and
Information Analysis, pages 59-68.

- ACL Anthology: https://aclanthology.org/2021.louhi-1.7/
- arXiv: https://arxiv.org/abs/2011.11226

## Why This Paper Was Chosen

The Murarka et al. paper directly matches the project domain: transformer-based
classification of mental health-related social media posts. It provides a clear
multiclass task, a public dataset, and a relevant RoBERTa-based baseline that
can be extended with domain-adversarial learning.

## Dataset

The project expects the Murarka et al. dataset files:

- `data/both_train.csv`
- `data/both_val.csv`
- `data/both_test.csv`

Dataset links:

- https://github.com/amurark/mental-health-classification
- https://drive.google.com/drive/folders/11aW_fpXjA-O51uv3xYY3xj6NWGh1VYh_?usp=sharing
- https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset

Classes:

- ADHD
- Anxiety
- Bipolar
- Depression
- PTSD
- None

## Why This Dataset Was Chosen

The dataset was chosen because it is associated with the reference paper,
contains Reddit title/post text, and supports a six-class mental health
classification task. It is suitable for comparing a strong transformer baseline
against a GRL-based extension without changing the underlying classification
objective.

## GitHub Repository

Repository: https://github.com/Akashbera28/mental-health-classification.git

## Models

Baseline:

- Backbone: `distilroberta-base`
- Objective: supervised multiclass cross-entropy
- Model selection: best validation macro F1

GRL:

- Backbone: `distilroberta-base`
- Primary head: mental health classification
- Auxiliary head: proxy-domain classification
- GRL lambda: 0.1
- Domain loss weight: 0.1
- Proxy domain: text-length bins using the training median

## Training and Evaluation

Standard commands:

```bash
python validate_dataset.py
python train_baseline.py
python train_grl.py
python evaluate.py
python generate_figures.py
```

The final audit did not retrain models or regenerate experiments. Saved result
files were treated as the source of truth.

## Outputs

Important outputs:

- `outputs/reports/metrics.json`
- `outputs/reports/comparison.csv`
- `outputs/reports/baseline_classification_report.csv`
- `outputs/reports/grl_classification_report.csv`
- `outputs/reports/multiseed/`
- `outputs/figures/`
- `RESULTS.md`

## Final Results

Single-seed test results show the GRL model ahead:

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| DistilRoBERTa | 84.01% | 84.02% | 84.02% | 97.78% |
| DistilRoBERTa + GRL | 84.61% | 84.61% | 84.61% | 97.87% |

Three-seed test averages show the baseline slightly ahead:

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| DistilRoBERTa | 84.81 +/- 0.71 | 84.83 +/- 0.71 | 84.83 +/- 0.71 | 97.81 +/- 0.03 |
| DistilRoBERTa + GRL | 84.59 +/- 0.84 | 84.61 +/- 0.81 | 84.61 +/- 0.81 | 97.79 +/- 0.16 |

No statistical significance test is included.

## Limitations

- The system is for research only and is not a diagnostic tool.
- The dataset labels are not clinical diagnoses.
- The default GRL setup uses proxy-domain labels from text length, not genuine
  platform/source domains.
- The observed baseline-vs-GRL difference is small.
- Statistical significance testing is not included.

## Future Work

- Evaluate with explicit source/domain labels.
- Add statistical significance testing across seeds.
- Test larger or newer transformer backbones.
- Explore class-balanced losses.
- Add calibration and uncertainty analysis.
- Evaluate fairness and subgroup robustness where ethical metadata is
  available.

## Final Conclusion

The repository provides a reproducible DistilRoBERTa baseline and a GRL-based
extension for six-class mental health text classification. The saved results
show that GRL can outperform the baseline in one seed, but the baseline has a
small average advantage across three test seeds. Under the current proxy-domain
setup, GRL does not provide a consistent improvement.
