# Mental Health Classification Using DistilRoBERTa and GRL

A reproducible research project for multiclass mental health text
classification using DistilRoBERTa and a Gradient Reversal Layer (GRL).

The project compares:

- A supervised DistilRoBERTa baseline.
- A DistilRoBERTa + GRL model trained with proxy-domain adversarial learning.

The GRL experiment is intentionally described as proxy-domain adversarial
training because the Murarka `both_*.csv` files do not include a genuine
source/domain column. The default proxy domain is derived from text-length bins
using the training split median only.

## Reference Paper

This project is based on:

Murarka, A., Radhakrishnan, B., and Ravichandran, S. (2021).
Classification of mental illnesses on social media using RoBERTa.
Proceedings of the 12th International Workshop on Health Text Mining and
Information Analysis, pages 59-68.

- Paper: https://aclanthology.org/2021.louhi-1.7/
- arXiv: https://arxiv.org/abs/2011.11226

## Dataset

The expected dataset is the Murarka et al. Reddit mental health classification
dataset.

- Original dataset repository: https://github.com/amurark/mental-health-classification
- Google Drive dataset link: https://drive.google.com/drive/folders/11aW_fpXjA-O51uv3xYY3xj6NWGh1VYh_?usp=sharing
- Related Hugging Face dataset page: https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset

Place the CSV files at the repository root under `data/`:

```text
data/both_train.csv
data/both_val.csv
data/both_test.csv
```

The expected CSV columns are:

- `ID`
- `title`
- `post`
- `class_name`
- `class_id`

The model input is built from `title + post`; `class_name` is used as the
target label.

## Classes

All experiments use the six Murarka classes:

- ADHD
- Anxiety
- Bipolar
- Depression
- PTSD
- None

The raw CSV/report files store these labels in lowercase.

## Project Structure

```text
mental-health-classification/
├── configs/
├── data/
├── datasets/
├── models/
├── trainers/
├── utils/
├── tests/
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── reports/
├── train_baseline.py
├── train_grl.py
├── evaluate.py
├── generate_figures.py
├── validate_dataset.py
├── RESULTS.md
├── MENTOR_SUBMISSION.md
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/Akashbera28/mental-health-classification.git
cd mental-health-classification
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducible Workflow

Validate the dataset:

```bash
python validate_dataset.py
```

Train the baseline:

```bash
python train_baseline.py
```

Train the GRL model:

```bash
python train_grl.py
```

Evaluate trained checkpoints:

```bash
python evaluate.py
```

Generate figures from saved reports:

```bash
python generate_figures.py
```

Run code checks:

```bash
python -m compileall .
python -m unittest discover -s tests
pytest -q
```

Do not run training commands unless the dataset and compute environment are
ready. The audit artifacts in `outputs/reports/` are treated as the source of
truth for the reported metrics.

## Results Summary

Single-seed test results from `outputs/reports/comparison.csv`:

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| DistilRoBERTa | 84.01% | 84.02% | 84.02% | 97.78% |
| DistilRoBERTa + GRL | 84.61% | 84.61% | 84.61% | 97.87% |

Three-seed test averages from
`outputs/reports/multiseed/paper_ready_test_results.csv`:

| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| DistilRoBERTa | 84.81 +/- 0.71 | 84.83 +/- 0.71 | 84.83 +/- 0.71 | 97.81 +/- 0.03 |
| DistilRoBERTa + GRL | 84.59 +/- 0.84 | 84.61 +/- 0.81 | 84.61 +/- 0.81 | 97.79 +/- 0.16 |

The GRL model wins in the single seed preserved in `comparison.csv`, while the
baseline has a small average advantage across the three test seeds. No
statistical significance test is included, so the difference should be treated
as descriptive rather than statistically established.

See `RESULTS.md` for the full metric summary.

## Outputs

Generated artifacts are stored under `outputs/`:

- `outputs/reports/metrics.json`
- `outputs/reports/comparison.csv`
- `outputs/reports/*_classification_report.csv`
- `outputs/reports/multiseed/`
- `outputs/figures/`

Dataset CSVs and model checkpoints are intentionally excluded from Git.

## Limitations

- This repository is for research and education only.
- Model predictions are not medical diagnoses and must not be used for clinical
  decision-making.
- Labels are derived from Reddit communities and dataset annotation heuristics,
  not clinical assessments.
- The default GRL setup uses proxy-domain labels, not true cross-domain
  adaptation.
- The current repository reports descriptive multi-seed results but does not
  include a statistical significance test.

## License

This project is released under the MIT License. See `LICENSE` for details.
