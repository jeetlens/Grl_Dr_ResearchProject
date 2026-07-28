# 🧠 Mental Health Classification using DistilRoBERTa and Gradient Reversal Layer (GRL)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-success)

A deep learning research project for **multiclass mental health text classification** using **DistilRoBERTa** and **Domain-Adversarial Learning** with a **Gradient Reversal Layer (GRL)**.

This repository compares a standard DistilRoBERTa classifier with a domain-adversarial version that aims to learn domain-invariant representations while maintaining high classification performance.

---

# 📌 Overview

Mental health disorders are increasingly discussed on online platforms such as Reddit and other social media. Automatically identifying posts related to different mental health conditions can assist researchers in studying trends and developing supportive technologies.

This project investigates whether **Domain-Adversarial Training (GRL)** can improve the robustness of transformer-based mental health classification compared to a standard DistilRoBERTa model.

---

# 🎯 Objectives

- Build a strong DistilRoBERTa baseline.
- Implement Gradient Reversal Layer (GRL).
- Compare baseline and GRL models.
- Perform multi-seed evaluation.
- Generate publication-quality experimental results.
- Ensure reproducibility.

---

# 🧠 Mental Health Classes

The classifier predicts one of the following categories:

- Normal
- Depression
- Anxiety
- Stress
- Bipolar Disorder
- Personality Disorder
- Suicidal

---

# 🏗️ Project Structure

```text
mental-health-classification/
│
├── configs/
├── data/
│   ├── raw/
│   └── processed/
│
├── datasets/
├── models/
├── trainers/
├── utils/
├── tests/
│
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── reports/
│
├── train_baseline.py
├── train_grl.py
├── evaluate.py
├── generate_figures.py
├── validate_dataset.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚙️ Installation

Clone the repository.

```bash
git clone https://github.com/Akashbera28/mental-health-classification.git

cd mental-health-classification
```

Create virtual environment.

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset

The original dataset is **not included** in this repository due to licensing and size limitations.

Place your dataset inside

```
data/raw/
```

Validate the dataset using

```bash
python validate_dataset.py
```

---

# 🚀 Training

## Baseline DistilRoBERTa

```bash
python train_baseline.py
```

---

## DistilRoBERTa + Gradient Reversal Layer

```bash
python train_grl.py
```

---

# 📊 Evaluation

Evaluate trained models

```bash
python evaluate.py
```

Evaluation reports are automatically saved in

```
outputs/reports/
```

---

# 📈 Experimental Results

Average performance across multiple random seeds.

| Model | Accuracy | Macro F1 | Weighted F1 |
|--------|----------|----------|-------------|
| DistilRoBERTa | **84.81%** | **84.83%** | **84.83%** |
| DistilRoBERTa + GRL | 84.59% | 84.61% | 84.61% |

The baseline model achieved slightly better overall performance, while the GRL model demonstrated comparable results under the current proxy-domain setup.

---

# 📉 Generated Outputs

The project automatically generates

- Classification Report
- Confusion Matrix
- ROC Curves
- Training Curves
- Loss Curves
- Performance Comparison Figures
- Multi-seed Result Tables

All outputs are stored in

```
outputs/
```

---

# 🔬 Research Highlights

- Transformer-based Mental Health Classification
- DistilRoBERTa Fine-tuning
- Domain-Adversarial Learning
- Gradient Reversal Layer
- Multi-seed Evaluation
- Reproducible Experiments
- Publication-ready Results

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilRoBERTa
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Google Colab

---

# ⚠️ Limitations

- This project is intended for research purposes only.
- Predictions should **not** be interpreted as medical diagnoses.
- Performance depends on the quality and diversity of the training dataset.
- The current domain-adversarial setup uses proxy-domain labels.

---

# 🚀 Future Work

- Cross-platform domain adaptation
- Explainable AI (SHAP / Integrated Gradients)
- Larger transformer models
- Statistical significance testing
- Bias and fairness analysis
- Multilingual mental health classification

---

# 📄 Citation

If you use this repository in your research, please cite it as:

```text
Akash Bera.

Mental Health Classification using DistilRoBERTa and Gradient Reversal Layer.

GitHub Repository.

2026.
```

---

# 📜 License

This project is released under the **MIT License**.

See the LICENSE file for details.

---

# 👨‍💻 Author

**Akash Bera**

Final Year Research Project

Artificial Intelligence & Machine Learning

---

## ⭐ Support

If you found this repository useful,

⭐ Star this repository on GitHub.

It helps others discover the project.

---

## 📬 Contact

For questions or collaborations, feel free to open an Issue or submit a Pull Request.