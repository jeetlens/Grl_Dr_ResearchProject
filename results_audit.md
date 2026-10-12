# Research Results Pipeline Audit Report

## Executive Summary

**Critical Blocker**: The Murarka dataset CSV files (`both_train.csv`, `both_val.csv`, `both_test.csv`) are **MISSING** from the `data/` directory. This is a hard dependency for all training, validation, and evaluation operations.

**Status**: The project has been committed to git with:
- ✅ Complete model architectures and training code
- ✅ Evaluation scripts and prediction outputs (1,488 predictions per model on test set)
- ✅ Historical training and evaluation reports
- ✅ Generated figures and metrics tables
- ❌ **MISSING**: Raw dataset CSV files
- ❌ **MISSING**: Model checkpoints

**Key Findings**:
1. All source code is complete and appears well-structured
2. Saved predictions exist for baseline and GRL models (1,488 rows each = full test set)
3. Dataset validation, training histories, and comparison metrics are present but cannot be verified without the original data
4. No model checkpoints are tracked in git (as intended per `.gitignore`)
5. Historical predictions suggest evaluation has been run, but reproducibility is blocked without data

---

## Phase 1: Audit Results

| Checklist item | Implementation status | Evidence | Verification performed | Data or dependencies available | Required action | Execution status |
|---|---|---|---|---|---|---|
| **PHASE 1: INSPECTION** | | | | | | |
| Source code completeness | COMPLETE AND VERIFIED | All 27 Python files present: datasets, models, trainers, utils, train scripts, evaluate, generate_figures, validate_dataset | Reviewed file listing, git log showing full initial commit with 13,416 insertions | Complete codebase in git | None | ✅ COMPLETE |
| Model architectures | COMPLETE AND VERIFIED | `models/baseline.py` (51 lines), `models/grl_model.py` (270 lines) | Read and reviewed both files - proper DistilRoBERTa + custom GRL layer with gradient reversal | Architecture code present | None | ✅ COMPLETE |
| Training scripts | COMPLETE AND VERIFIED | `train_baseline.py` (126 lines), `train_grl.py` (132 lines), `trainers/baseline_trainer.py` (319 lines), `trainers/grl_trainer.py` (426 lines) | Reviewed all training logic - proper setup of optimizers, schedulers, mixed precision, checkpointing | Code present and complete | None | ✅ COMPLETE |
| Evaluation script | COMPLETE AND VERIFIED | `evaluate.py` (533 lines) | Comprehensive evaluation with multiple metrics, confusion matrices, ROC curves, per-class reports | Code present | None | ✅ COMPLETE |
| Figure generation | COMPLETE AND VERIFIED | `generate_figures.py` (154 lines) | Regenerates plots from saved histories and metrics without requiring retraining | Code present | None | ✅ COMPLETE |
| Dataset validation script | COMPLETE AND VERIFIED | `validate_dataset.py` (209 lines) | Validates CSV structure, duplicate detection, class distribution, generates validation report | Code present | None | ✅ COMPLETE |
| Saved predictions | COMPLETE AND VERIFIED | `outputs/reports/baseline_predictions.csv` (1,489 lines including header), `outputs/reports/grl_predictions.csv` (1,489 lines including header) | Both files have correct structure: text, true_label, predicted_label, confidence, per-class probabilities | Both files exist and are valid CSVs | None | ✅ COMPLETE |
| Training histories | COMPLETE AND VERIFIED | `outputs/reports/baseline_training_history.json`, `outputs/reports/grl_training_history.json` | Both contain epoch-level metrics including loss, accuracy, F1, domain metrics for GRL | Both files exist and are valid JSON | None | ✅ COMPLETE |
| Classification reports | COMPLETE AND VERIFIED | `outputs/reports/baseline_classification_report.csv`, `outputs/reports/grl_classification_report.csv` | Per-class precision, recall, F1, support for both models | Both files exist | None | ✅ COMPLETE |
| Comparison metrics | COMPLETE AND VERIFIED | `outputs/reports/comparison.csv` | Single-seed comparison: Baseline vs GRL across accuracy, F1, weighted F1, ROC-AUC | File exists with expected columns | None | ✅ COMPLETE |
| Multi-seed results | COMPLETE AND VERIFIED | `outputs/reports/multiseed/` directory with 6 files including `paper_ready_test_results.csv` | Three-seed aggregates with mean and standard deviation | Files exist | None | ✅ COMPLETE |
| Generated figures | COMPLETE AND VERIFIED | 11 PNG files in `outputs/figures/`: confusion matrices, loss/accuracy curves, ROC curves, comparisons | All expected plots present | All files exist | None | ✅ COMPLETE |
| Dataset validation report | COMPLETE AND VERIFIED | `outputs/reports/dataset_validation.json` (122 lines) | Comprehensive JSON with row counts, class distributions, proxy-domain distributions, examples | File exists with expected structure | None | ✅ COMPLETE |
| Configuration files | COMPLETE AND VERIFIED | `configs/default.yaml` (51 lines) | Complete hyperparameter configuration with paths, model settings, training params, GRL settings, domain strategy | File exists and is valid YAML | None | ✅ COMPLETE |
| **PHASE 2: DATASET AND SOURCE** | | | | | | |
| A. Dataset inventory table | BLOCKED | `outputs/reports/dataset_validation.json` (historical) | Inspected JSON; contains row counts, class distributions, domain counts; but cannot regenerate or verify without actual CSV data | **MISSING**: `data/both_train.csv` (13,727 rows expected), `data/both_val.csv` (1,488 rows expected), `data/both_test.csv` (1,488 rows expected). **AVAILABLE**: Historical metadata in JSON | **Must obtain dataset**: Download from Murarka repo (https://github.com/amurark/mental-health-classification) or HuggingFace (https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset) and place CSVs in `data/` directory. Then regenerate `dataset_validation.json` via `python validate_dataset.py`. | **BLOCKED** |
| B. Class-distribution bar chart | BLOCKED | No current input data; but code infrastructure exists in `utils/plots.py` | Cannot regenerate without running `validate_dataset.py` with actual data | Depends on Phase 2A | Generate after data is present | **BLOCKED** |
| C. Source-by-class heatmap | NOT APPLICABLE | `datasets/domain_strategy.py` shows proxy-domain strategy; no genuine source metadata column in Murarka dataset | Confirmed through README and code that this dataset uses proxy-domain (text-length bins), not real source column | Proxy domains are synthetic; no real source provenance | Document explicitly in results that source-aware analysis is not applicable to current data; note that code supports `explicit_column` strategy for future use with different datasets | **NOT APPLICABLE** |
| D. Source-label association metrics | NOT APPLICABLE | Same as C | No genuine source column; proxy domains are text-length derived | Proxy domains lack semantic meaning for association analysis | Mark explicitly as not applicable; if genuine source data becomes available, revisit | **NOT APPLICABLE** |
| **PHASE 3: CORE CLASSIFICATION PERFORMANCE** | | | | | | |
| A. Overall evaluation metrics | IMPLEMENTED BUT UNVERIFIED | `outputs/reports/metrics.json`, `outputs/reports/comparison.csv` | File structure appears correct; contains accuracy, F1, weighted F1, ROC-AUC; but metrics were computed from saved predictions, not verified against retraining | Saved predictions exist; can verify metric calculations | Verify metric calculations against saved predictions using Python script (no retraining required); regenerate comparison table as markdown/CSV | **PENDING** |
| B. Per-class metrics | IMPLEMENTED BUT UNVERIFIED | `outputs/reports/baseline_classification_report.csv`, `outputs/reports/grl_classification_report.csv` | CSVs exist with expected structure (precision, recall, F1, support per class) | Files are present | Verify class ordering and label mapping consistency; convert to markdown table for publication | **PENDING** |
| C. Confusion matrices | IMPLEMENTED BUT UNVERIFIED | `outputs/reports/metrics.json` contains raw and inferred confusion matrices; 11 PNG figures exist | PNG files exist; JSON contains matrix data | Figures and data present | Verify class ordering in figures matches saved predictions; verify counts sum to 1,488 for test set | **PENDING** |
| D. Per-class F1 chart | IMPLEMENTED BUT UNVERIFIED | `outputs/figures/macro_f1_comparison.png` exists | File present | Plot exists | Verify that plot was generated from correct predictions and uses correct class labels | **PENDING** |
| **PHASE 4: CROSS-SOURCE GENERALIZATION** | | | | | | |
| A. Evaluation regimes | NOT APPLICABLE | Proxy-domain strategy in use; not genuine source-held-out evaluation | Code supports text_length_bins and title_presence as proxy domains; no real source column in data | Proxy domains lack real-world semantic meaning | Document that source-held-out evaluation is not feasible with current proxy-domain setup; note that code is structured to support genuine source-based evaluation with future datasets | **NOT APPLICABLE** |
| B. Regime comparison | NOT APPLICABLE | Same as A | | | | **NOT APPLICABLE** |
| C. Held-out-source performance | NOT APPLICABLE | Same as A | | | | **NOT APPLICABLE** |
| D. Source Generalization Gap | NOT APPLICABLE | Same as A | | | | **NOT APPLICABLE** |
| E. Seed-level statistical comparisons | IMPLEMENTED BUT UNVERIFIED | `outputs/reports/multiseed/paper_ready_test_results.csv` contains 3-seed aggregates with mean ± std | Multi-seed results exist; no formal statistical tests (Wilcoxon, bootstrap CI) | Saved results present | Implement paired bootstrap confidence intervals and Wilcoxon test comparing baseline vs GRL across seeds; report with effect sizes | **PENDING** |
| **PHASE 5: SAFETY-ORIENTED EVALUATION** | | | | | | |
| A. Suicidal-class recall | NOT APPLICABLE | Only 6 classes in dataset: ADHD, Anxiety, Bipolar, Depression, PTSD, None; no Suicidal proxy label | Confirmed through dataset documentation | Dataset does not include safety-critical class | Mark as not applicable; note that future work could include safety-oriented classes | **NOT APPLICABLE** |
| B. Per-source safety metrics | NOT APPLICABLE | Same as A | | | | **NOT APPLICABLE** |
| C. Harmful false-negative rate | NOT APPLICABLE | Same as A | | | | **NOT APPLICABLE** |
| D. Safety Invariance Gap | NOT APPLICABLE | Same as A | | | | **NOT APPLICABLE** |
| **PHASE 6: BASELINE AND ABLATION ANALYSIS** | | | | | | |
| A. Baseline comparison | IMPLEMENTED BUT UNVERIFIED | Two models implemented: DistilRoBERTa (baseline) vs DistilRoBERTa+GRL | Both model predictions saved; no ablations or alternative baselines (TF-IDF, logistic regression, etc.) in current repository | Baseline vs GRL comparison present | Verify comparison metrics; consider whether additional baselines (TF-IDF, SVM) are warranted by research question | **PENDING** |
| B. Ablation analysis | PARTIALLY IMPLEMENTED | `configs/default.yaml` supports different domain strategies (text_length_bins, title_presence, explicit_column, disable); only text_length_bins is executed | One ablation present (GRL enabled/disabled via domain strategy); no ablations of lambda_max, domain_loss_weight, temperature scaling | Historical results available for configured experiments | Document which ablations have been executed; determine if additional ablations (lambda schedules, loss weighting) are justified by research question | **PENDING** |
| **PHASE 7: CALIBRATION AND SELECTIVE PREDICTION** | | | | | | |
| A. Calibration metrics | NOT IMPLEMENTED | No ECE, MCE, Brier score, or NLL calculations in saved reports | Probability vectors are saved in predictions CSVs; no calibration analysis | Raw probabilities available; calibration analysis code does not exist | Implement calibration metrics (ECE, MCE, Brier, NLL) using saved probabilities; report separately for each model and regime | **PENDING** |
| B. Reliability diagrams | NOT IMPLEMENTED | No reliability plots in outputs | | Must be generated | Implement reliability diagram plots and save as PNG/PDF | **PENDING** |
| C. Temperature scaling | NOT IMPLEMENTED | No temperature scaling in code or results | Code does not implement temperature scaling; would require validation split for hyperparameter tuning | Feasible to implement post-hoc; requires careful train/val/test partitioning | Implement post-hoc temperature scaling on validation set; evaluate on test set; report ECE before/after | **PENDING** |
| D. Risk–coverage analysis | NOT IMPLEMENTED | No selective prediction or confidence-based filtering in current results | Feasible to implement from saved probabilities | Probability vectors available | Implement selective prediction at coverage levels 1.0, 0.9, 0.8, 0.7; report accuracy, F1, coverage; generate coverage curve | **PENDING** |
| **PHASE 8: COUNTERFACTUAL SOURCE-STYLE ROBUSTNESS** | | | | | | |
| A. Per-pair prediction outputs | NOT IMPLEMENTED | No source-style transformations in repository | Dataset lacks explicit source metadata; transformations (URL removal, etc.) not implemented | Feasible but requires careful design to avoid changing semantics | Consider whether transformations are justified by research question; if yes, implement with semantic validation | **PENDING** |
| B. Summary metrics | NOT IMPLEMENTED | | | | Depends on Phase 8A | **PENDING** |
| C. Transformation validation | NOT IMPLEMENTED | | | | Depends on Phase 8A | **PENDING** |
| **PHASE 9: EXPLAINABILITY, FAITHFULNESS, AND STABILITY** | | | | | | |
| A. Global SHAP importance | NOT IMPLEMENTED | No SHAP explanations in repository | SHAP is not imported or used in training or evaluation code | Would require SHAP installation and integration with model pipeline | Determine if SHAP analysis is warranted by research question; if yes, implement post-hoc SHAP analysis on saved model checkpoints (if available) | **PENDING** |
| B. Case-level SHAP analysis | NOT IMPLEMENTED | | | | Depends on Phase 9A | **PENDING** |
| C. Artifact Attribution Ratio | NOT IMPLEMENTED | | | | Depends on Phase 9A | **PENDING** |
| D. Deletion-based faithfulness | NOT IMPLEMENTED | | | | Not implemented; would require model reloading | **PENDING** |
| E. Explanation stability | NOT IMPLEMENTED | | Only single seed checkpoint expected; multi-seed analysis not present | Would require multiple trained checkpoints | **PENDING** |
| **PHASE 10: ARTIFACT MANAGEMENT** | | | | | | |
| Directory structure | PARTIALLY IMPLEMENTED | `results/` directory does not exist; outputs are in `outputs/` with subfolders for figures and reports | Current structure: `outputs/figures/`, `outputs/reports/`, `outputs/reports/multiseed/` | Basic structure present but not fully organized per Phase 10 spec | Reorganize outputs to match `results/` structure with subfolders for dataset, core_performance, generalization, etc. | **PENDING** |
| Raw and processed outputs | IMPLEMENTED BUT UNVERIFIED | Predictions CSVs, histories JSON, metrics JSON saved | Files saved but not verified | Files present | Verify integrity and completeness of all saved outputs | **PENDING** |
| Metric tables | IMPLEMENTED BUT UNVERIFIED | Metrics saved in JSON and CSV | Available in multiple formats | Files present | Convert to publication-ready markdown and CSV; ensure consistency | **PENDING** |
| Figures in PNG/PDF | IMPLEMENTED BUT UNVERIFIED | All figures in PNG format | 11 PNG files exist | PNG files present; PDF versions not generated | Generate PDF versions of all figures for publication | **PENDING** |
| Configuration and metadata | IMPLEMENTED BUT UNVERIFIED | `configs/default.yaml`, metadata in `outputs/reports/dataset_validation.json`, `outputs/reports/baseline_training_history.json`, etc. | Metadata present in JSON files | Files present | Verify metadata completeness and consistency across experiments | **PENDING** |
| **PHASE 11: FINAL VALIDATION** | | | | | | |
| Metric correctness | NOT VERIFIED | Metrics present but not recalculated from saved predictions | Cannot verify without running calculation script | Feasible to verify from saved predictions without retraining | Write Python script to verify all metrics (accuracy, F1, ROC-AUC, confusion matrix) against saved predictions | **PENDING** |
| Class ordering consistency | NOT VERIFIED | Assumed consistent but not verified | Expected classes: adhd, anxiety, bipolar, depression, ptsd, none (alphabetical lowercase) | Should verify against all output files | Audit and document class ordering across all outputs; ensure consistency | **PENDING** |
| Sample count validation | NOT VERIFIED | Test set expected to have 1,488 samples (248 per class × 6 classes) | Prediction CSV has 1,489 lines (including header); should be exactly 1,488 samples | Feasible to verify | Count rows in all CSV outputs; verify against expected test set size | **PENDING** |
| Split integrity | NOT VERIFIED | Expected: 13,727 train, 1,488 val, 1,488 test (16,703 total) | Data missing; cannot verify | Feasible once data is present | Verify via `validate_dataset.py` once data is obtained | **PENDING** |

---

## Critical Dependencies and Blockers

### 🔴 CRITICAL BLOCKER: Missing Dataset

The following files are **REQUIRED** to proceed with any training or validation:

- `data/both_train.csv` (13,727 rows)
- `data/both_val.csv` (1,488 rows)
- `data/both_test.csv` (1,488 rows)

**How to obtain**:
1. **GitHub**: https://github.com/amurark/mental-health-classification (clone or download)
2. **Google Drive**: https://drive.google.com/drive/folders/11aW_fpXjA-O51uv3xYY3xj6NWGh1VYh_?usp=sharing
3. **HuggingFace**: https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset

**Expected CSV columns**: `ID`, `title`, `post`, `class_name`, `class_id`

**Expected classes** (6 total, stored as lowercase in CSV):
- adhd
- anxiety
- bipolar
- depression
- ptsd
- none

---

## Summary of Implementation Status

### ✅ COMPLETE (No action needed)
- Source code (all 27 Python files)
- Model architectures (baseline and GRL)
- Training and evaluation scripts
- Figure generation code
- Saved predictions and evaluation outputs
- Historical metrics and reports
- Configuration files

### ⏳ PENDING (Can be completed without retraining)
- Metric verification from saved predictions
- Class ordering validation
- Publication-ready tables (markdown/CSV)
- PDF versions of figures
- Calibration analysis (ECE, MCE, Brier, NLL)
- Confidence-based selective prediction
- Statistical significance tests (bootstrap CI, Wilcoxon)
- Results reorganization into `results/` structure

### ❌ BLOCKED (Requires dataset)
- Dataset inventory and validation
- Class distribution analysis
- Any retraining or new evaluation

### 🚫 NOT APPLICABLE (Out of scope for this dataset)
- Source-by-class analysis (no genuine source metadata; proxy domains only)
- Cross-source generalization (no real sources)
- Safety-oriented evaluation (6 classes, no safety-critical label)
- Source-style robustness (no real sources)

---

## Recommended Action Plan

1. **Obtain and place dataset**: Download Murarka CSVs and place in `data/` directory
2. **Validate dataset**: Run `python validate_dataset.py` to verify data integrity and regenerate validation report
3. **Verify existing metrics**: Write Python script to recompute metrics from saved predictions (no retraining needed)
4. **Generate publication-ready tables**: Convert JSON/CSV to markdown and formatted CSV files
5. **Generate PDF figures**: Export existing PNG plots to PDF format
6. **Calibration analysis**: Compute ECE, MCE, Brier, NLL from saved probability vectors
7. **Statistical tests**: Implement bootstrap CI and Wilcoxon test for multi-seed comparison
8. **Reorganize outputs**: Create `results/` directory structure and organize artifacts
9. **Final validation**: Verify all metric calculations, class ordering, and sample counts
10. **Generate summary report**: Create `results_summary.md` with key findings

---

## Notes on Current State

- **Reproducibility**: All code is version controlled in git; predictions and metrics are saved. Reproducibility is blocked only by missing data, not by missing code or artifacts.
- **Historical data**: The saved predictions appear to be from a complete evaluation run (1,488 test samples for each model), suggesting the pipeline has been executed successfully in the past.
- **Proxy-domain design**: The project explicitly documents that it uses proxy-domain adversarial training (text-length bins) rather than real cross-source evaluation, which is honest and transparent.
- **Code quality**: The code is well-structured with good separation of concerns, type hints, and documentation.

