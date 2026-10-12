# Implementation Plan: Research Results Pipeline

**Date**: October 12, 2026  
**Status**: Dataset validated and ready for processing

## Phase Summary

This document outlines the complete implementation plan to audit, complete, and validate the research results pipeline for the mental health classification project.

---

## Executive Status

| Phase | Status | Priority | Dependencies |
|---|---|---|---|
| **Phase 1: Inspection** | ✅ COMPLETE | DONE | None |
| **Phase 2A: Dataset Inventory** | ⏳ PENDING | HIGH | Dataset validated ✅ |
| **Phase 3: Core Performance** | ⏳ PENDING | HIGH | Metric verification script |
| **Phase 4: Cross-Source** | 🚫 NOT APPLICABLE | — | Proxy domains only |
| **Phase 5: Safety** | 🚫 NOT APPLICABLE | — | No safety-critical label |
| **Phase 6: Baselines/Ablations** | ⏳ PENDING | MEDIUM | Existing results |
| **Phase 7: Calibration** | ⏳ PENDING | HIGH | Probability vectors ✅ |
| **Phase 8: Counterfactuals** | 🚫 NOT APPLICABLE | — | No real sources |
| **Phase 9: Explainability** | ⏳ PENDING | LOW | Optional for paper |
| **Phase 10: Artifact Management** | ⏳ PENDING | MEDIUM | All components |
| **Phase 11: Final Validation** | ⏳ PENDING | HIGH | All components |

---

## Detailed Implementation Schedule

### STEP 1: Generate Dataset Inventory Table (Phase 2A)
- **Action**: Create CSV and Markdown tables from validation report
- **Input**: `outputs/reports/dataset_validation.json`
- **Output**: 
  - `results/dataset/dataset_inventory.csv`
  - `results/dataset/dataset_inventory.md`
- **Effort**: Script-based, ~30 minutes
- **Status**: READY TO EXECUTE

### STEP 2: Generate Class Distribution Chart (Phase 2B)
- **Action**: Create bar chart of class counts
- **Input**: `outputs/reports/dataset_validation.json`
- **Output**: `results/dataset/class_distribution.png`, `.pdf`
- **Effort**: Plotting script, ~30 minutes
- **Status**: READY TO EXECUTE

### STEP 3: Verify Metric Calculations (Phase 3A-D)
- **Action**: Recompute all metrics from saved predictions
- **Input**: `outputs/reports/baseline_predictions.csv`, `outputs/reports/grl_predictions.csv`
- **Output**: Verification script with computed metrics
- **Effort**: ~1 hour
- **Status**: READY TO EXECUTE

### STEP 4: Generate Publication-Ready Tables (Phase 3)
- **Action**: Convert metrics to markdown/CSV with proper formatting
- **Output**: 
  - `results/core_performance/overall_metrics.md`
  - `results/core_performance/per_class_metrics.csv`
  - `results/core_performance/confusion_matrices.md`
- **Effort**: ~45 minutes
- **Status**: READY TO EXECUTE

### STEP 5: Generate PDF Figures (Phase 10)
- **Action**: Export PNG figures to PDF format
- **Input**: All 11 PNG files in `outputs/figures/`
- **Output**: `.pdf` versions
- **Effort**: ~30 minutes
- **Status**: READY TO EXECUTE

### STEP 6: Compute Calibration Metrics (Phase 7A)
- **Action**: Calculate ECE, MCE, Brier, NLL from probabilities
- **Input**: `outputs/reports/baseline_predictions.csv`, `outputs/reports/grl_predictions.csv`
- **Output**: `results/calibration/calibration_metrics.csv`
- **Effort**: ~1 hour
- **Status**: READY TO EXECUTE

### STEP 7: Generate Reliability Diagrams (Phase 7B)
- **Action**: Create reliability plots
- **Output**: `results/calibration/reliability_diagrams.png`, `.pdf`
- **Effort**: ~45 minutes
- **Status**: READY TO EXECUTE

### STEP 8: Temperature Scaling (Phase 7C)
- **Action**: Post-hoc temperature scaling on validation set
- **Input**: `outputs/reports/baseline_training_history.json` for validation metrics
- **Output**: `results/calibration/temperature_scaling_results.csv`
- **Effort**: ~1.5 hours (requires careful implementation)
- **Status**: READY TO EXECUTE

### STEP 9: Risk-Coverage Analysis (Phase 7D)
- **Action**: Selective prediction at coverage levels 1.0, 0.9, 0.8, 0.7
- **Output**: `results/calibration/coverage_analysis.csv`, chart
- **Effort**: ~1 hour
- **Status**: READY TO EXECUTE

### STEP 10: Statistical Significance Tests (Phase 4E, Phase 6)
- **Action**: Bootstrap CI and Wilcoxon test for multi-seed comparison
- **Input**: `outputs/reports/multiseed/paper_ready_test_results.csv`
- **Output**: `results/statistics/significance_tests.md`
- **Effort**: ~1.5 hours
- **Status**: READY TO EXECUTE

### STEP 11: Reorganize Outputs (Phase 10)
- **Action**: Create `results/` directory structure and organize all artifacts
- **Output**: Organized `results/` directory
- **Effort**: ~30 minutes
- **Status**: READY TO EXECUTE

### STEP 12: Final Validation & Summary (Phase 11)
- **Action**: Verify all metrics, class ordering, sample counts
- **Output**: `results_summary.md`
- **Effort**: ~1.5 hours
- **Status**: READY TO EXECUTE

---

## Total Estimated Effort

- **Total Time**: ~10 hours (mostly automation/scripts)
- **High Priority Items**: Steps 1-4, 10-12 (metric verification, publication tables, summary)
- **Medium Priority Items**: Steps 5-7, 9 (calibration, figures)
- **Low Priority Items**: Step 8 (nice-to-have but not critical)

---

## Execution Order

1. Steps 1-2: Dataset analysis (HIGH, foundation for paper)
2. Steps 3-4: Metric verification and tables (HIGH, core results)
3. Steps 10-12: Statistics and validation (HIGH, confidence in results)
4. Steps 5-7: Figures and calibration (MEDIUM, supporting material)
5. Step 8: Temperature scaling (LOW, optional)

---

## Success Criteria

- [ ] All metrics from saved predictions verified and match saved reports
- [ ] Publication-ready markdown/CSV tables generated
- [ ] All figures exported to PNG and PDF
- [ ] Calibration analysis complete with ECE/MCE/Brier scores
- [ ] Statistical tests with proper confidence intervals reported
- [ ] Results organized into clean `results/` directory structure
- [ ] Final summary document with all key findings
- [ ] All outputs traceable to source data and code

---

## Notes

- **No retraining required**: All steps use saved predictions and existing checkpoints
- **Reproducibility**: Every output is generated from version-controlled code and saved artifacts
- **Out-of-scope for this run**: SHAP analysis, additional baselines, source-aware evaluation (not applicable to proxy domains)

