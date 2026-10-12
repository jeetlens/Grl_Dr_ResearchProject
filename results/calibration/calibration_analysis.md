# Calibration Analysis

## Calibration Metrics Comparison

| Metric | Baseline | GRL |
|---|---:|---:|
| ECE | 0.0386 | 0.0669 |
| MCE | 0.7108 | 0.7199 |
| Brier Score | 0.0371 | 0.0372 |
| Negative Log-Likelihood | 0.4472 | 0.4659 |
| Mean Confidence | 0.8777 | 0.9121 |

### Metric Definitions

- **ECE (Expected Calibration Error)**: Measures the gap between predicted confidence and actual accuracy across confidence bins (lower is better)
- **MCE (Maximum Calibration Error)**: Maximum calibration gap across all bins (lower is better)
- **Brier Score**: Mean squared error between predicted probabilities and true labels (lower is better)
- **NLL (Negative Log-Likelihood)**: Cross-entropy loss; measures how well predicted probabilities match true distribution (lower is better)
- **Mean Confidence**: Average predicted confidence across all samples

### Interpretation

Lower ECE, MCE, Brier Score, and NLL indicate better calibration. Models with high confidence predictions that are frequently incorrect are poorly calibrated.

## Baseline Model - Calibration by Confidence Bin

| Bin Range | Samples | Accuracy | Confidence | Gap |
|---|---:|---:|---:|---:|
| 0.20-0.30 | 1 | 1.0000 | 0.2892 | 0.7108 |
| 0.30-0.40 | 29 | 0.3103 | 0.3600 | 0.0497 |
| 0.40-0.50 | 71 | 0.4366 | 0.4548 | 0.0182 |
| 0.50-0.60 | 76 | 0.3816 | 0.5505 | 0.1689 |
| 0.60-0.70 | 70 | 0.4429 | 0.6525 | 0.2096 |
| 0.70-0.80 | 101 | 0.6238 | 0.7533 | 0.1296 |
| 0.80-0.90 | 123 | 0.8374 | 0.8530 | 0.0156 |
| 0.90-1.00 | 1017 | 0.9666 | 0.9779 | 0.0113 |

## GRL Model - Calibration by Confidence Bin

| Bin Range | Samples | Accuracy | Confidence | Gap |
|---|---:|---:|---:|---:|
| 0.20-0.30 | 1 | 1.0000 | 0.2801 | 0.7199 |
| 0.30-0.40 | 17 | 0.2941 | 0.3501 | 0.0559 |
| 0.40-0.50 | 42 | 0.2857 | 0.4553 | 0.1696 |
| 0.50-0.60 | 52 | 0.3846 | 0.5467 | 0.1621 |
| 0.60-0.70 | 63 | 0.3968 | 0.6530 | 0.2562 |
| 0.70-0.80 | 68 | 0.5441 | 0.7517 | 0.2076 |
| 0.80-0.90 | 113 | 0.7257 | 0.8524 | 0.1268 |
| 0.90-1.00 | 1132 | 0.9514 | 0.9848 | 0.0334 |
