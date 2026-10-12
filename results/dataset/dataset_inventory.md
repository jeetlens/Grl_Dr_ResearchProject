# Dataset Inventory

| Split | File | Raw Rows | Usable Rows | Duplicates | Missing Values |
|---|---|---:|---:|---:|---:|
| Train | data/both_train.csv | 13727 | 13727 | 3 | 0 |
| Validation | data/both_validation.csv | 1488 | 1488 | 0 | 0 |
| Test | data/both_test.csv | 1488 | 1488 | 0 | 0 |

## Class Distribution

| Split | ADHD | Anxiety | Bipolar | Depression | PTSD | None | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2465 | 2422 | 2407 | 2450 | 2001 | 1982 | 13727 |
| Validation | 248 | 248 | 248 | 248 | 248 | 248 | 1488 |
| Test | 248 | 248 | 248 | 248 | 248 | 248 | 1488 |

## Proxy-Domain Distribution (Text Length)

| Split | Short (<=Median) | Long (>Median) | Total |
|---|---:|---:|---:|
| Train | 6866 | 6861 | 13727 |
| Validation | 738 | 750 | 1488 |
| Test | 722 | 766 | 1488 |

## Notes

- Proxy Domain: Text-length bins computed from training set median (135.0 words)
- Classes: 6 mental health categories (ADHD, Anxiety, Bipolar, Depression, PTSD, None)
- Training: 13,727 samples (with 3 duplicate combined texts)
- Validation: 1,488 samples (balanced, 248 per class)
- Test: 1,488 samples (balanced, 248 per class)
- Total: 16,703 unique samples
