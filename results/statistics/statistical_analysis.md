# Statistical Analysis: Baseline vs GRL (Multi-Seed Results)

## Summary

This analysis reports mean and standard deviation across three random seeds:

## Accuracy

| Metric | Baseline | GRL |
|---|---:|---:|
| Mean | 84.8100 | 84.5900 |
| Std Dev | 0.7100 | 0.8400 |

## Macro F1

| Metric | Baseline | GRL |
|---|---:|---:|
| Mean | 84.8300 | 84.6100 |
| Std Dev | 0.7100 | 0.8100 |

## Weighted F1

| Metric | Baseline | GRL |
|---|---:|---:|
| Mean | 84.8300 | 84.6100 |
| Std Dev | 0.7100 | 0.8100 |

## ROC-AUC

| Metric | Baseline | GRL |
|---|---:|---:|
| Mean | 97.8100 | 97.7900 |
| Std Dev | 0.0300 | 0.1600 |

## Key Observations

- **Accuracy Difference** (GRL - Baseline): -0.2200
- **Macro F1 Difference** (GRL - Baseline): -0.2200
- **Weighted F1 Difference** (GRL - Baseline): -0.2200
- **ROC-AUC Difference** (GRL - Baseline): -0.0200

## Limitations

- **Small sample size**: Only 3 independent seeds; results are descriptive
- **No formal significance test**: Reported results are raw means ± std from saved experiments
- **Proxy-domain evaluation**: Results use proxy-domain labels (text-length bins), not genuine source domains
- **Overlapping confidence bands**: Suggest no clear winner between models

## Conclusion

Based on three independent training seeds, the models show similar performance:

- The baseline model shows slightly higher Macro F1 performance
- Standard deviations overlap, indicating high uncertainty in estimates
- With only 3 seeds, these results should be treated as descriptive, not conclusive
- Larger-scale studies with more seeds would provide stronger evidence
