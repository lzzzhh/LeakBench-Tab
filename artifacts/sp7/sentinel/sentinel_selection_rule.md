# SP7-D Sentinel Selection Rule (preregistered)

**Deterministic, NO harm data.**
**Date: 2026-07-14**

## Rule
1. Compute per-dataset training row count (`n_samples * 0.6`) and feature count (`n_original`).
2. Stratify: `size_stratum` = `pd.qcut(n_train, 3)` → small/medium/large.
3. Stratify: `dim_stratum` = `pd.qcut(n_features, 2)` → low/high.
4. Within each of 3×2=6 strata, select the dataset with the smallest `dataset_index` (deterministic tiebreaker).

## Selected sentinel panels

| ds | size | dim | n_train | n_features |
|----|------|-----|---------|-----------|
| 0  | small | low  | 600  | 12 |
| 2  | small | high | 720  | 20 |
| 4  | medium| low  | 840  | 12 |
| 7  | medium| high | 792  | 24 |
| 9  | large | low  | 924  | 16 |
| 14 | large | high | 1008 | 20 |

No `paired_harm`, model performance, or mechanism detectability was used.
