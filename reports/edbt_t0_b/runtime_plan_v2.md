# T0-B Runtime Plan V2

## Execution Summary

| Stage | Fits | Description |
|-------|------|-------------|
| B1 LR | 803,000 | Full-registry, both contracts, all budgets |
| B2 RF+LGBM | 286,000 | Semantic-group 20%, two learners |
| **Total** | **1,089,000** | |

Ranking model fits: 22,000 (P5 LR × 5,500 + P6 3-fold RF × 5,500).

## Per-Fit Estimates (based on historical runner performance)

| Model | ~time/fit | Notes |
|-------|-----------|-------|
| LR | ~0.3s | StandardScaler + LogisticRegression(max_iter=2000) |
| RF | ~2.0s | 250 estimators, min_samples_leaf=2 |
| LightGBM | ~1.5s | 250 estimators, CPU, early_stopping=30 |

## Wall-Clock Estimates

| Stage | Fits | Est. Time |
|-------|------|-----------|
| B1 LR | 803,000 | ~67 hours (single-thread) / ~3.5 hours (20 workers) |
| B2 RF | 143,000 | ~80 hours (single) / ~4 hours (20 workers) |
| B2 LGBM | 143,000 | ~60 hours (single) / ~3 hours (20 workers) |
| Ranking | 22,000 | ~2 hours |

**Recommended:** 20 worker processes, ~12-15 hours wall-clock total.

## Resume Strategy

- CSV append mode with run_id deduplication
- Checkpoint every 500 keys
- Deterministic run_id formula: SHA256(key || policy || contract || budget || seed)[:20]

## Memory

- LR: ~100MB per worker
- RF: ~500MB per worker (250 trees × dataset size)
- LightGBM: ~300MB per worker
- Bundle loading: ~50MB per bundle, cache up to 10 bundles in memory

**Recommended:** 32GB RAM minimum for 20 workers.

## Risk: P6 Permutation Cost

P6 cross-fitted RF permutation importance is the most expensive ranking operation:
5,500 keys × 3 folds × 5 repeats = 82,500 RF fits for ranking alone.
Estimated ~5 hours additional for ranking RF fits.

P6 ranking RF uses 100 estimators (not 250) to control cost.
This is a ranking-only factory, not the downstream canonical RF.
