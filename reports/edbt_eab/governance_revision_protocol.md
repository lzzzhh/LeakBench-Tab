# EDBT Governance Revision Protocol v1

**Status:** FROZEN_BEFORE_EXPERIMENT
**Date:** 2026-07-17
**Git commit:** ff667cd

## Research Questions

1. **GQ1:** Does the blind-MI governance advantage over random (P3−P2) persist
   when P2 random-seed variance is properly integrated?
2. **GQ2:** Is the structured-family null result due to low initial repair
   opportunity, detection failure, or genuine mechanism properties?
3. **GQ3:** Are the LR governance conclusions learner-conditional
   (RF vs LightGBM)?
4. **GQ4:** Is the overall effect driven by a single archetype?

## Workstream A: No new experiments (existing 55,000 cells)

### A1. Mechanism-level decomposition (20% budget)
Per mechanism: initial_gap, P2 SDR, P3 SDR, paired effect D_k,
task-cluster bootstrap 95% CI, P3-better probability, MI recall, retention.

### A2. Initial-gap stratification
Quartile analysis + binned regression. Exploratory, not confirmatory.

### A3. Archetype robustness
Five archetypes (linear/interaction/nonlinear/sparse/drifting).
Leave-one-archetype-out. Task-level bootstrap.

## Workstream B1: Multi-seed P2

### Design
- P2 governance seeds: 20 (`GOV_SEEDS = [2026071700 + i for i in range(20)]`)
- Same LR, 20 tasks, 11 mechanisms, 5 strengths, 5 training seeds
- Budgets: 1%, 5%, 10%, 20%
- Encoded-column matched cost
- Same strict/full references as frozen SP8
- P3 unchanged (no re-fit needed)

### Primary estimator
For each key k, average P2 SDR over governance seeds S:
  P2_bar_k = mean(P2_SDR_k_s for s in S)
  D_k = P3_SDR_k - P2_bar_k

Primary inference: dataset/task cluster bootstrap.

## Workstream B2: Cross-learner

### Models
- RF (sklearn.RandomForestClassifier, default params + seed control)
- LightGBM (lightgbm.LGBMClassifier, default params + seed control)

### Scope (confirmatory)
- 20 tasks × 11 mechanisms × 5 strengths × 5 training seeds
- Budget: 20% (primary)
- P3 + P2 multi-seed (20 gov seeds)
- Encoded-column matched cost
- Canonical strict/full references from SP5 frozen cells

### Expected cells
20×11×5×5 = 5,500 keys per model per policy
× (1 P0 + 1 P3 + 20 P2 seeds) = 22 gov fits per key
× 2 models = 242,000 fits
Each fit: strict+full retrain on post-removal features
Total: ~550,000 cell records

## Forbidden
- No hand-edited claims JSON
- No point±0.01 CIs
- No tolerance relaxation in validators
- No changing protocol after seeing results
- No cherry-picking mechanisms/budgets

## Success criteria
- A1/A2/A3 complete
- B1 multi-seed P2 complete
- B2 RF + LightGBM complete or honestly blocked
- Canonical table free of duplicates/missing
- Claims builder-derived
- Manifest fully bound
- All tests pass
