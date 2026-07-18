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

## Post-run protocol deviation disclosure (2026-07-18)

**Status:** DISCLOSED_AFTER_RESULTS; this section is not represented as a
prospectively frozen amendment.

The B2 runner did not numerically reuse strict/full AUROCs from the SP5
canonical table as stated on line 58. It re-fitted strict and full RF and
LightGBM baselines from the same immutable SP6 bundle views before fitting each
governed view. The re-fit is internally consistent with the SDR definition used
within each B2 key, but it is a deviation from the frozen wording. Therefore:

- B2 is described as a pre-specified cross-learner extension with a disclosed
  baseline-re-fit deviation, not as a bitwise reuse of SP5 model outputs.
- Cross-learner claims are restricted to within-run P3-minus-mean(P2) contrasts
  at the 20% encoded-column budget.
- The manuscript must not imply numerical identity between B2 strict/full
  baselines and SP5 canonical strict/full cells.

The original B1/B2 outputs also omitted valid selection-mask hashes. Because
the selections are deterministic functions of the immutable bundle, recorded
budget, MI random state, and governance seed, hashes are reconstructed without
model fitting by `scripts/backfill_governance_selection_hashes.py`. The
backfill metadata records pre/post hashes, the hash scheme, bundle-manifest
binding, and cross-model mask equality checks.

For asset governance, the canonical revision dataset is the manifest-bound
partition set `{b1_multiseed_p2.csv, b2_rf.csv, b2_lgbm.csv}` rather than a
fourth concatenated copy of the same 709,500 rows.
