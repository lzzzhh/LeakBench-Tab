# SP7 Hypothesis Lock

**Status: FROZEN_BEFORE_ANY_NEW_EXPERIMENT**
**Date: 2026-07-14**
**SP5 ledger: ccb2549f490e95cb**
**SP6 ledger: aef477b99ea62cd0**

## Six Candidate Hypotheses

| ID | Hypothesis | Priority |
|----|-----------|----------|
| H1 | Temporal Association Instability | primary |
| H2 | Weak-Signal Dilution / Feature Competition | primary |
| H3 | Finite-Sample Variance | primary |
| H4 | Optimization / Early-Stopping Path | model-process |
| H5 | Retrieval / Neighborhood Distortion | model-process |
| H6 | Regularization Interaction | model-process |

All priors: the M04/M05 negative-harm signature is robust (observed across
TabM, ModernNCA, TabR). No causal mechanism is claimed.

## Analysis Parameters
- Bootstrap seed: 20260715, unit: dataset_index, 10000 reps
- Multiple testing: Holm (confirmatory), BH-FDR (exploratory)
- Sentinel selection: 3 size strata × 2 dimensionality strata = 6 datasets (NO harm data)

## Forbidden Operations
- No test labels in model fitting
- No cross-split permutation
- No cross-view candidate/retrieval sharing
- No intervention definition changed after seeing results
