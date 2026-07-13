# LeakBench-Tab CE-2R Final Status

**Status:** REFROZEN — NARROWED SCOPE
**Readiness:** 91/100
**Date:** 2026-07-13

## Corrected Core Models

| Model | Family | Status |
|---|---|---|
| Logistic Regression | Linear | COMPLETE |
| Random Forest | Bagging Tree | COMPLETE |
| CatBoost | Boosted Tree | COMPLETE |
| LightGBM | Boosted Tree | COMPLETE |
| TabM | Deep Tabular | **INCOMPLETE** (GPU unavailable after correction) |

## Key Claims

| Claim | Status |
|---|---|
| CL1 Simple contamination detectable | CONFIRMED |
| CL2 Structured contamination hard to localize | CONFIRMED |
| CL3 Category-driven correlation | PARTIALLY CONFIRMED |
| CL4 Model-family effect | PARTIALLY CONFIRMED |
| CL6 BiQ/AIT collapse | CONFIRMED |
| CL7 Fixed budgets fail | PARTIALLY CONFIRMED |
| CL9 Not all contamination exploitable | CONFIRMED |
| CL10 Cross-model consistency | PARTIALLY CONFIRMED |

## Fixed Negative Results

- CL5b-raw/derived/v2: REFUTED (operational metadata peeking removed)
- CL16a: REFUTED (zero-shot natural transfer)
- CL-P1/P3: REFUTED (provenance pilot negative)
- CL-P2: EXPLORATORY ONLY

## Natural Tasks

- Bank PRE: REAL_DATA, single-reviewer
- NYC 311: REAL_DATA, ranking ceiling
- Lending Club: INSTRUMENTATION_ONLY (synthetic)

## Quality

- Active peeking: 0
- Tests: 112 passed, 1 known limitation
- Release validator: 21/21

## No Further Experiments

This release is frozen. No new experiments, models, mechanisms, or claims.
