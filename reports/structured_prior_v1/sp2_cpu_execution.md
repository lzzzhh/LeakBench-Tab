# Gate SP2 — Structured Prior v1 CPU Execution

**Status: CPU COMPLETE (4/5 models). TabM PENDING (GPU). M08 analysis INCOMPLETE.**
**Date:** 2026-07-13
**Runtime:** ~1816s (~30 min) for 6000 cells.

## Result

| Model | Expected | Success | Failure | Integrity verified | Duplicates | Status |
|-------|----------|---------|---------|--------------------|-----------| -------|
| lr | 1500 | 1500 | 0 | 1500 | 0 | COMPLETE |
| rf | 1500 | 1500 | 0 | 1500 | 0 | COMPLETE |
| lightgbm | 1500 | 1500 | 0 | 1500 | 0 | COMPLETE |
| catboost | 1500 | 1500 | 0 | 1500 | 0 | COMPLETE |
| **tabm** | 1500 | 0 | 0 | 0 | 0 | **PENDING_GPU** |

- Total CPU cells: **6000 / 6000 SUCCESS**, all integrity-verified.
- 0 duplicates (unique `task_hash × model` = 6000).
- 0 non-finite AUCs, 0 constant-0.5 predictions.
- Coverage uniform: 500 cells per (mechanism ∈ {M04,M05,M08}) × model.
- Runner: `run_structured_prior_v1_bundle.py --allow-run`, read-only, per-fit
  hash verification of bundle / task / strict-view / full-view / mask.
- Metric: `paired_harm = full_auc − strict_auc`.

## Provisional M08 numbers (NOT for claims)

CPU-only, no CI, TabM missing — recorded only to show plausibility, must not be
cited as the M08 result:

| Model | mean paired_harm | median | positive-cell rate |
|-------|-----------------:|-------:|-------------------:|
| lr | +0.0039 | +0.0038 | 0.66 |
| rf | +0.0057 | +0.0071 | 0.63 |
| lightgbm | +0.0023 | +0.0043 | 0.57 |
| catboost | +0.0080 | +0.0077 | 0.61 |

These are far smaller than the excluded interim numbers (+0.03…+0.05), as
expected: the frozen mechanism uses an outcome-independent constant-0.5 prior on
real confirmatory panels, not the ad-hoc entity-mean leak on synthetic data.

## Gate status

- SP2 CPU: **PASS** (6000/6000).
- SP3 TabM (GPU): **PENDING** — 1500 cells required before any M08 statistic.
- M08 / CL2 / CL3 / CL4 / CL10: remain PENDING until the 7500-cell merge (SP4).
