# SP5-F — Complete Five-Model Eleven-Mechanism Evidence Pool

**Status: COMPLETE. 27500 / 27500 formal cells.**
**Date:** 2026-07-14

## Result

| Check | Value |
|-------|-------|
| Total formal cells | 27500 / 27500 |
| Models | lr, rf, lightgbm, catboost, tabm |
| Mechanisms | M01–M11 |
| Coverage | 500 per mechanism × model (uniform, no gaps) |
| Duplicate primary keys | 0 |
| Missing keys | 0 |
| Replacement conflicts | 0 |
| Code-drift checkpoint rows | 0 |
| Smoke rows | 0 |

Primary key: `[dataset_index, mechanism, strength, model, seed]`.
Canonical inputs: `artifacts/sp5/claim_ledger_inputs_v2.{csv,parquet}`.

## Source composition (replacement architecture)

| Source | Cells | Role |
|--------|------:|------|
| core_cpu | 14000 | base: 7 non-amended mechs × 4 CPU models |
| base7_tabm | 3500 | base: 7 non-amended mechs × TabM (reproducible, code_hash 289e590d) |
| sp4_frozen | 7500 | exact replacement: M04/M05/M08 × 5 models |
| m10_amendment | 2500 | exact replacement: M10 × 5 models |

Replacement precedence: M10 amendment > SP4 (M04/M05/M08) > base. No cell matched
more than one replacement source (verified: base excludes M04/M05/M08/M10).

## TabM mechanism source registry (5-model completeness)

| Mechanism | TabM source | Cells |
|-----------|-------------|------:|
| M01,M02,M03,M06,M07,M09,M11 | BASE7_V2 | 500 each |
| M04,M05,M08 | SP4_FROZEN | 500 each |
| M10 | M10_AMENDMENT | 500 each |

Each mechanism has exactly one formal TabM source; 11/11 covered; 0 multiple-source.

## Metric consistency

All sources normalized to `paired_harm = full_auc − strict_auc`, where the strict
baseline is the mask-derived strict view. Verified earlier that core `clean_auc`
== SP4 `strict_auc` (|diff| = 0) for the shared mechanisms, so the metric is
identical across the whole pool.

## Excluded (not in pool)

- Old TabM confirmatory checkpoint (5228 cells, code_hash 99b17868) — CODE_DRIFT_EXCLUDED.
- Interim entity-mean M08 — INTERIM_EXCLUDED.
- Legacy ce2r_neural / pre-corrected_v2 aggregates — SUPERSEDED.
- All smoke cells — in `_excluded_smoke/`.
