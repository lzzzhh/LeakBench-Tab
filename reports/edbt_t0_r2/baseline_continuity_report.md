# T0-R2.1 Baseline Continuity Report

**Status:** COMPLETE  
**Date:** 2026-07-20  
**Audit:** T0-R2.1 Evidence-Chain Repair

---

## 1. Method

Compared `artifacts/sp8/governance_clean.csv` (P0_keep rows) with 
`results/edbt_eab_revision/b1_multiseed_p2.csv` (P0_keep rows at budget 0.0).

Key columns:
- SP8: `dataset_index, mechanism, strength, seed` (renamed `seed` → `training_seed`)
- B1: `dataset_index, mechanism, strength, training_seed`

Merged on all four key columns (outer join).

## 2. Results

| Metric | Value |
|--------|-------|
| Expected keys | 5,500 |
| Matched keys | 5,500 |
| SP8-only keys | 0 |
| B1-only keys | 0 |
| Duplicate keys (SP8) | 0 |
| Duplicate keys (B1) | 0 |
| Exact match (strict+full) | 5,500 / 5,500 |
| Within 1e-8 | 5,500 / 5,500 |
| Within 1e-6 | 5,500 / 5,500 |
| Max strict_auc absolute difference | 0.000000000000 |
| Max full_auc absolute difference | 0.000000000000 |
| Max initial_gap absolute difference | 0.000000000000 |

## 3. Verdict

**PASS — ALL 5,500 KEYS NUMERICALLY IDENTICAL.**

B1 multi-seed LR re-fitted strict/full baselines from the frozen SP6 bundles
rather than loading them numerically from the frozen SP8 governance CSV.
However, the resulting values are bitwise identical (all differences exactly 0.0).

## 4. Disclosure

The B1 protocol stated "Same strict/full references as frozen SP8" (line 37 of
governance_revision_protocol.md). This phrasing is technically misleading — the
mechanism was re-fit, not load. However, the numerical result is identical, so
the downstream usage of these baselines in P3-P2 paired analysis is valid.

**Recommendation:** Revise the manuscript to state "baselines were re-fitted from
frozen bundles and numerically match the SP8 ledger (max abs diff 0.0)" rather
than "reused."

## 5. Output

Detailed per-key comparison: `results/edbt_t0_r2/b1_sp8_baseline_continuity.csv`
