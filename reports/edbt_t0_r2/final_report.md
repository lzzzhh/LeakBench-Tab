# T0 R2 — Repair Construct-Validity Audit: Final Report

**Status:** COMPLETE_POSTRUN_CORRECTIVE_AUDIT
**Date:** 2026-07-20
**Branch:** t0/repair-construct-validity-r2
**Evidence commit:** f623b92
**Parent head:** e52d445
**Protocol:** reports/edbt_t0_r2/protocol.md (frozen)

---

## 1. Baseline Continuity (T0-A0)

**VERDICT: PASS — ALL 5,500 KEYS NUMERICALLY IDENTICAL**

B1 multi-seed LR re-fitted strict/full baselines from frozen SP6 bundles.
Values are bitwise identical to frozen SP8: max abs diff = 0.0 for both
strict_auc and full_auc. 5,500/5,500 keys matched, 0 duplicates, 0 missing.

The B1 protocol stated "same strict/full references as frozen SP8" — the
mechanism was re-fit, not load, but the result is numerically identical.

---

## 2. Selection Reconstruction (T0-A1)

**VERDICT: PASS — ALL 709,500 ROWS (ALL BUDGETS)**

Full reconstruction across all B1 budgets (0.0, 1%, 5%, 10%, 20%) and all
three learners (LR, RF, LightGBM): 709,500 rows, **0 selection hash mismatches,
0 bundle SHA-256 mismatches**. Every bundle load verified `bundle_sha256`
against the manifest.

---

## 3. R2 Metric Vector (T0-A2/A3)

### 3.1 Directional Decomposition (LR, representative)

| Component | P3 (MI) | P2 mean | Δ(P3−P2) |
|-----------|---------|---------|-----------|
| Legacy SDR | +0.059 | +0.004 | **+0.043** [0.004,0.077] |
| Directional repair | +0.116 | +0.031 | **+0.085** [0.070,0.098] |
| Same-side residual | +0.016 | +0.101 | −0.085 |
| Overcorrection | +0.055 | +0.014 | **+0.041** [0.018,0.068] |

MI-guided removal reduces residual leakage (directional repair +0.085) more
than random, but also produces more overcorrection (+0.041). The legacy SDR
(+0.043) is the net difference.

### 3.2 Mask-Grounded Metrics

| Metric | P3 | P2 mean |
|--------|-----|---------|
| Leak recall | 60.1% | 19.8% |
| Deletion precision | 31.0% | 10.4% |
| Legit retention | 85.1% | 79.9% |

P3 achieves 3× the leak recall and 3× the deletion precision, with 5% better
legitimate retention. However, P3 still misses ~40% of leak columns and 69%
of removed columns are legitimate features.

---

## 4. Paired P3−P2 Analysis (T0-A4)

### 4.1 Overall

| Δ metric | LR | RF | LightGBM |
|----------|-----|-----|----------|
| Δlegacy_sdr | +0.043 | +0.055 | +0.056 |
| Δdirectional_repair | +0.085 | +0.096 | +0.101 |
| Δleak_recall | +0.403 | +0.403 | +0.403 |
| Δovercorrection | +0.041 | +0.042 | +0.045 |
| Δlegit_retention | +0.052 | +0.052 | +0.052 |

All mask-grounded metrics (leak_recall, precision, retention) are identical
across learners — they depend on data and selections, not the model.

### 4.2 Archetype-Level (LR, canonical modulo-5 mapping)

| Archetype | ΔSDR | CI | N tasks |
|-----------|------|-----|---------|
| drifting | +0.084 | [+0.065,+0.104] | 4 |
| interaction | +0.107 | [+0.093,+0.125] | 4 |
| linear | +0.074 | [+0.068,+0.080] | 4 |
| nonlinear | +0.070 | [+0.045,+0.086] | 4 |
| **sparse** | **−0.118** | **[−0.160,−0.093]** | 4 |

**Sparse is reliably negative** for all three learners:
- LR: −0.118 [−0.160,−0.093]
- RF: −0.068 [−0.083,−0.044]
- LightGBM: −0.054 [−0.083,−0.009]

**Note on earlier T0-R2 result:** The initial T0-R2 analysis reported a near-zero
sparse ΔSDR due to an incorrect contiguous-block archetype mapping (datasets
0-3 all labeled "linear" instead of the canonical modulo-5 rotation). Under the
canonical mapping (0→linear, 1→interaction, 2→nonlinear, 3→sparse, ...),
the sparse archetype is unequivocally negative.

### 4.3 Mechanism-Family (LR, canonical mapping)

| Family | ΔSDR | Mechanisms |
|--------|------|------------|
| simple | +0.094 | M01, M02, M06, M10 |
| boundary | +0.039 | M03, M07, M11 |
| structured | −0.004 | M04, M05, M08, M09 |

The structured family mean near zero mixes low-gap negatives (M04/M05/M08)
with the strong positive M09 (+0.149).

---

## 5. False-Repair Audit (FR1–FR6)

### 5.1 All-Key Prevalence (5,500 keys)

| Category | LR | RF | LightGBM |
|----------|----|----|----------|
| FR1 (SDR↑, recall not better) | 9.4% | 8.8% | 8.9% |
| FR2 (P3 SDR>0, zero leak removal) | 1.4% | 7.7% | 8.0% |
| FR3 (SDR↑, residual not better) | 2.0% | 2.4% | 1.5% |
| FR4 (overcorrection) | 29.2% | 29.7% | 31.4% |
| FR5 (SDR↑, retention worse) | 9.0% | 8.7% | 8.7% |
| FR6 (M09 partial, ΔSDR>0) | 7.7% | 7.7% | 7.7% |

### 5.2 Conditional Prevalence (among eligible)

| Category | LR cond. | RF cond. | LGBM cond. | Eligible |
|----------|---------|---------|-----------|----------|
| FR1 | 15.3% | 14.3% | 14.4% | ΔSDR > 0 |
| FR2 | 2.6% | 12.3% | 12.8% | P3 SDR > 0 |
| FR3 | 3.3% | 3.8% | 2.5% | ΔSDR > 0 |
| FR4 | 47.6% | 48.3% | 51.0% | ΔSDR > 0 |
| FR5 | 14.7% | 14.1% | 14.2% | ΔSDR > 0 |
| FR6 | 92.4% | 92.8% | 93.1% | M09 ∧ ΔSDR>0 |

FR4 conditional: among keys with positive ΔSDR, 48–51% exhibit overcorrection.

FR2 is non-zero for tree-based models: positive strict-distance reduction can
occur without removing any oracle-labeled leakage column, showing that score
proximity alone does not identify semantic repair.

---

## 6. M09 Semantic-Group Analysis

| Metric | P3 | P2 mean | Δ | CI |
|--------|-----|---------|---|-----|
| Full-group removed rate | 0.0% | 0.0% | 0.000 | [0,0] |
| Any-hit rate | 99.4% | 88.1% | +0.114 | [+0.100,+0.128] |
| Partial removal rate | 99.4% | 88.1% | +0.114 | [+0.100,+0.128] |

At 20% budget (k≈4 columns removed from ~20 total), removing all 8 M09
one-hot columns is structurally impossible (requires k ≥ 8). Full-group
repair is infeasible at this budget. MI improves any-hit localization
(Δ=+0.114) but does not achieve full semantic-group repair.

---

## 7. Answers to Research Questions

### RQ1: Baseline Continuity
PASS. All 5,500 keys numerically identical to SP8.

### RQ2: Selection Reconstruction
PASS. All 709,500 rows across all budgets reconstructed, 0 hash mismatches,
0 bundle SHA mismatches.

### RQ3: What Does ΔSDR Mean?
Positive ΔSDR (+0.043) = directional repair (+0.085) minus overcorrection (+0.041).

### RQ4: Semantic Corroboration
**C1 status: SCORE_RECOVERY_ONLY.** Overcorrection gate (Δovercorrection ≤ 0)
failed for all learners. Semantic evidence (Δleak_recall=+0.40, Δdirectional_repair=+0.085)
is reported as descriptive subclaims.

### RQ5: How Much Repair Is Spurious?
- FR4: 29–31% of all keys, 48–51% of keys with positive ΔSDR
- FR2: 1.4% (LR) – 8.0% (LGBM) all-key; tree-based learners achieve P3 SDR>0 without removing leaks
- FR6: 92–93% among eligible M09 keys (partial removal, full removal zero)

### RQ6: Sparse Archetype
**Sparse is reliably negative** under correct canonical mapping:
LR −0.118 [−0.160,−0.093]; RF −0.068 [−0.083,−0.044]; LGBM −0.054 [−0.083,−0.009].
The earlier T0-R2 +0.022 result was caused by an incorrect contiguous-block archetype mapping.

### RQ7: M09 Semantic-Group
Any-hit localization improved (Δ=+0.114), but full-group repair not corroborated
(both P3 and P2 full-group removal = 0.0 at 20% budget).

### RQ8: Learner Consistency
Consistent direction across all three learners. No reliable learner interaction
detected for legacy SDR contrast.

---

## 8. Claim Recommendations

| Claim | Status | Rationale |
|-------|--------|-----------|
| C1 | **SCORE_RECOVERY_ONLY** | Δovercorrection > 0 for all learners |
| C1 descriptive | Descriptive subclaims | Δleak_recall, Δdirectional_repair are positive but do not satisfy the joint semantic gate |
| C2 (learner interaction) | No reliable interaction detected | Legacy SDR CIs cross zero |
| Sparse archetype | Negative | Confirmed with canonical mapping |
| M09 | Any-hit only | Full-group repair not corroborated |

---

## 9. Summary

The governance evidence is numerically intact: baselines continuous, selections
reconstructable, all 709,500 rows verified. Under R2 directional metrics:

- The legacy SDR advantage (+0.043–0.056) is real but overstates repair quality
  by hiding overcorrection (+0.041–0.045).
- MI leak recall (60%) substantially exceeds random (20%), but 69% of columns
  removed by MI are legitimate features.
- 29–31% of all keys satisfy the FR4 overcorrection condition; among keys with
  positive ΔSDR, 48–51% show overcorrection.
- Sparse archetype is reliably negative (LR −0.118).
- M09 any-hit localization is improved, but full semantic-group repair is
  structurally infeasible at 20% budget.
