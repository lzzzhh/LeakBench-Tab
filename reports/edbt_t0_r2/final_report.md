# T0 R2 — Repair Construct Validity Audit: Final Report

**Status:** COMPLETE
**Date:** 2026-07-20
**Branch:** t0/repair-construct-validity-r2
**Protocol:** reports/edbt_t0_r2/protocol.md (frozen)

---

## 1. Baseline Continuity Verdict (T0-A0)

**VERDICT: PASS — ALL 5,500 KEYS NUMERICALLY IDENTICAL**

B1 multi-seed LR re-fitted strict/full baselines from the frozen SP6 bundles rather than
loading them from the frozen SP8 governance CSV. However, the resulting values are
**bitwise identical** (all differences exactly 0.0) across all 5,500 keys.

This means:
- The B1 "reused baseline" phrase is technically misleading about the *mechanism*
  (re-fit rather than load) but numerically correct about the *result*.
- The same LR hyperparameters (max_iter=2000 vs 1000 in SP8) yielded identical AUROCs
  on these datasets, indicating convergence is not sensitive to max_iter.
- No key coverage issues: 5,500/5,500 matched, 0 duplicates, 0 missing.

**Recommendation:** Revise the manuscript to state "baselines were re-fitted from frozen
bundles and numerically match the SP8 ledger (max abs diff 0.0)" rather than "reused."

---

## 2. Selection Reconstruction Verdict (T0-A1)

**VERDICT: PASS — ALL 346,500 ROWS RECONSTRUCTED CORRECTLY**

All selection hashes at 20% budget were deterministically reconstructed:
- P3 (blind MI): mutual_info_classif with random_state=42, argsort descending.
- P2 (random): seed = (gov_seed × 100 + ds × 7 + ts × 13) % (2³¹−1).
- 346,500 rows checked (5,500 keys × 21 fits × 3 models).
- **0 mismatches**.

Cross-model hash consistency: confirmed identical P3 hashes and P2 hashes across
LR/RF/LightGBM (as expected — selections depend only on data and seeds, not learner).

---

## 3. R2 Metric Vector Findings (T0-A2/A3)

### 3.1 Legacy SDR Decomposition

The legacy SDR formula `|full−strict| − |governed−strict|` hides two distinct effects:

| Effect | P3 (MI) | P2 mean (Random) | Δ(P3−P2) |
|--------|---------|------------------|-----------|
| Legacy SDR | +0.059 | +0.004 | **+0.043** |
| Directional repair | +0.116 | +0.031 | **+0.085** |
| Same-side residual | +0.016 | +0.101 | **−0.085** |
| Overcorrection | +0.055 | +0.014 | **+0.041** |

(Values shown for LR; RF and LightGBM follow the same pattern.)

**Key insight:** MI-guided removal does 2.7× more directional repair (reducing residual
leakage) than random removal (+0.085). This is partially offset by 2.9× more
overcorrection (governed score overshooting the strict reference, +0.041). The net
legacy SDR gain (+0.043) is the difference between these two larger opposing effects.

### 3.2 Mask-Grounded Metrics

| Metric | P3 (MI) | P2 mean (Random) |
|--------|---------|------------------|
| Leak recall | 60.1% | 19.8% |
| Deletion precision | 31.0% | 10.4% |
| Legit retention | 85.1% | 79.9% |
| Residual leak columns | 39.9% of leak | 80.2% of leak |

P3 achieves **3× the leak recall** and **3× the deletion precision** of random removal
while retaining **5% more legitimate columns**. However:
- P3 still misses ~40% of leak columns.
- P3 deletes mostly legitimate features (69% of removed columns are legitimate).
- The absolute deletion precision (31%) is low — most removed columns are not leaks.

### 3.3 Overcorrection Prevalence

| Learner | Overcorrected rows | Mean overcorrection |
|---------|--------------------|---------------------|
| LR | 16.9% | +0.023 |
| RF | 20.3% | +0.043 |
| LightGBM | 21.2% | +0.046 |

Overcorrection is **systematic across all three learners**: tree-based models show
more overcorrection than linear models.

---

## 4. Paired P3−P2 Analysis (T0-A4)

### 4.1 Overall

| Metric (Δ = P3−P2) | LR | RF | LightGBM |
|---------------------|-----|-----|----------|
| Δlegacy_sdr | +0.043 [0.004,0.077] | +0.055 [0.024,0.082] | +0.056 [0.027,0.081] |
| Δdirectional_repair | +0.085 [0.070,0.098] | +0.096 [0.086,0.107] | +0.101 [0.090,0.111] |
| Δleak_recall | +0.403 [0.375,0.430] | +0.403 [0.375,0.430] | +0.403 [0.375,0.430] |
| Δovercorrection | +0.041 [0.018,0.068] | +0.042 [0.021,0.066] | +0.045 [0.023,0.070] |
| Δlegit_retention | +0.052 [0.048,0.055] | +0.052 [0.048,0.055] | +0.052 [0.048,0.055] |

All task-reweighting intervals for directional repair, leak recall, and legit retention
are strictly positive. Overcorrection is also strictly positive for all learners.

Mask-grounded metrics (leak_recall, deletion_precision, legit_retention) are identical
across learners by construction — they depend only on data and selections, not on the
model.

### 4.2 Mechanism-Level

**Mechanisms with positive ΔSDR AND positive Δleak_recall:**
- M01 (+0.069, +0.694 leak recall)
- M02 (+0.087, +0.766)
- M06 (+0.110, +0.679)
- M07 (+0.018, +0.650) — CI crosses zero
- M09 (+0.149, +0.261)
- M10 (+0.110, +0.786)
- M11 (+0.155, +0.554)

**Mechanisms with negative ΔSDR:**
- M03 (−0.057, −0.111 leak recall)
- M04 (−0.057, −0.093)
- M05 (−0.056, −0.093)
- M08 (−0.052, +0.338 leak recall) — leak recall positive but SDR negative

M08 is a unique case: MI correctly identifies leak columns (Δleak_recall=+0.34) but
ΔSDR is negative. This suggests that M08's leak columns overlap with features that are
important for model performance — removing them hurts more than the leakage helps,
even though they are correctly identified as leaks.

### 4.3 Archetype-Level

| Archetype | ΔSDR (LR) | CI |
|-----------|-----------|-----|
| drifting | +0.039 | [−0.058,+0.092] |
| interaction | +0.081 | [+0.063,+0.101] |
| linear | +0.022 | [−0.113,+0.094] |
| nonlinear | +0.053 | [−0.046,+0.123] |
| sparse | +0.022 | [−0.049,+0.072] |

**Critical finding:** The sparse archetype is NOT reliably negative. The previously
reported −0.118 (SP8 single-seed P2) was an artifact of relying on a single frozen
P2 governance seed. Under properly integrated multi-seed P2 averaging, the sparse
archetype CI crosses zero.

---

## 5. False-Repair Audit

### 5.1 Summary Counts (5,500 keys per model)

| Category | LR | RF | LightGBM | Description |
|----------|-----|-----|----------|-------------|
| FR1: SDR↑ but recall not better | 516 (9.4%) | 485 (8.8%) | 488 (8.9%) | Positive ΔSDR with no leak recall gain |
| FR3: SDR↑ but residual not better | 111 (2.0%) | 130 (2.4%) | 84 (1.5%) | Same residual with positive SDR |
| **FR4: SDR↑ with overcorrection** | **1,606 (29.2%)** | **1,634 (29.7%)** | **1,726 (31.4%)** | **Overshoots strict reference** |
| FR5: SDR↑ but retention worse | 496 (9.0%) | 476 (8.7%) | 481 (8.7%) | Trade-off: SDR gain vs legit data loss |

FR4 is the most prevalent false-repair category: nearly 30% of all keys show positive
legacy SDR that is partially attributable to overcorrection.

### 5.2 Mechanism-Level FR4

FR4 is concentrated in mechanisms with large initial gaps:
- M01, M02: high initial gap → MI removes strong features → overcorrection
- M06, M10, M11: similar pattern
- M03, M04, M05, M08: low initial gap → SDR is negative, so FR4 doesn't apply

---

## 6. Answers to Research Questions

### RQ1: Baseline Continuity
**YES.** All 5,500 B1 baselines numerically identical to SP8 (max abs diff = 0.0).
The mechanism was re-fit, not load, but the result is bitwise identical.

### RQ2: Selection Reconstruction
**YES.** All 346,500 selection hashes reconstructed with 0 mismatches.

### RQ3: What Does Positive SDR Actually Mean?
Positive legacy ΔSDR (+0.043) is the net of:
- **Strong positive**: directional repair (+0.085) — MI genuinely reduces residual leakage.
- **Partial offset**: overcorrection (+0.041) — MI also drives governed score past strict.
Legacy SDR understates the repair magnitude while hiding the overcorrection cost.

### RQ4: Semantic Corroboration
**PARTIALLY CORROBORATED.** Leak recall (+0.40), directional repair (+0.085), and
legit retention (+0.052) all point in the same direction. But overcorrection (+0.041)
is a countervailing force. The evidence tier is
SEMANTICALLY_CORROBORATED_WITH_OVERCORRECTION_CAVEAT.

### RQ5: How Much Repair Is Spurious?
- FR4 (overcorrection): 29–31% of keys with positive SDR
- FR1 (no recall gain): 9%
- FR5 (retention trade-off): 9%
- FR3 (no residual improvement): 1.5–2.4%
Most "repair" is not spurious, but a substantial minority mixes genuine repair with
overcorrection.

### RQ6: Sparse Archetype Negativity
The sparse archetype is NOT reliably negative under multi-seed P2 (ΔSDR=+0.022,
CI[−0.049,+0.072]). The previously reported −0.118 was a single-seed artifact.

### RQ7: M09 Semantic-Group Robustness
**CONFIRMED.** M09 remains a strong positive outlier under R2 metrics (Δlegacy_sdr=+0.149,
Δleak_recall=+0.261). Confirms prior revision finding.

### RQ8: Learner Consistency Under R2 Metrics
**CONSISTENT.** All three learners (LR, RF, LightGBM) show the same directional pattern:
positive directional repair, positive leak recall, positive overcorrection, positive
legit retention. Tree-based models show slightly larger overcorrection (20–21% vs 17%).

---

## 7. Paper Claim Recommendations

### C1 (MULTI-LEARNER GOVERNANCE)
**Revise to:** SEMANTICALLY_CORROBORATED_WITH_OVERCORRECTION_CAVEAT
- Positive directional repair and leak recall are genuine.
- Overcorrection is a systematic side effect (~30% of keys).
- The legacy SDR understates the repair while hiding the cost.

### C2 (NO LEARNER INTERACTION)
**Keep as SUPPORTED.** Consistent across learners.

### C3 (STRUCTURED HETEROGENEITY)
**Keep as NARROWED.** Mechanism-level patterns confirmed.

### C4 (ARCHETYPE SENSITIVITY — SPARSE)
**REVISE.** Sparse is not reliably negative. The single-seed P2 finding does not
replicate. LOAO-sparse previously appeared positive (+0.084) because sparse was thought
to be strongly negative — it was near zero all along.

### C5 (NATURAL GOVERNANCE)
Unchanged by this audit.

### C6 (SEMANTIC GROUP BUDGET)
Unchanged by this audit.

---

## 8. Summary Verdict

The existing governance evidence (B1/B2) is **numerically correct** (baselines continuous,
selections reconstructable) and the **legacy SDR direction is reproducible** under R2
metrics. However, the legacy SDR metric **conflates directional repair with
overcorrection**, leading to a systematically incomplete picture.

MI-guided removal genuinely reduces residual leakage (directional repair +0.085) and
correctly identifies leak columns (leak recall 3× random), but it also consistently
produces more overcorrection (+0.041) than random removal. Approximately 30% of keys
with positive legacy SDR show overcorrection.

The paper's core claim (C1) should be revised from SUPPORTED to
SEMANTICALLY_CORROBORATED_WITH_OVERCORRECTION_CAVEAT, and the sparse archetype
assessment (C4) needs correction.

**The CDXR experiment on the separate branch is irrelevant to these findings — this
audit is complete and self-contained within frozen SP8/B1/B2 evidence.**
