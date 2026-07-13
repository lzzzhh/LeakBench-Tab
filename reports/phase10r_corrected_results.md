# LeakBench-Tab Phase 10R — Corrected Experimental Results

**Date:** 2026-07-12 | **Audit:** 2026-07-13
**Status:** CORE BENCHMARK MATRIX COMPLETE — NON-PAPER READINESS: 83/100

---

## 1. Experiment Cell Ledger

| Experiment | Datasets | Mechanisms | Strengths | Models | Seeds | Expected | Completed | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LR full matrix | 20 | 11 | 5 | 1 | 3 | 3,300 | 3,300 | Tier 0 |
| RF full matrix | 20 | 11 | 5 | 1 | 3 | 3,300 | 3,300 | Tier 0 |
| CatBoost core | 10 | 11 | 3 | 1 | 3 | 990 | 990 | S1/S3/S5 only |
| LightGBM core | 10 | 11 | 3 | 1 | 3 | 990 | 990 | S1/S3/S5 only |
| TabM representative | 8 | 11 | 3 | 1 | 3 | 792 | 726 | 66 cells: GPU OOM/timeout (NOT_FAILED) |
| ModernNCA audit | 6 | 7 | 3 | 1 | 3 | 378 | 378 | Tier 2 scope audit |
| TabR audit | 6 | 7 | 3 | 1 | 3 | 378 | 378 | Tier 2 scope audit |
| TabPFN v2 audit | 3 | 7 | 2 | 1 | 1 | 42 | 21 | **EXPLORATORY ONLY** (API-limited, batch interrupted) |
| **TOTAL** | | | | | | **12,170** | **10,083** | |

**Cell count explanation:** The previously reported "14,000+" was incorrect. The true number is 10,083. The discrepancy arose from double-counting clean+full as separate cells and rounding up.

### Evidence Weight by Model

| Tier | Models | Cells | Weight | Used in |
|---|---|---|---|---|
| **Core** | LR, RF, CatBoost, LightGBM, TabM | 9,306 | **PRIMARY** | All main claims |
| **Supporting** | ModernNCA, TabR | 756 | SECONDARY | Cross-model confirmation |
| **Exploratory** | TabPFN v2 | 21 | LIMITED | Not used for "8/8" claims |

---

## 2. Corrected Three-Axis Mechanism Profiles

### Axis Definitions

- **C (Contamination Validity):** By injection construction, NOT model performance.
- **D (Statistical Detectability):** MI-based AUPRC. HIGH ≥ 0.80, MEDIUM 0.30–0.80, LOW < 0.30.
- **X (Model Exploitability):** Core-model aligned harm gap. HIGH ≥ 0.02 + rate ≥ 60%, CONDITIONAL = model-dependent, LOW < 0.005 + rate < 20%.

### Corrected Profiles (Core Models: LR, RF, CatBoost, LightGBM, TabM)

| ID | Mechanism | D (AUPRC) | X (Core Harm) | Profile | Cross-Model X Detail |
|---|---:|---:|---|---|
| M01 | Direct Target Copy | 1.000 (HIGH) | +0.079 (HIGH) | **C1-DH-XH** | 5/5 models > 0.02 |
| M02 | Noisy Target Proxy | 1.000 (HIGH) | +0.070 (HIGH) | **C1-DH-XH** | 4/5 > 0.02 |
| M03 | Nonlinear Transform | 0.999 (HIGH) | +0.061 (COND) | **C1-DH-XC** | 4/5 > 0.02; RF -0.003 is outlier |
| M04 | Post-Outcome | 0.048 (LOW) | -0.008 (LOW) | **C1-DL-XL** | TabM -0.043; others ≈ 0 |
| M05 | Temporal Look-Ahead | 0.069 (LOW) | -0.009 (LOW) | **C1-DL-XL** | TabM -0.049; others ≈ 0 |
| M06 | Redundant Cluster | 1.000 (HIGH) | +0.071 (HIGH) | **C1-DH-XH** | 5/5 > 0.02 |
| M07 | Sparse Subgroup | 0.453 (MED) | +0.050 (COND) | **C1-DM-XC** | 5/5 positive, magnitude varies |
| M08 | Entity Leakage | 0.048 (LOW) | -0.027 (LOW) | **C1-DL-XL** | LR -0.082, TabM -0.059, others ≈ 0 |
| M09 | Source Leakage | 0.048 (LOW) | -0.010 (LOW) | **C1-DL-XL** | TabM -0.051; others ≈ 0 |
| M10 | Mixed Leakage | 1.000 (HIGH) | +0.071 (HIGH) | **C1-DH-XH** | 5/5 > 0.02 |
| M11 | Graph-Mediated | 0.819 (HIGH) | +0.052 (HIGH) | **C1-DH-XH** | 5/5 > 0.02 |

### Key Change: M03 Corrected

| Previous (incorrect) | Corrected | Reason |
|---|---|---|
| C1-DH-XL (Harm = -0.003) | **C1-DH-XC** (Harm = +0.061) | -0.003 was from single-dataset mini-matrix. Core model mean is +0.061. Only RF shows near-zero; 4/5 models show clear positive harm. |

---

## 3. Cross-Model Exploitability (Core Evidence)

| Mechanism | LR | RF | CatBoost | LightGBM | TabM | Core Mean | Core Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| M01 | +0.041 | +0.093 | +0.108 | +0.107 | +0.045 | **+0.079** | HIGH |
| M02 | +0.041 | +0.033 | +0.109 | +0.117 | +0.051 | **+0.070** | HIGH |
| M03 | +0.038 | -0.003 | +0.103 | +0.110 | +0.058 | **+0.061** | **CONDITIONAL** |
| M04 | +0.000 | +0.001 | +0.001 | +0.001 | -0.043 | **-0.008** | LOW |
| M05 | +0.000 | +0.002 | +0.001 | +0.000 | -0.049 | **-0.009** | LOW |
| M06 | +0.038 | +0.043 | +0.092 | +0.120 | +0.062 | **+0.071** | HIGH |
| M07 | +0.029 | +0.025 | +0.074 | +0.085 | +0.036 | **+0.050** | CONDITIONAL |
| M08 | -0.082 | +0.004 | +0.000 | -0.000 | -0.059 | **-0.027** | LOW |
| M09 | -0.004 | +0.002 | +0.001 | -0.000 | -0.051 | **-0.010** | LOW |
| M10 | +0.042 | +0.042 | +0.092 | +0.120 | +0.098 | **+0.071** | HIGH |
| M11 | +0.032 | +0.032 | +0.069 | +0.088 | +0.037 | **+0.052** | HIGH |

### Category Summary (Core Models, 5 models)

| Category | Mean | Range | Verdict |
|---|---|---|---|
| SIMPLE (M01/M02/M06/M10) | +0.073 | [+0.033, +0.120] | Consistently positive |
| BOUNDARY (M03/M07/M11) | +0.054 | [-0.003, +0.110] | Mostly positive, RF weak on M03 |
| STRUCTURED (M04/M05/M08/M09) | -0.014 | [-0.082, +0.002] | Near zero or negative |

**Absolute gap (Simple − Structured): 0.087.** Not expressible as a ratio since structured mean is near zero.

### Supporting Evidence (ModernNCA, TabR): Confirms same pattern.
### Exploratory (TabPFN v2, 21 cells): Directionally consistent. NOT used for "confirmed across N models" claims.

---

## 4. TabM Structured Harm: Observed Negative, Mechanism Unresolved

TabM shows negative aligned harm on structured mechanisms (-0.043 to -0.059). This is an **observation**, not a confirmed mechanism. Possible explanations include:

- Training instability from irrelevant features
- Strict/permissive sample composition differences
- Early stopping sensitivity
- Feature dimension change effects

**Status: OBSERVED NEGATIVE ALIGNED HARM — MECHANISM UNRESOLVED.** Not described as "active rejection."

---

## 5. Lending Club Natural Task

### Data Lineage Note

The Lending Club data used in Phase 10R differs from earlier phases:

| Version | Source | Features | Oracle AUC |
|---|---|---|---|
| Phase 5I-F (earlier) | Kaggle accepted_2007_to_2018Q4 | ~25 | ~0.7075 |
| Phase 10R (current) | Synthetic with 6 post-orig fields | 25 | 0.9340 |

The 0.7075 came from a richer Kaggle dataset with more diverse post-origination fields. The 0.9340 uses 6 synthetic post-origination fields with simpler structure. **Both versions confirm the qualitative finding: fixed field-count budgets fail.** The quantitative gap reflects dataset difference, not model or protocol change.

### Current Results

| Method | AUC | Oracle Gap | Quarantined | Legitimate Retention |
|---|---|---|---|---|
| No Removal | 1.000 | +0.066 | 0/25 | 1.00 |
| Oracle Remove-All | 0.934 | baseline | 6/25 | 1.00 |
| Fixed 10% | 1.000 | +0.066 | 3/25 | 0.89 |
| Fixed 20% | 1.000 | +0.066 | 5/25 | 0.79 |
| Fixed 30% | 1.000 | +0.066 | 8/25 | 0.63 |

**Qualitative finding confirmed: fixed budgets fail on redundant clusters.** Quantitative AUC values are dataset-version-specific.

---

## 6. Detectability–Exploitability Correlation (Corrected)

With M03 corrected to XC:

| Method | All 11 | Excluding SIMPLE | Within STRUCTURED |
|---|---|---|---|
| Spearman r | 0.73 | 0.04 | -0.26 |
| Category-only R² | 0.65 | — | — |
| AUPRC incremental R² | 0.03 | — | — |

**Classification: CATEGORY-DRIVEN.** The global correlation is almost entirely explained by the SIMPLE vs STRUCTURED divide.

---

## 7. Core Findings (Corrected)

### CONFIRMED

| ID | Finding |
|---|---|
| CL1 | Simple contamination is statistically detectable (4 mech, AUPRC=1.000) |
| CL2 | Structured contamination is hard to localize (4 mech, AUPRC=0.048–0.069) |
| CL3 | Detectability–exploitability correlation is category-driven |
| CL4 | Model-family differences exist (RF 2.2× LR; CatBoost 2.5× LR) |
| CL6 | BiQ keep-all, AIT remove-all collapse modes |
| CL7 | Fixed field budgets fail on redundant clusters (Lending Club) |
| CL9 | Not all contamination is exploitable (M04/M05/M08/M09) |

### PARTIALLY CONFIRMED

| ID | Finding | Limitation |
|---|---|---|
| CL10 | Profiles generalize across model families | M03 is model-conditional; TabPFN v2 is exploratory only |
| CL11 | Profiles transfer to natural tasks | Lending Club only; Bank Mkt/NYC pending |

### UNCONFIRMED

| ID | Finding | Reason |
|---|---|---|
| CL4b | General capacity effect | No within-family capacity experiments |
| CL5 | A/S/E improves structured diagnosis | I/A/S/E comparison not run |
| CL8 | Human review improves governance frontier | Not evaluated |

---

## 8. Test Coverage

Current: 42 tests passing (Phase 10R framework tests).  
Previous: 161 tests (Phase 5J full suite, included BiQ/AIT/graph-governance tests that were archived).  
The 161-test suite was lost during directory corruption. The 42 tests cover the active LeakBench-Tab framework. Full test restoration requires rebuilding archived-module tests.

---

## 9. Profile Distribution (Corrected)

```
C1-DH-XH: M01, M02, M06, M10, M11 (5 mechanisms)
C1-DH-XC: M03 (1 mechanism)                    ← CORRECTED from C1-DH-XL
C1-DM-XC: M07 (1 mechanism)
C1-DL-XL: M04, M05, M08, M09 (4 mechanisms)
C1-DL-XH: EMPTY (0 mechanisms)
```

**C1-DL-XH quadrant remains empty.** This finding is robust to the M03 correction.

---

## 10. Corrected Non-Paper Readiness Score

| Dimension | Score | Notes |
|---|---|---|
| Semi-synthetic coverage | 94 | 11 mechanisms × 5 strengths |
| Model coverage | 92 | 5 core + 2 supporting + 1 exploratory |
| Core matrix completeness | 84 | 10K cells; CatBoost/LGBM at 10 datasets |
| Statistical correlation audit | 90 | Category-driven confirmed |
| **Result consistency** | **68→78** | M03 corrected; Lending Club lineage documented |
| Natural task evidence | 52 | Lending Club only |
| Governance evaluation | 45 | Framework built, not run |
| Contextual diagnostics | 40 | I/A/S/E pending |
| Engineering + tests | 75 | 42 tests; release validator 21/21 |
| Reproducibility | 83 | Configs, manifests, hashes present |
| Negative results | 96 | BiQ/AIT well-documented |

**Corrected overall: 83/100** (was self-rated 88/100).

---

## 11. Remaining P0 Blockers

| # | Issue | Status |
|---|---|---|
| P0-1 | M03 profile corrected to C1-DH-XC | ✅ Fixed |
| P0-2 | Cell ledger reconciled (10,083) | ✅ Fixed |
| P0-3 | CatBoost/LGBM documented as 10-dataset | ✅ Fixed |
| P0-4 | TabM 66 missing cells documented | ✅ Fixed |
| P0-5 | TabPFN v2 downgraded to exploratory | ✅ Fixed |
| P0-6 | Lending Club lineage documented | ✅ Fixed |
| P0-7 | Test count discrepancy explained | ✅ Fixed |
| P0-8 | 19× claim removed | ✅ Fixed |
| P0-9 | TabM "active rejection" downgraded | ✅ Fixed |

### P1 Remaining

| # | Issue |
|---|---|
| P1-1 | Bank Marketing deep audit |
| P1-2 | NYC Taxi deep audit |
| P1-3 | I-only vs I+A+S+E diagnostic comparison |
| P1-4 | Governance strategy matrix |
| P1-5 | Human-in-the-loop frontier |
| P1-6 | Restore full test suite (161 → 200+) |
