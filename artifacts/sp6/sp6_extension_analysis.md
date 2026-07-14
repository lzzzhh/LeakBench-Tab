# SP6 Extension Analysis — ModernNCA (first batch)

**Status:** ModernNCA formal complete (5500/5500). TabR pending environment gate.
**Evidence tier:** SP6_MODEL_EXPANSION (SP5 core unchanged, 27500 rows byte-identical).
**Metric:** paired_harm = full_auc − strict_auc (identical to SP5).

## Q1 — Does SP5 mechanism-level exploitability ranking extend to ModernNCA?
**Yes, strongly.** Spearman(SP5-core-mean, ModernNCA) = **0.973**.
Per SP5-model vs ModernNCA: LightGBM 1.00, RF 0.97, CatBoost 0.96, TabM 0.96,
LR 0.73 (LR mildly lower, as in SP5). ModernNCA reproduces the mechanism ordering.

## Q2 — Is CL4's modest model-family heterogeneity preserved with ModernNCA?
**Yes.** ModernNCA mean paired_harm = **+0.1636**, sitting inside the SP5 range
[0.1343, 0.1752], between RF (0.162) and CatBoost (0.175):

| Model | mean |
|-------|-----:|
| lr | +0.1343 |
| tabm | +0.1531 |
| rf | +0.1622 |
| **modernnca** | **+0.1636** |
| catboost | +0.1745 |
| lightgbm | +0.1752 |

No order-of-magnitude effect; ModernNCA does not overturn the small-magnitude finding.

## Q3 — Does 3-axis consistency drop with the new model?
**No — it slightly increases.** 6-model mean pairwise Spearman = **0.872**
(SP5 5-model was 0.845); min pairwise 0.591 (unchanged, still the LR-related pair).
ModernNCA vs each SP5 model: mean 0.925, min 0.727 (LR). Adding ModernNCA does
not reduce cross-model profile consistency.

## Q4 — ModernNCA-specific anomalies / negative harm?
ModernNCA shows negative mean harm on **M04 (−0.0052) and M05 (−0.0073)** — the
**same temporal-structured mechanisms** where SP5 TabM was negative. This is a
consistent cross-architecture pattern (neural/metric-learning models slightly
hurt by the corrected temporal-structured leak), not a ModernNCA-specific
failure. No other anomalies; all 5500 cells finite, CUDA, train-only.

## Interpretation (SP6 EXTENSION RESULT, not SP5 core claim)
ModernNCA is a **robustness confirmation** of the SP5 core findings: mechanism
ranking (Q1), modest model-family magnitude (Q2), and profile consistency (Q3)
all extend to a metric-learning neural model, and the TabM M04/M05 negative-harm
signature replicates (Q4). SP5 Claim Matrix V2 is unchanged; a Claim Matrix V3
is not proposed until TabR (and any further models) complete.
