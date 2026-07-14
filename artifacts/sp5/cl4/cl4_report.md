# CL4 — Model-Family Heterogeneity (SP5 recompute)

**Status: CONFIRMED_WITH_REVISED_MAGNITUDES**
**Metric:** paired_harm; cluster bootstrap over datasets (10000 reps, seed 20260714);
paired comparisons on identical (dataset,mechanism,strength,seed) keys.

## Previous claim
"Model family affects exploitation magnitude; RF = 2.2×LR; CatBoost = 2.5×LR."

## New evidence (5 models, 27500 cells)

| Model | mean paired_harm | 95% CI | pos-rate |
|-------|-----------------:|--------|---------:|
| lr | +0.1343 | [0.1223, 0.1463] | — |
| tabm | +0.1531 | [0.1412, 0.1656] | — |
| rf | +0.1622 | [0.1491, 0.1755] | — |
| catboost | +0.1745 | [0.1592, 0.1895] | — |
| lightgbm | +0.1752 | [0.1592, 0.1909] | — |

Paired differences (all CIs exclude 0 except catboost−lightgbm):

| Comparison | mean diff | 95% CI | sig |
|-----------|----------:|--------|-----|
| rf − lr | +0.0279 | [0.0207, 0.0348] | ✓ |
| lightgbm − lr | +0.0409 | [0.0301, 0.0523] | ✓ |
| catboost − lr | +0.0402 | [0.0323, 0.0485] | ✓ |
| tabm − lr | +0.0188 | [0.0119, 0.0252] | ✓ |
| catboost − rf | +0.0123 | [0.0066, 0.0182] | ✓ |
| catboost − lightgbm | −0.0007 | [−0.0064, 0.0049] | ✗ |
| catboost − tabm | +0.0214 | [0.0135, 0.0295] | ✓ |
| rf − tabm | +0.0091 | [0.0023, 0.0152] | ✓ |

Ratios (LR denominator stable, mean 0.134): RF/LR **1.21**, LightGBM/LR **1.30**,
CatBoost/LR **1.30**, TabM/LR **1.14**.

Interaction model: model main effect present; model×mechanism incremental R²
in `cl4_interaction_models.csv`.

## Key finding
Model family **does** measurably affect exploitation (all pairwise vs LR
significant), but **magnitudes are far smaller than previously reported**:
CatBoost/LR = 1.30 (not 2.5), RF/LR = 1.21 (not 2.2). Boosting trees (LightGBM,
CatBoost) are statistically indistinguishable from each other and are the most
exploitative; LR is the least; TabM sits between LR and the trees.

## Verdict
**Revised wording:** *"Model family is associated with exploitation magnitude:
boosting trees (LightGBM/CatBoost) exploit contamination most, logistic
regression least, with the neural model (TabM) intermediate. Absolute differences
are small (≈0.02–0.04 AUROC) and relative ratios modest (1.1–1.3×), substantially
smaller than earlier estimates."*

CL4b (capacity causally drives exploitation): remains **NOT upgraded** — no
within-family capacity-gradient experiment exists.
