# CL10 — Three-Axis Profile Consistency (SP5 recompute)

**Status: BROADLY_CONSISTENT_WITH_EXCEPTIONS**
**Scope:** 5 models × 11 mechanisms = 55 profiles. Axes: construction (invalid=1
for all), detectability (corrected AUPRC), exploitability (paired_harm).

## Previous claim
"Three-axis mechanism profiles are consistent across core models (5 models, 11 mechanisms)."

## New evidence
- Mean pairwise cross-model Spearman (mechanism exploitability ranking): **0.845**.
- Minimum pairwise Spearman: **0.591**.
- Kendall's W (concordance): **0.876**.
- Quadrant (median-split D×X) agreement: **9 / 11** mechanisms have all 5 models
  in the same quadrant.
- Leave-one-model-out mean pairwise Spearman: drop LR 0.95, drop RF 0.81,
  drop LightGBM 0.81, drop CatBoost 0.83, drop TabM 0.83 (LR is the mild outlier;
  removing it raises agreement).

## Exceptions
- Quadrant disagreement at **M02, M03** (models split across the median boundary;
  M02 sits near the exploitability median, M03 is the low-D/high-X outlier).
- **TabM negative harm** at **M04 (−0.0025), M05 (−0.0024)** — the corrected
  temporal-structured mechanisms slightly *hurt* TabM; other models are ~0 there.
  This is a genuine model×mechanism exception, not overall inconsistency.

## M08 / M10 profile change (vs superseded profiles_v2)
- **M08**: old core_mean_harm −0.027 (interim/invalid) → new +0.0045; detectability
  0.048→0.431. M08 moves from "invisible/no-harm" to "moderately detectable,
  near-zero harm".
- **M10**: old 0.071 → new +0.226 (amendment strict-view; old base-clean baseline
  understated harm). M10 remains a simple/high-exploitability mechanism.

## Verdict
Profiles are **broadly consistent** across models (mean Spearman 0.85, W 0.88),
but not unconditionally: 2/11 mechanisms show quadrant disagreement and TabM
shows negative harm on M04/M05.

**Revised wording:** *"Three-axis mechanism profiles are broadly consistent across
the five models (mean cross-model Spearman ≈ 0.85, Kendall W ≈ 0.88), with
identifiable mechanism-specific exceptions — notably TabM's slightly negative harm
on temporal-structured mechanisms (M04/M05) and boundary-median disagreement at
M02/M03."*

CL13 (TabM negative-harm causal explanation): remains **UNCONFIRMED** — no
dedicated mechanism experiment; SP5 does not assign causation.
