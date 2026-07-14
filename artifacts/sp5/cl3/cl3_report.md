# CL3 — Detectability–Exploitability Relationship (SP5 recompute)

**Status: CATEGORY_DRIVEN_NUMERICALLY_REVISED → PARTIALLY_CONFIRMED (within-category claim REFUTED)**
**Unit:** mechanism-level (11 points); exploitability model-averaged; detectability = corrected AUPRC.

## Previous claim
"Detectability–exploitability correlation is category-driven; global r = 0.73;
within-category r ≈ 0."

## New evidence
- Global Pearson r = **0.692** (95% CI [0.157, 0.951]); Spearman 0.645; Kendall 0.47.
- Within-category Pearson: **simple 0.949, structured 0.862, boundary −0.044**.
- Regression R²: detectability-only 0.478, category-only 0.514, both 0.619,
  interaction 0.803. Incremental ΔR² of detectability **after** category = **0.105**.
- Partial correlation (detectability vs exploitability | category) ≈ retained positive.

## Key finding — the "within-category ≈ 0" claim is REFUTED
Within simple (r=0.95) and structured (r=0.86) categories there IS a strong
positive detectability–exploitability association. Detectability adds ΔR²=0.105
**even after** conditioning on category. So the relationship is **not** merely an
artifact of category membership; category and detectability both contribute.

Only the **boundary** category shows no within-category association (r=−0.04),
driven by M03 (low detectability 0.13 but high exploitability 0.22 — Cook's D=1.1,
the dominant influence point).

## Sensitivity
- LOMO global Pearson range: **0.602–0.886**.
- Leave-M08-out: global r=0.695 (within-structured r→1.00).
- Leave-M10-out: global r=0.677.
- Leave-one-model-out: 0.66–0.71 (stable).
- Highest influence: **M03** (Cook's D 1.1), then M08/M05/M04 (~0.11–0.14).

## Verdict
Global positive association survives (r≈0.69, but wide CI due to only 11 points).
The specific "category-driven with zero within-category correlation" story is
**refuted**: within-category correlations are strong for simple and structured.

**Revised wording:** *"Detectability and exploitability are positively associated
overall (r≈0.69, mechanism-level, wide CI), with strong positive within-category
associations for simple and structured mechanisms; detectability retains
incremental explanatory power beyond category (ΔR²≈0.11). Boundary mechanisms are
the exception, dominated by the low-detectability/high-exploitability outlier M03."*

Caveat: n=11 mechanisms → wide CIs; treat magnitudes as indicative.
