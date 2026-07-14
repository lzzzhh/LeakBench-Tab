# CL2 — Structured Detectability (SP5 recompute)

**Status: DOWNGRADED_PARTIAL / WORDING_NARROWED**
**Metric:** detector AUPRC = `mutual_info_classif(X_train,y_train)` vs leakage_mask.
Cluster bootstrap over datasets (10000 reps, seed 20260714).

## Previous claim
"Structured contamination is difficult to localize statistically; AUPRC ≈ 0.05–0.07."

## New evidence (corrected mechanisms)
Structured mean AUPRC = **0.346** (95% CI [0.315, 0.376]), range **[0.13, 0.69]**.

Per-mechanism (corrected):

| Mechanism | AUPRC | detectable? |
|-----------|------:|-------------|
| M04 Post-Outcome Agg | 0.130 | hard |
| M05 Temporal Look-Ahead | 0.136 | hard |
| M08 Entity Leakage | 0.431 | moderate |
| M09 Source Leakage | 0.687 | fairly detectable |

Simple mean AUPRC = 0.930; simple − structured = **0.584** (95% CI [0.562, 0.606]).

## Why it changed
The old "≈0.05–0.07" was computed on pre-correction M04/M05/M08 mechanisms.
Corrected structured mechanisms are more detectable, especially M08 (0.43, from
frozen SP4 bundles) and M09 (0.69, canonical core). M09 was never amended; its
current core detectability is 0.69, far above the old reported 0.048.

## Verdict
The blanket claim "structured contamination is hard to detect" is **not
supported**. Structured mechanisms are **heterogeneous**: only the temporal
family (M04/M05) remains hard (~0.13); entity/source leakage is moderately to
clearly detectable.

**Revised wording:** *"Simple contamination is near-perfectly localizable
(AUPRC ≈ 0.93), while structured contamination is markedly less localizable on
average (≈0.35) but heterogeneous — temporal look-ahead/aggregation remain hard
(≈0.13) whereas entity and source leakage are moderately detectable
(0.43–0.69)."*

Sensitivity: leave-one-mechanism-out and leave-one-dataset-out in
`cl2_lomo.csv` / `cl2_lodo.csv`; structured mean stays 0.24–0.42 depending on
which structured mechanism is removed (M09 most influential).
