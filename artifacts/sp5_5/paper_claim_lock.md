# Paper Claim Lock (SP5.5)

Source: claim_ledger_v2.csv sha ccb2549f490e95cb

## CL2 — DOWNGRADED_PARTIAL

**Canonical:** Simple contamination is nearly perfectly localizable on average (AUPRC ~0.93), whereas structured contamination is markedly less detectable on average (AUPRC ~0.35) but highly heterogeneous. Temporal M04/M05 remain difficult (~0.13-0.14); entity/source M08/M09 are moderately detectable (~0.43-0.69).

**Numeric:** simple 0.93; structured mean 0.346; M08 0.43; M09 0.69 | **CI:** simple-structured 0.584 CI[0.562,0.606]

**Prohibited:** Structured contamination is statistically undetectable / AUPRC 0.05-0.07 / all structured hard to localize.

## CL3 — PARTIALLY_CONFIRMED

**Canonical:** Detectability and exploitability exhibit a positive overall association across 11 mechanisms (Pearson r ~0.69), with wide uncertainty (n=11). Category explains substantial variance but detectability retains incremental value (dR2 ~0.11). Strong within simple and structured; boundary is exception (M03 influential).

**Numeric:** global Pearson 0.69; Spearman 0.65; incremental dR2 0.11 | **CI:** Pearson CI[0.157,0.951]

**Prohibited:** Purely category-driven / within-category correlation ~zero / causal.

## CL4 — CONFIRMED_WITH_REVISED_MAGNITUDES

**Canonical:** Model family is associated with exploitation magnitude (boosting trees largest, LR smallest, TabM intermediate), but differences are modest (~0.02-0.04 AUROC; ratios ~1.1-1.3x), not 2.2-2.5x.

**Numeric:** LR 0.134 RF 0.162 LightGBM 0.175 CatBoost 0.175 TabM 0.153 | **CI:** ratios RF1.21 CatBoost1.30 TabM1.14

**Prohibited:** RF 2.2x / CatBoost 2.5x / capacity causally drives exploitation.

## CL10 — BROADLY_CONSISTENT_WITH_EXCEPTIONS

**Canonical:** Three-axis mechanism profiles are broadly consistent across five model families (mean pairwise Spearman ~0.85, Kendall W ~0.88), not universal: TabM slightly negative on M04/M05; disagreement near M02/M03.

**Numeric:** mean Spearman 0.845; min 0.591; W 0.876; quadrant 9/11 | **CI:** TabM M04 -0.0025, M05 -0.0024

**Prohibited:** Fully consistent / invariant across all models.

