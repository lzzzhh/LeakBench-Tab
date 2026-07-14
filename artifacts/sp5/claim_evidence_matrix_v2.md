# Claim-Evidence Matrix V2 (SP5-G freeze)

**Date:** 2026-07-14 | **Evidence:** `claim_ledger_v2` (27500 cells, 5 models × 11 mechanisms)
**Bootstrap:** cluster over datasets, 10000 reps, seed 20260714.

This supersedes prior claim status for CL2/CL3/CL4/CL10. Does not overwrite frozen
prior matrices. Machine-readable: `claim_evidence_matrix_v2.{csv,json}`.

## Summary of changes

| Claim | Prev | New | Class |
|-------|------|-----|-------|
| CL2 | CONFIRMED | DOWNGRADED_PARTIAL | DOWNGRADED |
| CL3 | CONFIRMED | PARTIALLY_CONFIRMED | NUMERICALLY_REVISED |
| CL4 | CONFIRMED | CONFIRMED_WITH_REVISED_MAGNITUDES | NUMERICALLY_REVISED |
| CL10 | CONFIRMED | BROADLY_CONSISTENT_WITH_EXCEPTIONS | WORDING_NARROWED |
| CL4b | UNCONFIRMED | UNCONFIRMED | UNCHANGED |
| CL9 | CONFIRMED | UNCHANGED | UNCHANGED |
| CL13 | UNCONFIRMED | UNCONFIRMED | UNCHANGED |

## CL2 — DOWNGRADED_PARTIAL
- Old: structured hard to detect, AUPRC 0.05–0.07.
- New: structured mean 0.346 [0.315, 0.376]; heterogeneous (M04/M05 ~0.13,
  M08 0.43, M09 0.69). Simple 0.93; gap 0.584.
- Reason: old numbers used pre-correction mechanisms.
- Paper impact: abstract, main results, figure, discussion.

## CL3 — PARTIALLY_CONFIRMED (within-category≈0 refuted)
- Old: category-driven, global r=0.73, within-category r≈0.
- New: global r=0.692 [0.157, 0.951]; within simple 0.95, structured 0.86,
  boundary −0.04; detectability incremental ΔR²=0.11 after category.
- Reason: strong within-category associations contradict the "≈0" story.
- Paper impact: main results, figure, discussion.

## CL4 — CONFIRMED_WITH_REVISED_MAGNITUDES
- Old: RF=2.2×LR, CatBoost=2.5×LR.
- New: RF/LR 1.21, CatBoost/LR 1.30, LightGBM/LR 1.30, TabM/LR 1.14; all
  pairwise vs LR significant; absolute diffs 0.02–0.04.
- Reason: corrected, complete 5-model evidence; magnitudes much smaller.
- Paper impact: main results, table, figure.

## CL10 — BROADLY_CONSISTENT_WITH_EXCEPTIONS
- Old: consistent across models.
- New: mean pairwise Spearman 0.845, Kendall W 0.876, quadrant 9/11;
  exceptions TabM negative on M04/M05, quadrant disagreement M02/M03.
- Reason: full 5-model (incl. real TabM) reveals mechanism-specific exceptions.
- Paper impact: main results, figure, discussion.

## Related (not upgraded)
- **CL4b** (capacity causal): UNCONFIRMED — no within-family capacity experiment.
- **CL9** (Simple > Structured): UNCHANGED, consistent with corrected evidence.
- **CL13** (TabM negative-harm cause): UNCONFIRMED — observational only.

## Paper impact rollup
- Abstract: CL2 (detectability nuance).
- Main results/figures: CL2, CL3, CL4, CL10.
- Tables: CL4 magnitudes.
- Discussion/limitations: CL3 within-category, CL10 exceptions, CL13.
