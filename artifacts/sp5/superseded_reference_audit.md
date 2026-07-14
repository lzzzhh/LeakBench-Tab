# Superseded Reference Audit (SP5-G)

## Method
Searched active code/docs (scripts/*.py, artifacts/sp5/*.md,*.yaml) for superseded
HEADLINE references. Data CSVs are excluded from text search because floating-point
AUC values contain coincidental digit substrings (e.g. '0.0283', '...5228...') that
are NOT references to the interim +0.028 headline or the 5228-row checkpoint.

## Result
Active code/docs superseded-reference hits (excluding exclusion records): 0

- NONE — active analysis pipeline clean

## Data-level verification
- Ledger M08 mean paired_harm = 0.0045 (corrected), NOT interim 0.028.
- claim_ledger_v2 source column ∈ {core_cpu, base7_tabm, sp4_frozen, m10_amendment} only.
- Old checkpoint (99b17868 / 5228 rows) archived + gitignored; not read by any SP5 script.
- Interim entity-mean M08 archived under archive/invalid_interim; not in ledger.
- Legacy profiles_v2 / ce2r_neural excluded via source_registry; not read by SP5 scripts.

## Conclusion
0 superseded references reachable by the active analysis/claim pipeline.
