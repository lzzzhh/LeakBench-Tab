# Notes: EDBT governance revision repair

## Confirmed pre-fix defects

- LR formal analysis pooled 1%, 5%, 10%, and 20% budgets while RF and LightGBM used only 20%.
- `P3_better` independently resampled P3 and P2 task sets instead of preserving pairing.
- Learner interaction did not separate observed difference from bootstrap mean.
- LOAO positive-task proportion was hard-coded to zero.
- Revision claim state was missing and manifest status was stronger than its known gaps allowed.
- The formal script produced 22,000 LR rows versus 5,500 RF and LightGBM rows because it did not filter LR to the primary 20% budget.
- The frozen 5,500-key bundle manifest is available at `artifacts/sp6/sp6_bundle_manifest.csv`, so deterministic selection hashes can be reconstructed without model fitting.
- P0 rows record `governed_auc=strict_auc`; no-removal semantics require `governed_auc=full_auc` while SDR remains zero.

## Target primary values

These values are audit targets only; regenerated artifacts remain authoritative.

- LR 20%: approximately +0.0434, CI approximately [+0.004, +0.078].
- RF 20%: approximately +0.0547, CI approximately [+0.025, +0.082].
- LightGBM 20%: approximately +0.0561, CI approximately [+0.028, +0.082].
- LR M09 20%: approximately +0.149.
- LR sparse archetype 20%: approximately -0.118.
- LR high-gap quartile 20%: approximately +0.203.

## Verified post-fix results

- LR 20%: +0.043413, 95% task-cluster CI [+0.004153, +0.077511].
- RF 20%: +0.054744, CI [+0.025084, +0.081983].
- LightGBM 20%: +0.056119, CI [+0.027780, +0.081745].
- Direct learner contrasts all cross zero; this supports only a no-detected-difference statement, not equivalence.
- LR M09: +0.148903, CI [+0.125821, +0.170974].
- LR structured family: -0.003940, CI [-0.038503, +0.025709].
- LR high-gap quartile: +0.202625, CI [+0.179430, +0.223163].
- LR sparse archetype: -0.118332, CI [-0.159720, -0.092976].
- LOAO-sparse overall: +0.083850, CI [+0.072562, +0.094772].

## Provenance repair

- Reconstructed selection hashes for all 709,500 rows without model fitting.
- Verified 115,500 matched 20% P2/P3 rows between LR and each of RF and LightGBM.
- Corrected P0 `governed_auc` to equal `full_auc` while preserving zero SDR.
- Bound the analysis, claim state, runners, protocol disclosure, raw partitions, and derived tables in the revision manifest.
