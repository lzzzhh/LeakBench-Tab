# EDBT Governance Revision Repair Report

## Status

`COMPLETE_WITH_DISCLOSED_LIMITATIONS`

The governance revision analysis now uses the same 5,500 keys and the same 20%
encoded-column budget for LR, RF, and LightGBM. No downstream model fit was
rerun during this repair.

## Primary results

| Learner | P3 - mean(P2) | 95% task-cluster CI | Keys | P2 seeds |
|---|---:|---:|---:|---:|
| LR | +0.043413 | [+0.004153, +0.077511] | 5,500 | 20 |
| RF | +0.054744 | [+0.025084, +0.081983] | 5,500 | 20 |
| LightGBM | +0.056119 | [+0.027780, +0.081745] | 5,500 | 20 |

Direct paired learner contrasts all cross zero. The supported statement is
that no reliable learner difference was detected, not that the learners are
equivalent.

## Decision-changing sensitivities

| Scope | Effect | 95% CI | Interpretation |
|---|---:|---:|---|
| LR structured family | -0.003940 | [-0.038503, +0.025709] | no reliable family-level advantage |
| LR M09 | +0.148903 | [+0.125821, +0.170974] | structured counterexample |
| LR initial-gap Q4 | +0.202625 | [+0.179430, +0.223163] | high repair opportunity |
| LR sparse archetype | -0.118332 | [-0.159720, -0.092976] | negative regime |
| LR LOAO-sparse | +0.083850 | [+0.072562, +0.094772] | overall effect without sparse |

## Evidence package

- Canonical revision dataset: three manifest-bound CSV partitions, 709,500 rows.
- Selection hashes: complete for every row and matched across learners.
- Formal statistics: `results/edbt_eab_revision/analysis_summary.json`.
- Derived claims: `results/edbt_eab_revision/claim_state.json`.
- Provenance manifest: `results/edbt_eab_revision/manifest.json`.
- Dedicated regression tests: `tests/test_governance_revision.py`.

## Disclosed limitations

- B2 re-fitted strict/full baselines instead of numerically reusing SP5 cells;
  this post-run protocol deviation is disclosed in the protocol.
- Natural-data governance was not run.
- Semantic-group budget sensitivity was not run.
- Gap strata and mechanism-level decomposition are sensitivity analyses and do
  not establish that initial gap alone causally determines repairability.

## Validation

`PYTHONPATH=. pytest -q` completed with 228 passed and 0 failed.
