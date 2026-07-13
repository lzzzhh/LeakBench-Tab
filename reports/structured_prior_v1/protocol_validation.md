# Gate SP0 — Structured Prior v1 Protocol Integrity Validation

**Verdict: `READY_TO_RUN`**
**Date:** 2026-07-13
**Machine artifact:** `results/structured_prior_replacement_v1/protocol_validation.json`

## Protocol status

| Field | Value |
|-------|-------|
| Freeze status | `FROZEN_BEFORE_ANY_MODEL_RUN` |
| Models executed | 0 |
| Model results observed | false |
| Frozen files verified | 14 / 14 |
| All hashes + sizes match | **YES** |

All 14 frozen files match both SHA256 and byte size against
`freeze_manifest_v1.json`. No file has drifted since freeze.

## Key hashes

| Artifact | SHA256 |
|----------|--------|
| inference_protocol_v1.json | `6368d95d…b23200` |
| task plan (replacement) | `7ded0fd8…6dba58` |
| injector (structured_prior_v1.py) | `533cc542…37d8cc` |
| runner (run_structured_prior_v1_bundle.py) | `a5b1d416…da9ec4` |
| core_models.py | `a1a795d8…11444e` |
| official_tabm.py | `d1e9ff88…8c0f1b` |

## Task plan dimensions (read from frozen CSV, not derived)

| Dimension | Value |
|-----------|-------|
| Total task variants | 1500 |
| Unique variant IDs | 1500 |
| Duplicate variant IDs | 0 |
| Missing | 0 |
| Datasets (dataset_index) | 20 (0–19) |
| Mechanisms | M04, M05, M08 |
| Strengths | S1, S2, S3, S4, S5 |
| Seeds | 13, 42, 2026, 3407, 7777 |
| Archetypes | drifting, interaction, linear, nonlinear, sparse |
| Models per variant | lr, rf, catboost, lightgbm, tabm |

Grid check: 20 × 3 × 5 × 5 = **1500** ✓ (exact, no gaps/dupes).
Per mechanism × strength: 100 each (uniform).

**Note:** The replacement plan covers all three amended future-outcome
mechanisms (M04/M05/M08), not M08 alone. M08 is 500 of the 1500 variants
(2500 of 7500 model cells).

## Config cross-check (all match)

| Config field | Value | Matches plan |
|--------------|-------|--------------|
| expected_task_variants | 1500 | ✓ |
| expected_model_cells | 7500 | — |
| expected_cpu_model_cells | 6000 | — |
| expected_tabm_model_cells | 1500 | — |
| mechanisms | M04/M05/M08 | ✓ |
| seeds | 13/42/2026/3407/7777 | ✓ |
| dataset_count | 20 | ✓ |
| core_models | lr/rf/catboost/lightgbm/tabm | — |

## Metric definition (frozen)

- strict view: `task.X[:, ~task.leakage_mask]`
- full view: `task.X`
- primary metric: `paired_harm = full_auc − strict_auc`

## Authorization

`--allow-run` is GRANTED under the conditions in the execution directive.
All protocol-substitution prohibitions remain in force.

**No model has been run yet.** Next gate: SP1 (export immutable bundles).
