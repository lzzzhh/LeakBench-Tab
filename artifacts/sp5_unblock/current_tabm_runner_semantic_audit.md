# SP5-A1 — Current TabM Runner Semantic Audit

**Classification: `PROTOCOL_SEMANTICS_MATCH`**
**Date:** 2026-07-13

## Runner

- Path: `experiments/leakbench/run_corrected_tabm.py`
- Runner SHA256: (see `current_tabm_runner_semantic_audit.json`)
- code_hash (current committed): `289e590d03b8ebe5…`
- Config: `configs/paper/corrected_v2.yaml` (config_hash `71b210b0…`, matches core_cpu + old checkpoint)
- Imported frozen sources: `datasets.py`, `mechanisms/__init__.py`, `official_tabm.py` — all match the structured_prior freeze manifest hashes.

## Verified invariants

| Invariant | Result |
|-----------|--------|
| strict view == base.X for base-7 | ✅ all 7 (M01,M02,M03,M06,M07,M09,M11) |
| clean_auc IS the mask-strict AUC | ✅ (clean==strict, |diff|=0) |
| paired_harm = full − clean = full − strict | ✅ consistent with SP4 + whole ledger |
| chronological split frozen | ✅ split invariance guard present + enforced |
| CPU↔TabM-runner split_hash parity | ✅ 28/28 sampled cells match |
| run_id includes code_hash | ✅ (so drift is detectable) |
| CUDA fail-closed, no CPU/MPS fallback | ✅ official_tabm raises if CUDA unavailable |

## Risk scan

- Runtime injection: YES via base `LeakBenchInjector` — this is the **correct**
  implementation for the base-7 mechanisms (the amended mechanisms M04/M05/M08/M10
  are excluded and come from SP4/amendment replacements).
- Target leakage beyond mechanism design: none.
- Silent dataset substitution: none.
- Automatic device fallback: none (fails closed).

## Scope

base-7 only: M01, M02, M03, M06, M07, M09, M11.
M04/M05/M08 → SP4 frozen replacement. M10 → M10 amendment replacement.

## Verdict

The current committed runner reproduces the protocol semantics and produces
strict/full views identical in construction to the CPU reference pipeline. It is
eligible for a **prospective freeze** (SP5-A2) and to generate new base-7 TabM
evidence from zero.
