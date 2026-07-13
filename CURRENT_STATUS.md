# LeakBench-Tab — Current Status

**Date:** 2026-07-13
**Status:** NOT FROZEN

## Known Issues (pending fix)

1. **M08 entity generator** — computes entity_rates but never uses them. Feature = random entity IDs.
2. **Governance scripts** — run_meta_tier.py still uses old implementation (int indices as strings, seed=42, M04 centered conv, S-axis reads leak_mask).
3. **Natural tasks** — Bank PRE field name is bm_19 (synthetic fallback indicator). Lending adapter has undefined `ext` variable.
4. **TabPFN v2** — 594/594 cells returned constant 0.000 (API failures).

## Currently Valid

- C/D/X three-axis framework
- Simple contamination is easily detectable and exploitable
- Old operational metadata gain (+0.098) is invalid (feature-name peeking)
- Corrected operational gain is small (+0.017, +0.031) — insufficient for reliable claim
- Structured mechanisms show low/inconsistent exploitability (M08 excluded)

## Withdrawn/Pending

- 8-model consistency claim
- CL3/CL4/CL10 CONFIRMED
- CL14-R/G/T governance claims
- CL16a natural transfer (not properly evaluated)
- REFROZEN status
- 91/100, 93/100, 94/100 readiness scores

## 2026-07-13 Update

### Interim M08 runs EXCLUDED (INVALID_PROTOCOL_SUBSTITUTION)
Ad-hoc CPU/GPU M08 runs were produced before discovering that corrected M08 is
governed by the frozen `structured_prior_replacement_v1` protocol. They violated
the frozen protocol on every dimension (mechanism/seed/strength/task/dataset/
metric/model substitution, no integrity hashes) and are **withdrawn**.

- Excluded files archived under `archive/invalid_interim/m08/` (not consumed by
  any script, glob, or validator).
- Exclusion record: `reports/excluded_results/m08_temporary_runs_exclusion.md`.
- Withdrawn numbers (must not be cited): interim `+0.037`, six-model `+0.0277`,
  `ModernNCA −0.0092`, and per-model interim inflations.

### Canonical M08 path (frozen protocol)
Corrected M08 evidence will come solely from executing the frozen protocol:
- Protocol: `protocols/structured_prior_v1/inference_protocol_v1.json`
  (`FROZEN_BEFORE_ANY_MODEL_RUN`)
- Task plan: `structured_prior_replacement_v1_tasks.csv` (1500 task variants)
- Injector: `StructuredPriorV1Injector` (constant 0.5 prior)
- Runner: `run_structured_prior_v1_bundle.py --allow-run` (read-only, hash-verified)
- Scope: M04/M05/M08 × S1–S5 × seeds [13,42,2026,3407,7777] × {lr,rf,catboost,lightgbm,tabm}
- Cells: 7500 total (6000 CPU + 1500 TabM GPU)
- Metric: `paired_harm = full_auc − strict_auc`

### Claim status (until 7500-cell closure)
- M08: `PENDING FROZEN-PROTOCOL EXECUTION`
- CL2 / CL3 / CL4 / CL10: `PENDING RECOMPUTATION`
- Eight-model consistency claim: `WITHDRAWN` (protocol has 5 models; at most a
  five-model frozen-protocol M08 result can be restored)

### Other pending (separate tasks, not this track)
- Governance (Track B): audit reuse of bundle infra for M06/M09/M10/M11
- Natural tasks (Bank/Lending adapters): deferred to a separate task
- Old eight-model aggregates remain `SUPERSEDED`
