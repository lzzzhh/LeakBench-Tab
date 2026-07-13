# M08 Temporary Runs — Scientific Exclusion Record

**Status:** `INVALID_PROTOCOL_SUBSTITUTION` — EXCLUDED FROM SCIENTIFIC EVIDENCE
**Date:** 2026-07-13
**Decision authority:** Explicit user directive ("DISCARD TEMPORARY M08 EVIDENCE")

## Summary

Two interim M08 result files were produced by ad-hoc runner scripts (CPU and
Windows GPU) before the audit discovered that the canonical corrected M08
evidence is governed by a pre-existing **frozen protocol**
(`protocols/structured_prior_v1/inference_protocol_v1.json`,
status `FROZEN_BEFORE_ANY_MODEL_RUN`).

The interim runs do **not** satisfy the frozen protocol and are therefore
scientifically invalid. They are retained only as an auditable exclusion record
and must never re-enter the evidence base.

## Excluded files

| File | SHA256 (short) | Rows | Git-tracked |
|------|----------------|------|-------------|
| `results/ce2r/m08_rerun.csv` | `de07c7c2…` | 216 cells | yes (commit `2059308`) |
| `results/ce2r/m08_neural.csv` | `0620ec27…` | 108 cells | no (untracked) |

Full metadata: `m08_temporary_runs_manifest.csv` (same directory).

## Why they are invalid (protocol substitutions)

The frozen protocol explicitly forbids substitution. The interim runs violated
every dimension:

| Dimension | Frozen protocol | Interim runs | Violation |
|-----------|-----------------|--------------|-----------|
| Injector | `StructuredPriorV1Injector` (constant 0.5 prior) | base `LeakBenchInjector` + `ENTITY_LEAK` | mechanism_substitution |
| Seeds | `[13, 42, 2026, 3407, 7777]` | `[13, 42, 2026]` | seed_substitution |
| Strengths | `S1–S5` (0.2, 0.4, 0.6, 0.8, 1.0) | `[0.1, 0.5, 1.0]` | strength_substitution |
| Datasets | 20 frozen confirmatory bundles | self-generated synthetic (n=350–400) | dataset/task_substitution |
| Metric | `paired_harm = full_auc − strict_auc` (hashed views) | ad-hoc `infl` (leave-one-out drop) | metric_substitution |
| Models | lr, rf, catboost, lightgbm, tabm | + TabR, ModernNCA, TabICL | model_out_of_protocol |
| Integrity | task_hash / split_hash / bundle_sha256 required | none | no hashes |

## Numbers that must NOT be cited

The following interim numbers are **withdrawn** and must not appear in any
report, Claim Matrix, model comparison, or aggregate:

- Six-model mean inflation `+0.0277`
- Per-model: LightGBM +0.0501, CatBoost +0.0372, RF +0.0326, LR +0.0281, TabR +0.0273
- `ModernNCA −0.0092`
- Any "eight-model" or "six-model" M08 restoration claim

## Disposition

- Files moved to `archive/invalid_interim/m08/` (excluded from all analysis
  scripts, glob discovery, and validator valid-results scope).
- Interim runner scripts on the GPU node (`m08_n.py`, `run_m08_neural.py`)
  are non-canonical and are not part of the evidence base.
- Canonical M08 evidence will be produced solely by executing the frozen
  `structured_prior_replacement_v1` protocol (7500 prespecified cells).

## Claim impact

Until the 7500-cell frozen protocol completes:

- M08: `PENDING FROZEN-PROTOCOL EXECUTION`
- CL2 / CL3 / CL4 / CL10: `PENDING RECOMPUTATION`
- Eight-model consistency claim: `WITHDRAWN`
