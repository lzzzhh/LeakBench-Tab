# SP5-A Runner Audit — TabM Confirmatory (code drift) + M10 amendment

**Date:** 2026-07-13

## TabM confirmatory runner — code drift found

| Field | Value |
|-------|-------|
| Runner | `experiments/leakbench/run_corrected_tabm.py` |
| config_hash | `71b210b0…` — matches checkpoint (same protocol) |
| **code_hash (current)** | `289e590d…` |
| **code_hash (in checkpoint)** | `99b17868…` |
| code_hash match | ❌ NO |
| Only drifted file | `run_corrected_tabm.py` (the 3 src files match frozen manifest) |
| Classification | `FROZEN_BUT_INCOMPATIBLE` (was never hash-bound; runner drifted) |

The 5228-cell checkpoint was produced by an unavailable runner version. `run_id`
embeds `code_hash`, so `--resume` cannot match them and the cells cannot be
reproduced from the current repository → **CODE_DRIFT_EXCLUDED**.

Contrast: CPU core cells (`run_corrected_core.py`) code_hash `1bdb0bf2…`
**matches** current committed code → CPU evidence is reproducible and clean.

## Decision (user Option 3)

- Old 5228 checkpoint: archived `CODE_DRIFT_EXCLUDED`, never claim-eligible.
- Regenerate only the 7 base mechanisms' TabM cells from zero, under a
  prospective freeze of the current committed runner.
- M04/M05/M08 TabM ← SP4 frozen. M10 TabM ← M10 amendment.

## M10 amendment runner (preliminary)

- Path: `experiments/leakbench/run_m10_amendment.py`
- Input mode: immutable npz bundle (verifies bundle SHA256) — clean architecture.
- TabM support: **to be audited before SP5 M10-B run** (Gate SP5 M10-A).
