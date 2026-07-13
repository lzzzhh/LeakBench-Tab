# SP5-A5 — Base-7 TabM Confirmatory (reproducible, CUDA)

**Status: COMPLETE. 3500/3500 SUCCESS on CUDA under prospective freeze.**
**Date:** 2026-07-14
**Freeze commit:** `915af85` | **code_hash:** `289e590d03b8ebe5`

## Result

| Metric | Value |
|--------|-------|
| Cells | 3500 / 3500 |
| SUCCESS | 3500 |
| FAILURE | 0 |
| Duplicates (run_id) | 0 |
| Non-finite | 0 |
| CPU/MPS fallback | 0 |
| Mechanisms | M01, M02, M03, M06, M07, M09, M11 (500 each) |
| Contains M04/M05/M08/M10 | NO (no replacement overlap) |
| Device | cuda only |
| Ledger SHA256 | `384c30452008c03f…` |

## Provenance

- Runner: `run_corrected_tabm.py`, prospectively frozen (SP5-A2), code_hash
  `289e590d` — reproducible against current repo (unlike the excluded 5228
  checkpoint, code_hash `99b17868`).
- Fresh formal directory, started empty, no resume from old checkpoint.
- Metric `paired_harm = full_auc − clean_auc`, where clean == mask-strict view
  for base-7 (verified in SP5-A1).
- WSL2 Ubuntu + RTX 4060 + torch 2.5.1+cu121, `/root/LeakBench-Tab-sp5` (ext4).
- Runtime ~10640s (~3h).

## Per-mechanism mean paired_harm (base-7 TabM)

| M01 | M02 | M03 | M06 | M07 | M09 | M11 |
|----:|----:|----:|----:|----:|----:|----:|
| +0.164 | +0.188 | +0.267 | +0.262 | +0.116 | +0.215 | +0.264 |

These are the simple / more-exploitable mechanisms, with large harms — a sharp
contrast to the small structured M04/M05/M08 harms (~+0.004 from SP4). This is
the expected Simple ≫ Structured pattern and will feed CL2/CL3/CL10.

## Role in ledger

Provides the TabM evidence for 7 of 11 mechanisms. Combined with:
- M04/M05/M08 ← SP4 frozen (TabM +0.0026 etc.)
- M10 ← M10 amendment (TabM: pending, Gate SP5 M10-B)
to complete the 5-model × 11-mechanism TabM coverage.
