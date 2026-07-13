# TabM Confirmatory Checkpoints — CODE_DRIFT_EXCLUDED

**Status:** `CODE_DRIFT_EXCLUDED` / `NON_REPRODUCIBLE` / `NOT CLAIM ELIGIBLE`
**Date:** 2026-07-13
**Decision:** SP5-A Code Drift Recovery (Option 3), explicit user directive.

These checkpoint cells were produced by an unavailable runner version
whose code hash does not match the current committed runner.

- Stored `code_hash`: `99b1786882445b85e9ef49e8d57eb8eb9867a2f12c47dd3ca194fb8eec2818dc`
- Current committed `code_hash`: `289e590d03b8ebe5…`
- The original `run_corrected_tabm.py` bytes that generated these cells are not
  recoverable from git history.
- `run_id` embeds `code_hash`, so these cells cannot be matched, resumed, or
  reproduced against the current repository.

Max checkpoint = 5228 / 5500 cells (incomplete). 20 checkpoint files, all
carrying the same drifted code_hash.

They are retained for audit history only.

They are excluded from all:
- formal evidence pools
- claim ledgers
- statistical analyses
- paper tables
- paper figures
- appendices
- claim-evidence matrices

Full per-file inventory + SHA256: `exclusion_manifest.json` (same directory).

Replacement: a fresh, reproducible base-7 TabM run
(`tabm_confirmatory_base7_v2`) is produced under a prospective freeze of the
current committed runner. See `artifacts/sp5_unblock/`.
