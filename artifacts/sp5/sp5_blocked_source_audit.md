# SP5 — BLOCKED at Source Audit (Section 17 stop)

**Status: BLOCKED. Awaiting protocol decision before building claim_ledger_v2.**
**Date:** 2026-07-13
**Gate reached:** SP5.1 (source registry) complete; SP5.1b (evidence completeness stop-check) triggered stop conditions.

## What the ledger needs

CL2/CL3/CL4/CL10 require a unified 5-model × 11-mechanism formal ledger:
`5 models × 11 mech × 5 strengths × 5 seeds × 20 datasets = 27500 cells`.

## Intended composition (determined from configs, not guessed)

`configs/paper/structured_prior_replacement_v1.yaml` declares
`role: exact_replacement_only`, `replaces_corrected_v2_mechanisms: [M04,M05,M08]`,
`replacement_key: [dataset_index, mechanism, strength, model, seed]`.
`configs/paper/m10_amendment_v1.yaml` declares
`purpose: replace_base_clean_with_mask_derived_strict_view` for M10.

So the ledger = core_cpu_cells + TabM confirmatory, with M04/M05/M08 replaced by
SP4 and M10 replaced by the m10 amendment.

## Metric consistency: RESOLVED (not a blocker)

Verified empirically: core `clean_auc` == SP4 `strict_auc` exactly
(mean |diff| = 0.0 over 6000 overlapping cells). For the 7 non-amended
mechanisms the core `paired_harm = full − clean` **is** the mask-derived strict
harm. Metric is consistent across the whole ledger. The M10 amendment exists
only because M10's construction made base-clean ≠ strict for that one mechanism.

## BLOCKERS (Section 17)

### Blocker 1 — TabM confirmatory is INCOMPLETE (stop cond. #7-adjacent / missing cells)
- Only checkpoint files exist: max `tabm_cells_checkpoint_005228.csv` = **5228 / 5500** cells (272 missing, ~475/mech vs 500 target).
- No finalized `tabm_cells.csv` result file, no run manifest.
- TabM is one of the 5 models CL4/CL10 require across all 11 mechanisms.
- Impact: CL4 (model family, needs TabM) and CL10 (cross-model profiles, needs TabM on all mechs) **cannot be computed at full 5-model scope** for the 7 non-amended + M10 mechanisms.

### Blocker 2 — M10 amendment has NO TabM cells
- `m10_amendment_confirmatory/cpu_cells.csv` = 2000 cells, **CPU models only** (lr/rf/lightgbm/catboost). No TabM M10 amendment run exists.
- Since the amendment *replaces* core M10, and core M10 TabM (in the incomplete confirmatory) uses the un-amended clean baseline, there is **no valid strict-view TabM M10 cell** at all.
- Impact: M10 has 4-model coverage only; TabM×M10 strict evidence is absent.

## Consequence

A complete, protocol-consistent **5-model × 11-mechanism** ledger cannot be
assembled from existing formal results. Two sub-options are possible but **both
require your authorization** because they change scope or require new runs:

- **Option 1 — Complete the TabM evidence (new runs).** Finish the 272 missing
  TabM confirmatory cells AND run the M10 amendment on TabM (2000 cells) via the
  same WSL2+CUDA path. Then build the full 27500-cell ledger. (~1–2 h GPU.)
  Requires confirming the TabM confirmatory runner is frozen/compatible and that
  I'm authorized to run it.

- **Option 2 — Build a 4-model (CPU) ledger now + TabM where complete.**
  Compute CL2/CL3/CL4/CL10 primarily on the 4 CPU models (fully complete for all
  11 mechanisms after amendment), and report TabM only for M04/M05/M08 (SP4, the
  only complete TabM evidence). Explicitly scope claims to "4-model core; TabM on
  structured mechanisms only." No new runs.

## Recommendation

Given the project's strictness, **Option 1** yields the honest full-scope result
the claims were originally about, and reuses the now-proven WSL2+CUDA frozen
infrastructure. But it needs (a) your `--allow-run` authorization for the TabM
confirmatory + M10-amendment-TabM, and (b) confirmation those runners are
frozen/compatible (I have not yet audited the TabM confirmatory runner's freeze
state or its resume/dedup behavior).

## Not done (correctly deferred)
- No claim_ledger_v2 built (would require manual gap-filling → stop cond. #5).
- No CL2/3/4/10 computed.
- No claim status changed.

## Artifacts
- `artifacts/sp5/source_registry.csv` — classified source inventory.
- This report.
