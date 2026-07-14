# SP5-G Reproducibility Report

**Date:** 2026-07-14

## Single-command rebuild
```bash
bash artifacts/sp5/commands.sh
```
runs, in order:
1. `assemble_claim_ledger_inputs_v2.py` — merge 27500-cell pool (exact replacements)
2. `compute_sp4_detectability.py` — corrected M04/M05/M08 detectability from frozen bundles
3. `build_claim_ledger_v2.py` — enrich to claim_ledger_v2 (+categories, axes, masks)
4. `recompute_sp5_claims.py` — CL2/CL3/CL4/CL10 with cluster bootstrap
5. `render_sp5_figures.py` — figures from ledger (no manual numbers)
6. `pytest tests/test_sp5_ledger.py` — integrity/coverage/reproducibility

## Determinism
- Analysis/bootstrap seed: **20260714** (fixed in scripts + manifest).
- Cluster bootstrap unit: dataset_index; 10000 primary reps.
- `test_bootstrap_reproducible` asserts identical CI on repeated runs.
- claim_ledger_v2 CSV/Parquet verified equal.

## Environment
See `environment.txt` (python 3.x, numpy/pandas/scipy/sklearn/statsmodels/
matplotlib/pyarrow versions).

## Hashes
- Inputs: `input_hashes.json` (5 source ledgers).
- Outputs: `output_hashes.json` (all SP5 artifacts).
- Manifest: `sp5_manifest.json` + `.sha256`.

## Notes
- No notebook hidden state; all analysis is script-driven.
- Figures record source table + ledger hash in `figures/figure_lineage.json`.
