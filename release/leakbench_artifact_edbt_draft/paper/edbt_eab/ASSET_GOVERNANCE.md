# EDBT EA&B Asset Governance

This directory is the paper-facing asset boundary for the EDBT EA&B version.
It does not replace or delete the frozen evidence under `results/` and
`artifacts/`; it provides a small, deterministic set of tables that authors may
cite without choosing among overlapping legacy summaries.

For the project-wide evidence, paper, release, and superseded-material map, use
`reports/edbt_eab/PROJECT_MATERIALS.md` as the canonical navigation page.

## Authoritative paper tables

Only the following three generated CSV files are paper-table inputs:

1. `source_data/generated/main_results.csv` combines the frozen directional
   category contrast, all eleven mechanism C/D/X profiles, and their ordered
   strength slopes. It replaces separate mechanism, strength, and claim tables.
2. `results/edbt_t0_b_full_b1_analysis/paper_table_1_policy.csv` is the primary
   policy repair-vector table; its contract and archetype sensitivities are the
   two adjacent compact Full-B1 paper tables.
3. `source_data/generated/natural_cases.csv` contains the five fixed real-data
   cases. These rows remain case-study-only evidence.

The generator fails closed unless `paper_claims.json` equals
`claim_state.json`, the corrected-v2 release is validated, the canonical SHA is
bound, all eleven mechanisms are present, and the SP8 claim registry has the
expected G1--G4 states.

```bash
python paper/edbt_eab/source_data/build_paper_assets.py
python paper/edbt_eab/source_data/build_paper_assets.py --check
```

## Assets that do not consume a paper table

- `mechanism_model_summary.csv` and `diagnostic_method_by_mechanism.csv` remain
  machine-readable robustness matrices and figure sources.
- The 5,500-row task manifest remains an artifact registry.
- Claim scope is written in prose and remains machine-readable in
  `paper_claims.json`; it is not repeated as a display table.
- Model and diagnostic identities are described in Methods. The legacy
  `baseline_matrix.csv` is supporting documentation, not a result table.

## Legacy dispositions

- `artifacts/edbt_eab/claim_evidence_matrix.csv` is stale and must not be cited.
  Its intervals and some C2a statuses do not match the current claim state.
- `artifacts/edbt_eab/mechanism_contract_matrix.csv` covers only seven of the
  eleven mechanisms and is artifact-only until rebuilt from a complete contract
  registry.
- Both legacy `provenance_inventory.json` copies under `reports/edbt_eab/` and
  `artifacts/edbt_eab/` are pre-build snapshots that incorrectly label current
  canonical outputs as missing. Retain them for history, but do not cite them as
  the current inventory.
- Venue-obsolete paper tables, templates, and release packages are not active
  evidence. The EDBT manuscript uses only the three governed paper-facing CSVs
  and the generated outputs bound by the current manifest.

The measurement/natural disposition is recorded in
`source_data/generated/paper_asset_manifest.json`. Full-B1 source and output
hashes are recorded in `results/edbt_t0_b_full_b1_analysis/analysis_manifest.json`
and the generated paper-artifact manifest.
