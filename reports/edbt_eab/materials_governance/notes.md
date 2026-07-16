# Notes: EDBT EA&B Project Material Governance

## Sources

Repository evidence and findings will be recorded here during the audit.

## Classification Contract

- **Canonical**: frozen source of truth used to generate or validate claims.
- **Paper input**: compact artifact directly consumed by manuscript generation.
- **Derived**: reproducible output generated from canonical inputs.
- **Supporting**: protocol, contract, inventory, or documentation needed to interpret evidence.
- **Superseded**: retained for provenance but prohibited from current claims.
- **Local-only**: required or informative but not redistributable or not Git-tracked.
- **Working**: draft, temporary, cache, or intermediate material with no claim authority.

## Audit Findings

- The EDBT paper boundary already reduces numeric manuscript inputs to three
  governed CSVs with 12, 7, and 5 rows.
- `canonical_cells.csv` is current at 27,500 rows with SHA-256 `25c21440...`.
- `paper_claims.json` and `claim_state.json` are byte-identical with SHA-256
  `ce88fca4...`; the EDBT paper asset check passes.
- SP8 clean governance contains 55,000 successful rows, 5,500 keys, and zero
  duplicates; the governance evidence is LR-only.
- The two `provenance_inventory.json` copies are stale pre-build snapshots that
  still mark now-present outputs as missing.
- Venue-obsolete paper and release artifacts were removed after the EDBT-only
  packages were built and verified.
- `CURRENT_STATUS.md` and `HANDOFF.md` had stale test counts, and HANDOFF still
  recommended the under-audit governance runner.

## Verification Notes

- `build_paper_assets.py --check` returned `EDBT_EAB_PAPER_ASSETS_CURRENT`.
- `python -m pytest tests -q` returned 281 passed and 3 skipped.
- The EDBT manuscript compiled to 11 pages with no undefined citations,
  undefined references, or overfull horizontal boxes.
- All 14 canonical paths referenced by the project-material index exist.
- Running `generate_paper_artifacts.py --help` regenerated deterministic assets
  because the script has no argument parser. No evidence source was changed;
  document this behavior to prevent future accidental writes.
