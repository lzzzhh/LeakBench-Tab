# LeakBench-Tab EDBT EA&B Project Materials

**Status:** evidence chain current; official-template draft and EDBT-only draft package built

**Last audited:** 2026-07-17

**Target:** EDBT 2027 Experiments, Analysis & Benchmarks

This is the canonical navigation page for project authors, reviewers, and
artifact evaluators. It adds no new scientific result. It identifies which
existing files may support the paper and which files are historical,
superseded, incomplete, or local-only.

## Start Here

Use this path for all paper work:

```text
frozen cells and protocols
  -> canonical manifest and claim state
  -> three governed paper CSVs
  -> generated tables, figures, and macros
  -> EDBT manuscript and PDF
```

1. **Manuscript:** `paper/edbt_eab/main.tex`
2. **Rendered draft:** `paper/edbt_eab/output/official/main.pdf`
3. **Paper asset policy:** `paper/edbt_eab/ASSET_GOVERNANCE.md`
4. **Machine-readable paper claims:** `results/corrected_v2/paper_claims.json`
5. **Canonical measurement ledger:** `results/corrected_v2/canonical_cells.csv`
6. **SP8 governance manifest:** `artifacts/sp8/governance_clean_manifest.json`
7. **Standalone source ZIP:** `release/LeakBench-Tab-EDBT-2027-source.zip`

Do not choose a convenient number from a pilot, legacy table, or historical
report. A manuscript sentence must be supported by the claim state or by one of
the three governed paper CSVs below.

## Smallest Paper-Facing Asset Set

Only three CSV files provide numeric manuscript results:

| File | Rows | Purpose |
|---|---:|---|
| `paper/edbt_eab/source_data/generated/main_results.csv` | 12 | Directional category contrast plus M01--M11 C/D/X and strength profiles |
| `paper/edbt_eab/source_data/generated/governance_results.csv` | 7 | Four SP8 budgets plus the 20% category breakdown |
| `paper/edbt_eab/source_data/generated/natural_cases.csv` | 5 | Five fixed natural case studies; not a population sample |

Their hashes and upstream bindings are recorded in
`paper/edbt_eab/source_data/generated/paper_asset_manifest.json`. Generated
LaTeX tables, result macros, and multi-format figures are derived outputs and
must not be edited by hand.

## Evidence Layers

### 1. Confirmatory measurement

- `results/corrected_v2/canonical_cells.csv`: 27,500 successful cells.
- `results/corrected_v2/canonical_manifest.json`: 20 controlled datasets,
  11 mechanisms, five strengths, five models, and five seeds.
- Canonical SHA-256:
  `25c2144064c47fd2b2965fea6a609d08017f87c038f61456c54e8711eb9b49d7`.
- `results/corrected_v2/claim_state.json` and
  `results/corrected_v2/paper_claims.json` are byte-identical and define allowed
  and prohibited wording.

The canonical cell ledger is the source of truth for controlled measurement.
The raw CPU, TabM, and M10 amendment files remain provenance inputs rather than
alternative paper tables.

### 2. Diagnostic robustness

- `results/corrected_v2/diagnostic_canonical_cells.csv` contains the frozen
  diagnostic panel.
- `results/corrected_v2/statistics/diagnostic_method_by_mechanism.csv` and
  `results/corrected_v2/statistics/mechanism_model_summary.csv` are
  machine-readable robustness matrices or figure sources.
- Diagnostic method comparisons remain descriptive unless the claim state
  explicitly allows stronger wording.

Pilot diagnostic directories are retained for provenance and are not paper
evidence.

### 3. SP8 matched-cost governance

- `artifacts/sp8/governance_clean.csv`: 55,000 successful policy cells over
  5,500 task keys, with zero duplicates.
- `artifacts/sp8/governance_clean_manifest.json`: binds the clean runner, CSV,
  policy registry, bootstrap analysis, and G1--G4 claim matrix.
- `artifacts/sp8/bootstrap_analysis.json`: task-level dataset-cluster bootstrap.
- `artifacts/sp8/claims/claim_evidence_matrix_sp8.json`: G1/G3/G4 supported;
  G2 inconclusive.

The governance evidence is LR-only. It does not establish learner-independent
governance effectiveness. The old meta-tier runner and the old bundle runner
remain under integrity restrictions and must not be used to refresh claims.

### 4. Natural case studies

- `results/corrected_v2/public_natural/natural_task_summary.csv`
- `results/corrected_v2/public_natural/natural_statistics.json`
- `results/corrected_v2/public_natural/public_natural_provenance_manifest.json`

These five tasks broaden context but do not define a sampling frame. The local
Lending Club mirror remains non-redistributable until its source and license are
verified.

### 5. Paper and reproducibility outputs

- `paper/edbt_eab/generated/paper_artifact_manifest.json` binds all current
  figures, tables, macros, palette values, and source hashes.
- `paper/edbt_eab/output/official/main.pdf` is the official-template A4 draft:
  ten pages total, with the body ending on page 10 before the references.
- `release/LeakBench-Tab-EDBT-2027-source.zip` is the compact, independently
  compilable LaTeX source package.

## Disposition: Do Not Cite or Promote

The following assets may remain for provenance, but they are not current EDBT
paper evidence:

- `artifacts/edbt_eab/claim_evidence_matrix.csv`: stale claim intervals/statuses.
- `artifacts/edbt_eab/mechanism_contract_matrix.csv`: incomplete, covering only
  seven of eleven mechanisms.
- `reports/edbt_eab/provenance_inventory.json` and its duplicate under
  `artifacts/edbt_eab/`: pre-build snapshot that still labels now-present
  canonical outputs as missing.
- `results/corrected_v2/*pilot*`: pilot or smoke evidence.
- `results/corrected_v2/superseded_snapshots/` and the 23 entries governed by
  `results/corrected_v2/superseded_evidence.json`.
- `artifacts/sp8/governance_clean_LABEL_METADATA_INVALID.csv`, legacy
  governance ledgers, and excluded smoke results.

## Current Readiness Verdict

**Conditionally ready, but not submit-ready.** The evidence, paper-facing CSVs,
official EDBT 2027 A4 template, title prefix, manuscript build, and EDBT-only
draft artifact have passed their current checks. The only submission-blocking
paper edit is author front matter:

1. Replace `Author Names Pending`, affiliation, city, country, and email.
2. Confirm every author has a CMT account and ORCID, as required by the call.
3. After author insertion, rebuild the PDF and promote the draft artifact only
   after producing a fresh EDBT-specific final manifest.

`release/leakbench_artifact_edbt_draft.zip` is the evidence review package and
`release/LeakBench-Tab-EDBT-2027-source.zip` is the compact LaTeX source package.
Both are EDBT-only. The evidence package excludes pilots, superseded snapshots,
excluded smoke runs, and the raw Lending Club mirror.

The architecture figure is intentionally simple and can be replaced later; it
is not a submission blocker.

## Reproduction Commands

```bash
# Fail-closed check of the three governed paper CSVs
python paper/edbt_eab/source_data/build_paper_assets.py --check

# Regenerate derived paper artifacts
python paper/edbt_eab/source_data/generate_paper_artifacts.py

# Compile the EDBT draft
cd paper/edbt_eab
tectonic -X compile main.tex --outdir output/official

# Build the explicitly non-final EDBT-only artifact
cd ../..
python paper/edbt_eab/source_data/build_edbt_draft_package.py

# Build the compact, standalone LaTeX source ZIP
python paper/edbt_eab/source_data/build_edbt_source_package.py

# Repository tests
python -m pytest -q
```

`generate_paper_artifacts.py` regenerates files immediately; it does not provide
a read-only `--check` or `--help` mode.

## Ownership Rules

- Change claims through the canonical claim builder, not by editing prose-only
  summaries.
- Change numeric paper inputs through `build_paper_assets.py`, not generated
  LaTeX tables or figures.
- Preserve frozen protocols, manifests, and superseded-evidence records.
- Rebuild and revalidate the EDBT release package after manuscript/front-matter
  finalization.
- Treat `CURRENT_STATUS.md`, `HANDOFF.md`, and this page as navigation; machine
  manifests remain authoritative when prose and hashes disagree.
