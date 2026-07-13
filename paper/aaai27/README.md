# LeakBench-Tab AAAI-27 paper workspace

This directory uses the unmodified official AAAI-27 LaTeX and bibliography
styles downloaded from the AAAI-27 Author Kit on 2026-07-13.  The style requires
PDFLaTeX; Tectonic/XeTeX is rejected by the official engine guard.

## Evidence gate

The manuscript has one numerical source:
`results/corrected_v2/paper_claims.json`.  Generate LaTeX macros only after the
complete confirmatory release exists:

```bash
python source_data/generate_result_macros.py --check-only
python source_data/generate_result_macros.py
```

The generator rejects an absent or non-confirmatory release, an incomplete
27,500-cell core matrix, an incomplete 22,000-cell diagnostic matrix, non-final
main claims, unordered/non-finite intervals, and the wrong five-model identity
set.  It never reads pilot outputs or legacy metadata/governance tables.
It also rejects the raw task-seeded-MI diagnostic table: final diagnostics must
come from the scope-locked fixed-seed-42 canonical amendment and its manifest.

From the repository root, rebuild and analyze that deterministic amendment
from the checked-in freeze:

```bash
python scripts/build_diagnostic_rng_amendment.py --overwrite
python scripts/analyze_diagnostic_suite.py \
  --input results/corrected_v2/diagnostic_canonical_cells.csv \
  --namespace confirmatory \
  --output-dir results/corrected_v2/statistics \
  --repetitions 20000 --seed 20260713 --require-complete
python scripts/analyze_diagnostic_rng_amendment_impact.py
```

Do not recreate `diagnostic_rng_amendment_freeze.json` during reproduction.
The unamended `diagnostic_confirmatory_cells.csv` remains only as source
provenance for the three preserved methods and the disclosed MI correction.

The final statistics additionally require the frozen post-audit inference
amendment.  After the complete canonical table and all 5,000 retained M08/M09
prediction files exist, run:

```bash
python scripts/analyze_corrected_v2_amendment.py \
  --canonical results/corrected_v2/canonical_cells.csv \
  --bootstrap-reps 20000 --permutation-reps 20000 --seed 20260713
python scripts/analyze_cluster_sensitivity_amendment_v2.py \
  --canonical results/corrected_v2/canonical_cells.csv \
  --task-manifest results/corrected_v2/task_bundles/task_manifest.csv \
  --prediction-dirs results/corrected_v2/predictions \
    results/corrected_v2/tabm_bundle_confirmatory/tabm_cells_predictions \
  --inner-reps 200 --outer-reps 5000 --seed 20260713
```

Do not recreate `statistical_amendment_protocol_v2_freeze.json`.  The amended
manifests bind code, canonical/config hashes, all consumed prediction hashes,
all twenty task bundles, direct task-array lineage, and output hashes.  The M08
entity draw is shared across all seeds, models, and strengths within a dataset.
The release validator deep-reruns every paper-facing statistical analysis.

Natural-case claims are accepted only from
`natural_protocol_v2_freeze.json` / `natural_trainfit_categories_v2`.  The v1
full-table categorical-vocabulary outputs are superseded; public artifacts
contain path-redacted lineage copies rather than local usernames or source
locations.

Without `generated/result_macros.tex`, both PDFs still compile for structural
review, but display an explicit **RESULTS BLOCKED** banner and render empirical
values as `PENDING`.  Such a PDF is not submission-ready.

## Build

The supported build is fail closed and runs from the repository root.  First
generate all paper-facing artifacts from the final claim release:

```bash
python paper/aaai27/source_data/generate_result_macros.py --check-only
python paper/aaai27/source_data/generate_result_macros.py
python paper/aaai27/source_data/generate_result_tables.py
python scripts/generate_corrected_v2_figures.py
```

Then build the pinned TeX image and run the attested two-build check:

```bash
python scripts/build_aaai27_paper.py --build-image
```

The Dockerfile pins the `texlive/texlive:latest-small` base image by digest;
the build manifest records both that digest and the resulting local image ID.
The absence of host PDFLaTeX is not a release exception: use this pinned
container build instead.

If Poppler is not on `PATH`, pass `--poppler-bin /path/to/poppler/bin`.
The builder requires byte-identical independent PDFs, all fonts embedded, no
Type 3 fonts, no undefined citations/references or overfull boxes, anonymous
text, no private paths or blocked-result markers, at most seven main-content
pages and at most nine total main-PDF pages.  Only after every check passes does
it replace `output/pdf/main.pdf`, `output/pdf/supplement.pdf`, and write
`output/pdf/paper_build_manifest.json`.

For non-numerical layout work, `python scripts/build_aaai27_paper.py
--build-image --draft-only` writes only under `tmp/pdfs/` and can never create a
submission-ready manifest.

`main.tex` is the anonymous Main Technical Track manuscript.  The controlled
model matrix (27,500 cells), diagnostic sensitivity matrix (22,000 cells), and
five fixed real-data case studies are distinct evidence units and must never be
summed into a headline “experiment count.”

The checked-in files currently under `output/pdf/` predate this corrected
scaffold and are never submission artifacts.  Packaging excludes the entire
directory unless fresh PDFs are explicitly admitted by a validated paper-build
manifest after final evidence macros and all three data figures are generated.
