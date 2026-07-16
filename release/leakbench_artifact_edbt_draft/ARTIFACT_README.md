# LeakBench-Tab EDBT 2027 draft artifact

Status: `EDBT_DRAFT` -- not a final submission package.

This package deliberately contains only the EDBT manuscript, the three compact
paper-facing CSVs, and their allowlisted evidence chain. It excludes pilots,
superseded snapshots, excluded smoke runs, and the non-redistributable Lending
Club source file.

The remaining submission blocker is author front matter: replace the pending
author, affiliation, email, and ORCID fields before creating a final package.

From the repository root, verify governed paper inputs with:

```bash
python paper/edbt_eab/source_data/build_paper_assets.py --check
```

Compile from `paper/edbt_eab/` with:

```bash
tectonic -X compile main.tex --outdir output/official
```
