# EDBT-Only Cleanup and Source Package Report

## Outcome

The repository now has one active manuscript tree: `paper/edbt_eab/`. The
venue-obsolete paper directory, template, rendered outputs, release package,
builders, validators, and coupled tests were removed. Corrected-v2, SP8, natural
case, and frozen protocol evidence were preserved.

## Standalone Source Package

- ZIP: `release/LeakBench-Tab-EDBT-2027-source.zip`
- SHA-256: `c8af9b73def6034496c3ef32f7db1a71bc1fdb988ce6c443d2630f16547cbe96`
- Entrypoint: `main.tex`
- Contents: 16 files including `SOURCE_MANIFEST.json`
- Compile command: `tectonic -X compile main.tex --outdir output`

The package contains the official EDBT template files, bibliography, generated
tables and macros, three figure PDFs, and the local Libertinus Math font plus
its OFL license. It has no repository-relative compilation dependency.

## Verification

The ZIP was extracted into `/tmp/leakbench-edbt-source-verify`, then compiled
from that extracted directory. The result is a ten-page A4 PDF titled
`[EA&B] When Blind Feature Removal Mitigates Tabular Leakage`.

Repository verification after deletion: `221 passed` with
`python -m pytest -q`.

## Remaining Human Blocker

The source still contains placeholder author, affiliation, location, and email
fields. Author metadata and ORCID confirmation remain required before final
submission packaging.
