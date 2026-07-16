# Notes: AAAI Removal and EDBT Source Package

## Scope

The user explicitly authorized deletion of AAAI material. Evidence ledgers and
protocol records are retained unless they exist only to build or validate the
AAAI manuscript.

## Findings

- `paper/aaai27/` contains the historical manuscript, supplement, generated
  tables/figures, rendered PDFs, and template files.
- Seven scripts are coupled to the AAAI paper/release tree: the paper builder,
  corrected-v2 public artifact builder/verifier/CI, release validator, old
  figure generator, and SP5 paper macro generator.
- Seven tests import or inspect those venue-specific scripts and paths.
- The EDBT manuscript itself is already self-contained relative to
  `paper/edbt_eab/`; it needs three generated table files, one macro file, three
  figure PDFs, bibliography, official template files, and the local Libertinus
  math font for Tectonic.
- Scientific results and protocol records may contain historical venue labels.
  They remain evidence provenance and are not deleted or rewritten.
- Deleted the venue-obsolete manuscript tree, seven paper/release scripts, the
  mismatched release directory/ZIP, and seven coupled test modules.
- The standalone ZIP contains 16 files including its manifest. Its SHA-256 is
  `c8af9b73def6034496c3ef32f7db1a71bc1fdb988ce6c443d2630f16547cbe96`.
- A clean extraction compiled with Tectonic to a 10-page A4 PDF.
- The remaining repository suite passes: 221 tests, 0 failures.
