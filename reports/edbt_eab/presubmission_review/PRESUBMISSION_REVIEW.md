# EDBT 2027 EA&B Pre-Submission Review

**Verdict:** conditional pass; the paper and evidence package are technically
ready for author completion, but the manuscript must not be submitted with the
current placeholder front matter.

**Audit date:** 2026-07-17

## Submission Boundary

The review follows the official EDBT/ICDT 2027 call and template. The source now
uses the required `[EA&B]` title prefix, the official A4 geometry and conference
macros, and the non-anonymous author layout. The compiled PDF is ten A4 pages;
the body is below the 12-page limit and references begin on page 10.

- Official conference site: <https://edbticdt2027.github.io/>
- Official call: <https://edbticdt2027.github.io/?contents=EDBT_CFP.html>

## Gate Results

| Gate | Result | Evidence |
|---|---|---|
| Official template and prefix | PASS | A4 `acmart`; title begins `[EA&B]`; official template hashes recorded in the paper README |
| Page limit | PASS | 10 pages total; body below 12 pages |
| Front matter | BLOCKED | author, affiliation, city, country, and email are placeholders; author CMT/ORCID status requires human confirmation |
| Numeric assets | PASS | three governed paper CSVs; `build_paper_assets.py --check` returned current |
| Claim scope | PASS | central statements stay within corrected-v2 claims and frozen SP8 G1/G3/G4; G2 remains inconclusive |
| Citations | PASS | 18 cited keys, no missing BibTeX entries; one unused entry is nonblocking |
| Figures and tables | PASS | four figures and three compact paper tables; generated outputs are manifest-bound |
| Visual PDF review | PASS | all ten pages rendered; no clipped figures, tables, equations, or references observed |
| EDBT draft artifact | PASS | explicit manifest; excludes pilots, superseded snapshots, smoke outputs, and raw Lending Club data |
| Repository tests | PASS | 283 passed, 3 skipped, 0 failed with root `python -m pytest -q` |

## Claim Audit

The paper's primary controlled result is the prespecified simple-minus-
structured harm contrast. The governance claim remains LR-only and is anchored
to the predeclared 20% P3-versus-P2 contrast. Small-budget intervals and the 1%
recall/retention vector are reported as observed registry results, not promoted
to learner-independent or population claims. Natural cases remain descriptive
case studies. The structured interval crossing zero is worded as “no reliable
advantage,” not equivalence.

One conclusion sentence was tightened during this audit: it now distinguishes
positive observed intervals at all nominal budgets from support for the
predeclared 20% claim. No frozen result, protocol, claim state, or governance
asset was modified.

## Artifact Audit

The replacement review package is
`release/leakbench_artifact_edbt_draft.zip`. Its manifest status is
`EDBT_DRAFT`. It contains the EDBT manuscript and PDF, official template files,
the three governed paper CSVs and generators, the canonical measurement ledger,
the clean SP8 governance ledger and bootstrap/claim matrix, compact natural-case
evidence, and only the code needed to audit those bindings.

ZIP SHA-256:
`cdc5fcf39f1961ca106d96e4e0c67f1615bfb32b372f4f21caf127949824221d`.

The compact compilation package is
`release/LeakBench-Tab-EDBT-2027-source.zip`, SHA-256
`c8af9b73def6034496c3ef32f7db1a71bc1fdb988ce6c443d2630f16547cbe96`.

## Nonblocking Follow-Ups

- Replace the simple architecture figure later if desired; the current figure
  is readable and deliberately bounded.
- Remove the unused `roth2026leakage` BibTeX entry if bibliography hygiene is
  desired. It does not appear in the rendered references.
- Run one final official pdfLaTeX/Overleaf build after author insertion. The
  current Tectonic build succeeds but emits ordinary box warnings; visual review
  found no clipping.

## Required Human Action

Before submission, provide the final author order, affiliation(s), city,
country, email address(es), and confirm a CMT account plus ORCID for every
author. Then rebuild the PDF and generate a new package labeled `final`; do not
rename the current draft ZIP.
