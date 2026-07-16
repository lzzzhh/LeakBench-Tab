# Notes: EDBT EA&B Pre-Submission Review

## Operating Mode

- Claude Scholar mode: camera-ready audit.
- Claim authority: corrected-v2 claim state plus frozen SP8 claim matrix.
- Citation authority: programmatic metadata and DOI/arXiv resolution; no BibTeX generated from memory.

## Findings

- Official 2027 rules require the exact `[EA&B]` prefix, A4 official template,
  visible authors/affiliations, and a 12-page body limit with unlimited
  references. The paper now uses those format requirements.
- Final PDF is A4 and ten pages. The paper body and conclusion occupy page 10;
  references begin on page 10. All pages were rendered and visually inspected.
- The manuscript cites 18 of 19 BibTeX entries. Every cited key exists; the only
  unused entry is `roth2026leakage`. This is nonblocking.
- Claims remain inside the corrected-v2 state and frozen SP8 matrix. The
  conclusion now distinguishes observed small-budget intervals from the
  predeclared 20% governance claim.
- Venue-obsolete paper and release assets were removed after the EDBT source
  and evidence packages were verified.
- The new `release/leakbench_artifact_edbt_draft.zip` contains only EDBT paper
  material and no pilot, superseded-snapshot, excluded-smoke, or raw Lending
  Club inputs.
- Repository tests: 283 passed, 3 skipped, 0 failed with root `pytest -q` after
  constraining discovery to the authoritative `tests/` tree.
