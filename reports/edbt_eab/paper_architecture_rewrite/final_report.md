# Final report

## Outcome

The manuscript is now method-first rather than benchmark-first. Its title is
**[EA&B] CDXR: A Contract-Grounded Evaluation Architecture for Tabular Leakage
Repair**. CDXR separates construction validity, blind detectability,
learner-conditional exploitability, and matched-cost repair response;
LeakBench-Tab is the controlled instantiation used to evaluate those objects.

## Material changes

- Rebuilt the abstract, introduction, method, results, related work,
  discussion, limitations, and conclusion around CDXR.
- Formalized repair opportunity, the repair-response vector, matched-cost
  repair advantage, and a general intervention cost map.
- Added a minimum information-contract schema, selector/evaluator/verifier
  interfaces, policy access rules, a claim certificate, and explicit claim
  status/falsification gates.
- Made semantic precedence, oracle isolation, matched cost, and paired
  reference evaluation explicit architecture invariants.
- Replaced stale SP8 paper values with the final 709,500-row governance
  revision evidence across LR, RF, and LightGBM.
- Limited the main paper to three evidence tables and four figures, including
  the intentionally simple CDXR architecture diagram.
- Added data-management positioning against production ML data validation and
  data-quality verification systems.
- Preserved all negative and narrowing evidence: sparse archetype failure,
  mixed natural cases, semantic-cost aggregate uncertainty, and the disclosed
  cross-learner baseline deviation.
- Added separately manifested, hash-verified sparse and NYC311 failure anatomy
  without refitting a downstream model or changing confirmatory claim state.

## Verification

- Paper-facing asset builder: current and manifest-bound.
- Tests: 237 passed.
- LaTeX: compiles without errors or undefined references.
- Layout: 12 A4 review pages; all pages were rendered and inspected, with no
  visible clipping, overlap, or unreadable figure/table.
- Claim scan: no unsupported positive generalization found.
- Complete package: 50 files independently compiled and hash-verified; the final archive
  is `release/CDXR-EDBT-2027-complete.zip` with SHA-256
  `5e9e0fef44bea47c83273369eae9f2027db6d9c96c95de0496d41758fbd44c06`.

## Remaining author action

Replace the front-matter author, affiliation, city, country, email, and ORCID
placeholders before submission. The architecture diagram can optionally be
replaced with a more elaborate author-drawn version without changing the story.
