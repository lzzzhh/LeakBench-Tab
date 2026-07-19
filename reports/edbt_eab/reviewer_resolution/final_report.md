# Reviewer-risk resolution report

## Verdict

The six substantive reviewer concerns are resolved as far as the frozen
evidence permits. The paper now presents CDXR as a role-separated, falsifiable
evaluation architecture rather than as a renamed taxonomy, and it treats MI as
one conventional test policy rather than the contribution. No claim was
strengthened beyond its evidence tier.

## Resolution by concern

| Concern | Resolution | Remaining boundary |
|---|---|---|
| Architecture novelty | Added a minimum information-contract schema, selector/evaluator/verifier interfaces, policy access control, a claim certificate, and INVALID/SUPPORTED/UNSUPPORTED/NARROWED/MIXED gates. Lineage-, time-, and group-aware policies plug into the same selector interface without receiving the oracle mask. | The current instantiation tests MI; CDXR does not claim a new detector. |
| Small positive margin | The primary claim is explicitly LR under the 20% encoded-column contract: +0.043, 95% task-cluster interval [+0.004,+0.078]. Semantic-group recomposition is reported as crossing zero. | No representation-invariant or generally reliable repair claim. |
| Sparse failure | A hash-verified post-hoc reconstruction matches 1,100/1,100 P3 masks. The effect is negative in 4/4 tasks and 9/11 mechanisms; P3 removes 1.92 of three clean signal fields on average, including `x_000` in 89.3% and `x_003` in 84.9% of keys. | This is a construction-grounded diagnosis, not a randomized causal decomposition. |
| NYC311 failure | All three P3 hashes match. With opportunity 0.019 and budget 8/40, P3 removes one of two invalid fields plus seven contract-valid fields, yielding recall 50.0%, retention 81.6%, and repair advantage -0.108. | One fixed case; contract-valid does not mean deployment-optimal. |
| Natural generalization | The five cases are consistently labeled selected fixed-case audits with mixed governance evidence. | No population or external-generalization claim. |
| RF/LightGBM amendment | The exact chronology and implementations are disclosed: post-LR design freeze, RF outcome, failed GPU attempt, CPU LightGBM before any completed LightGBM outcome, and later discovery of separate 100-estimator baseline refits. | RF/LightGBM are corroborative within-run contrasts, not baseline-continuity, equivalence, or learner-invariance evidence. |

## New evidence boundary

The failure-anatomy analysis performs zero downstream model fits. It is stored
under `results/edbt_eab_revision/failure_anatomy/` with a separate manifest,
status `POST_HOC_DESCRIPTIVE_DIAGNOSTIC`, and no authority to promote the
confirmatory claim state. The paper asset builder verifies this manifest and
incorporates the diagnostic fields into the existing governance asset, keeping
the paper-facing table count at three.

## Verification

- Failure selections: 1,100 sparse and 3 NYC311 hashes matched.
- Paper assets: `EDBT_EAB_PAPER_ASSETS_CURRENT`, three tables.
- Tests: 237 passed, 0 failed.
- Static checks: Python compilation and `git diff --check` passed.
- PDF: 12 A4 pages, visually inspected page by page; no clipping, overlap, or
  unreadable table/figure was observed.
- Complete package: 50 manifest-bound files verified and compiled from a fresh
  extraction to a 12-page PDF.
- PDF SHA-256: `1bacb4b8a3c9bf6b5a77fc192a88a61123423b092d95954eb025a133901b7dc0`.
- Package SHA-256: `5e9e0fef44bea47c83273369eae9f2027db6d9c96c95de0496d41758fbd44c06`.

## Human-only blocker

The author, affiliation, city, country, email, and required author identifiers
remain placeholders. They must be supplied before submission; no evidence or
code change can resolve that metadata.
