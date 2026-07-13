# Corrected-v2 paper-number contract

This file is a human-readable contract, not a numerical input.  Every empirical
number in LaTeX must be generated from
`results/corrected_v2/paper_claims.json` by
`source_data/generate_result_macros.py`.

## Frozen design dimensions

| Evidence unit | Frozen design | Count | May be combined? |
|---|---:|---:|---|
| Paired core model-training cells | 20 tasks x 11 mechanisms x 5 strengths x 5 models x 5 seeds | 27,500 | No |
| Blind diagnostic cells | 20 tasks x 11 mechanisms x 5 strengths x 5 seeds x 4 rankers | 22,000 | No |
| Fixed real-data case studies | Five declared task boundaries; model/seed coverage comes from final JSON | 5 cases | No; case-study only |

The four diagnostics are mutual information (frozen primary), absolute
correlation, logistic-regression coefficient magnitude, and random-forest
permutation importance.  All are oracle-blind and pre-test; the first three use
training rows only, while the forest is fitted on training rows and permutation
importance is evaluated on frozen validation labels.  Diagnostic rows are not
model-training cells.

The paper-facing diagnostic input is
`results/corrected_v2/diagnostic_canonical_cells.csv`.  A scope-locked,
post-unblinding provenance amendment replaces every and only the 5,500 raw MI
rows with fixed-seed-42 metrics already present in the frozen task manifest;
the other 16,500 rows and the four-method set are preserved.  The raw
task-seeded MI table is provenance, not final evidence.

The paper-facing category tests come from
`statistics/category_contrasts_amended.csv`: exact two-sided sign flips over 20
task effects, with Holm correction over the three declared contrasts.  The
D--X interval in `correlation_analysis_amended.json` jointly resamples both
axes.  M08 uses one synchronized entity draw across all five seeds, models, and
strengths within each dataset, retaining seed-specific effects; M09's
synchronized source-category interval is descriptive designed-category
reweighting only and cannot support the counterexample claim.  The superseded
bootstrap-tail p-values, independent-per-cell cluster intervals, and first
seed-independent M08 amendment are barred.  Only the pre-existing directional
simple-versus-structured claim has a support rule: exact Holm-adjusted
`p <= 0.05` and `CI low > 0`.  M03, M08, M09, D--X, diagnostic-method, and
model-specific summaries remain descriptive.

## Required final release blocks

- `protocol_integrity`: exact complete 27,500-cell confirmatory matrix and the
  five model identities (`lr`, `rf`, `catboost`, `lightgbm`, `tabm`).
- `claims.simple_vs_structured`: effect, hierarchical interval, exact
  task-sign-flip Holm-adjusted p-value, and model-direction counts.
- `claims.m03_profile`, `claims.m08_profile`, and
  `claims.m09_counterexample`: primary-MI D and paired X with intervals.
- `claims.detectability_exploitability_relation`: descriptive global,
  category-adjusted, permutation, and LOMO diagnostics.
- `claims.D_METHOD_CONDITIONAL`: descriptive-only four-ranker sensitivity;
  without paired simultaneous method-comparison intervals it cannot authorize
  an inferential ranking or best-method selection.
- `diagnostic_sensitivity`: complete 22,000-cell integrity block and M03/M04/M05
  summaries for all four rankers.
- `natural`: exactly five fixed case studies, status `CASE_STUDY_ONLY`.
- `provenance`: non-empty canonical source/checksum information.

## Superseded numerical sources

The following must not source the manuscript: the 10,083-cell legacy Core
ledger, 4,032-row Meta ledger, their 14,115 total, eight-model summaries,
surrogate neural outputs, old metadata/governance gains, zero-shot natural
transfer tables, the pre-bundle partial TabM run, and every pilot statistic.
They may appear only in an integrity/supersession audit outside the paper's
empirical claims.

## Release rule

If the final JSON is missing, non-confirmatory, incomplete, contains a blocked
main claim, or disagrees with any frozen dimension, the macro generator exits
nonzero.  The fallback PDF displays `RESULTS BLOCKED` and `PENDING`; it is not a
submission artifact.
