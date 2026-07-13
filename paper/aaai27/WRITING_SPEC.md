# AAAI-27 corrected-v2 writing specification

- Track: AAAI-27 Main Technical Track.
- Paper type: benchmark/measurement paper with critical empirical analysis.
- Title: *LeakBench-Tab: Separating Contamination Validity, Statistical
  Detectability, and Model Exploitability in Tabular Learning*.
- Central thesis: prediction-time validity is semantic; detectability and
  exploitability are distinct, diagnostic- and model-conditional empirical
  measurements.

## Research questions

1. Which mechanisms can oracle-blind, pre-test development diagnostics locate,
   and how diagnostic- and representation-conditional is that conclusion? MI,
   absolute correlation, and LR coefficients use training rows only; RF
   permutation is evaluated on frozen validation labels.
2. Which mechanisms are exploited by the five frozen model families?
3. Which C/D/X profiles persist across strengths, model families, and five
   explicitly bounded real-data case studies?

## Contribution scope

1. Benchmark-level operationalization of C/D/X; do not claim invention of
   leakage legitimacy, the no-time-machine principle, or downstream
   exploitation as a concept (Magar and Schwartz, ACL 2022).
2. Twenty controlled panel tasks, eleven construction-audited mechanisms, five
   strengths, five core models, and five injection seeds: 27,500 paired
   model-training cells.
3. A separate 22,000-cell, four-ranker diagnostic sensitivity matrix on the
   same immutable tasks.  Mutual information is primary; absolute correlation,
   LR coefficient magnitude, and RF permutation importance are sensitivity
   analyses.  Never present the best method per mechanism as a deployable
   selector.
4. Hierarchical, multiplicity-aware analysis of category contrasts plus
   descriptive crossed profiles, structured counterexamples, and
   model/diagnostic sensitivity.
   Category p-values are exact task-level sign flips with Holm correction; D--X
   intervals jointly resample both axes.  M09 source-category reweighting is
   descriptive only.
5. Five fixed real-data case studies as boundary-specific external validity,
   not population inference or zero-shot transfer.

## Evidence and wording rules

- `results/corrected_v2/paper_claims.json` is the only numerical source.
- Do not transcribe pilot values or draw confirmatory wording from pilot runs.
- The 27,500 model cells and 22,000 diagnostic cells are distinct units; do not
  sum them into a marketing number.
- Controlled tasks are synthetic panel tasks, not twenty real datasets.
- D must always identify the diagnostic.  Headline profiles use primary MI.
- D--X regression/correlation remains descriptive even if nominally
  significant; eleven designed mechanisms do not establish zero-shot transfer.
- Never use the superseded bootstrap-tail category p-values or independent
  per-cell or seed-independent cluster intervals.  M08 draws must be
  synchronized across all seeds, models, and strengths within a dataset;
  M09's eight designed categories are not a source population.
- Only simple-versus-structured may receive `SUPPORTED`/`NOT_SUPPORTED`, using
  exact Holm-adjusted p <= 0.05 and CI low > 0.  M03, M08, M09, D--X,
  diagnostic-method, and model-specific summaries remain descriptive.
- Natural evidence must remain `CASE_STUDY_ONLY`.
- `NOT_SUPPORTED` is reportable only for the directional category claim.
  `PENDING`, `REFUTED`,
  `INTEGRITY_HOLD`, and incomplete claims cannot enter the paper.

## Main-text prohibitions

Do not include internal phase numbers, readiness scores, gates, claim IDs,
agent language, full ledgers, validator pass counts, credentials, or incident
details.  Do not describe the work as a new detector, new governance algorithm,
complete real-world validation, a uniform eight-model benchmark, a metadata
transfer study, or proof that governance does or does not work.  Metadata and
governance remain outside the corrected-v2 main claim set.
