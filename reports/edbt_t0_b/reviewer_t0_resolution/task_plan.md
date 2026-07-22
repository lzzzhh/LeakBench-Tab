# Task Plan: Resolve Reviewer T0 Issues RC1, RC3, and RC2

## Goal
Produce an auditable evidence chain that resolves repair construct validity,
random-baseline robustness, and cross-learner repair generalization, then update
the EDBT manuscript and reproducible source package without overstating results.

## Acceptance Phases
- [x] Phase 1: Inventory the current RC1/RC3/RC2 evidence and identify blocking contract gaps.
- [ ] Phase 2: Close the Full-B1 production plan, runner, shard-admission, merge, and physical-CSV contracts.
- [ ] Phase 3: Add fail-closed negative tests, bind validation receipts to the implementation, and pass a production canary.
- [ ] Phase 4: Execute, merge, freeze, and analyze RC3 Full-B1.
- [ ] Phase 5: Execute and analyze RC2 RF/LightGBM confirmation.
- [ ] Phase 6: Synthesize RC1 construct-validity evidence and update the manuscript/package.
- [ ] Phase 7: Audit every reviewer requirement against authoritative artifacts.

## Reviewer Acceptance Questions
1. Does the repair estimand track removable leakage distortion rather than deletion volume, noise, or threshold artifacts?
2. Do learned policies outperform a 20-seed matched-random policy across budgets, mechanisms, and cost contracts?
3. Are effect sizes and policy rankings stable across LR, RF, and LightGBM, with interactions reported rather than hidden?
4. Can plan corruption, shard corruption, recovery, or merging ever publish an invalid scientific result?

## Decisions Made
- RC3 infrastructure is a hard gate before any Full-B1 outcome is generated.
- The production builder output itself, not only a synthetic fixture, must pass the same schema consumed by runner/admission/merge.
- `--synthetic` is an explicit mode boundary; absence means production.
- Header-only CSV is exactly `header\n`; `header\n\n` is an invalid blank physical row.
- Source-aggregate verification must be genuinely streaming for both source and candidate ledgers.

## Errors Encountered
- Current formal Full-B1 manifest fails the R10c schema because the production builder still emits the legacy format.
- Runner `--validate-only` reports PASS for that same inadmissible manifest because it bypasses the shared plan gate.
- R10c source-aggregate validation still materializes decompressed ledgers despite being labelled streaming.
- Current committed test receipt predates R10c and is not bound to HEAD.
- Strict-mode regression initially exposed three legacy tests that invoked a
  synthetic validate-only plan without `--synthetic`; the tests were corrected
  to exercise the now-explicit mode contract.
- Enforcing the physical CSV rule exposed that the shard producer itself wrote
  empty failure ledgers as `run_id\n\n`; the producer was corrected to emit the
  canonical `run_id\n` representation.

## Status
**Currently in Phase 2** — repairing the shared production contract before any scientific execution.
