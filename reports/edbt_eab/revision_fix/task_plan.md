# Task Plan: EDBT governance revision repair

## Goal

Produce a reproducible, 20%-budget governance analysis with paired inference, derived claims, protocol amendment, bound provenance, and dedicated tests without rerunning model fits.

## Phases

- [x] Phase 1: Inspect current analysis, protocol, manifests, and test patterns
- [x] Phase 2: Repair analysis semantics and add regression tests
- [x] Phase 3: Regenerate tables, summary, claim state, and manifest
- [x] Phase 4: Verify hashes, statistics, tests, and residual blockers

## Key Questions

1. Are all three learner comparisons restricted to the same 5,500 keys at the 20% budget?
2. Are bootstrap draws paired at the task level and are observed and bootstrap means separated?
3. Can selection-mask hashes be reconstructed without model retraining?
4. Does the evidence package satisfy the frozen protocol after an explicit amendment?

## Decisions Made

- Do not rerun RF or LightGBM model fits.
- Treat 20% as the primary matched cross-learner budget; retain the four-budget LR curve only as a separate descriptive analysis.
- Derive claims and manifest from validated artifacts rather than editing them manually.

## Errors Encountered

- None yet.

## Status

**Complete** - all repair phases passed; remaining limitations are explicitly recorded as not run.
