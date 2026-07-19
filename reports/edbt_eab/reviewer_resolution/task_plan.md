# Task Plan: Resolve CDXR reviewer risks

## Goal

Turn the six reviewer concerns into an evidence-bound manuscript revision that
strengthens CDXR as an evaluation architecture, explains the two negative
regimes, and preserves the frozen scope of every empirical claim.

## Phases

- [x] Phase 1: Lock scope, inspect current manuscript, evidence, and provenance.
- [x] Phase 2: Build post-hoc sparse and NYC311 failure-anatomy artifacts.
- [x] Phase 3: Formalize CDXR interfaces, access rules, claim certificate, and falsification gates.
- [x] Phase 4: Revise failure analysis, natural-case explanation, amendment timeline, and claim wording.
- [x] Phase 5: Regenerate paper assets, compile, visually inspect, and run claim/statistical QA.
- [x] Phase 6: Rebuild and independently compile the complete paper package.

## Key Questions

1. Can every new numerical statement be reconstructed from frozen inputs?
2. Does the architecture define inputs, policy-visible information, interfaces,
   output certificate, and invalid/unsupported/narrowed conditions?
3. Is the sparse explanation diagnostic rather than falsely causal?
4. Is the B2 timeline explicit that the extension was frozen after LR but before
   RF/LightGBM outcomes, with the baseline-refit deviation discovered later?

## Decisions Made

- No new downstream model training: the reviewer risks can be resolved with
  deterministic selection reconstruction and existing governed outputs.
- New failure-anatomy outputs are post-hoc descriptive evidence in a separate
  manifest; they do not overwrite the confirmatory revision manifest.
- Keep the title term `architecture` only if the manuscript exposes a concrete
  contract schema, role-separated interfaces, and a machine-checkable claim certificate.
- Retain three compact paper-facing CSV assets; reviewer diagnostics will be
  incorporated into the existing governance asset rather than creating a fourth core table.

## Errors Encountered

- Initial sparse B1 row expectation counted P0 inside the 20% slice; P0 is
  recorded at budget 0. Corrected the expected slice from 22 to 21 rows per key.
- The first independent-package check invoked Tectonic from the repository root;
  the second created the extraction-local output directory and compiled from the
  package root under `set -euo pipefail`.

## Status

**Complete.** The manuscript, post-hoc diagnostic manifest, paper assets, PDF,
tests, and independently compiled complete package all pass their declared gates.
