# EDBT EA&B paper architecture rewrite plan

## Objective

Reframe the manuscript from a benchmark report into a method-first paper built
around an evidence-backed architecture for tabular leakage governance, while
preserving the 12-page limit and every frozen empirical boundary.

## Plan

- [completed] Audit the current manuscript, generated assets, and frozen claim state.
- [completed] Define the method architecture, title, research canon, argument map, and section contracts.
- [completed] Rewrite the manuscript and synchronize paper-facing tables/macros with the final evidence.
- [completed] Compile, inspect statistics and claims, and keep the paper within 12 pages.
- [completed] Run a hostile self-review, rebuild the complete LaTeX package, and report the final state.

## Guardrails

- Do not invent an unimplemented detector, policy, or deployment claim.
- Treat semantic validity, detectability, exploitability, repair opportunity,
  and repair response as distinct objects.
- Keep natural-data governance descriptive (`MIXED`) and semantic-cost evidence
  as a boundary on the aggregate claim.
- Preserve the negative sparse-archetype result and the M09 counterexample.
- Keep no more than three main result tables; move detailed registries to the appendix/package.
