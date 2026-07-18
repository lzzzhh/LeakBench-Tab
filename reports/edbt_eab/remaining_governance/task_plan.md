# Remaining Governance Experiments

## Objective

Complete the two disclosed EDBT governance sensitivities without changing frozen evidence: natural-case matched-cost governance and semantic-group budget governance.

## Plan

- [x] Inventory local natural-case data, labels, boundary fields, and frozen provenance.
- [x] Inventory encoded-column-to-semantic-group mappings and establish whether they are reconstructible without guesswork.
- [x] Freeze prospective protocols, estimands, exclusions, seeds, and failure conditions.
- [x] Implement deterministic runners and focused tests.
- [x] Run both experiment families and validate completeness, duplicates, hashes, and provenance.
- [x] Perform task-clustered/descriptive analyses appropriate to each evidence tier.
- [x] Update revision claim state, manifest, paper-facing base tables, and documentation.
- [x] Run the full validation suite and record the final git state.

## Guardrails

- Frozen SP5--SP8 evidence remains immutable.
- Natural cases are case studies, not a population sample.
- Semantic groups must come from auditable source/encoder metadata; no inferred grouping from names alone unless the encoder contract proves it.
- P2 and P3 always use the same cost unit within a comparison.
- Zero-opportunity cells are retained and reported, not silently filtered.
