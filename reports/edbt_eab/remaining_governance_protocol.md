# EDBT Remaining Governance Protocol

Status: prospective protocol, frozen before execution on 2026-07-18.

## Natural-case governance

The five existing boundary-specific public case studies are retained without changing raw files, splits, labels, leakage ground truth, or train-fitted preprocessing. Logistic regression is evaluated at the 20% retained-feature budget. P3 ranks features by mutual information computed on training rows only. P2 removes the same number of retained features under 20 deterministic governance seeds. The estimand is P3 strict-distance reduction minus the within-key mean P2 strict-distance reduction.

Three frozen training seeds are retained to align with the existing natural harm audit. Strict and full baselines are reused only after matching task, seed, source hash, and preprocessing hash. Results are summarized per case; the five cases are not treated as an iid sample from a dataset population. A case-level bootstrap and exact sign-flip calculation are reported only as descriptive sensitivity summaries.

## Semantic-group budget

The sensitivity changes the cost unit from encoded columns to semantic feature groups. The frozen synthetic registry uses one-to-one semantic/encoded mappings except M09, where one source field expands to eight complete one-hot columns. Consequently, only M09 requires refitting; the full 5,500-key panel is recomposed by replacing the 500 encoded-cost M09 effects with semantic-cost M09 effects and retaining the 5,000 identity-mapped keys.

For M09, the 12 original columns remain singleton groups and the eight injected one-hot columns form one source-field group. A group is scored by the maximum train-side MI of its columns, preventing group width from mechanically increasing its score. P2 samples semantic groups uniformly without replacement. P2 and P3 remove the same number of groups, with `max(1, round(0.20 * n_groups))`.

## Integrity rules

- All input and code hashes are bound in `remaining_governance_protocol_freeze.json`.
- Any failed, duplicate, missing, or hash-mismatched cell blocks claim derivation.
- Frozen SP5--SP8 outputs are read-only.
- Outcomes do not alter task inclusion, group definitions, budgets, seeds, or estimators.

