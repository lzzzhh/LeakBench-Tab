# StructuredPriorV1Injector — M08 Semantic Audit

**Purpose:** Document the exact semantics of the corrected M08 (entity leakage)
mechanism before the frozen protocol is executed, and record which invariant
tests certify each property.

**Source:** `src/leakbench/mechanisms/structured_prior_v1.py` (SHA256 `533cc542…37d8cc`, frozen)
**Amendment version:** `structured_constant_prior_v1`
**Constant prior:** `0.5`

## What M08 computes

For each row, M08 produces a single injected feature
(`contam_M08_future_entity_rate`) equal to:

```
signal = 0.5 + strength * (future_rate − 0.5) + Normal(0, noise_std)
```

where `future_rate` is a **strictly-future, same-entity, constant-shrinkage
mean of the label**:

1. Rows are grouped by `entity_id`.
2. Within an entity, rows are ordered by timestamp.
3. For row *r*, only labels of same-entity rows at **strictly later**
   timestamps are eligible (`searchsorted(..., side="right")` excludes the
   current row **and** any row sharing its timestamp).
4. The eligible future labels are averaged with a fixed pseudo-count shrinkage
   toward the constant prior 0.5:
   `future_rate = (Σ future_y + prior_weight·0.5) / (count + prior_weight)`.
5. If a row has **no** eligible strict future (e.g. the last row of an entity),
   `future_rate = 0.5` exactly (the constant prior), and `count = 0`.

## Invariant checklist

| Property | How enforced | Certifying test |
|----------|--------------|-----------------|
| Entity structure actually drives the feature | future_rate computed per entity from that entity's future labels | `test_m08_uses_only_future_same_entity_labels_and_constant_shrinkage` |
| Frozen 0.5 prior is used (not a data-derived prior) | `CONSTANT_PRIOR = 0.5` hard-coded in shrinkage + fallback | `test_no_eligible_future_has_noiseless_constant_value`; `test_m08_strength_zero_collapses_signal_to_constant_prior` |
| No self-label access | strict-future window excludes current row | `test_flipping_current_label_cannot_change_that_rows_feature` |
| No same-timestamp label access | `side="right"` excludes ties | `test_same_timestamp_labels_are_excluded` |
| No full-table target mean | only strictly-later same-entity labels used | (construction; covered by the two exclusion tests) |
| Test-split labels do not corrupt train-row features | features are a pure function of (timestamp order, entity, future labels); split is chronological 60/20/20 and frozen; each row's feature depends only on **later** rows regardless of split membership | `test_flipping_current_label_cannot_change_that_rows_feature` (any single-row label flip cannot change that row's own feature); protocol strict view removes the leaked column entirely |
| Unseen / no-future entity row | falls back to constant prior 0.5 | `test_m08_unseen_future_entity_row_uses_constant_prior` |
| Strength changes signal amplitude | linear multiplier on deviation from prior | `test_m08_strength_scales_deviation_from_prior_deterministically` |
| Mask marks exactly the injected field(s) | `n_leak == 1`, original columns all legitimate | `test_m08_mask_marks_exactly_the_injected_field` |
| Reproducible by seed | deterministic given seed | `test_m08_reproducible_by_seed`; `test_structured_amendment_is_deterministic` |

## Clarification: "constant 0.5 prior" ≠ "feature is always 0.5"

The constant prior is the **shrinkage target and empty-window fallback**, not
the feature value. When an entity has observable future labels, the feature
deviates from 0.5 toward that entity's future outcome rate, scaled by
`strength`. Only rows with zero eligible future (and `strength=0`) produce
exactly 0.5. This is what makes M08 a genuine — but outcome-independent-prior —
structured leak: it can leak an entity's *later* label distribution without ever
reading the current row's own label, same-timestamp labels, or a global target
mean.

## Test result

`tests/test_structured_prior_v1.py` (5) +
`tests/test_m08_section6_invariants.py` (8) +
`tests/test_structured_prior_protocol_v1.py` (6) = **19 passing**.
