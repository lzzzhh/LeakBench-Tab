# T0-B Semantic Lineage Audit V2

**Source:** `src/leakbench/mechanisms/__init__.py`
**Source SHA-256:** `0f1c605ea4afeeb70cb8fa300f58f97c9e347762f6943b3cd93a21f500e9227f`

## M06 — Redundant Leakage Cluster

**Lineage:** Lines 287-297. All injected columns are generated from a single
loop: `for i in range(count)`, each column = `strength * (y - prior) + N(0, noise_std * (1 + i/count))`.
All share the same underlying signal (`y - prior`) with only escalating noise.

**Verdict:** One atomic group. All M06 injected columns share a single source
field (the target proxy) and differ only in noise scale. V1 and V2 agree.

## M09 — Source Leakage (One-Hot)

**Lineage:** Lines 358-406. A single categorical variable ("source") is
one-hot encoded into 8 binary columns. The assignment is outcome-dependent
but comes from a single `np.random.choice` over source categories.

**Verdict:** One atomic group (8 columns). All 8 one-hot columns originate
from one logical source field. Neither V1 nor V2 treat them as singletons.
Atomic removal under semantic-group contract is required.

## M10 — Mixed Legit + Contaminant

**Lineage:** Lines 408-423. Two columns injected:
- `mixed_legitimate_clean_0`: copy of `X[:, 0]` (clean feature). Labeled "legitimate", `available_at_prediction=True`.
- `contam_M10_target_proxy`: `strength * (y - prior) + noise`. Labeled "M10", masked as leak.

**Verdict:** Two SEPARATE singleton groups. These columns have different
origins (clean feature copy vs target proxy). No shared source lineage.
V2 treats them as independent singletons (V1 also had them as separate groups).

## M11 — Graph-Mediated Leakage

**Lineage:** Lines 425-444. All columns are projections of the same signal
`(y - prior) * projections * strength + noise` through different random
directions. Generated in a single loop from one leakage block.

**Verdict:** One atomic group. All M11 columns share the same underlying
leakage signal and differ only in projection direction. V1 and V2 agree.

## Original Columns

All original columns (indices 0 through n_original-1) are singleton groups
regardless of mechanism. They originate from the dataset generator
(`src/leakbench/datasets.py`) as independent features.

## Coverage Verification

5,500/5,500 keys have complete group mappings. Every encoded column belongs
to exactly one group. 0 missing, 0 overlap, 0 duplicate group IDs per key.
