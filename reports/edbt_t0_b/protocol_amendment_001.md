# T0-B Protocol Amendment 001 — Pre-Outcome Supersession of V1

**Status:** PRE_OUTCOME_CORRECTIVE
**Date:** 2026-07-20
**Superseded freeze:** fbaa9f3

---

## 1. What Changed

V1 (fbaa9f3) was a draft protocol freeze with eight completeness gaps.
V2 replaces it entirely with a consolidated protocol package.

## 2. Issues Found and Resolved

### 2.1 POLICY_VISIBLE_ORACLE_LABELS
V1's `semantic_group_registry.json` contained `contamination_labels` in the same
file as `policy_visible` groups. Non-oracle selectors could access leak_mask by
reading the same JSON. V2 splits this into two physically separate files:
`policy_group_registry_v2.json` (oracle-isolated) and
`semantic_evaluation_labels_v2.json` (evaluation-only).

### 2.2 P2_NAMESPACE_INCOMPLETE
V1 reused the B1 simple arithmetic formula `(gov_seed * 100 + ds * 7 + ts * 13)`
which does not include `mechanism`, `strength`, `cost_contract_id`, or `budget`.
V2 uses SHA256-based cryptographic derivation with all namespace dimensions.

### 2.3 SEMANTIC_LINEAGE_UNBOUND
V1 did not bind group definitions to generator source code or provide per-mechanism
lineage evidence. V2 provides `semantic_lineage_audit_v2.md` with source SHA-256
and per-mechanism rationale.

### 2.4 BUDGET_ROUNDING_AMBIGUOUS
V1 did not specify rounding behavior. V2 uses explicit integer half-up rounding
with no Python `round()`.

### 2.5 EXECUTION_COUNT_INCORRECT
V1 estimated ~814,000 fits for B1. Corrected estimate: 1,089,000 total downstream
fits (803,000 B1 + 286,000 B2).

### 2.6 CLAIM_GATE_OVERLAP
V1 claim statuses were unordered. V2 defines a strict precedence tree with 6
mutually exclusive statuses.

### 2.7 FREEZE_HASH_CLOSURE_INCOMPLETE
V1 did not bind source hashes. V2 binds bundle manifest SHA, generator source SHA,
core_models SHA, R2 manifest SHA, and R2 validation receipt SHA.

## 3. What Did NOT Change

- Policy set: P0-P6 unchanged
- Cost contracts: semantic-group + encoded-column unchanged
- Budgets: 5%, 10%, 20% unchanged
- Models: LR, RF, LightGBM unchanged (250 estimators each)
- Claim estimands: unchanged
- Pareto dimensions: unchanged

## 4. Why Pre-Outcome (No Researcher Degrees of Freedom)

All issues were identified by static analysis of the frozen V1 artifacts
against the repository source code. No outcomes were generated, read, or
analyzed before the amendment was written. The corrections are structural
(seed namespace, oracle isolation, rounding determinism) and do not depend
on any experimental results.

## 5. Effect on Existing Results

None. No T0-B outcomes exist. No frozen SP5-SP8 or T0-R2 evidence was modified.
