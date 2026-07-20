# T0-B V2 — Consolidated Protocol

**Status:** FROZEN_BEFORE_EXPERIMENT (V2, supersedes fbaa9f3)
**Date:** 2026-07-20
**Git:** fbaa9f3 (V1, SUPERSEDED) → this commit (V2, effective)

Full protocol details in companion files:
- Policy registry: `configs/edbt_t0_b/policy_registry_v2.yaml`
- Semantic groups: `configs/edbt_t0_b/policy_group_registry_v2.json`
- Evaluation labels: `configs/edbt_t0_b/semantic_evaluation_labels_v2.json`
- Cost contracts: `configs/edbt_t0_b/cost_contracts_v2.json`
- Model factories: `configs/edbt_t0_b/model_factories_v2.json`
- Claim gates: `configs/edbt_t0_b/claim_gates_v2.json`
- Execution matrix: `configs/edbt_t0_b/execution_matrix_v2.json`
- Pareto contract: `configs/edbt_t0_b/pareto_contract_v2.json`
- Lineage audit: `reports/edbt_t0_b/semantic_lineage_audit_v2.md`
- Runtime plan: `reports/edbt_t0_b/runtime_plan_v2.md`
- Preflight report: `reports/edbt_t0_b/static_preflight_report_v2.md`

## Policy Matrix (P0-P6)

Seven policies. P2 is matched random ×20 governance seeds per key.
P3-P6 are blind deterministic rankers using training rows only.

## Cost Contracts

Semantic-group (primary, atomic group deletion) and encoded-column (secondary).
Both at 5%, 10%, 20% budget. Explicit integer half-up rounding.

## Execution

1,089,000 total downstream fits (803,000 B1 LR + 286,000 B2 RF/LGBM).
22,000 ranking model fits. P3-P6 scores computed once per key, reused across budgets/contracts/learners.

## Claim Gates

Six mutually exclusive statuses in precedence order:
NOT_EVALUABLE → SEMANTICALLY_CORROBORATED → TRADEOFF → LOCALIZATION_ONLY → SCORE_RECOVERY_ONLY → NEGATIVE.

## Forbidden

No post-outcome status creation, no oracle-as-deployable, no population CI language,
no mechanism filtering, no budget cherry-picking.
