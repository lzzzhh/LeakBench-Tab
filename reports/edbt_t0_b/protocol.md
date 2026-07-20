# T0-B — Multi-Policy Comparative Evaluation under Semantic-Group and Encoded-Column Cost Contracts

**Status:** FROZEN_BEFORE_EXPERIMENT
**Date:** 2026-07-20
**Branch:** t0/multipolicy-semantic-cost
**Protocol SHA:** (computed at freeze)

---

## 1. Research Questions

1. **RQ1 (Policy Dominance):** Does MI (P3) outperform other blind ranking policies
   (P4–P6) under both cost contracts, or does a different policy provide better
   semantic repair at the same cost?

2. **RQ2 (Cost Contract Sensitivity):** Does the encoded-column contract's
   systematic overcorrection and partial semantic-group violations resolve
   under the semantic-group atomic-deletion contract?

3. **RQ3 (M09 Atomic Repair):** Under the semantic-group contract, does the M09
   eight-column source group get atomically deleted by MI-guided selection at
   achievable budgets (5%, 10%, 20%), eliminating the partial-removal problem
   observed at T0-R2?

4. **RQ4 (Pareto Trade-off):** Across policies, is there a universally optimal
   policy, or do policies form a Pareto frontier trading off full-group recall,
   directional repair, overcorrection, and legitimate retention?

5. **RQ5 (Learner Consistency):** Are multi-policy conclusions consistent across
   LR, RF, and LightGBM downstream learners?

---

## 2. Frozen Inputs

| Input | Path | SHA-256 | Status |
|-------|------|---------|--------|
| Bundle manifest | artifacts/sp6/sp6_bundle_manifest.csv | 304c96ae... | FROZEN |
| Canonical cells | results/corrected_v2/canonical_cells.csv | 25c21440... | FROZEN |
| SP8 governance clean | artifacts/sp8/governance_clean.csv | 6e3aa4c7... | FROZEN |
| B1 multi-seed (LR) | results/edbt_eab_revision/b1_multiseed_p2.csv | 22aae323... | FROZEN |
| B2 RF | results/edbt_eab_revision/b2_rf.csv | b2f8ecb5... | FROZEN |
| B2 LightGBM | results/edbt_eab_revision/b2_lgbm.csv | a51e1f53... | FROZEN |
| T0-R2 claim state | results/edbt_t0_r2/claim_state_r2.json | 8e8b8896... | FROZEN |
| T0-R2 manifest | results/edbt_t0_r2/manifest.json | c2875501... | FROZEN |

---

## 3. Policy Matrix

Six policies (P0–P6) defined in `configs/edbt_t0_b/policy_registry.yaml`.

| ID | Name | Oracle? | Deployable? | Ranker |
|----|------|---------|-------------|--------|
| P0 | KEEP_ALL | No | No | None (baseline) |
| P1 | ORACLE_REMOVE_ALL | Yes | No | Oracle mask (upper bound) |
| P2 | RANDOM_MATCHED | No | No | 20 governance seeds (negative control) |
| P3 | MUTUAL_INFORMATION | No | Yes | `mutual_info_classif(train, rs=42)` |
| P4 | ABS_POINT_BISERIAL | No | Yes | `abs(pearsonr(X_col, y))` |
| P5 | STANDARDIZED_LR_COEF | No | Yes | `abs(coef)` after StandardScaler+LR |
| P6 | CROSS_FITTED_RF_PERM | No | Yes | 3-fold CV permutation importance |

### 3.1 Tie-Breaking (all deterministic policies)

1. Score descending
2. Unit index ascending (stable sort)

### 3.2 Group Aggregation (all policies)

Group score = `max(member column scores)`.

Mean aggregation is reserved as a sensitivity analysis and must be pre-specified
in an amendment if used.

---

## 4. Semantic-Group Registry

Defined in `configs/edbt_t0_b/semantic_group_registry.json`.

Key rules:
- M01–M05, M07, M08: each injected column is a singleton group
- M06: all M06 redundant columns form one group (atomic removal)
- M09: 8 one-hot columns form `M09_source_one_hot` (atomic removal required)
- M10: legitimate copy and contaminant are SEPARATE singleton groups
- M11: all M11 graph-projection columns form one group
- n_original columns: each is a singleton group
- Total groups = n_original + count(policy_visible_groups)

Policy selectors receive `group_id`, `group_size`, `member_encoded_indices`.
They do NOT receive `contaminated_status`, `leak_mask`, or `source_role`.

---

## 5. Cost Contracts

### 5.1 PRIMARY — Semantic-Group Contract

```
k_group = max(1, round(budget_fraction × total_semantic_groups))
```

Each policy removes exactly `k_group` COMPLETE semantic groups.

Recorded for every governed fit:
- nominal_group_budget
- realized_group_count (= k_group)
- realized_encoded_count (sum of group sizes of selected groups)
- realized_encoded_fraction

### 5.2 SECONDARY — Encoded-Column Contract

```
k_column = max(1, round(budget_fraction × total_encoded_columns))
```

Each policy removes exactly `k_column` encoded columns.

Recorded semantic metrics for every governed fit:
- full_group_removals (groups where all members removed)
- any_hit_groups (groups with at least one member removed)
- partial_group_violations (groups with some but not all members removed)

### 5.3 Budgets

Both contracts: **5%, 10%, 20%**.

### 5.4 Matched Random Cost

P2 uses the same k_group or k_column as the corresponding deterministic policy.
For semantic-group contract, P2 randomly selects k_group complete groups.
For encoded-column contract, P2 randomly selects k_column columns.

---

## 6. Downstream Models

Canonical factories from `src/leakbench/models/core_models.py :: fit_predict_core_model`.

| Model | Key Parameters |
|-------|---------------|
| LR | StandardScaler + LogisticRegression(max_iter=2000, C=1.0, random_state=training_seed) |
| RF | RandomForestClassifier(n_estimators=250, min_samples_leaf=2, max_features="sqrt", random_state=training_seed, n_jobs=1) |
| LightGBM | LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=31, max_depth=6, min_child_samples=20, random_state=training_seed, n_jobs=1, verbosity=-1, early_stopping=30) |

### 6.1 Baseline Rules

Strict/full baselines are re-fitted per key using the canonical factory.
They are NOT loaded from frozen SP5/SP8 CSVs. Numerical identity with SP8 is
verified before governance runs (baseline parity check).

---

## 7. Execution Matrix

### Stage B1 — Primary Full-Registry LR

| Dimension | Value |
|-----------|-------|
| Keys | 5,500 (20 datasets × 11 mechanisms × 5 strengths × 5 training seeds) |
| Primary contract | Semantic-group |
| Secondary contract | Encoded-column |
| Budgets | 5%, 10%, 20% |
| Policies | P0 (1×), P1 (1×), P2 (20 seeds), P3–P6 (1× each) |
| Fits per key | 1 + 1 + 20×3 + 4×3 = 74 per contract |
| Learner | LR only |
| Total fits | ~5,500 × 74 × 2 contracts ≈ 814,000 |

### Stage B2 — Cross-Learner Confirmation

| Dimension | Value |
|-----------|-------|
| Keys | 5,500 |
| Primary contract | Semantic-group only |
| Budget | 20% only |
| Policies | P0–P6 |
| Learners | RF, LightGBM |
| Total fits | ~5,500 × 25 × 2 learners ≈ 275,000 |

### Stage B3 — Budget Sensitivity

Analysis only (no new fits). Compares 5%/10%/20% under semantic-group contract
for full-group recall, partial violations, directional repair, overcorrection.

---

## 8. Primary Estimands

For each deterministic policy P and each contract C:

```
Δ = P − mean(P2 over 20 matched random seeds)
```

Computed per key, then aggregated:

| Metric | Direction |
|--------|-----------|
| Δlegacy_sdr | Larger positive = better |
| Δdirectional_repair | Larger positive = better |
| Δsame_side_residual | Larger negative = better |
| Δovercorrection | Larger negative = better (or ≤ 0) |
| Δleak_recall | Larger positive = better |
| Δdeletion_precision | Larger positive = better |
| Δlegit_retention | Larger positive = better |
| Δsemantic_group_recall_full | Larger positive = better |
| Δsemantic_group_recall_any | Larger positive = better |
| Δpartial_group_violation | Larger negative = better |
| Δintroduced_distortion | Larger negative = better (or ≤ 0) |

### 8.1 Inference

- Independent unit: controlled task (dataset_index)
- Primary estimand: exact equal-task mean
- Task-reweighting bootstrap: 20,000 reps, seed=20260719
- Report: reweighting 2.5%/97.5% interval, positive-task fraction
- No "population confidence interval" language

---

## 9. Claim Gates (frozen)

### SEMANTICALLY_CORROBORATED (under semantic-group contract)
All of:
- Δdirectional_repair mean > 0 AND lower bound > 0
- Δfull_group_recall mean > 0 AND lower bound > 0
- Δovercorrection mean <= 0
- Δlegit_retention mean >= 0
- Zero-opportunity introduced_distortion not increased

### LOCALIZATION_ONLY
- Full-group recall or any-hit significantly improved
- But directional repair, overcorrection, or retention gate failed

### SCORE_RECOVERY_ONLY
- Δlegacy_sdr mean > 0 AND lower bound > 0
- Semantic localization gate not passed

### TRADEOFF
- Semantic localization or directional repair improved
- But overcorrection or legit retention clearly worse

### NEGATIVE
- Score, localization, and retention all not better than random

### NOT_EVALUABLE
- Cost contract infeasible, mapping not auditable, zero opportunity

No custom post-hoc statuses. No single weighted score declaring "best policy."

---

## 10. Pareto Analysis

Four-dimensional Pareto frontier per budget, contract, and learner:
- full_group_recall ↑ (maximize)
- directional_repair ↑ (maximize)
- overcorrection ↓ (minimize)
- legit_retention ↑ (maximize)

Report:
- Pareto-optimal policies
- Dominated policy count
- Mechanism-specific winners
- No-universal-winner verdict (only if data supports)

---

## 11. Recorded Metrics

### Identity
run_id, dataset_index, mechanism, strength, training_seed, governance_seed,
learner, policy, ranker, cost_contract, nominal_budget, realized_group_count,
realized_encoded_count, selection_hash, group_selection_hash, status, failure_reason

### Performance
strict_auc, full_auc, governed_auc, legacy_sdr, directional_repair,
same_side_residual, overcorrection, introduced_distortion

### Semantic
removed_leak_count, removed_legit_count, leak_recall, deletion_precision,
legit_retention, semantic_leak_group_total, semantic_group_recall_full,
semantic_group_recall_any, partial_group_violation_count

### Secondary
AUPRC, log_loss, Brier_score, fixed_bin_ECE (not used in claim gates)

---

## 12. Hard Stops

Any of the following blocks all downstream analysis:

- Non-oracle policy accesses leak_mask
- Policy reads test labels
- Semantic mapping missing or non-unique
- Group policy removes partial group members
- Matched random cost not equal
- Duplicate run_id
- Missing key
- Selection hash unstable
- Baseline parity failure (>1e-6 diff from SP8)
- Bundle SHA mismatch
- Split hash mismatch
- Model factory drift
- Post-outcome protocol modification
- Failed rows silently dropped
- Policy coverage incomplete

---

## 13. Forbidden Actions

- Filter out sparse, low-opportunity, or negative-result keys
- Only report positive budgets
- Choose group aggregation (mean/max/sum) based on outcomes
- Choose "best" learner based on results
- Label task-reweighting interval as population CI
- Present oracle policy as deployable
- Add custom claim statuses after seeing outcomes

---

## 14. Dry Run

Before full execution:
- 2 datasets
- 2 mechanisms (must include M09)
- 1 strength (S3)
- 1 training seed (42)
- All policies
- Both cost contracts
- LR only

Namespace: `results/edbt_t0_b_dryrun/`

Dry-run outcomes do not enter formal claims.

---

## 15. Outputs

```
reports/edbt_t0_b/protocol.md
reports/edbt_t0_b/final_report.md
reports/edbt_t0_b/policy_comparison_report.md
reports/edbt_t0_b/cost_contract_report.md
reports/edbt_t0_b/failure_report.md

configs/edbt_t0_b/policy_registry.yaml
configs/edbt_t0_b/semantic_group_registry.json
configs/edbt_t0_b/model_factories.json

results/edbt_t0_b/protocol_freeze.json
results/edbt_t0_b/selection_ledger.csv.gz
results/edbt_t0_b/lr_cells.csv.gz
results/edbt_t0_b/rf_cells.csv.gz
results/edbt_t0_b/lightgbm_cells.csv.gz
results/edbt_t0_b/task_effects.csv
results/edbt_t0_b/policy_summary.csv
results/edbt_t0_b/cost_contract_summary.csv
results/edbt_t0_b/m09_summary.json
results/edbt_t0_b/pareto_frontier.csv
results/edbt_t0_b/failure_ledger.csv
results/edbt_t0_b/analysis_summary.json
results/edbt_t0_b/claim_state.json
results/edbt_t0_b/manifest.json
results/edbt_t0_b/environment.json
```

---

## 16. Paper Constraint

No modification to `paper/edbt_eab/main.tex`, paper-facing tables, figures, or
claim macros until formal claim state is reviewed and a separate
paper-integration branch is opened.
