# T0 R2 — Repair Construct-Validity Audit Protocol

**Status:** DRAFT (not yet frozen)
**Date:** 2026-07-20
**Branch:** t0/repair-construct-validity-r2
**Repository:** LeakBench-Tab

---

## 0. Repository Inventory

### 0.1 Canonical Ledgers

| Ledger | Path | Rows | Key Schema | Status |
|--------|------|------|------------|--------|
| SP8 governance (clean) | `artifacts/sp8/governance_clean.csv` | 55,000 | `(dataset_index, mechanism, strength, seed)` | FROZEN |
| B1 multi-seed LR | `results/edbt_eab_revision/b1_multiseed_p2.csv` | 467,500 | `(dataset_index, mechanism, strength, training_seed, governance_seed, budget_fraction)` | FROZEN |
| B2 cross-learner RF | `results/edbt_eab_revision/b2_rf.csv` | 121,000 | `(dataset_index, mechanism, strength, training_seed, governance_seed, model)` | FROZEN |
| B2 cross-learner LGBM | `results/edbt_eab_revision/b2_lgbm.csv` | 121,000 | same as B2 RF | FROZEN |
| SP5 canonical cells | `results/corrected_v2/canonical_cells.csv` | 27,500 | `(dataset_index, mechanism, strength, seed)` | FROZEN |

### 0.2 Bundle Manifest

| File | Rows | Key Columns |
|------|------|-------------|
| `artifacts/sp6/sp6_bundle_manifest.csv` | 5,500 | `dataset_index, mechanism, strength, seed, bundle_key, bundle_path, bundle_sha256, n_original, n_injected, n_leak` |

### 0.3 Key Schemas

**SP8 governance_clean.csv** (column name: `seed`):
`run_id, dataset_index, mechanism, strength, seed, policy, budget_k, budget_fraction, status, strict_auc, full_auc, governed_auc, strict_distance_reduction, utility_loss, residual_harm, removed_count, removed_leak_count, removed_legit_count, leak_recall, legit_retention, oracle_policy`

**B1 multiseed_p2.csv** (column name: `training_seed`):
`run_id, dataset_index, mechanism, strength, training_seed, governance_seed, policy, budget_k, budget_fraction, status, strict_auc, full_auc, governed_auc, strict_distance_reduction, initial_gap, removed_count, selection_mask_hash`

**B2 rf.csv / lgbm.csv** (adds `model` column):
`run_id, dataset_index, mechanism, strength, training_seed, governance_seed, model, policy, budget_k, budget_fraction, status, strict_auc, full_auc, governed_auc, strict_distance_reduction, initial_gap, removed_count, selection_mask_hash`

**Bundle .npz files** contain: `base_X, block__<key>, y, train_idx, val_idx, test_idx, leak_mask__<key>`

### 0.4 Selection Reconstruction Logic

**P3 (blind MI):**
```python
mi = mutual_info_classif(X[train_idx], y[train_idx], random_state=42)
mi = np.nan_to_num(mi, nan=0.0)
indices = np.argsort(mi)[::-1][:k]
```

**P2 (random, 20 seeds):**
```python
rng = np.random.RandomState((gov_seed * 100 + dataset_index * 7 + training_seed * 13) % (2**31 - 1))
indices = rng.choice(n_features, k, replace=False)
```

**Selection hash:**
```python
payload = b'encoded_column_indices_v1\0' + np.sort(indices).astype(np.int64).tobytes()
hash = hashlib.sha256(payload).hexdigest()
```

### 0.5 Model Identities

| Model ID | Implementation | Parameters |
|----------|---------------|------------|
| LR (B1, SP8) | `sklearn.linear_model.LogisticRegression` | `max_iter=1000` (SP8), `max_iter=2000` (B1) |
| RF (B2) | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=100, random_state=<seed>` |
| LightGBM (B2) | `lightgbm.LGBMClassifier` | `n_estimators=100, device='cpu', random_state=<seed>` |

### 0.6 Frozen Namespaces

| Namespace | Path | Frozen |
|-----------|------|--------|
| SP5 core evidence | `artifacts/sp5/`, `artifacts/sp5_5/` | YES |
| SP6 bundles | `artifacts/sp6/` | YES |
| SP8 clean governance | `artifacts/sp8/`, `results/corrected_v2/canonical_cells.csv` | YES |
| EDBT revision (B1/B2) | `results/edbt_eab_revision/` | YES |
| EDBT paper | `paper/edbt_eab/` | READ-ONLY for this audit |

### 0.7 Superseded Namespaces

| Namespace | Superseded By | Reason |
|-----------|---------------|--------|
| SP8 legacy (77k rows) | SP8 clean (55k) | Oracle-path leaked leakage_mask to non-oracle policies |
| CDXR cross-learner v2 | TBD (not yet complete) | In-progress experiment on separate branch |

### 0.8 Existing Metric Definitions

**Legacy SDR (strict distance reduction):**
```
legacy_sdr = |full_auc − strict_auc| − |governed_auc − strict_auc|
```
= `opportunity − |governed_offset|`

**Paired effect (P3 − mean P2):**
```
P2_bar_k = mean(P2_legacy_sdr_k_s for s in 20 governance seeds)
D_k = P3_legacy_sdr_k − P2_bar_k
```

### 0.9 Known Protocol Deviations

1. **B2 baseline re-fit:** B2 RF/LGBM strict/full were re-fitted per key rather than loaded from SP5 canonical cells. This is disclosed in `governance_revision_protocol.md` and in `manifest.json`.
2. **B1 baseline re-fit:** B1 also re-fitted strict/full LR per key (not loaded from SP8). This is NOT disclosed in the protocol — it claims "Same strict/full references as frozen SP8" (protocol line 37).
3. **B1 SP8 `seed` vs B1 `training_seed`:** Column renaming from `seed` to `training_seed` between SP8 and B1 CSVs.
4. **B1 `max_iter` change:** SP8 uses `max_iter=1000`, B1 uses `max_iter=2000` for LR.

---

## 1. Research Questions

### RQ1 (T0-A0): Baseline Continuity
Does B1 actually reuse frozen SP8 strict/full baselines, or were they re-fitted?
If re-fitted, are the resulting values numerically identical to SP8?

### RQ2 (T0-A1): Selection Reconstruction
Can all 709,500 selection hashes be deterministically reconstructed from the
frozen bundle manifest? Are there any mismatches?

### RQ3 (T0-A2/A3): What Does "Positive SDR" Actually Mean?
When P3 SDR > P2 SDR, does the governed score move toward the strict reference
(directional repair of residual leakage) or does it overshoot (overcorrection)?
How much leakage is actually removed vs. legitimate signal discarded?

### RQ4 (T0-A4): Semantic Corroboration
Does the P3 advantage over P2 persist when measured by:
- residual leakage reduction (same-side residual)
- leak localization (recall, precision)
- legitimate signal retention
- semantic-group recall
rather than just legacy SDR?

### RQ5 (False Repair): How Much "Repair" Is Spurious?
What fraction of positive ΔSDR is attributable to:
- legitimate strong-feature deletion (FR1)
- blind luck with zero leak removal (FR2)
- unchanged residual (FR3)
- overcorrection past the strict reference (FR4)
- worse legitimate retention (FR5)
- partial semantic-group hits (FR6)

### RQ6 (Sparse Archetype): Why Negative?
Is the sparse archetype's negative ΔSDR driven by low opportunity, legitimate
strong-feature deletion, low leak recall, overcorrection, or multiple factors?

### RQ7 (M09): Semantic-Group Robustness
Does M09's strong positive encoded-column result hold under semantic full-group recall?

### RQ8 (Learner Consistency): Do LR/RF/LGBM Agree Under R2 Metrics?
Do the three learners show the same directional pattern under R2 metrics?

---

## 2. Frozen Inputs

### 2.1 Data Sources (read-only, must not modify)

| Input | Path | SHA-256 (from manifest) |
|-------|------|-------------------------|
| SP8 governance clean | `artifacts/sp8/governance_clean.csv` | `6e3aa4c7...` |
| B1 multi-seed | `results/edbt_eab_revision/b1_multiseed_p2.csv` | `22aae323...` |
| B2 RF | `results/edbt_eab_revision/b2_rf.csv` | `b2f8ecb5...` |
| B2 LGBM | `results/edbt_eab_revision/b2_lgbm.csv` | `a51e1f53...` |
| Bundle manifest | `artifacts/sp6/sp6_bundle_manifest.csv` | `304c96ae...` |
| Canonical cells | `results/corrected_v2/canonical_cells.csv` | `25c21440...` |
| Bundle .npz files | `results/corrected_v2/task_bundles/panel_*.npz` | (per manifest) |

### 2.2 Constants (frozen, from original runners)

```
GOV_SEEDS = [2026071700 + i for i in range(20)]  # B1/B2
MI_SEED = 42                                       # mutual_info_classif random_state
P2_SEED_FORMULA = (gov_seed * 100 + dataset_index * 7 + training_seed * 13) % (2**31 - 1)
PRIMARY_BUDGET = 0.20
HASH_SCHEME = "sha256(encoded_column_indices_v1\\0 || sorted_int64_le_indices)"
BOOTSTRAP_SEED = 20260719   # for T0 audit (new, not from original)
BOOTSTRAP_REPS = 20000      # for T0 audit
```

---

## 3. Metric Definitions

### 3.1 Core Quantities

```
signed_gap = full_auc − strict_auc
opportunity = abs(signed_gap)
governed_offset = governed_auc − strict_auc
```

### 3.2 R2 Directional Repair Metrics

When `opportunity > 1e-12`:

```
direction = sign(signed_gap)
same_side_residual = max(direction * governed_offset, 0)
overcorrection = max(−direction * governed_offset, 0)
directional_repair = opportunity − same_side_residual
legacy_sdr = opportunity − abs(governed_offset)
directional_repair_fraction = directional_repair / opportunity  (may be < 0, not clipped)
```

When `opportunity <= 1e-12`:

```
direction = 0
directional_repair_fraction = NA
same_side_residual = 0
overcorrection = 0
introduced_distortion = abs(governed_offset)
opportunity_class = "zero"
```

For all rows: `introduced_distortion = 0` (except zero-opportunity rows).

### 3.3 Mask-Grounded Semantic Metrics

```
total_leak_columns = sum(leak_mask)
total_legit_columns = sum(~leak_mask)
removed_leak_count = sum(leak_mask[removed_indices])
removed_legit_count = sum(~leak_mask[removed_indices])
leak_recall = removed_leak_count / total_leak_columns
deletion_precision = removed_leak_count / removed_count
legit_retention = 1 − removed_legit_count / total_legitimate_columns
residual_leak_column_count = total_leak_columns − removed_leak_count
residual_leak_fraction = residual_leak_column_count / total_leak_columns
```

### 3.4 Semantic-Group Metrics

A semantic group is a set of encoded columns that originate from the same source field.

For each semantic group g with members M_g:

```
semantic_leak_group_total = count of groups g where any column in M_g is a leak column
semantic_leak_group_full_removed = count of groups g where all columns in M_g are removed
semantic_leak_group_any_hit = count of groups g where at least one column in M_g is removed
semantic_leak_group_partial_removed = count of groups g where some but not all columns in M_g are removed
semantic_group_recall_full = semantic_leak_group_full_removed / semantic_leak_group_total
semantic_group_recall_any = semantic_leak_group_any_hit / semantic_leak_group_total
partial_group_violation_count = semantic_leak_group_partial_removed
```

### 3.5 Paired P3 − mean(P2) Analysis

For each key k with P3 metrics M_P3 and 20 P2 metrics M_P2_s:

```
P2_bar[k] = mean(M_P2_s for s in 1..20)
paired[k] = M_P3[k] − P2_bar[k]
```

Analyze for: legacy_sdr, directional_repair, same_side_residual, overcorrection,
leak_recall, deletion_precision, legit_retention, semantic_group_recall_full,
partial_group_violation, introduced_distortion.

### 3.6 Finite-Registry Estimands

Primary estimand: **finite-registry equal-task mean** of paired differences across
all controlled tasks in the registry.

Inference: **task-reweighting bootstrap** (20,000 reps), where each rep resamples
the 20 dataset indices with replacement and computes the mean of task-level paired
differences.

Do NOT report as "population-level confidence intervals" or "p-values."
Report as:
- finite-registry mean
- task-reweighting 2.5% / 97.5% interval
- P3 > mean(P2) bootstrap fraction
- leave-one-task-out range

### 3.7 Zero-Opportunity Handling

When `opportunity <= 1e-12`:
- directional_repair_fraction = NA
- These keys are excluded from directional_repair and same_side_residual analyses
- They are counted and reported separately
- introduced_distortion is computed and reported

---

## 4. Semantic Grouping Rules

### 4.1 Identity-Mapped Mechanisms (M01–M08, M11)

For M01 through M08 and M11, each encoded column corresponds to exactly one
semantic source field. The semantic group mapping is an identity mapping:
each column is its own group.

### 4.2 M09 (One-Hot Indicators)

M09 encodes a categorical entity leak via 8 one-hot indicator columns.
These 8 columns form a single semantic group.

Group definition:
```
M09_group = {column indices 12, 13, 14, 15, 16, 17, 18, 19}  (columns within the M09-injected block)
```

Full removal of M09 requires all 8 columns to be removed.

### 4.3 M10 (Legal + Contaminated Fields)

M10 includes both legitimate and leak columns as separate semantic groups.
They must be treated as distinct groups, not merged.

### 4.4 Natural Cases (NOT EVALUABLE for Semantic-Group)

Natural datasets do not have an explicit, auditable source-column mapping in
the current preprocessing artifacts. They are marked NOT_EVALUABLE for
semantic-group metrics unless a lineage artifact is found.

---

## 5. Claim-State Rules

Each claim uses the following statuses:

1. **SEMANTICALLY_CORROBORATED**: All of:
   - legacy ΔSDR exact mean > 0 AND reweighting lower bound > 0
   - Δresidual_reduction exact mean > 0 AND reweighting lower bound > 0
   - Δleak_recall exact mean > 0 AND reweighting lower bound > 0
   - Δovercorrection exact mean <= 0
   - Δlegit_retention exact mean >= 0

2. **SCORE_RECOVERY_ONLY**: Legacy SDR conditions met, but semantic corroboration fails.

3. **MIXED**: Conflicting directions across tasks, mechanisms, or metrics.

4. **NEGATIVE**: P3 residual and leak localization both not better than P2.

5. **NOT_EVALUABLE**: Missing auditable mapping, zero opportunity, or incomplete coverage.

---

## 6. Exclusions

The following are excluded from the audit scope:
- Natural governance (5 case studies) — marked NOT_EVALUABLE for semantic-group, but legacy metrics are computed
- CDXR cross-learner v2 experiment (separate branch, not yet frozen)
- Budget curves other than 20% (primary budget only for R2 audit)
- SP8 legacy (77k rows, oracle-contaminated)

---

## 7. Failure Conditions (Hard Stops)

The following conditions block further analysis:

1. T0-A0: Any key coverage ≠ 5,500 for B1/SP8 match
2. T0-A0: Any duplicate key in SP8 or B1
3. T0-A0: Any absolute difference > 1e-6 between SP8 and B1 strict/full AUC
4. T0-A1: Any selection hash mismatch between reconstruction and recorded value
5. T0-A1: Bundle hash mismatch
6. T0-A1: Missing leak mask
7. Any protocol deviation discovered after freeze

---

## 8. Expected Coverage

| Audit | Expected Keys | Expected Rows |
|-------|--------------|---------------|
| T0-A0 (B1/SP8 continuity) | 5,500 | 5,500 |
| T0-A1 (selection reconstruction) | 5,500 × 3 models | 709,500 |
| T0-A2 (metric vector) | 709,500 SUCCESS rows | 709,500 |
| T0-A3 (mask metrics) | same as A2 | 709,500 |
| T0-A4 (paired analysis) | 5,500 per model | 16,500 keys |

---

## 9. Planned Outputs

### Reports
- `reports/edbt_t0_r2/protocol.md` (this file)
- `reports/edbt_t0_r2/final_report.md`
- `reports/edbt_t0_r2/baseline_continuity_report.md`
- `reports/edbt_t0_r2/false_repair_report.md`

### Results
- `results/edbt_t0_r2/protocol_freeze.json`
- `results/edbt_t0_r2/b1_sp8_baseline_continuity.csv`
- `results/edbt_t0_r2/task_effects_r2.csv`
- `results/edbt_t0_r2/mechanism_summary_r2.csv`
- `results/edbt_t0_r2/archetype_summary_r2.csv`
- `results/edbt_t0_r2/false_repair_summary.csv`
- `results/edbt_t0_r2/false_repair_examples.csv`
- `results/edbt_t0_r2/analysis_summary_r2.json`
- `results/edbt_t0_r2/claim_state_r2.json`
- `results/edbt_t0_r2/manifest.json`

### Scripts
- `scripts/audit_repair_construct_validity.py`
- `scripts/analyze_repair_r2.py`
- `scripts/build_repair_r2_claim_state.py`
- `scripts/build_repair_r2_manifest.py`

### Tests
- `tests/test_repair_r2.py`
