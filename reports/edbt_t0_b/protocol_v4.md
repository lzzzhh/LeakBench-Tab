# T0-B V4 — Multi-Policy Semantic-Cost Evaluation (Complete Protocol)

**Status:** FROZEN_BEFORE_EXPERIMENT

## Research Questions
RQ1: Policy dominance (P3 vs P4-P6). RQ2: Cost contract sensitivity. RQ3: M09 atomic repair. RQ4: Pareto trade-off. RQ5: Learner consistency.

## Policies (P0-P6)
P0: KEEP_ALL baseline. P1: ORACLE upper bound. P2: RANDOM_MATCHED ×20 governance seeds (SHA256 cryptographic seed). P3: MUTUAL_INFORMATION (mi, rs=42). P4: ABS_POINT_BISERIAL (abs(pearsonr), constant→0). P5: STANDARDIZED_LR_COEF (StandardScaler train-only + LogisticRegression C=1.0 max_iter=2000 rs=42). P6: CROSS_FITTED_RF_PERM (3-fold StratifiedKFold rs=42, n_repeats=5, scoring=roc_auc, ranking RF n_estimators=100).

Tie-breaking: score descending, opaque_group_id ascending, stable sort.

## Cost Contracts
Semantic-group (primary): k_group = max(1, floor((n_groups * bp + 5000) / 10000)). Encoded-column (secondary): k_column = same formula. Budgets: 500bp(5%), 1000bp(10%), 2000bp(20%). Matched random uses same k.

## Oracle Isolation
PolicyGroupView receives only opaque_group_id, member_encoded_indices, group_size. SemanticEvaluationLabels is a separate file, never passed to P2-P6.

## Selection Hash
SHA256 with V3 contract prefix, canonical key, policy, contract, budget, sorted indices/group IDs.

## Model Factories
LR: StandardScaler + LogisticRegression(max_iter=2000, C=1.0, rs=training_seed). RF: 250 estimators, min_samples_leaf=2, max_features=sqrt. LightGBM: 250 estimators, early_stopping=30, validation-only.

## Execution
B1 LR: 803,000 downstream fits (11,000 baseline + 792,000 governed). B2 RF+LGBM: 286,000. Grand total: 1,089,000. Ranking: 22,000. P0/P1 are virtual references, not additional fits.

## Dry Run
4 keys: datasets [0,3] × mechanisms [M01,M09] × strength S3 × seed 42. Expected: 584 downstream fits, 16 ranking fits. Exact bundle SHAs bound in dryrun_matrix_v4.json.

## Claim Gates
Precedence: NOT_EVALUABLE → SEMANTICALLY_CORROBORATED → TRADEOFF → LOCALIZATION_ONLY → SCORE_RECOVERY_ONLY → NEGATIVE. Pareto: 6D frontier, strict/weak dominance, tolerance 1e-12.

## Forbidden
Post-outcome protocol changes, old P2 formula, oracle words in policy artifacts, custom claim statuses, population CI language, mechanism filtering.
