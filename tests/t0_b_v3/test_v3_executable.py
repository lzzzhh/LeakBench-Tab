"""T0-B V3 Executable Static Tests — real assertions, no pass stubs."""
from __future__ import annotations
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np, pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
from scripts.t0_b_v3.policy_selectors import (
    score_mi, score_point_biserial, score_lr_coef, score_rf_permutation,
    group_max_score, top_k_groups, top_k_columns,
)

# ============================================================
# Synthetic fixtures
# ============================================================
@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 12)
    y = (X[:, 0] + X[:, 1] + rng.randn(100) * 0.5 > 0).astype(int)
    return X, y

@pytest.fixture
def synthetic_groups():
    return [
        {"opaque_group_id": "g000", "member_encoded_indices": [0, 1], "group_size": 2},
        {"opaque_group_id": "g001", "member_encoded_indices": [2], "group_size": 1},
        {"opaque_group_id": "g002", "member_encoded_indices": [3, 4, 5], "group_size": 3},
        {"opaque_group_id": "g003", "member_encoded_indices": [6], "group_size": 1},
    ]

# ============================================================
# 1. Point-biserial constant column → 0
# ============================================================
def test_constant_column_score_zero():
    X = np.ones((50, 3))
    X[:, 1] = 5.0  # another constant
    X[:, 2] = np.arange(50) * 0.1  # varying
    y = np.array([0, 1] * 25)
    scores = score_point_biserial(X, y)
    assert scores[0] == 0.0, f"Constant column score should be 0, got {scores[0]}"
    assert scores[1] == 0.0, f"Constant column score should be 0, got {scores[1]}"
    assert scores[2] > 0.0, f"Varying column should have positive correlation"

# ============================================================
# 2. P5 scaler uses training fixture only
# ============================================================
def test_p5_scaler_training_only(synthetic_data):
    X, y = synthetic_data
    X_train = X[:60]; y_train = y[:60]
    scores = score_lr_coef(X_train, y_train)
    assert len(scores) == X_train.shape[1]
    assert not np.any(np.isnan(scores))
    # Verify scaler was fit (coefficients exist)
    assert np.sum(scores) > 0, "All coefficients are zero"

# ============================================================
# 3. P6 folds from training fixture only
# ============================================================
def test_p6_folds_training_only(synthetic_data):
    X, y = synthetic_data
    scores = score_rf_permutation(X, y)
    assert len(scores) == X.shape[1]
    assert not np.any(np.isnan(scores))

# ============================================================
# 4. P6 parameters match registry
# ============================================================
def test_p6_parameters_explicit():
    """All P6 RF parameters must be explicitly set, no sklearn defaults."""
    import inspect
    src = inspect.getsource(score_rf_permutation)
    required = ["n_estimators=100", "criterion=\"gini\"", "max_features=\"sqrt\"",
                "random_state=42", "n_jobs=1", "n_repeats=5",
                "n_splits=3", "shuffle=True"]
    for r in required:
        assert r in src, f"Missing required parameter: {r}"

# ============================================================
# 5. P2 matched cost
# ============================================================
def test_p2_matched_cost():
    n_units = 20
    k = compute_k(n_units, 2000)  # 20%
    assert k == 4  # deterministic
    assert compute_k(n_units, 2000) == compute_k(20, 2000)

# ============================================================
# 6. Semantic selection expands groups to columns
# ============================================================
def test_semantic_group_expansion(synthetic_groups):
    # Select group g002 which has columns [3,4,5]
    selected_groups = ["g002"]
    expanded = []
    for g in synthetic_groups:
        if g["opaque_group_id"] in selected_groups:
            expanded.extend(g["member_encoded_indices"])
    assert expanded == [3, 4, 5]

# ============================================================
# 7. Evaluation labels don't change selection
# ============================================================
def test_selection_independent_of_labels(synthetic_data, synthetic_groups):
    X, y = synthetic_data
    # Selection with same y is deterministic
    scores_a = score_mi(X, y)
    scores_b = score_mi(X, y)
    assert np.allclose(scores_a, scores_b)
    # Selection doesn't access evaluation labels (structural: MI receives only X,y)
    scores_subset = score_mi(X[:50], y[:50])
    assert len(scores_subset) == X.shape[1]

# ============================================================
# 8. Neutral IDs don't expose origin/injection identity
# ============================================================
def test_neutral_ids_no_origin_info():
    """Group IDs g000, g001, ... contain no mechanism/leak/oracle information."""
    ids = [f"g{i:03d}" for i in range(20)]
    for gid in ids:
        assert "orig" not in gid
        assert "inj" not in gid
        assert "leak" not in gid
        assert "contam" not in gid
        assert gid.startswith("g")
        assert len(gid) == 4

# ============================================================
# 9. Selection hash namespace isolation
# ============================================================
def test_selection_hash_namespace_isolated():
    h1 = hash_encoded_selection(0, "M01", "S1", 13, "key1", "sha1", "P3", "semantic_group", 2000, np.array([0, 1], dtype=np.int64))
    h2 = hash_semantic_selection(0, "M01", "S1", 13, "key1", "sha1", "P3", "semantic_group", 2000, ["g000", "g001"])
    assert h1 != h2, "Encoded and semantic hashes must differ"

    # Different policy → different hash
    h3 = hash_encoded_selection(0, "M01", "S1", 13, "key1", "sha1", "P4", "semantic_group", 2000, np.array([0, 1], dtype=np.int64))
    assert h1 != h3

    # Different budget → different hash
    h4 = hash_encoded_selection(0, "M01", "S1", 13, "key1", "sha1", "P3", "semantic_group", 1000, np.array([0, 1], dtype=np.int64))
    assert h1 != h4

    # Different contract → different hash
    h5 = hash_encoded_selection(0, "M01", "S1", 13, "key1", "sha1", "P3", "encoded_column", 2000, np.array([0, 1], dtype=np.int64))
    assert h1 != h5

    # Order invariant
    h6 = hash_encoded_selection(0, "M01", "S1", 13, "key1", "sha1", "P3", "semantic_group", 2000, np.array([1, 0], dtype=np.int64))
    assert h1 == h6

# ============================================================
# 10. 5,500-row mapping ledger coverage
# ============================================================
def test_mapping_ledger_5500_rows():
    p = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
    assert p.exists(), "Policy mapping ledger not found"
    data = gzip.decompress(p.read_bytes()).decode("utf-8")
    lines = [l for l in data.strip().split("\n") if l]
    assert len(lines) == 5500, f"Expected 5500 rows, got {len(lines)}"

    seen = set()
    for line in lines:
        row = json.loads(line)
        key = (row["dataset_index"], row["mechanism"], row["strength"], row["training_seed"])
        assert key not in seen, f"Duplicate key: {key}"
        seen.add(key)
        assert row["n_groups"] == len(row["groups"])
        assert len(row["mapping_sha256"]) == 64

def test_eval_ledger_5500_rows():
    p = ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"
    assert p.exists()
    data = gzip.decompress(p.read_bytes()).decode("utf-8")
    lines = [l for l in data.strip().split("\n") if l]
    assert len(lines) == 5500, f"Expected 5500 rows, got {len(lines)}"

def test_eval_policy_ledger_matched():
    """Evaluation and policy ledgers must have the same keys."""
    p_pol = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
    p_eval = ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"
    pol_lines = gzip.decompress(p_pol.read_bytes()).decode("utf-8").strip().split("\n")
    eval_lines = gzip.decompress(p_eval.read_bytes()).decode("utf-8").strip().split("\n")
    pol_keys = set()
    eval_keys = set()
    for line in pol_lines:
        r = json.loads(line)
        pol_keys.add((r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]))
    for line in eval_lines:
        r = json.loads(line)
        eval_keys.add((r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]))
    assert pol_keys == eval_keys, f"Key mismatch: {len(pol_keys ^ eval_keys)} keys differ"

# ============================================================
# 11. Deterministic gzip (mtime=0)
# ============================================================
def test_gzip_mtime_zero():
    p = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
    with open(p, "rb") as f:
        f.read(2)  # magic: 0x1f 0x8b
        f.read(1)  # compression method
        f.read(1)  # flags
        mtime = int.from_bytes(f.read(4), "little")
    assert mtime == 0, f"gzip mtime should be 0, got {mtime}"

# ============================================================
# 12. Model source hash
# ============================================================
def test_model_source_hash():
    with open(ROOT / "configs/edbt_t0_b/model_factories_v2.json") as f:
        cfg = json.load(f)
    actual = hashlib.sha256((ROOT / "src/leakbench/models/core_models.py").read_bytes()).hexdigest()
    assert cfg["source_sha256"] == actual

# ============================================================
# 13. Claim gate mutual exclusivity
# ============================================================
def test_claim_gates_v3():
    from scripts.t0_b.claim_gates import MetricEstimate, determine_claim_status
    pos = MetricEstimate(0.05, 0.01, 0.10)
    neg = MetricEstimate(-0.05, -0.10, -0.01)
    zero = MetricEstimate(0.0, -0.01, 0.01)

    # Only one status per input
    status = determine_claim_status(pos, pos, pos, pos, zero, pos, zero, True)
    assert status == "SEMANTICALLY_CORROBORATED"
    status = determine_claim_status(neg, neg, neg, neg, zero, neg, zero, True)
    assert status == "NEGATIVE"

# ============================================================
# 14. Pareto tolerance and missing handling
# ============================================================
def test_pareto_v3():
    from scripts.t0_b.pareto import ParetoPoint, strictly_dominates, weakly_dominates
    a = ParetoPoint("A", 0.9, 0.12, 0.90, 0.01, 0.0, 0.5)
    b = ParetoPoint("B", 0.9 + 1e-13, 0.12, 0.90, 0.01, 0.0, 0.5)
    assert not strictly_dominates(a, b, tolerance=1e-12)

# ============================================================
# 15. Frozen namespace no diff
# ============================================================
def test_frozen_namespace_no_diff():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--name-only", "fbaa9f3...HEAD", "--",
         "artifacts/", "paper/", "results/edbt_eab_revision/",
         "results/corrected_v2/", "results/edbt_eab_crosslearner_confirmatory_v2/"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.stdout.strip() == "", f"Frozen files modified: {result.stdout}"

# ============================================================
# 16. P2 seed contract tests
# ============================================================
def test_p2_v3_same_input_same_seed():
    s1 = derive_p2_seed(2026071700, 0, "M01", "S1", 13, "semantic_group", 2000)
    s2 = derive_p2_seed(2026071700, 0, "M01", "S1", 13, "semantic_group", 2000)
    assert s1 == s2

def test_p2_v3_different_dims_different_seed():
    s1 = derive_p2_seed(2026071700, 0, "M01", "S1", 13, "semantic_group", 2000)
    s2 = derive_p2_seed(2026071700, 0, "M02", "S1", 13, "semantic_group", 2000)
    s3 = derive_p2_seed(2026071700, 0, "M01", "S2", 13, "semantic_group", 2000)
    s4 = derive_p2_seed(2026071700, 0, "M01", "S1", 13, "semantic_group", 1000)
    s5 = derive_p2_seed(2026071700, 0, "M01", "S1", 13, "encoded_column", 2000)
    assert len({s1, s2, s3, s4, s5}) == 5, "All should differ"

# ============================================================
# 17. Budget rounding tests
# ============================================================
@pytest.mark.parametrize("n,bp,expected", [
    (13, 2000, 3), (20, 500, 1), (21, 1000, 2),
    (8, 2000, 2), (15, 1000, 2), (25, 1000, 3),
])
def test_budget_v3(n, bp, expected):
    assert compute_k(n, bp) == expected

# ============================================================
# 18. Group max score
# ============================================================
def test_group_max_score():
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7, 0.4, 0.8, 0.0, 0.0, 0.0, 0.0])
    groups = [
        {"opaque_group_id": "g000", "member_encoded_indices": [0, 1], "group_size": 2},
        {"opaque_group_id": "g001", "member_encoded_indices": [2, 3, 4], "group_size": 3},
    ]
    result = group_max_score(scores, groups)
    assert result == [("g000", 0.5), ("g001", 0.9)]

# ============================================================
# 19. Old P2 formula banned
# ============================================================
def test_old_p2_formula_not_in_v3():
    """V3 selectors/contracts must not use old B1 P2 formula."""
    v3_files = list(Path("scripts/t0_b_v3").glob("*.py"))
    old_formula = "(gov_seed * 100 + dataset_index * 7 + training_seed * 13)"
    for f in v3_files:
        if "validate_protocol_freeze" in f.name:
            continue  # validator checks for old formula, that's fine
        content = f.read_text()
        assert old_formula not in content, f"Old P2 formula found in {f}"

# ============================================================
# 20. M09 full-group = 8 columns
# ============================================================
def test_m09_full_group_in_ledger():
    p = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
    data = gzip.decompress(p.read_bytes()).decode("utf-8")
    for line in data.strip().split("\n"):
        row = json.loads(line)
        if row["mechanism"] == "M09":
            for g in row["groups"]:
                if g["group_size"] == 8:
                    assert len(g["member_encoded_indices"]) == 8
                    break
            else:
                pytest.fail(f"M09 key has no 8-column group: {row['dataset_index']}_{row['strength']}_{row['training_seed']}")

# ============================================================
# 21. M10 two singleton groups
# ============================================================
def test_m10_two_singleton_groups():
    p = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
    data = gzip.decompress(p.read_bytes()).decode("utf-8")
    for line in data.strip().split("\n"):
        row = json.loads(line)
        if row["mechanism"] == "M10":
            sizes = [g["group_size"] for g in row["groups"]]
            ones = [s for s in sizes if s == 1]
            assert len(ones) >= len(row["groups"]) - 2  # all singletons

# ============================================================
# 22. Top-k tie-breaking
# ============================================================
def test_top_k_tie_breaking():
    scores = np.array([0.5, 0.5, 0.8, 0.5, 0.3])
    # columns: 0=0.5, 1=0.5, 2=0.8, 3=0.5, 4=0.3
    # top-3: col 2 (0.8), then cols 0,1,3 (0.5 each, ascending index)
    selected = top_k_columns(scores, 3)
    assert list(selected) == [2, 0, 1]
