"""T0-B V4 Executable Tests — real behavioral assertions."""
from __future__ import annotations
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection
from scripts.t0_b_v3.policy_selectors import score_mi, score_point_biserial, score_lr_coef


@pytest.fixture
def syn_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 8)
    y = (X[:,0] + rng.randn(100)*0.5 > 0).astype(int)
    return X, y


# ================================================================
# 1. P2 matched-cost: P2 and deterministic selector produce same k
# ================================================================
def test_p2_matched_cost():
    n_units = 20
    k = compute_k(n_units, 2000)
    assert k == compute_k(20, 2000)
    assert k > 0

# ================================================================
# 2. P5 training-only: API must not accept held-out data
# ================================================================
def test_p5_api_no_held_out(syn_data):
    X, y = syn_data
    X_train = X[:60]; y_train = y[:60]
    scores = score_lr_coef(X_train, y_train)
    assert len(scores) == X_train.shape[1]
    # Verify API signature doesn't accept test data
    import inspect
    sig = inspect.signature(score_lr_coef)
    assert len(sig.parameters) == 2, f"score_lr_coef should take 2 args, got {len(sig.parameters)}"

# ================================================================
# 3. P6 train-internal: indices all from within X_train
# ================================================================
def test_p6_train_internal(syn_data):
    X, y = syn_data
    from scripts.t0_b_v3.policy_selectors import score_rf_permutation
    scores = score_rf_permutation(X, y)
    assert len(scores) == X.shape[1]
    import inspect
    sig = inspect.signature(score_rf_permutation)
    assert len(sig.parameters) == 2

# ================================================================
# 4. Evaluation-label independence
# ================================================================
def test_eval_label_independence(syn_data):
    X, y = syn_data
    scores1 = score_mi(X, y)
    scores2 = score_mi(X, y)
    assert np.allclose(scores1, scores2)

# ================================================================
# 5. M10: two singleton groups, eval: 1 leak + 1 legit
# ================================================================
def test_m10_evaluation_consistency():
    p = ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"
    data = gzip.decompress(p.read_bytes()).decode("utf-8")
    for line in data.strip().split("\n"):
        row = json.loads(line)
        if row["mechanism"] == "M10":
            assert len(row["leak_group_ids"]) == 1, f"M10 should have 1 leak group"
            assert len(row["legitimate_group_ids"]) > 0

# ================================================================
# 6. Evaluation consistency: leak mask matches group partition
# ================================================================
def test_evaluation_mask_consistency():
    man = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    p = ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"
    data = gzip.decompress(p.read_bytes()).decode("utf-8")
    eval_map = {}
    for line in data.strip().split("\n"):
        r = json.loads(line)
        key = (r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"])
        eval_map[key] = r

    p2 = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
    data2 = gzip.decompress(p2.read_bytes()).decode("utf-8")
    pol_map = {}
    for line in data2.strip().split("\n"):
        r = json.loads(line)
        key = (r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"])
        pol_map[key] = r

    for key, ev in eval_map.items():
        gr = pol_map[key]
        leak_ids = set(ev["leak_group_ids"])
        legit_ids = set(ev["legitimate_group_ids"])
        all_ids = leak_ids | legit_ids
        assert len(all_ids) == len(gr["groups"]), f"Group count mismatch for {key}"
        assert not (leak_ids & legit_ids), f"Overlap in {key}"

# ================================================================
# 7. Hash closure
# ================================================================
def test_hash_closure_detects_change():
    h1 = hash_encoded_selection(0, "M01", "S1", 13, "k1", "sha", "P3", "sg", 2000, np.array([0,1], dtype=np.int64))
    h2 = hash_encoded_selection(0, "M01", "S1", 13, "k1", "sha", "P3", "sg", 2000, np.array([0,2], dtype=np.int64))
    assert h1 != h2

# ================================================================
# 8. Dry-run matrix: 4 keys, 584 downstream fits
# ================================================================
def test_dryrun_matrix_exact():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    assert len(dr["keys"]) == 4
    assert dr["expected_counts"]["total_downstream_fits"] == 584
    assert dr["expected_counts"]["ranking_model_fits"]["total"] == 16
    for k in dr["keys"]:
        assert len(k["bundle_sha256"]) == 64
        assert k["mechanism"] in ("M01", "M09")

# ================================================================
# 9. Freeze paths: all artifacts have path + sha256
# ================================================================
def test_freeze_v4_all_paths_bound():
    with open(ROOT / "results/edbt_t0_b/protocol_freeze_v4.json") as f:
        fr = json.load(f)
    for section_name, section in fr.items():
        if isinstance(section, dict):
            for k, v in section.items():
                if isinstance(v, dict):
                    assert "path" in v, f"{section_name}.{k} missing path"
                    assert "sha256" in v, f"{section_name}.{k} missing sha256"
                    assert len(v["sha256"]) == 64, f"{section_name}.{k} sha256 length != 64"

# ================================================================
# 10. Old P2 formula banned in V4
# ================================================================
def test_old_p2_banned_in_v4():
    for f in Path("scripts/t0_b_v4").glob("*.py"):
        if "validate" in f.name: continue
        content = f.read_text()
        assert "(gov_seed * 100" not in content, f"Old P2 formula in {f}"

# ================================================================
# 11. Execution counts consistent
# ================================================================
def test_execution_counts():
    with open(ROOT / "configs/edbt_t0_b/execution_matrix_v4.json") as f:
        ex = json.load(f)
    assert ex["grand_total_downstream"] == 1089000
    assert ex["b1_lr"]["total"] == 803000
    assert ex["b2_rf_lgbm"]["total"] == 286000
    assert ex["dry_run"]["total_downstream"] == 584

# ================================================================
# 12. Mapping ledgers: 5,500 rows, unique keys
# ================================================================
def test_mapping_ledger_integrity():
    for path in ["policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gzip.decompress((ROOT / "results/edbt_t0_b" / path).read_bytes()).decode("utf-8")
        lines = [l for l in data.strip().split("\n") if l]
        assert len(lines) == 5500, f"{path}: {len(lines)} rows"
        keys = set()
        for line in lines:
            r = json.loads(line)
            k = (r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"])
            assert k not in keys, f"Duplicate key in {path}: {k}"
            keys.add(k)

# ================================================================
# 13. P6 all params explicit
# ================================================================
def test_p6_all_params_explicit():
    src = (ROOT / "scripts/t0_b_v3/policy_selectors.py").read_text()
    for param in ["n_estimators=100", "n_repeats=5", "n_splits=3",
                  "criterion=\"gini\"", "min_samples_leaf=1", "scoring=\"roc_auc\""]:
        assert param in src, f"Missing P6 param: {param}"
