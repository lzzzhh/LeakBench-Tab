"""T0-B1 Dry-Run Runner Tests — verify imports and frozen config integrity before execution."""
import json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_dryrun_matrix_has_4_keys():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    assert len(dr["keys"]) == 4

def test_dryrun_budgets_bp():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    assert dr["budgets_bp"] == [500, 1000, 2000]

def test_dryrun_contracts():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    assert set(dr["contracts"]) == {"semantic_group", "encoded_column"}

def test_dryrun_p2_seeds_count():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    assert len(dr["p2_governance_seeds"]) == 20

def test_dryrun_expected_counts():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    ec = dr["expected_counts"]
    assert ec["total_downstream_rows"] == 584
    assert ec["total_downstream_fits"] == 584
    assert ec["ranking_model_fits"]["total"] == 16

def test_imports_from_frozen_modules():
    """Runner must import from frozen V3/V4 modules."""
    from scripts.t0_b_v3.budget_contract import compute_k
    from scripts.t0_b_v3.seed_contract import derive_p2_seed
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
    from scripts.t0_b_v3.policy_selectors import score_mi, score_point_biserial, score_lr_coef, score_rf_permutation
    assert compute_k(20, 2000) == 4

def test_selection_order_invariant():
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection
    import numpy as np
    h1 = hash_encoded_selection(0, "M01", "S1", 13, "k", "s", "P3", "sg", 2000, np.array([0,1,2], dtype=np.int64))
    h2 = hash_encoded_selection(0, "M01", "S1", 13, "k", "s", "P3", "sg", 2000, np.array([2,0,1], dtype=np.int64))
    assert h1 == h2

def test_p2_governance_seeds():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    seeds = dr["p2_governance_seeds"]
    assert seeds[0] == 2026071700
    assert seeds[-1] == 2026071719
