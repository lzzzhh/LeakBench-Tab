"""T0-B1R Runner Tests."""
import json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_runner_imports():
    from scripts.t0_b_v3.budget_contract import compute_k
    from scripts.t0_b_v3.seed_contract import derive_p2_seed
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection
    assert compute_k(20, 2000) == 4

def test_dryrun_matrix_keys():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    assert len(dr["keys"]) == 4
    assert dr["expected_counts"]["total_downstream_rows"] == 584

def test_factory_hash():
    from scripts.t0_b_dryrun_r1.run_t0_b1_dryrun_r1 import V4_FACTORY_HASH
    assert len(V4_FACTORY_HASH) == 64

def test_selection_order_invariant():
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection
    import numpy as np
    h1 = hash_encoded_selection(0, "M01", "S1", 13, "k", "s", "P3", "sg", 2000, np.array([0,1,2], dtype=np.int64))
    h2 = hash_encoded_selection(0, "M01", "S1", 13, "k", "s", "P3", "sg", 2000, np.array([2,0,1], dtype=np.int64))
    assert h1 == h2
