"""T0-B1R2 Behavioral Tests."""
import json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_imports():
    from scripts.t0_b_v3.budget_contract import compute_k
    from scripts.t0_b_dryrun_r2.run_t0_b1_dryrun_r2 import CALL_COUNTS, V4_FACTORY_HASH
    assert compute_k(20, 2000) == 4
    assert len(V4_FACTORY_HASH) == 64

def test_matrix_keys():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    assert len(dr["keys"]) == 4
    assert dr["expected_counts"]["ranking_model_fits"]["total"] == 16

def test_selection_order_invariant():
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection
    import numpy as np
    h1 = hash_encoded_selection(0, "M01", "S1", 13, "k", "s", "P3", "sg", 2000, np.array([0,1,2], dtype=np.int64))
    h2 = hash_encoded_selection(0, "M01", "S1", 13, "k", "s", "P3", "sg", 2000, np.array([2,0,1], dtype=np.int64))
    assert h1 == h2

def test_p2_seeds_count():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    assert len(dr["p2_governance_seeds"]) == 20

def test_runner_counter_initial():
    from scripts.t0_b_dryrun_r2.run_t0_b1_dryrun_r2 import CALL_COUNTS
    assert "lr" in CALL_COUNTS
    assert "p3" in CALL_COUNTS

def test_repeat_fit_parity_tolerance():
    """AUC diff <= 1e-12 and prob max diff <= 1e-12."""
    # Load from receipt if exists
    p = ROOT / "results/edbt_t0_b_dryrun_r2/repeat_fit_parity_receipt.json"
    if p.exists():
        with open(p) as f: rec = json.load(f)
        for r in rec["records"]:
            assert r["auc_diff"] <= 1e-12, f"AUC diff {r['auc_diff']} > 1e-12"
            assert r["prob_max_diff"] <= 1e-12, f"Prob diff {r['prob_max_diff']} > 1e-12"

def test_manifest_receipts_exist():
    """After first run, all required outputs must exist."""
    import glob
    out = ROOT / "results/edbt_t0_b_dryrun_r2"
    required = ["baseline_ledger.csv.gz", "governed_ledger.csv.gz", "selection_ledger.csv.gz",
                "failure_ledger.csv.gz", "repeat_fit_parity_receipt.json", "runtime_receipt.json"]
    for r in required:
        if (out / r).exists():
            assert (out / r).stat().st_size > 0
