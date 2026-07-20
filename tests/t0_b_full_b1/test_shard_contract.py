"""T0-B Shard Contract Tests."""
import hashlib, sys
from pathlib import Path; import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_balanced_shard_counts():
    from scripts.t0_b_full_b1.execution_contract import balanced_shard_assignment
    keys = [{"canonical_key_id":hashlib.sha256(f"k{i}".encode()).hexdigest(),"n_original":12,"n_injected":1} for i in range(5500)]
    assignments = balanced_shard_assignment(keys, 64)
    import pandas as pd
    counts = pd.DataFrame(assignments).groupby("shard_id").size()
    assert counts.max() - counts.min() <= 1

def test_key_not_cross_shard():
    from scripts.t0_b_full_b1.execution_contract import balanced_shard_assignment
    keys = [{"canonical_key_id":hashlib.sha256(f"k{i}".encode()).hexdigest(),"n_original":12,"n_injected":1} for i in range(100)]
    assignments = balanced_shard_assignment(keys, 8)
    seen = {}
    for a in assignments:
        assert a["canonical_key_id"] not in seen
        seen[a["canonical_key_id"]] = a["shard_id"]

def test_shard_assignment_deterministic():
    from scripts.t0_b_full_b1.execution_contract import balanced_shard_assignment
    keys = [{"canonical_key_id":hashlib.sha256(f"k{i}".encode()).hexdigest(),"n_original":12,"n_injected":1} for i in range(100)]
    a1 = balanced_shard_assignment(keys, 8)
    a2 = balanced_shard_assignment(keys, 8)
    assert a1 == a2

def test_workload_estimates():
    from scripts.t0_b_full_b1.execution_contract import workload_estimate
    assert workload_estimate({"n_original":24,"n_injected":8}) > workload_estimate({"n_original":12,"n_injected":1})
