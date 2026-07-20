"""Full-result validator tests — synthetic fixtures and mutation detection."""
import gzip, io, json, subprocess, sys, tempfile
from pathlib import Path; import pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_not_executed_returns_42():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 42

def test_synthetic_fixture_passes():
    """Validator must PASS on a correct synthetic fixture."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # Small valid fixture: 10 baseline, 100 governed, all unique
        bl = pd.DataFrame({"run_id":[f"bl_{i:05d}" for i in range(10)], "dataset_index":list(range(10)), "mechanism":["M01"]*10, "strength":["S1"]*10, "training_seed":[13]*10})
        gl = pd.DataFrame({"run_id":[f"gl_{i:06d}" for i in range(100)], "policy":["P2"]*100, "contract":["semantic_group"]*100, "mechanism":["M01"]*100, "governance_seed":list(range(100)), "selection_hash":[f"sh_{i:06d}" for i in range(100)], "budget_bp":[2000]*100, "realized_cost":[4]*100, "dataset_index":[0]*100, "strength":["S1"]*100, "training_seed":[13]*100, "learner":["lr"]*100, "strict_auc":[0.7]*100, "full_auc":[0.8]*100, "governed_auc":[0.75]*100, "legacy_sdr":[0.05]*100})
        for name, df in [("baseline_ledger",bl),("governed_ledger",gl)]:
            buf = io.StringIO(); df.to_csv(buf, index=False)
            (tdp/f"{name}.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))
        # Create failure ledger (empty)
        fl = pd.DataFrame(columns=["run_id"])
        buf = io.StringIO(); fl.to_csv(buf, index=False)
        (tdp/"failure_ledger.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))
        r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py"),
            "--synthetic-fixture", td], capture_output=True, text=True, cwd=ROOT)
        assert "PASS" in r.stdout or r.returncode == 0

def test_duplicate_run_ids_fail():
    """Validator must FAIL on duplicate run_ids."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        bl = pd.DataFrame({"run_id":["dup"]*10, "dataset_index":list(range(10)), "mechanism":["M01"]*10, "strength":["S1"]*10, "training_seed":[13]*10})
        gl = pd.DataFrame({"run_id":["dup"]*100, "policy":["P2"]*100, "contract":["semantic_group"]*100, "mechanism":["M01"]*100, "governance_seed":list(range(100)), "selection_hash":[f"sh_{i}" for i in range(100)], "budget_bp":[2000]*100, "realized_cost":[4]*100, "dataset_index":[0]*100, "strength":["S1"]*100, "training_seed":[13]*100, "learner":["lr"]*100, "strict_auc":[0.7]*100, "full_auc":[0.8]*100, "governed_auc":[0.75]*100, "legacy_sdr":[0.05]*100})
        for name, df in [("baseline_ledger",bl),("governed_ledger",gl)]:
            buf = io.StringIO(); df.to_csv(buf, index=False)
            (tdp/f"{name}.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))
        fl = pd.DataFrame(columns=["run_id"])
        buf = io.StringIO(); fl.to_csv(buf, index=False)
        (tdp/"failure_ledger.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))
        r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py"),
            "--synthetic-fixture", td], capture_output=True, text=True, cwd=ROOT)
        assert "FAIL" in r.stdout or r.returncode != 0
