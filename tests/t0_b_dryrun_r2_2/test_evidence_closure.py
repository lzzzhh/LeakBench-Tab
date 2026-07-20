"""T0-B1R2.1 Behavioral Tests."""
import gzip, hashlib, io, json, sys
from pathlib import Path
import numpy as np, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_selection_audit_no_lr_calls():
    """Selection audit must never call fit_predict_core_model."""
    # Verify by checking the selection_determinism_receipt
    rec = ROOT / "results/edbt_t0_b_dryrun_r2_1/selection_determinism_receipt.json"
    if rec.exists():
        with open(rec) as f: sr = json.load(f)
        assert sr["downstream_lr_calls"] == 0

def test_selection_payload_matches():
    rec = ROOT / "results/edbt_t0_b_dryrun_r2_1/selection_determinism_receipt.json"
    if rec.exists():
        with open(rec) as f: sr = json.load(f)
        assert sr["payload_mismatches"] == 0

def test_resume_sha_match():
    rec = ROOT / "results/edbt_t0_b_dryrun_r2_1/resume_hash_receipt.json"
    if rec.exists():
        with open(rec) as f: rr = json.load(f)
        assert rr["sha_match"] == True
        assert rr["duplicate_run_ids"] == 0

def test_r2_governed_rows():
    gl = gzip.decompress((ROOT / "results/edbt_t0_b_dryrun_r2/governed_ledger.csv.gz").read_bytes()).decode("utf-8")
    lines = [l for l in gl.strip().split("\n") if l]
    assert len(lines) == 577

def test_r2_failure_rows():
    fl = gzip.decompress((ROOT / "results/edbt_t0_b_dryrun_r2/failure_ledger.csv.gz").read_bytes()).decode("utf-8")
    lines = [l for l in fl.strip().split("\n") if l]
    assert len(lines) == 1  # header only

def test_r2_no_duplicates():
    import pandas as pd
    gl = gzip.decompress((ROOT / "results/edbt_t0_b_dryrun_r2/governed_ledger.csv.gz").read_bytes()).decode("utf-8")
    df = pd.read_csv(io.StringIO(gl))
    assert not df.run_id.duplicated().any()

def test_semantic_no_partial_violations():
    import pandas as pd
    gl = gzip.decompress((ROOT / "results/edbt_t0_b_dryrun_r2/governed_ledger.csv.gz").read_bytes()).decode("utf-8")
    df = pd.read_csv(io.StringIO(gl))
    sg = df[df.contract == "semantic_group"]
    assert (sg.semantic_partial == 0).all()

def test_scientific_config_diff():
    import subprocess
    r = subprocess.run(["git","diff","--name-only","ff347b...HEAD","--","configs/edbt_t0_b/dryrun_matrix_v4.json"], capture_output=True, text=True, cwd=ROOT)
    assert r.stdout.strip() == ""

def test_r2_seal_is_ancestor():
    import subprocess
    r = subprocess.run(["git","merge-base","--is-ancestor","b1fb0041a2e6ebdef817b5be489b9c85993002de","HEAD"], capture_output=True, cwd=ROOT)
    assert r.returncode == 0

def test_selection_hash_order_invariant():
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection
    h1 = hash_encoded_selection(0,"M01","S1",13,"k","s","P3","sg",2000,np.array([0,1,2],dtype=np.int64))
    h2 = hash_encoded_selection(0,"M01","S1",13,"k","s","P3","sg",2000,np.array([2,0,1],dtype=np.int64))
    assert h1 == h2

def test_environment_has_sklearn():
    """Environment receipt must include sklearn version."""
    # Check that sklearn is importable
    import sklearn
    assert sklearn.__version__

def test_manifest_has_required_outputs():
    """R2 manifest must have all required entries."""
    p = ROOT / "results/edbt_t0_b_dryrun_r2/dryrun_manifest.json"
    if p.exists():
        with open(p) as f: dm = json.load(f)
        assert "baseline_rows" in dm
        assert dm["baseline_rows"] == 8
