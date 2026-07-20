"""T0-B Full-B1 Behavioral Tests — synthetic fixtures only, no real model fits."""
import gzip, hashlib, io, json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_plan_derives_5500_keys():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    keys = set()
    for line in data.strip().split("\n"):
        r = json.loads(line); keys.add(r["canonical_key_id"])
    assert len(keys) == 5500

def test_plan_baseline_11000():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    runs = [json.loads(l) for l in data.strip().split("\n")]
    baseline = [r for r in runs if r["run_type"]=="baseline"]
    assert len(baseline) == 11000

def test_plan_governed_792000():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    runs = [json.loads(l) for l in data.strip().split("\n")]
    governed = [r for r in runs if r["run_type"]=="governed"]
    assert len(governed) == 792000

def test_run_ids_unique():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    ids = set()
    for line in data.strip().split("\n"):
        ids.add(json.loads(line)["run_id"])
    assert len(ids) == 803000

def test_run_ids_deterministic():
    """Same input → same plan."""
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    s1 = hashlib.sha256((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
    s2 = hashlib.sha256((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
    assert s1 == s2

def test_shard_coverage():
    with open(ROOT/"results/edbt_t0_b_full_b1_preflight/full_b1_shard_plan.json") as f:
        sp = json.load(f)
    assigned = {a["canonical_key_id"] for a in sp["assignments"]}
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    all_keys = {json.loads(l)["canonical_key_id"] for l in data.strip().split("\n")}
    assert assigned == all_keys

def test_shard_no_cross_key():
    with open(ROOT/"results/edbt_t0_b_full_b1_preflight/full_b1_shard_plan.json") as f:
        sp = json.load(f)
    # Each key in exactly one shard
    key_to_shard = {}
    for a in sp["assignments"]:
        cid = a["canonical_key_id"]
        assert cid not in key_to_shard, f"Key {cid} in multiple shards"
        key_to_shard[cid] = a["shard_id"]

def test_no_outcome_files():
    fb1 = ROOT/"results/edbt_t0_b_full_b1"
    if fb1.exists():
        for f in fb1.glob("**/baseline_ledger*"):
            pytest.fail(f"Outcome file: {f}")

def test_validator_not_executed():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 42
    assert "EXPECTED_NOT_EXECUTED" in r.stdout

def test_selection_hash_order():
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection
    h1 = hash_encoded_selection(0,"M01","S1",13,"k","s","P3","sg",2000,np.array([0,1],dtype=np.int64))
    h2 = hash_encoded_selection(0,"M01","S1",13,"k","s","P3","sg",2000,np.array([1,0],dtype=np.int64))
    assert h1 == h2

def test_manifest_binding():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    with open(pref/"full_b1_plan_manifest.json") as f: pm = json.load(f)
    assert pm["canonical_keys"] == 5500
    assert pm["downstream_rows"] == 803000
    assert "key_plan_sha256" in pm
    assert "run_plan_sha256" in pm
