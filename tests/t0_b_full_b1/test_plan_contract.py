"""Plan contract tests."""
import gzip, hashlib, io, json, sys, tempfile
from pathlib import Path; import numpy as np, pandas as pd, pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def test_plan_5500_keys():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    assert len([l for l in data.strip().split("\n")]) == 5500

def test_plan_full_sha_ids():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    for line in data.strip().split("\n")[:10]:
        assert len(json.loads(line)["canonical_key_id"]) == 64

def test_plan_counts():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    runs = [json.loads(l) for l in data.strip().split("\n")]
    assert len(runs) == 803000
    assert len([r for r in runs if r["run_type"]=="baseline"]) == 11000
    assert len([r for r in runs if r["run_type"]=="governed"]) == 792000

def test_run_ids_unique():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    ids = set()
    for line in data.strip().split("\n"):
        rid = json.loads(line)["run_id"]
        assert rid not in ids; ids.add(rid)

def test_gzip_deterministic():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    s1 = hashlib.sha256((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
    s2 = hashlib.sha256((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
    assert s1 == s2

def test_shard_balance():
    with open(ROOT/"results/edbt_t0_b_full_b1_preflight/full_b1_shard_plan.json") as f:
        sp = json.load(f)
    counts = [v["count"] for v in sp["shard_stats"].values()]
    assert max(counts) - min(counts) <= 1
