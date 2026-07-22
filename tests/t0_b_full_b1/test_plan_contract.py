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


def test_formal_plan_passes_shared_production_contract():
    from scripts.t0_b_full_b1.merge_contract import (
        validate_plan_schema, validate_plan, validate_global_scope,
    )
    pref = ROOT / "results/edbt_t0_b_full_b1_preflight"
    manifest = json.loads((pref / "full_b1_plan_manifest.json").read_text())
    errors = validate_plan_schema(manifest, "production")
    plan_errors, keys, runs = validate_plan(manifest, pref)
    errors.extend(plan_errors)
    if not plan_errors:
        errors.extend(validate_global_scope(manifest, keys, runs))
    assert errors == []
    assert len(keys) == 5500
    assert len(runs) == 803000
    assert {row["execution_contract_version"] for row in runs} == {"v1"}


def test_formal_plan_receipt_binds_manifest_and_tool_seal():
    pref = ROOT / "results/edbt_t0_b_full_b1_preflight"
    manifest_path = pref / "full_b1_plan_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((pref / "full_b1_plan_receipt.json").read_text())
    assert receipt["plan_manifest_sha256"] == hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    assert receipt["tool_seal_sha"] == manifest["tool_seal_sha"]
    assert len(manifest["tool_seal_sha"]) == 40
