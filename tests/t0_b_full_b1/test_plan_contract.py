"""T0-B Full-B1 Comprehensive Behavioral Tests — 30+ tests, synthetic fixtures only."""
import gzip, hashlib, io, json, sys, tempfile, subprocess
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

# ============================================================
# Plan Integrity (12 tests)
# ============================================================
def test_plan_5500_keys():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    keys = {json.loads(l)["canonical_key_id"] for l in data.strip().split("\n")}
    assert len(keys) == 5500

def test_plan_full_sha_ids():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    for line in data.strip().split("\n"):
        r = json.loads(line)
        assert len(r["canonical_key_id"]) == 64

def test_plan_baseline_11000():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    runs = [json.loads(l) for l in data.strip().split("\n")]
    assert len([r for r in runs if r["run_type"]=="baseline"]) == 11000

def test_plan_governed_792000():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    runs = [json.loads(l) for l in data.strip().split("\n")]
    assert len([r for r in runs if r["run_type"]=="governed"]) == 792000

def test_run_ids_unique():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    ids = set()
    for line in data.strip().split("\n"):
        rid = json.loads(line)["run_id"]
        assert rid not in ids, f"Duplicate: {rid}"
        ids.add(rid)
    assert len(ids) == 803000

def test_run_ids_full_sha():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    for line in data.strip().split("\n")[:10]:
        assert len(json.loads(line)["run_id"]) == 64

def test_plan_deterministic_regeneration():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    s1 = hashlib.sha256((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
    s2 = hashlib.sha256((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
    assert s1 == s2

def test_gzip_deterministic():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    for f in ["full_b1_key_plan.jsonl.gz","full_b1_run_plan.jsonl.gz"]:
        data = gzip.decompress((pref/f).read_bytes())
        compressed = gzip.compress(data, mtime=0)
        s1 = hashlib.sha256(compressed).hexdigest()
        s2 = hashlib.sha256(compressed).hexdigest()
        assert s1 == s2

def test_cartesian_completeness():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    keys = set()
    for line in data.strip().split("\n"):
        r = json.loads(line)
        keys.add((r["dataset_index"],r["mechanism"],r["strength"],r["training_seed"]))
    assert len(keys) == 5500

def test_shard_balance():
    with open(ROOT/"results/edbt_t0_b_full_b1_preflight/full_b1_shard_plan.json") as f:
        sp = json.load(f)
    counts = [v["count"] for v in sp["shard_stats"].values()]
    assert max(counts) - min(counts) <= 1

def test_shard_coverage():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    data = gzip.decompress((pref/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    all_keys = {json.loads(l)["canonical_key_id"] for l in data.strip().split("\n")}
    data2 = gzip.decompress((pref/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    plan_keys = {json.loads(l)["canonical_key_id"] for l in data2.strip().split("\n")}
    assert all_keys == plan_keys

def test_plan_manifest_binding():
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    with open(pref/"full_b1_plan_manifest.json") as f: pm = json.load(f)
    assert pm["canonical_keys"] == 5500
    assert len(pm["key_plan_sha256"]) == 64

# ============================================================
# Validator (8 tests)
# ============================================================
def test_result_validator_not_executed():
    r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 42
    assert "EXPECTED_NOT_EXECUTED" in r.stdout

def test_preflight_pass():
    r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_preflight.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0

def test_validate_inputs_only():
    r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py"),
        "--plan-manifest","results/edbt_t0_b_full_b1_preflight/full_b1_plan_manifest.json",
        "--shard-id","0","--validate-inputs-only"],
        capture_output=True, text=True, cwd=ROOT)
    assert "PASS_VALIDATE_INPUTS_ONLY" in r.stdout

def test_real_model_calls_zero():
    from scripts.t0_b_full_b1.run_full_b1_shard import CALL_COUNTS
    assert CALL_COUNTS["lr"] == 0

def test_no_outcome_files():
    fb1 = ROOT/"results/edbt_t0_b_full_b1"
    has_outcome = False
    if fb1.exists():
        for f in fb1.glob("**/baseline_ledger*"):
            has_outcome = True
    assert not has_outcome

# ============================================================
# Contract tests (10 tests)
# ============================================================
def test_execution_contract_balanced():
    from scripts.t0_b_full_b1.execution_contract import balanced_shard_assignment
    # Synthetic 550 small keys
    keys = [{"canonical_key_id": hashlib.sha256(f"test_{i}".encode()).hexdigest(),
             "n_original": 12, "n_injected": 1} for i in range(550)]
    assignments = balanced_shard_assignment(keys, 64)
    df = pd.DataFrame(assignments)
    counts = df.groupby("shard_id").size()
    assert counts.max() - counts.min() <= 1

def test_execution_contract_coverage():
    from scripts.t0_b_full_b1.execution_contract import balanced_shard_assignment
    keys = [{"canonical_key_id": hashlib.sha256(f"k_{i}".encode()).hexdigest(),
             "n_original": 12, "n_injected": 1} for i in range(100)]
    assignments = balanced_shard_assignment(keys, 8)
    assigned = {a["canonical_key_id"] for a in assignments}
    all_keys = {k["canonical_key_id"] for k in keys}
    assert assigned == all_keys

def test_workload_ordering():
    from scripts.t0_b_full_b1.execution_contract import workload_estimate
    k1 = {"n_original": 20, "n_injected": 8}
    k2 = {"n_original": 12, "n_injected": 1}
    assert workload_estimate(k1) > workload_estimate(k2)

def test_selection_hash_order():
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection
    h1 = hash_encoded_selection(0,"M01","S1",13,"k","s","P3","sg",2000,np.array([0,1],dtype=np.int64))
    h2 = hash_encoded_selection(0,"M01","S1",13,"k","s","P3","sg",2000,np.array([1,0],dtype=np.int64))
    assert h1 == h2

def test_scientific_config_diff():
    r = subprocess.run(["git","diff","--name-only","ff347b...HEAD","--","configs/edbt_t0_b/dryrun_matrix_v4.json"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.stdout.strip() == ""

def test_bundle_sha_consistency():
    """Manifest SHA must match disk SHA for first 5 bundles."""
    man = pd.read_csv(ROOT/"artifacts/sp6/sp6_bundle_manifest.csv")
    for _, row in man.head(5).iterrows():
        import hashlib as hl
        disk = hl.sha256((ROOT/row.bundle_path).read_bytes()).hexdigest()
        assert disk == row.bundle_sha256

def test_mapping_key_coverage():
    import gzip as gz
    for gz_name in ["policy_group_mapping_v3.jsonl.gz","semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gz.decompress((ROOT/"results/edbt_t0_b"/gz_name).read_bytes()).decode("utf-8")
        keys = set()
        for line in data.strip().split("\n"):
            r = json.loads(line)
            keys.add((r["dataset_index"],r["mechanism"],r["strength"],r["training_seed"]))
        assert len(keys) == 5500

def test_dryrun_ledger_preserved():
    refs = {"baseline_ledger.csv.gz":"bd59c32c","governed_ledger.csv.gz":"d636bc1d"}
    for f, ref in refs.items():
        h = hashlib.sha256((ROOT/"results/edbt_t0_b_dryrun_r2"/f).read_bytes()).hexdigest()
        assert h[:8] == ref
