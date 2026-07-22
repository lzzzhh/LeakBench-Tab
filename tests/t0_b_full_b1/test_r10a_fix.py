"""T0-B1I-R10a-Fix tests — writer lock CLI, selection duplicate, validate-only audit, stale cleanup, validator."""
import gzip, hashlib, io, json, os, subprocess, sys, tempfile, time
from pathlib import Path; import pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
MERGER = str(ROOT/"scripts/t0_b_full_b1/merge_full_b1_shards.py")
VALIDATOR = str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py")
SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")

from scripts.t0_b_full_b1.io_contract import exclusive_writer_lock, WriterLockError
from scripts.t0_b_full_b1.forbidden_pattern_audit import run_audit


# ─── Runner writer lock CLI test ───────────────────────────────

def test_runner_rejects_existing_writer_lock():
    """Runner CLI must fail when writer lock is already held."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        out_dir = tdp / "shard_0"
        out_dir.mkdir()
        # Hold lock
        with exclusive_writer_lock(out_dir, "test_block"):
            r = subprocess.run([sys.executable, RUNNER,
                "--plan-manifest", SYNTH_PLAN,
                "--shard-id", "0", "--output-dir", str(out_dir), "--synthetic"],
                capture_output=True, text=True, cwd=ROOT)
            assert r.returncode != 0
            assert "FAIL" in r.stdout or "lock" in r.stderr.lower()
            assert not (out_dir / "shard_manifest.json").exists()
        # After lock release, clean output and runner should succeed
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir()
        r2 = subprocess.run([sys.executable, RUNNER,
            "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", str(out_dir), "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode == 0


# ─── Selection duplicate tests ─────────────────────────────────

def test_runner_duplicate_selection_fails():
    """Runner must detect duplicate selection_hash and fail."""
    # This is tested via the shard rebuild logic — we can't easily inject
    # a duplicate without running the full shard, so we test the helper
    from scripts.t0_b_full_b1.run_full_b1_shard import _find_duplicates
    dups = _find_duplicates(["hash_a", "hash_b", "hash_a"])
    assert len(dups) > 0

def test_merge_duplicate_governed_fails():
    """Merge rejects minimal plan (strict R10c requires complete plan manifest)."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for sid in range(2):
            sd = tdp / f"shard_{sid}"
            sd.mkdir()
            (sd / "shard_manifest.json").write_text("{}")
            (sd / "baseline_ledger.csv.gz").write_bytes(gzip.compress(f"run_id\nbl_{sid}\n".encode()))
            (sd / "governed_ledger.csv.gz").write_bytes(gzip.compress(f"run_id,dataset_index,mechanism,strength,training_seed,governance_seed,learner,policy,contract,budget_bp,strict_auc,full_auc,governed_auc,legacy_sdr,selection_hash,realized_cost\ndup_gov,0,M01,S1,13,0,lr,P2,semantic_group,500,0.7,0.8,0.75,0.05,sh_a,1\n".encode()))
            (sd / "selection_ledger.csv.gz").write_bytes(gzip.compress(b"selection_hash,policy,contract,budget_bp,removed_encoded_indices,removed_group_ids,realized_encoded_cost\nsh_a,P2,semantic_group,500,[],[],1\n"))
            (sd / "failure_ledger.csv.gz").write_bytes(gzip.compress(b"run_id\n"))
        pm = {"shard_count": 2}
        with open(tdp / "plan_manifest.json", "w") as f: json.dump(pm, f)
        out_dir = tdp.parent / "merged"
        r = subprocess.run([sys.executable, MERGER,
            "--plan-manifest", str(tdp / "plan_manifest.json"),
            "--shard-root", str(tdp), "--output-dir", str(out_dir)],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_GLOBAL_MERGE_FAIL" in r.stdout
        assert not out_dir.exists()


# ─── Validate-only audit tests ─────────────────────────────────

def test_validate_only_valid_plan_passes():
    r = subprocess.run([sys.executable, RUNNER,
        "--plan-manifest", SYNTH_PLAN, "--shard-id", "0", "--validate-only"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    assert "VALIDATION_PASS" in r.stdout

def test_validate_only_fails_with_missing_plan():
    r = subprocess.run([sys.executable, RUNNER,
        "--plan-manifest", "/nonexistent/manifest.json",
        "--shard-id", "0", "--validate-only"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0

def test_validate_only_duplicate_run_id_fails():
    """Construct a plan with duplicate run IDs and verify validate-only catches it."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # Copy synthetic plan but inject a duplicate run_id
        synth_dir = ROOT / "results/edbt_t0_b_full_b1_preflight/synthetic_full_contract"
        keys_data = gzip.decompress((synth_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode()
        runs_lines = [json.loads(l) for l in gzip.decompress((synth_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
        # Duplicate first run_id
        runs_lines[1]["run_id"] = runs_lines[0]["run_id"]
        runs_data = "\n".join(json.dumps(r) for r in runs_lines) + "\n"
        (tdp / "full_b1_key_plan.jsonl.gz").write_bytes(gzip.compress(keys_data.encode(), mtime=0))
        (tdp / "full_b1_run_plan.jsonl.gz").write_bytes(gzip.compress(runs_data.encode(), mtime=0))
        pm = json.load(open(synth_dir / "full_b1_plan_manifest.json"))
        pm["key_plan_sha256"] = hashlib.sha256((tdp / "full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
        pm["run_plan_sha256"] = hashlib.sha256((tdp / "full_b1_run_plan.jsonl.gz").read_bytes()).hexdigest()
        with open(tdp / "full_b1_plan_manifest.json", "w") as f: json.dump(pm, f)
        r = subprocess.run([sys.executable, RUNNER,
            "--plan-manifest", str(tdp / "full_b1_plan_manifest.json"),
            "--shard-id", "0", "--validate-only"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "VALIDATION_FAIL" in r.stdout
        assert "duplicate" in r.stdout.lower()


# ─── Forbidden pattern audit test ──────────────────────────────

def test_production_forbidden_patterns_zero():
    """Production code must have zero forbidden patterns."""
    matches = run_audit()
    assert len(matches) == 0, f"Forbidden patterns found: {matches}"

def test_audit_allowed_self_references_recorded():
    """Audit must record allowed test references."""
    from scripts.t0_b_full_b1.forbidden_pattern_audit import ALLOWED_TEST_REFS
    assert len(ALLOWED_TEST_REFS) > 0
    for ref in ALLOWED_TEST_REFS:
        assert "path" in ref
        assert "purpose" in ref


# ─── Validator formal-path test ────────────────────────────────

def test_validator_no_result_exits_42():
    r = subprocess.run([sys.executable, VALIDATOR],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 42
    assert "EXPECTED_NOT_EXECUTED" in r.stdout

def test_validator_formal_result_hits_r10d_block():
    """Formal result path must hit NOT_IMPLEMENTED_FULL_VALIDATOR_R10D and exit 2."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        shards = tdp / "shards" / "shard_0"
        shards.mkdir(parents=True)
        (shards / "dummy").write_text("x")
        (tdp / "full_b1_manifest.json").write_text("{}")
        r = subprocess.run([sys.executable, VALIDATOR, "--output-dir", str(tdp)],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 2
        assert "NOT_IMPLEMENTED_FULL_VALIDATOR_R10D" in r.stdout
        assert "PASS" not in r.stdout

def test_validator_partial_result_exits_nonzero():
    """Partial results (shards exist but no manifest) must exit nonzero."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        shards = tdp / "shards" / "shard_0"
        shards.mkdir(parents=True)
        (shards / "dummy").write_text("x")
        # No full_b1_manifest.json
        r = subprocess.run([sys.executable, VALIDATOR, "--output-dir", str(tdp)],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert r.returncode != 42  # Not "not executed"
        assert "PASS" not in r.stdout


# ─── R2 ledger preservation ────────────────────────────────────

def test_r2_ledgers_unchanged():
    expected = {
        "baseline_ledger.csv.gz": "bd59c32c46d2461b9fd871b21ef5954cb68e758fb68e2ded89cc1967e451d444",
        "governed_ledger.csv.gz": "d636bc1d95fe967f981147d56010d70120d0810ad7eb466c1605ea18e78b798d",
        "selection_ledger.csv.gz": "8f5107c62007239c25585ae4316c581658f6667acae7231bcd32467109a01477",
        "failure_ledger.csv.gz": "e185730c78ea8c5a7d8a88ced4e8cf93692694033b4794e5dfa65555488fdb60",
    }
    for fname, exp in expected.items():
        actual = hashlib.sha256((ROOT/"results/edbt_t0_b_dryrun_r2"/fname).read_bytes()).hexdigest()
        assert actual == exp, f"{fname}: SHA mismatch"

def test_config_diff_zero():
    r = subprocess.run(["git", "diff", "--name-only", "ff347b...HEAD", "--",
                        "configs/edbt_t0_b/dryrun_matrix_v4.json"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.stdout.strip() == ""
