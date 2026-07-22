"""Tests for duplicate detection, failure preservation, validate-only, writer lock CLI."""
import gzip, hashlib, io, json, os, subprocess, sys, tempfile
from pathlib import Path; import pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
MERGER = str(ROOT/"scripts/t0_b_full_b1/merge_full_b1_shards.py")

from scripts.t0_b_full_b1.io_contract import exclusive_writer_lock, WriterLockError
from scripts.t0_b_full_b1.run_full_b1_shard import _find_duplicates


def test_find_duplicates_empty():
    assert _find_duplicates([]) == []

def test_find_duplicates_no_dups():
    assert _find_duplicates(["a", "b", "c"]) == []

def test_find_duplicates_one_dup():
    result = _find_duplicates(["a", "b", "a"])
    assert "a" in result

def test_find_duplicates_multiple_dups():
    result = _find_duplicates(["a", "b", "a", "c", "b", "a"])
    assert "a" in result
    assert "b" in result


def test_validator_no_result_exits_42():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([
            sys.executable,
            str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py"),
            "--output-dir", str(Path(td) / "not_executed"),
        ], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 42
        assert "EXPECTED_NOT_EXECUTED" in r.stdout


def test_validator_formal_result_does_not_false_pass():
    """Formal result path must NOT exit 0."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # Create a fake shards dir with a manifest to trigger formal path
        output_dir = tdp / "results/edbt_t0_b_full_b1"
        shards = output_dir / "shards" / "shard_0"
        shards.mkdir(parents=True)
        (shards / "dummy").write_text("x")
        (output_dir / "full_b1_manifest.json").write_text("{}")
        r = subprocess.run([
            sys.executable,
            str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py"),
            "--output-dir", str(output_dir),
        ], capture_output=True, text=True, cwd=tdp)
        assert r.returncode != 0
        assert "PASS" not in r.stdout


def test_validate_only_passes_with_valid_plan():
    """validate-only must actually validate plan SHA."""
    synth_plan = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")
    r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", synth_plan,
                        "--shard-id", "0", "--validate-only", "--synthetic"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0
    assert "VALIDATION_PASS" in r.stdout


def test_validate_only_fails_with_missing_plan():
    r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", "/nonexistent/manifest.json",
                        "--shard-id", "0", "--validate-only"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0


def test_merge_concurrent_writer_fails():
    """Second merge writer must fail with non-zero exit."""
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # Create fake shard structure
        for sid in range(2):
            sd = tdp / f"shard_{sid}"
            sd.mkdir()
            (sd / "shard_manifest.json").write_text("{}")
            (sd / "baseline_ledger.csv.gz").write_bytes(gzip.compress(b"run_id\n"))
            (sd / "governed_ledger.csv.gz").write_bytes(gzip.compress(b"run_id\n"))
            (sd / "selection_ledger.csv.gz").write_bytes(gzip.compress(b"selection_hash\n"))
            (sd / "failure_ledger.csv.gz").write_bytes(gzip.compress(b"run_id\n"))
        plan_manifest = tdp / "plan_manifest.json"
        with open(plan_manifest, "w") as f: json.dump({"shard_count": 2}, f)
        out_a = tdp / "merged_a"
        out_a.mkdir()
        # Hold lock on out_a
        with exclusive_writer_lock(out_a, "test_block"):
            r = subprocess.run([sys.executable, MERGER,
                "--plan-manifest", str(plan_manifest),
                "--shard-root", str(tdp), "--output-dir", str(out_a)],
                capture_output=True, text=True, cwd=ROOT)
            assert r.returncode != 0
            assert "FAIL" in r.stdout or "WriterLockError" in r.stderr


def test_config_diff_zero():
    r = subprocess.run(["git", "diff", "--name-only", "ff347b...HEAD", "--",
                        "configs/edbt_t0_b/dryrun_matrix_v4.json"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.stdout.strip() == ""


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


def test_no_forbidden_patterns():
    """Post-fix: production code must have zero forbidden patterns (use audit module)."""
    from scripts.t0_b_full_b1.forbidden_pattern_audit import run_audit
    matches = run_audit()
    assert len(matches) == 0, f"Forbidden patterns found in production: {matches}"
