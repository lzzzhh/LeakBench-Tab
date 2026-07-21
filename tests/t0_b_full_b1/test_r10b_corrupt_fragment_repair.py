"""R10b-3 Corrupt fragment repair tests — CLI-level quarantine and single-key recompute."""
import gzip, hashlib, io, json, os, subprocess, sys, tempfile
from pathlib import Path; import pytest
from scripts.t0_b_full_b1.resume_contract import (
    classify_completed_key_failure, ResumeReasonCode, ClassifiedValidationFailure,
)
from scripts.t0_b_full_b1.fragment_contract import CompletedKeyValidation

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")

def _first_exec(out):
    r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
        "--shard-id", "0", "--output-dir", out, "--synthetic"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[:300]


def test_repair_invalid_requires_resume():
    r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
        "--shard-id", "0", "--repair-invalid"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2
    assert "REPAIR_INVALID_REQUIRES_RESUME" in r.stdout


def test_default_resume_corrupt_fragment_stays_fail_closed():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _first_exec(out)
        # Corrupt first key's governed
        frags = Path(out) / "key_fragments"
        key_dir = sorted(frags.iterdir())[0]
        gpath = key_dir / "governed.csv.gz"
        data = bytearray(gpath.read_bytes())
        data[len(data)//2] ^= 0x01
        gpath.write_bytes(bytes(data))
        # Resume without repair
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "RESUME_VALIDATION_FAIL" in r.stdout


def test_corrupt_fragment_repairs_and_restores_clean_ledgers():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _first_exec(out)
        # Record clean shard SHAs
        clean_shas = {}
        for fname in ["baseline_ledger", "governed_ledger", "selection_ledger", "failure_ledger"]:
            fp = Path(out) / f"{fname}.csv.gz"
            clean_shas[fname] = hashlib.sha256(fp.read_bytes()).hexdigest()
        # Record valid key SHAs (keys 2-4)
        frags = Path(out) / "key_fragments"
        all_keys = sorted(frags.iterdir())
        corrupt_key = all_keys[0]
        valid_keys = all_keys[1:]
        valid_shas_before = {}
        for vk in valid_keys:
            for fname in ["baseline.csv.gz","governed.csv.gz","selection.csv.gz","failure.csv.gz",
                          "fragment_manifest.json","completion_receipt.json"]:
                valid_shas_before[f"{vk.name}/{fname}"] = hashlib.sha256((vk/fname).read_bytes()).hexdigest()
        # Corrupt corrupt_key's governed
        gpath = corrupt_key / "governed.csv.gz"
        data = bytearray(gpath.read_bytes())
        data[len(data)//2] ^= 0x01
        gpath.write_bytes(bytes(data))
        # Repair
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume", "--repair-invalid"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[:300]
        assert "Quarantined" in r.stdout
        # Check quarantine
        qdir = Path(out) / "quarantine"
        assert qdir.exists()
        qdirs = list(qdir.iterdir())
        assert len(qdirs) == 1
        assert (qdirs[0] / "fragment_sha_mismatch").exists()
        # Check valid keys unchanged
        for vk in valid_keys:
            for fname in ["baseline.csv.gz","governed.csv.gz","selection.csv.gz","failure.csv.gz",
                          "fragment_manifest.json","completion_receipt.json"]:
                after_sha = hashlib.sha256((vk/fname).read_bytes()).hexdigest()
                assert valid_shas_before[f"{vk.name}/{fname}"] == after_sha, f"{vk.name}/{fname} changed!"
        # Check clean ledger SHAs restored
        for fname in ["baseline_ledger", "governed_ledger", "selection_ledger", "failure_ledger"]:
            fp = Path(out) / f"{fname}.csv.gz"
            after_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
            assert clean_shas[fname] == after_sha, f"{fname} SHA differs after repair!"
        # Check resume receipt
        with open(Path(out)/"resume_receipt.json") as f: rr = json.load(f)
        assert rr["validated_complete"] == 3
        assert len(rr["validated_complete_key_ids"]) == 3
        assert rr["recomputed"] == 1
        assert len(rr["recomputed_key_ids"]) == 1
        assert rr["skipped"] == 3
        assert len(rr["skipped_key_ids"]) == 3
        assert rr["quarantined"] == 1
        assert len(rr["quarantined_key_ids"]) == 1
        assert rr["invalid"] == 1
        assert len(rr["invalid_key_ids"]) == 1
        assert rr["repairable_invalid"] == 1
        assert rr["unsupported_invalid"] == 0
        assert rr["mode"] == "partial_repair"
        assert rr["post_repair_all_keys_valid"] is True


def test_partial_repair_counter_delta():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _first_exec(out)
        frags = Path(out) / "key_fragments"
        corrupt_key = sorted(frags.iterdir())[0]
        gpath = corrupt_key / "governed.csv.gz"
        data = bytearray(gpath.read_bytes())
        data[0] ^= 0x01
        gpath.write_bytes(bytes(data))
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume", "--repair-invalid"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        # Check shard execution receipt
        with open(Path(out)/"shard_execution_receipt.json") as f: ser = json.load(f)
        scd = ser["synthetic_call_counter_delta"]
        assert scd["lr_calls"] == 146, f"lr={scd['lr_calls']}"
        assert scd["p3_calls"] == 1
        assert scd["p4_calls"] == 1
        assert scd["p5_calls"] == 1
        assert scd["p6_calls"] == 1
        pgd = ser["production_guard_delta"]
        assert pgd["real_bundle_loads"] == 0


def test_mixed_sha_receipt_unrepairable():
    """classify_completed_key_failure marks mixed SHA+non-SHA errors as unrepairable."""
    validation = CompletedKeyValidation(
        is_complete=False,
        errors=[
            "governed SHA mismatch",
            "completion receipt missing",
        ],
    )
    cf = classify_completed_key_failure("test_key_001", validation)
    assert cf.repairable is False
    assert cf.reason_code not in (ResumeReasonCode.FRAGMENT_SHA_MISMATCH,)
    # Should be RECEIPT_MISSING (first non-SHA error)
    assert cf.reason_code == ResumeReasonCode.RECEIPT_MISSING


def test_mixed_sha_run_id_unrepairable():
    """classify_completed_key_failure marks mixed SHA+run_id errors as unrepairable."""
    validation = CompletedKeyValidation(
        is_complete=False,
        errors=[
            "baseline SHA mismatch",
            "selection SHA mismatch",
            "planned run has null selection_hash",
        ],
    )
    cf = classify_completed_key_failure("test_key_002", validation)
    assert cf.repairable is False
    assert cf.reason_code not in (ResumeReasonCode.FRAGMENT_SHA_MISMATCH,)


def test_missing_receipt_remains_unsupported():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _first_exec(out)
        # Record shard-level artifact SHAs before
        shard_before = {}
        for fname in ["baseline_ledger.csv.gz", "governed_ledger.csv.gz",
                       "selection_ledger.csv.gz", "failure_ledger.csv.gz",
                       "shard_manifest.json", "shard_execution_receipt.json"]:
            fp = Path(out) / fname
            shard_before[fname] = hashlib.sha256(fp.read_bytes()).hexdigest()
        # Remove receipt from first key
        frags = Path(out) / "key_fragments"
        first_key = sorted(frags.iterdir())[0]
        (first_key / "completion_receipt.json").unlink()
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume", "--repair-invalid"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "RESUME_REPAIR_UNSUPPORTED" in r.stdout
        # Verify no quarantine created
        qdir = Path(out) / "quarantine"
        assert not qdir.exists(), "quarantine should not exist for unsupported repair"
        # Verify shard-level artifacts are immutable
        for fname in ["baseline_ledger.csv.gz", "governed_ledger.csv.gz",
                       "selection_ledger.csv.gz", "failure_ledger.csv.gz",
                       "shard_manifest.json", "shard_execution_receipt.json"]:
            fp = Path(out) / fname
            after = hashlib.sha256(fp.read_bytes()).hexdigest()
            assert shard_before[fname] == after, f"{fname} changed during unsupported repair!"
        # Verify no resume receipt was written
        assert not (Path(out) / "resume_receipt.json").exists()


# ═══════════════════════════════════════════════════════════════════
# NEW R10b-3-Fix-2 TESTS
# ═══════════════════════════════════════════════════════════════════

def _exec_repair(out):
    """First execution then repair with corrupted key."""
    _first_exec(out)
    frags = Path(out) / "key_fragments"
    corrupt_key = sorted(frags.iterdir())[0]
    gpath = corrupt_key / "governed.csv.gz"
    data = bytearray(gpath.read_bytes())
    data[len(data)//2] ^= 0x01
    gpath.write_bytes(bytes(data))
    r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
        "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume", "--repair-invalid"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[:300]
    return r


def test_partial_repair_receipt_contains_exact_key_sets():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _exec_repair(out)
        with open(Path(out)/"resume_receipt.json") as f: rr = json.load(f)
        # 3 valid + 1 invalid = 4 total
        assert rr["validated_complete"] == 3
        assert len(rr["validated_complete_key_ids"]) == 3
        assert rr["invalid"] == 1
        assert len(rr["invalid_key_ids"]) == 1
        # The corrupt key should be the same in all these lists
        corrupt_id = rr["recomputed_key_ids"][0]
        assert corrupt_id in rr["invalid_key_ids"]
        assert corrupt_id in rr["repairable_invalid_key_ids"]
        assert corrupt_id in rr["quarantined_key_ids"]
        assert corrupt_id not in rr["skipped_key_ids"]
        assert corrupt_id not in rr["validated_complete_key_ids"]
        # 1 repairable, 0 unsupported
        assert rr["repairable_invalid"] == 1
        assert len(rr["repairable_invalid_key_ids"]) == 1
        assert rr["unsupported_invalid"] == 0
        assert rr["unsupported_invalid_key_ids"] == []
        # recomputed = 1, skipped = 3, quarantined = 1
        assert rr["recomputed"] == 1
        assert rr["skipped"] == 3
        assert rr["quarantined"] == 1
        assert len(rr["recomputed_key_ids"]) == 1
        assert len(rr["skipped_key_ids"]) == 3
        assert len(rr["quarantined_key_ids"]) == 1


def test_partial_repair_receipt_binds_reason_and_relative_quarantine_path():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _exec_repair(out)
        with open(Path(out)/"resume_receipt.json") as f: rr = json.load(f)
        corrupt_id = rr["recomputed_key_ids"][0]
        # Reason code binding
        assert rr["reason_codes"][corrupt_id] == "fragment_sha_mismatch"
        # Quarantine path must be relative
        qpath = rr["quarantine_paths"][corrupt_id]
        assert not qpath.startswith("/"), f"path {qpath} should be relative"
        assert "quarantine" in qpath
        assert "fragment_sha_mismatch" in qpath


def test_partial_repair_new_rows_are_actual():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _exec_repair(out)
        with open(Path(out)/"resume_receipt.json") as f: rr = json.load(f)
        nr = rr["new_rows"]
        assert nr["baseline"] == 2, f"baseline={nr['baseline']}"
        assert nr["governed"] == 144, f"governed={nr['governed']}"
        assert nr["selection"] == 144, f"selection={nr['selection']}"
        assert nr["failure"] == 0, f"failure={nr['failure']}"
        fr = rr["final_rows"]
        assert fr["baseline"] == 8   # 4 keys x 2
        assert fr["governed"] == 576  # 4 keys x 144
        assert fr["selection"] == 576
        assert fr["failure"] == 0


def test_partial_repair_final_validation_results_are_four_of_four():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _exec_repair(out)
        with open(Path(out)/"resume_receipt.json") as f: rr = json.load(f)
        fvr = rr["final_validation_results"]
        assert len(fvr) == 4, f"expected 4 keys, got {len(fvr)}"
        for cid, v in fvr.items():
            assert v["is_complete"] is True, f"{cid} not complete"
            assert v["errors"] == [], f"{cid} has errors: {v['errors']}"
        assert rr["post_repair_all_keys_valid"] is True


def test_quarantine_recomputes_and_verifies_all_six_target_shas():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _exec_repair(out)
        qdirs = list((Path(out) / "quarantine").iterdir())
        assert len(qdirs) == 1
        qdir = qdirs[0] / "fragment_sha_mismatch"
        ts_dirs = list(qdir.iterdir())
        assert len(ts_dirs) == 1
        qr_path = ts_dirs[0] / "quarantine_receipt.json"
        with open(qr_path) as f: qr = json.load(f)
        assert qr["integrity_verified"] is True
        orig = qr["original_artifact_sha256"]
        moved = qr["moved_artifact_sha256"]
        assert set(orig.keys()) == set(moved.keys())
        for fname in orig:
            assert orig[fname] == moved[fname], f"{fname}: {orig[fname]} != {moved[fname]}"


def test_directory_fsync_error_is_not_silenced():
    """monkeypatch os.fsync to raise OSError — must raise RuntimeError and close fd."""
    from scripts.t0_b_full_b1 import resume_contract as rc_mod
    import builtins
    original_fsync = os.fsync
    close_calls = []
    original_close = os.close

    def raising_fsync(fd):
        raise OSError("simulated fsync failure")

    def tracking_close(fd):
        close_calls.append(fd)
        original_close(fd)

    try:
        os.fsync = raising_fsync
        os.close = tracking_close
        with pytest.raises(RuntimeError, match="directory fsync failed"):
            rc_mod._fsync_directory(Path("/tmp"))
        assert len(close_calls) == 1, "fd was not closed after fsync failure"
    finally:
        os.fsync = original_fsync
        os.close = original_close


def test_complete_resume_receipt_uses_real_validation_results():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _first_exec(out)
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        with open(Path(out)/"resume_receipt.json") as f: rr = json.load(f)
        assert rr["mode"] == "complete_resume"
        assert rr["validation_phase"] == "pre_resume"
        assert rr["validated_complete"] == 4
        assert len(rr["validated_complete_key_ids"]) == 4
        assert rr["invalid"] == 0
        assert rr["invalid_key_ids"] == []
        assert rr["recomputed"] == 0
        assert rr["recomputed_key_ids"] == []
        assert rr["skipped"] == 4
        assert len(rr["skipped_key_ids"]) == 4
        assert rr["quarantined"] == 0
        # Final validation results: all 4 keys complete
        fvr = rr["final_validation_results"]
        assert len(fvr) == 4
        for cid, v in fvr.items():
            assert v["is_complete"] is True
            assert v["errors"] == []
        assert rr["post_repair_all_keys_valid"] is True
        # Counter deltas must all be 0 for complete resume
        sd = rr["synthetic_call_counter_delta"]
        assert sd["lr_calls"] == 0; assert sd["p3_calls"] == 0; assert sd["p4_calls"] == 0
        assert sd["p5_calls"] == 0; assert sd["p6_calls"] == 0
        pd = rr["production_guard_delta"]
        assert pd["real_bundle_loads"] == 0; assert pd["real_model_calls"] == 0; assert pd["real_selector_calls"] == 0
        # New rows must be all 0 for complete resume
        nr = rr["new_rows"]
        assert nr["baseline"] == 0; assert nr["governed"] == 0
        assert nr["selection"] == 0; assert nr["failure"] == 0
