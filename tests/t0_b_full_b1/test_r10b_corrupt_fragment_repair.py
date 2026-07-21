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
        assert rr["recomputed"] == 1
        assert rr["skipped"] == 3


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
        frags = Path(out) / "key_fragments"
        first_key = sorted(frags.iterdir())[0]
        (first_key / "completion_receipt.json").unlink()
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume", "--repair-invalid"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "RESUME_REPAIR_UNSUPPORTED" in r.stdout
