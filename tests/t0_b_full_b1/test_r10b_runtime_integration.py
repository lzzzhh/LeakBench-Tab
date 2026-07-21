"""R10b-1 Runtime Integration Tests — CLI-level verification with counter checks."""
import gzip, hashlib, io, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path; import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")


def test_no_local_validate_completed_key():
    content = (ROOT / "scripts/t0_b_full_b1/run_full_b1_shard.py").read_text()
    lines = [l for l in content.split("\n") if l.startswith("def validate_completed_key")]
    assert len(lines) == 0


def test_no_self_counter_in_runner():
    """CallCounter must be removed from runner."""
    content = (ROOT / "scripts/t0_b_full_b1/run_full_b1_shard.py").read_text()
    assert "self.counter" not in content, "self.counter still referenced in runner"


def test_first_shard_counter_totals():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", f"{td}/shard_0", "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        with open(f"{td}/shard_0/shard_execution_receipt.json") as f: sr = json.load(f)
        scd = sr["synthetic_call_counter_delta"]
        assert scd["lr_calls"] == 584, f"LR={scd['lr_calls']}"
        assert scd["p3_calls"] == 4
        assert scd["p4_calls"] == 4
        assert scd["p5_calls"] == 4
        assert scd["p6_calls"] == 4
        pgd = sr["production_guard_delta"]
        assert pgd["real_bundle_loads"] == 0
        assert pgd["real_model_calls"] == 0
        assert pgd["real_selector_calls"] == 0


def test_each_completion_receipt_per_key_delta():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", f"{td}/shard_0", "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        frags = Path(f"{td}/shard_0/key_fragments")
        for cid_dir in sorted(frags.iterdir()):
            with open(cid_dir / "completion_receipt.json") as f: rec = json.load(f)
            scd = rec["synthetic_call_counter_delta"]
            assert scd["lr_calls"] == 146, f"{cid_dir.name}: lr={scd['lr_calls']}"
            assert scd["p3_calls"] == 1
            assert scd["p4_calls"] == 1
            assert scd["p5_calls"] == 1
            assert scd["p6_calls"] == 1


def test_complete_resume_all_delta_zero():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        with open(f"{out}/resume_receipt.json") as f: rr = json.load(f)
        assert rr["recomputed"] == 0
        sd = rr["synthetic_call_counter_delta"]
        assert sd["lr_calls"] == 0; assert sd["p3_calls"] == 0; assert sd["p4_calls"] == 0; assert sd["p5_calls"] == 0; assert sd["p6_calls"] == 0
        pd = rr["production_guard_delta"]
        assert pd["real_bundle_loads"] == 0; assert pd["real_model_calls"] == 0; assert pd["real_selector_calls"] == 0


def test_invalid_resume_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        frags = Path(out) / "key_fragments"
        first_key = sorted(frags.iterdir())[0]
        receipt_before_sha = hashlib.sha256((first_key / "completion_receipt.json").read_bytes()).hexdigest()
        (first_key / "completion_receipt.json").unlink()
        r2 = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode != 0
        assert "RESUME_VALIDATION_FAIL" in r2.stdout
        assert not (first_key / "completion_receipt.json").exists(), "receipt should NOT be auto-regenerated"


def test_invalid_resume_artifacts_unchanged():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        before_shas = {}
        for fname in ["baseline_ledger.csv.gz", "governed_ledger.csv.gz", "shard_manifest.json"]:
            before_shas[fname] = hashlib.sha256((Path(out) / fname).read_bytes()).hexdigest()
        frags = Path(out) / "key_fragments"
        first_key = sorted(frags.iterdir())[0]
        (first_key / "completion_receipt.json").unlink()
        r2 = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode != 0
        for fname in ["baseline_ledger.csv.gz", "governed_ledger.csv.gz", "shard_manifest.json"]:
            after_sha = hashlib.sha256((Path(out) / fname).read_bytes()).hexdigest()
            assert before_shas[fname] == after_sha, f"{fname}: SHA changed on invalid resume!"


def test_complete_resume_manifests_byte_identical():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        before_shas = {}
        frags = Path(out) / "key_fragments"
        for cid_dir in sorted(frags.iterdir()):
            for fname in ["baseline.csv.gz","governed.csv.gz","selection.csv.gz","failure.csv.gz",
                          "fragment_manifest.json","completion_receipt.json"]:
                before_shas[f"{cid_dir.name}/{fname}"] = hashlib.sha256((cid_dir/fname).read_bytes()).hexdigest()
        r2 = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode == 0
        for cid_dir in sorted(frags.iterdir()):
            for fname in ["baseline.csv.gz","governed.csv.gz","selection.csv.gz","failure.csv.gz",
                          "fragment_manifest.json","completion_receipt.json"]:
                after_sha = hashlib.sha256((cid_dir/fname).read_bytes()).hexdigest()
                assert before_shas[f"{cid_dir.name}/{fname}"] == after_sha, f"{cid_dir.name}/{fname}: SHA changed!"
