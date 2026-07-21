"""R10b-1 Runtime Integration Tests — CLI-level verification."""
import gzip, hashlib, io, json, os, subprocess, sys, tempfile
from pathlib import Path; import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")


def test_no_local_validate_completed_key():
    """Runner must not define its own validate_completed_key function."""
    content = (ROOT / "scripts/t0_b_full_b1/run_full_b1_shard.py").read_text()
    lines = [l for l in content.split("\n") if l.startswith("def validate_completed_key")]
    assert len(lines) == 0, f"Found local validate_completed_key: {lines}"


def test_first_execution_cli_exit_zero():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", f"{td}/shard_0", "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"stderr: {r.stderr[:300]}"


def test_first_execution_fragment_manifest_written():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", f"{td}/shard_0", "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        frags = Path(f"{td}/shard_0/key_fragments")
        for cid_dir in frags.iterdir():
            assert (cid_dir / "fragment_manifest.json").exists(), f"Missing manifest in {cid_dir.name}"
            assert (cid_dir / "completion_receipt.json").exists(), f"Missing receipt in {cid_dir.name}"
            assert (cid_dir / "baseline.csv.gz").exists()
            assert (cid_dir / "governed.csv.gz").exists()
            assert (cid_dir / "selection.csv.gz").exists()
            assert (cid_dir / "failure.csv.gz").exists()


def test_completion_receipt_binds_manifest_sha():
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", f"{td}/shard_0", "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        frags = Path(f"{td}/shard_0/key_fragments")
        for cid_dir in frags.iterdir():
            with open(cid_dir / "completion_receipt.json") as f: rec = json.load(f)
            actual_sha = hashlib.sha256((cid_dir / "fragment_manifest.json").read_bytes()).hexdigest()
            assert rec["fragment_manifest_sha256"] == actual_sha, f"{cid_dir.name}: receipt SHA mismatch"


def test_complete_resume_cli_exit_zero():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        r2 = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode == 0, f"stderr: {r2.stderr[:300]}"


def test_complete_resume_synthetic_delta_zero():
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
        assert rr["synthetic_delta"]["lr_calls"] == 0
        assert rr["synthetic_delta"]["p3_calls"] == 0
        assert rr["production_delta"]["real_bundle_loads"] == 0
        assert rr["production_delta"]["real_model_calls"] == 0


def test_complete_resume_manifests_byte_identical():
    """After complete resume, key artifact files must be unchanged."""
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
                key = f"{cid_dir.name}/{fname}"
                assert before_shas[key] == after_sha, f"{key}: SHA changed!"
