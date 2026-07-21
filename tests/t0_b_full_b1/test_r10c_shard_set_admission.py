"""R10c-1 Strict shard-set admission tests — full CLI integration."""
import gzip, hashlib, io, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path; import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
ADMIT_CLI = str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_shard_set.py")
SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")


def _generate_all_shards(shard_root, plan_path=SYNTH_PLAN):
    """Generate all synthetic shards via real runner."""
    plan = json.loads(Path(plan_path).read_text())
    plan_dir = Path(plan_path).parent
    keys = [json.loads(l) for l in gzip.decompress(
        (plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
    for sid in sorted(set(k["shard_id"] for k in keys)):
        out = str(Path(shard_root) / f"shard_{sid}")
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", plan_path,
            "--shard-id", str(sid), "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"shard {sid} failed: {r.stderr[:300]}"


def test_exact_synthetic_shard_set_pass():
    """Two synthetic shards pass strict admission."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"stdout={r.stdout[:500]}\nstderr={r.stderr[:300]}"
        assert "STRICT_SHARD_SET_ADMISSION_PASS" in r.stdout
        assert "validated_shards=2" in r.stdout


def test_missing_shard_fails():
    """Missing planned shard directory blocks admission."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        shutil.rmtree(f"{shard_root}/shard_1")
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "missing planned shard directory: shard_1" in r.stdout


def test_extra_shard_fails():
    """Extra unplanned shard directory blocks admission."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        # Copy shard_0 as shard_99 (extra)
        shutil.copytree(f"{shard_root}/shard_0", f"{shard_root}/shard_99")
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "extra unplanned shard directory: shard_99" in r.stdout


def test_non_canonical_shard_entry_fails():
    """Non-canonical shard_* entry blocks admission."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        # Create a non-canonical entry
        (Path(shard_root) / "shard_00").mkdir()
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "non-canonical shard entry: shard_00" in r.stdout


def test_shard_identity_mismatch_fails():
    """Shard manifest shard_id mismatch blocks admission."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        # Modify shard_0 manifest to claim shard_id=1
        sm = json.loads((Path(shard_root) / "shard_0" / "shard_manifest.json").read_text())
        sm["shard_id"] = 1
        (Path(shard_root) / "shard_0" / "shard_manifest.json").write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "shard identity mismatch" in r.stdout


def test_invalid_shard_blocks_set():
    """One invalid shard blocks global admission even if another is valid."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        # Tamper shard_0: self-consistent governed tamper
        gpath = Path(shard_root) / "shard_0" / "governed_ledger.csv.gz"
        raw = gzip.decompress(gpath.read_bytes()).decode("utf-8")
        lines = raw.split("\n")
        data = [l for l in lines[1:] if l != ""]
        parts = data[0].split(",")
        parts[13] = "0.999" if parts[13] != "0.999" else "0.001"
        data[0] = ",".join(parts)
        new_text = lines[0] + "\n" + "\n".join(sorted(data)) + "\n"
        new_bytes = gzip.compress(new_text.encode("utf-8"), mtime=0)
        gpath.write_bytes(new_bytes)
        # Update manifest governed SHA
        sm = json.loads((Path(shard_root) / "shard_0" / "shard_manifest.json").read_text())
        sm["governed_sha256"] = hashlib.sha256(new_bytes).hexdigest()
        (Path(shard_root) / "shard_0" / "shard_manifest.json").write_text(json.dumps(sm))

        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "shard 0 validation failed" in r.stdout


def test_global_count_mismatch_fails():
    """Global row-count mismatch blocks admission."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        # Modify the plan's baseline_rows but keep all SHA bindings matching original
        plan = json.loads(Path(SYNTH_PLAN).read_text())
        original_plan_sha = hashlib.sha256(Path(SYNTH_PLAN).read_bytes()).hexdigest()
        plan["baseline_rows"] = 999  # only change the count
        tmp_plan = Path(td) / "modified_plan.json"
        tmp_plan.write_text(json.dumps(plan))
        # Copy key/run plans for CLI to find
        plan_dir = Path(SYNTH_PLAN).parent
        shutil.copy(plan_dir / "full_b1_key_plan.jsonl.gz", Path(td) / "full_b1_key_plan.jsonl.gz")
        shutil.copy(plan_dir / "full_b1_run_plan.jsonl.gz", Path(td) / "full_b1_run_plan.jsonl.gz")
        plan["key_plan_sha256"] = hashlib.sha256((Path(td) / "full_b1_key_plan.jsonl.gz").read_bytes()).hexdigest()
        plan["run_plan_sha256"] = hashlib.sha256((Path(td) / "full_b1_run_plan.jsonl.gz").read_bytes()).hexdigest()
        tmp_plan.write_text(json.dumps(plan))
        # Update shard manifests and fragment manifests to bind the new plan SHA
        new_plan_sha = hashlib.sha256(tmp_plan.read_bytes()).hexdigest()
        from scripts.t0_b_full_b1.shard_contract import fragment_manifest_set_sha256
        for sid in [0, 1]:
            sm_path = Path(shard_root) / f"shard_{sid}" / "shard_manifest.json"
            sm = json.loads(sm_path.read_text())
            sm["plan_manifest_sha256"] = new_plan_sha
            # Also update fragment manifests
            for kdir in (Path(shard_root) / f"shard_{sid}" / "key_fragments").iterdir():
                if kdir.is_dir():
                    fm_path = kdir / "fragment_manifest.json"
                    fm = json.loads(fm_path.read_text())
                    fm["plan_manifest_sha256"] = new_plan_sha
                    fm_path.write_text(json.dumps(fm))
            # Update fragment manifest set digest
            key_dirs = sorted((Path(shard_root) / f"shard_{sid}" / "key_fragments").iterdir())
            key_dirs = [d for d in key_dirs if d.is_dir()]
            sm["fragment_manifest_set_sha256"] = fragment_manifest_set_sha256(key_dirs)
            sm_path.write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", str(tmp_plan),
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "global baseline_rows mismatch" in r.stdout
