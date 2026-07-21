"""R10b-5 Shard determinism tests — deterministic manifest, ledger closure, 4-path identity."""
import gzip, hashlib, io, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path; import pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")

from scripts.t0_b_full_b1.shard_contract import (
    build_shard_manifest, validate_shard_artifacts,
    ShardArtifactValidation, SHARD_MANIFEST_SCHEMA_VERSION,
)


def _fresh_exec(out):
    r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
        "--shard-id", "0", "--output-dir", out, "--synthetic"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[:300]


def _five_shas(out):
    out = Path(out)
    return {name: hashlib.sha256((out / f"{name}_ledger.csv.gz").read_bytes()).hexdigest()
            for name in ["baseline", "governed", "selection", "failure"]} | \
           {"shard_manifest": hashlib.sha256((out / "shard_manifest.json").read_bytes()).hexdigest()}


# ═══════════════════════════════════════════════════════════════════
# PATH A: Fresh execution
# ═══════════════════════════════════════════════════════════════════

def test_fresh_shard_manifest_valid():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _fresh_exec(out)
        out = Path(out)
        plan = json.loads(Path(SYNTH_PLAN).read_text())
        plan_sha = hashlib.sha256(Path(SYNTH_PLAN).read_bytes()).hexdigest()
        plan_dir = Path(SYNTH_PLAN).parent
        keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
        runs = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
        shard_keys = [k for k in keys if k.get("shard_id") == 0]
        shard_runs = [r for r in runs if r.get("shard_id") == 0]
        result = validate_shard_artifacts(
            output_dir=out, plan_manifest=plan, plan_manifest_sha256=plan_sha,
            shard_key_rows=shard_keys, shard_run_rows=shard_runs,
        )
        assert result.is_valid, result.errors[:5]
        assert result.baseline_rows == 8
        assert result.governed_rows == 576
        assert result.selection_rows == 576
        assert result.failure_rows == 0


def test_shard_manifest_contains_no_dynamic_fields():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _fresh_exec(out)
        sm = json.loads((Path(out) / "shard_manifest.json").read_text())
        forbidden = {"new_keys", "complete_keys", "recomputed", "skipped", "quarantined",
                     "invalid", "timestamp", "completed_utc", "resume", "repair"}
        for f in forbidden:
            assert f not in sm, f"found dynamic field {f} in shard manifest"
        # Verify presence of required fields
        for f in ["schema_version", "mode", "shard_id", "scientific_freeze_sha",
                   "baseline_sha256", "governed_sha256", "selection_sha256", "failure_sha256",
                   "completed_key_ids_sha256", "planned_run_ids_sha256", "produced_run_ids_sha256"]:
            assert f in sm, f"missing required field {f}"


# ═══════════════════════════════════════════════════════════════════
# PATH B: Untouched complete resume
# ═══════════════════════════════════════════════════════════════════

def test_complete_resume_five_artifacts_byte_identical():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _fresh_exec(out)
        shas_before = _five_shas(out)
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[:300]
        shas_after = _five_shas(out)
        for name in shas_before:
            assert shas_before[name] == shas_after[name], f"{name} changed after complete resume"


# ═══════════════════════════════════════════════════════════════════
# PATH C: Fragment SHA repair restores clean artifacts
# ═══════════════════════════════════════════════════════════════════

def test_fragment_sha_repair_restores_five_artifacts():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _fresh_exec(out)
        shas_clean = _five_shas(out)
        # Corrupt first key's governed
        frags = Path(out) / "key_fragments"
        corrupt_key = sorted(frags.iterdir())[0]
        gpath = corrupt_key / "governed.csv.gz"
        data = bytearray(gpath.read_bytes())
        data[len(data)//2] ^= 0x01
        gpath.write_bytes(bytes(data))
        # Repair
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume", "--repair-invalid"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[:300]
        shas_after = _five_shas(out)
        for name in shas_clean:
            assert shas_clean[name] == shas_after[name], f"{name} differs after fragment SHA repair"


# ═══════════════════════════════════════════════════════════════════
# PATH D: Missing receipt repair restores clean artifacts
# ═══════════════════════════════════════════════════════════════════

def test_missing_receipt_repair_restores_five_artifacts():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _fresh_exec(out)
        shas_clean = _five_shas(out)
        frags = Path(out) / "key_fragments"
        first_key = sorted(frags.iterdir())[0]
        (first_key / "completion_receipt.json").unlink()
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume", "--repair-invalid"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[:300]
        shas_after = _five_shas(out)
        for name in shas_clean:
            assert shas_clean[name] == shas_after[name], f"{name} differs after missing-receipt repair"


# ═══════════════════════════════════════════════════════════════════
# SHARD MANIFEST BINDING
# ═══════════════════════════════════════════════════════════════════

def test_shard_manifest_binds_four_ledger_shas():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _fresh_exec(out)
        out = Path(out)
        sm = json.loads((out / "shard_manifest.json").read_text())
        for name in ["baseline", "governed", "selection", "failure"]:
            actual = hashlib.sha256((out / f"{name}_ledger.csv.gz").read_bytes()).hexdigest()
            assert sm[f"{name}_sha256"] == actual, f"{name} SHA not bound"


def test_shard_manifest_binds_key_and_run_universes():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _fresh_exec(out)
        out = Path(out)
        sm = json.loads((out / "shard_manifest.json").read_text())
        plan_dir = Path(SYNTH_PLAN).parent
        keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
        runs = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
        shard_keys = [k for k in keys if k.get("shard_id") == 0]
        shard_runs = [r for r in runs if r.get("shard_id") == 0]
        from scripts.t0_b_full_b1.shard_contract import canonical_key_ids_sha256, sorted_lines_sha256
        assert sm["key_count"] == len(shard_keys)
        assert sm["completed_key_ids_sha256"] == canonical_key_ids_sha256(
            sorted(k["canonical_key_id"] for k in shard_keys))
        assert sm["planned_run_ids_sha256"] == sorted_lines_sha256(
            sorted(r["run_id"] for r in shard_runs))


def test_shard_manifest_binds_selection_multiset():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _fresh_exec(out)
        out = Path(out)
        sm = json.loads((out / "shard_manifest.json").read_text())
        from scripts.t0_b_full_b1.shard_contract import selection_hash_multiset_sha256
        sl_df = pd.read_csv(out / "selection_ledger.csv.gz")
        assert sm["selection_hash_multiset_sha256"] == selection_hash_multiset_sha256(sl_df)


def test_shard_manifest_binds_fragment_manifest_set():
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        _fresh_exec(out)
        out = Path(out)
        sm = json.loads((out / "shard_manifest.json").read_text())
        from scripts.t0_b_full_b1.shard_contract import fragment_manifest_set_sha256
        cids = sorted(d.name for d in (out / "key_fragments").iterdir() if d.is_dir())
        key_dirs = [out / "key_fragments" / cid for cid in cids]
        assert sm["fragment_manifest_set_sha256"] == fragment_manifest_set_sha256(key_dirs)


# ═══════════════════════════════════════════════════════════════════
# NEGATIVE TESTS
# ═══════════════════════════════════════════════════════════════════

def _setup_valid(out_path):
    _fresh_exec(str(out_path))


def _load_plan():
    plan = json.loads(Path(SYNTH_PLAN).read_text())
    plan_sha = hashlib.sha256(Path(SYNTH_PLAN).read_bytes()).hexdigest()
    plan_dir = Path(SYNTH_PLAN).parent
    keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
    runs = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
    shard_keys = [k for k in keys if k.get("shard_id") == 0]
    shard_runs = [r for r in runs if r.get("shard_id") == 0]
    return plan, plan_sha, shard_keys, shard_runs


def _validate(out, plan, plan_sha, shard_keys, shard_runs):
    return validate_shard_artifacts(
        output_dir=Path(out), plan_manifest=plan, plan_manifest_sha256=plan_sha,
        shard_key_rows=shard_keys, shard_run_rows=shard_runs,
    )


def test_tampered_ledger_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        # Corrupt baseline ledger
        bp = out / "baseline_ledger.csv.gz"
        data = bytearray(bp.read_bytes())
        data[10] ^= 0x01
        bp.write_bytes(bytes(data))
        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert any("baseline_sha256" in e for e in result.errors)


def test_tampered_manifest_provenance_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        sm = json.loads((out / "shard_manifest.json").read_text())
        sm["plan_manifest_sha256"] = "f" * 64
        (out / "shard_manifest.json").write_text(json.dumps(sm))
        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert any("plan_manifest_sha256" in e for e in result.errors)


def test_missing_or_extra_active_key_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        # Remove one key directory
        cids = sorted(d.name for d in (out / "key_fragments").iterdir() if d.is_dir())
        shutil.rmtree(out / "key_fragments" / cids[0])
        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert any("missing active key" in e for e in result.errors)


def test_missing_shard_manifest_is_structured():
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        (out / "shard_manifest.json").unlink()
        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert "shard_manifest.json missing" in result.errors


def test_tampered_fragment_manifest_set_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        # Tamper one fragment manifest
        cids = sorted(d.name for d in (out / "key_fragments").iterdir() if d.is_dir())
        mp = out / "key_fragments" / cids[0] / "fragment_manifest.json"
        data = json.loads(mp.read_text())
        data["baseline_rows"] = 999
        mp.write_text(json.dumps(data))
        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert any("fragment_manifest_set_sha256" in e for e in result.errors)


def test_selection_multiset_mutation_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        sl_df = pd.read_csv(out / "selection_ledger.csv.gz")
        # Change one selection hash and update manifest SHA to pass SHA check
        old_sha = hashlib.sha256((out / "selection_ledger.csv.gz").read_bytes()).hexdigest()
        sl_df.loc[0, "selection_hash"] = "mutated_selection_hash_0000000000000000000000000000000000000000"
        buf = io.StringIO(); sl_df.to_csv(buf, index=False, header=True)
        (out / "selection_ledger.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode(), mtime=0))
        new_sha = hashlib.sha256((out / "selection_ledger.csv.gz").read_bytes()).hexdigest()
        # Update manifest to keep SHA check passing
        sm = json.loads((out / "shard_manifest.json").read_text())
        sm["selection_sha256"] = new_sha
        (out / "shard_manifest.json").write_text(json.dumps(sm))
        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert any("multiset" in e for e in result.errors)
