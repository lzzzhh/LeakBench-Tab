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


def test_self_consistent_governed_shard_tamper_is_rejected_by_active_fragment_aggregate_closure():
    """Self-consistent shard ledger + manifest is rejected by fragment aggregate closure."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()

        # Read governed ledger, mutate governed_auc on first data row
        gl_path = out / "governed_ledger.csv.gz"
        raw = gzip.decompress(gl_path.read_bytes()).decode("utf-8")
        lines = raw.split("\n")
        # Find first data row and mutate governed_auc (14th comma-separated field)
        data_rows = [l for l in lines[1:] if l != ""]
        parts = data_rows[0].split(",")
        # governed_auc is at index 13 (0-based: run_id=0,...,legacy_sdr=13, selection_hash=14, realized_cost=15)
        old_auc = parts[13]
        new_auc = "0.999" if old_auc != "0.999" else "0.001"
        parts[13] = new_auc
        data_rows[0] = ",".join(parts)
        # Rebuild canonical text
        header = lines[0]
        new_text = header + "\n" + "\n".join(sorted(data_rows)) + "\n"
        new_bytes = gzip.compress(new_text.encode("utf-8"), mtime=0)
        gl_path.write_bytes(new_bytes)

        # Verify key invariants: row count, run IDs unchanged
        assert len(data_rows) == 576, f"governed row count changed: {len(data_rows)}"
        run_ids = [r.split(",")[0] for r in data_rows]
        assert len(set(run_ids)) == len(run_ids), "duplicate run IDs introduced"

        # Update shard manifest governed SHA
        sm = json.loads((out / "shard_manifest.json").read_text())
        sm["governed_sha256"] = hashlib.sha256(new_bytes).hexdigest()
        (out / "shard_manifest.json").write_text(json.dumps(sm))

        # Assert manifest self-consistent
        assert sm["governed_rows"] == 576
        assert sm["governed_sha256"] == hashlib.sha256(gl_path.read_bytes()).hexdigest()

        # Validate: must be rejected by fragment aggregate closure
        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert result.fragment_aggregate_valid is False
        assert any("governed" in e.lower() and "aggregate" in e.lower() for e in result.errors), \
            f"expected governed aggregate error, got: {result.errors[:5]}"
        # Must NOT be rejected by ordinary SHA mismatch
        assert not any("governed_sha256 mismatch" in e for e in result.errors), \
            f"should not fail on SHA check: {result.errors[:5]}"


def test_missing_planned_active_fragment_ledger_is_structured_invalid():
    """Missing active fragment file is rejected with structured error."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        cids = sorted(k["canonical_key_id"] for k in shard_keys)
        # Delete one active fragment
        sel_path = out / "key_fragments" / cids[0] / "selection.csv.gz"
        sel_path.unlink()
        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert result.fragment_aggregate_valid is False
        assert any("selection" in e.lower() and "missing" in e.lower() for e in result.errors), \
            f"expected selection missing error, got: {result.errors[:5]}"


def test_self_consistent_active_fragment_and_shard_tamper_is_rejected_by_fragment_manifest_source_binding():
    """Dual tamper: active fragment + shard agree, but fragment manifest is stale."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        cids = sorted(k["canonical_key_id"] for k in shard_keys)
        cid = cids[0]

        # Read fragment governed, mutate legacy_sdr (field index 13)
        fg_path = out / "key_fragments" / cid / "governed.csv.gz"
        fg_raw = gzip.decompress(fg_path.read_bytes()).decode("utf-8")
        fg_lines = fg_raw.split("\n")
        fg_data = [l for l in fg_lines[1:] if l != ""]
        parts = fg_data[0].split(",")
        old_sdr = parts[13]
        parts[13] = "0.999" if old_sdr != "0.999" else "0.001"
        fg_data[0] = ",".join(parts)
        fg_new_text = fg_lines[0] + "\n" + "\n".join(sorted(fg_data)) + "\n"
        fg_path.write_bytes(gzip.compress(fg_new_text.encode("utf-8"), mtime=0))

        # Apply identical mutation to shard governed ledger
        gl_path = out / "governed_ledger.csv.gz"
        gl_raw = gzip.decompress(gl_path.read_bytes()).decode("utf-8")
        gl_lines = gl_raw.split("\n")
        gl_data = [l for l in gl_lines[1:] if l != ""]
        # Find matching row by run_id
        target_rid = parts[0]
        for i, row in enumerate(gl_data):
            if row.split(",")[0] == target_rid:
                rp = row.split(",")
                rp[13] = parts[13]
                gl_data[i] = ",".join(rp)
                break
        gl_new_text = gl_lines[0] + "\n" + "\n".join(sorted(gl_data)) + "\n"
        gl_new_bytes = gzip.compress(gl_new_text.encode("utf-8"), mtime=0)
        gl_path.write_bytes(gl_new_bytes)

        # Update shard manifest governed SHA (NOT fragment manifest)
        sm = json.loads((out / "shard_manifest.json").read_text())
        sm["governed_sha256"] = hashlib.sha256(gl_new_bytes).hexdigest()
        (out / "shard_manifest.json").write_text(json.dumps(sm))

        # Assert: fragment manifest NOT changed
        fm = json.loads((out / "key_fragments" / cid / "fragment_manifest.json").read_text())
        assert fm["governed_sha256"] != hashlib.sha256(fg_path.read_bytes()).hexdigest(), \
            "fragment manifest governed_sha256 should be stale"

        # Validate
        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert result.fragment_manifest_set_valid is True
        assert result.fragment_sources_valid is False
        assert any(f"{cid}: governed.csv.gz sha256 mismatch against fragment manifest" in e
                   for e in result.errors), \
            f"expected source SHA mismatch, got: {result.errors[:5]}"


def test_active_fragment_malformed_csv_is_structured_invalid():
    """Malformed CSV (unterminated quote) is caught by strict csv.reader."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        cids = sorted(k["canonical_key_id"] for k in shard_keys)
        cid = cids[0]

        # Corrupt a selection fragment: unterminated quoted field
        sp = out / "key_fragments" / cid / "selection.csv.gz"
        raw = gzip.decompress(sp.read_bytes()).decode("utf-8")
        # Insert an unterminated quote in a data row
        lines = raw.split("\n")
        lines[1] = lines[1].replace(",", ',"unterminated,', 1)  # creates unterminated quote
        new_text = "\n".join(lines)
        sp.write_bytes(gzip.compress(new_text.encode("utf-8"), mtime=0))

        # Update fragment manifest selection SHA
        fm = json.loads((out / "key_fragments" / cid / "fragment_manifest.json").read_text())
        fm["selection_sha256"] = hashlib.sha256(sp.read_bytes()).hexdigest()
        (out / "key_fragments" / cid / "fragment_manifest.json").write_text(json.dumps(fm))

        # Update shard manifest fragment-manifest set
        from scripts.t0_b_full_b1.shard_contract import fragment_manifest_set_sha256
        key_dirs = [out / "key_fragments" / c for c in cids]
        sm = json.loads((out / "shard_manifest.json").read_text())
        sm["fragment_manifest_set_sha256"] = fragment_manifest_set_sha256(key_dirs)
        (out / "shard_manifest.json").write_text(json.dumps(sm))

        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert result.fragment_sources_valid is False
        assert any("CSV parse error" in e for e in result.errors), \
            f"expected CSV parse error, got: {result.errors[:5]}"


def test_active_fragment_blank_record_is_not_silently_normalized():
    """Blank data record in fragment CSV is rejected."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        cids = sorted(k["canonical_key_id"] for k in shard_keys)
        cid = cids[0]

        # Insert a blank line in baseline fragment
        bp = out / "key_fragments" / cid / "baseline.csv.gz"
        raw = gzip.decompress(bp.read_bytes()).decode("utf-8")
        lines = raw.split("\n")
        # Insert a completely blank record (empty line) after header
        lines.insert(2, "")
        new_text = "\n".join(lines)
        bp.write_bytes(gzip.compress(new_text.encode("utf-8"), mtime=0))

        # Update fragment manifest baseline SHA
        fm = json.loads((out / "key_fragments" / cid / "fragment_manifest.json").read_text())
        fm["baseline_sha256"] = hashlib.sha256(bp.read_bytes()).hexdigest()
        (out / "key_fragments" / cid / "fragment_manifest.json").write_text(json.dumps(fm))

        # Update shard manifest fragment-manifest set
        from scripts.t0_b_full_b1.shard_contract import fragment_manifest_set_sha256
        key_dirs = [out / "key_fragments" / c for c in cids]
        sm = json.loads((out / "shard_manifest.json").read_text())
        sm["fragment_manifest_set_sha256"] = fragment_manifest_set_sha256(key_dirs)
        (out / "shard_manifest.json").write_text(json.dumps(sm))

        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert result.fragment_sources_valid is False
        assert any("blank data record" in e.lower() for e in result.errors), \
            f"expected blank record error, got: {result.errors[:5]}"


def test_unplanned_empty_shard_id_fails_closed_without_publication():
    """Unplanned shard_id fails before any artifact is written."""
    plan_dir = Path(SYNTH_PLAN).parent
    keys = [json.loads(l) for l in gzip.decompress(
        (plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()
    ).decode("utf-8").strip().split("\n")]
    max_id = max(k.get("shard_id", -1) for k in keys)
    invalid_id = max_id + 1000

    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_{invalid_id}"
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", str(invalid_id), "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "NO_PLANNED_KEYS_FOR_SHARD" in r.stdout, f"stdout={r.stdout}"
        out_p = Path(out)
        for fname in ["baseline_ledger.csv.gz", "governed_ledger.csv.gz",
                       "selection_ledger.csv.gz", "failure_ledger.csv.gz",
                       "shard_manifest.json", "shard_execution_receipt.json",
                       "resume_receipt.json"]:
            assert not (out_p / fname).exists(), f"{fname} should not exist"


def test_shard_validator_rejects_empty_planned_scope():
    """Validator returns invalid when given empty key/run rows."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        result = validate_shard_artifacts(
            output_dir=out, plan_manifest=plan, plan_manifest_sha256=plan_sha,
            shard_key_rows=[], shard_run_rows=[],
        )
        assert not result.is_valid
        assert result.planned_scope_valid is False
        assert "no planned keys for shard" in result.errors
        assert result.fragment_sources_valid is False
        assert result.fragment_aggregate_valid is False


def test_quoted_multiline_fragment_record_is_rejected_before_aggregation():
    """Legally quoted multiline CSV record is rejected by physical-line parser."""
    import csv as csv_mod
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        cids = sorted(k["canonical_key_id"] for k in shard_keys)
        cid = cids[0]

        # Create a selection.csv.gz with a multiline quoted field
        sp = out / "key_fragments" / cid / "selection.csv.gz"
        raw = gzip.decompress(sp.read_bytes()).decode("utf-8")
        lines = raw.split("\n")
        # Build a data row with an embedded newline using csv.writer
        import csv
        row_with_newline = ['hash1', 'P2', 'semantic_group', '500', '[]', '[]', '0']
        row_with_newline[2] = 'line1\nline2'  # embedded newline
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(row_with_newline)
        multiline_row = buf.getvalue().rstrip("\n")  # physical repr spans lines
        new_text = lines[0] + "\n" + multiline_row + "\n"
        sp.write_bytes(gzip.compress(new_text.encode("utf-8"), mtime=0))

        # Update manifest SHAs
        fm = json.loads((out / "key_fragments" / cid / "fragment_manifest.json").read_text())
        fm["selection_sha256"] = hashlib.sha256(sp.read_bytes()).hexdigest()
        (out / "key_fragments" / cid / "fragment_manifest.json").write_text(json.dumps(fm))

        from scripts.t0_b_full_b1.shard_contract import fragment_manifest_set_sha256
        key_dirs = [out / "key_fragments" / c for c in cids]
        sm = json.loads((out / "shard_manifest.json").read_text())
        sm["fragment_manifest_set_sha256"] = fragment_manifest_set_sha256(key_dirs)
        (out / "shard_manifest.json").write_text(json.dumps(sm))

        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert result.fragment_sources_valid is False
        assert any("embedded newline" in e.lower() or "multiline" in e.lower()
                   or "csv parse error" in e.lower()
                   for e in result.errors), f"expected multiline rejection, got: {result.errors[:5]}"


def test_quoted_canonical_header_is_not_accepted_as_exact_header():
    """Quoted header fields (semantically equivalent) are rejected as exact mismatch."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(f"{td}/shard_0")
        _setup_valid(out)
        plan, plan_sha, shard_keys, shard_runs = _load_plan()
        cids = sorted(k["canonical_key_id"] for k in shard_keys)
        cid = cids[0]

        # Change baseline header: run_id -> "run_id"
        bp = out / "key_fragments" / cid / "baseline.csv.gz"
        raw = gzip.decompress(bp.read_bytes()).decode("utf-8")
        lines = raw.split("\n")
        header_parts = lines[0].split(",")
        header_parts[0] = '\"run_id\"'
        new_text = ",".join(header_parts) + "\n" + "\n".join(lines[1:])
        bp.write_bytes(gzip.compress(new_text.encode("utf-8"), mtime=0))

        # Update manifest SHAs
        fm = json.loads((out / "key_fragments" / cid / "fragment_manifest.json").read_text())
        fm["baseline_sha256"] = hashlib.sha256(bp.read_bytes()).hexdigest()
        (out / "key_fragments" / cid / "fragment_manifest.json").write_text(json.dumps(fm))

        from scripts.t0_b_full_b1.shard_contract import fragment_manifest_set_sha256
        key_dirs = [out / "key_fragments" / c for c in cids]
        sm = json.loads((out / "shard_manifest.json").read_text())
        sm["fragment_manifest_set_sha256"] = fragment_manifest_set_sha256(key_dirs)
        (out / "shard_manifest.json").write_text(json.dumps(sm))

        result = _validate(out, plan, plan_sha, shard_keys, shard_runs)
        assert not result.is_valid
        assert result.fragment_sources_valid is False
        assert any("exact header mismatch" in e for e in result.errors),             f"expected header mismatch, got: {result.errors[:5]}"
