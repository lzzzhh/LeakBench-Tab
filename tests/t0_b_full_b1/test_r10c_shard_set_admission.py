"""R10c-1 Strict shard-set admission tests — full CLI integration."""
import gzip, hashlib, io, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path; import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
ADMIT_CLI = str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_shard_set.py")
SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")
SYNTH_PLAN_DIR = Path(SYNTH_PLAN).parent


def _copy_complete_synthetic_plan_fixture(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in [
        "full_b1_plan_manifest.json",
        "full_b1_key_plan.jsonl.gz",
        "full_b1_run_plan.jsonl.gz",
        "full_b1_shard_plan.json",
        "synthetic_policy_group_mapping.jsonl.gz",
        "synthetic_semantic_evaluation_mapping.jsonl.gz",
    ]:
        shutil.copy2(SYNTH_PLAN_DIR / name, destination / name)
    return destination / "full_b1_plan_manifest.json"


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
        plan_path = _copy_complete_synthetic_plan_fixture(Path(td) / "plan")
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        # Modify the plan's baseline_rows but keep all SHA bindings matching original
        plan = json.loads(plan_path.read_text())
        original_plan_sha = hashlib.sha256(Path(SYNTH_PLAN).read_bytes()).hexdigest()
        plan["baseline_rows"] = 999  # only change the count, keep downstream consistent for schema
        plan["downstream_rows"] = 999 + plan["governed_rows"]
        tmp_plan = plan_path
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


def test_missing_required_global_count_field_fails_closed():
    """Missing required count field fails at plan schema stage."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f'{td}/shards'
        _generate_all_shards(shard_root)
        plan_d = Path(td)
        plan_dir = Path(SYNTH_PLAN).parent
        shutil.copy(plan_dir / 'full_b1_key_plan.jsonl.gz', plan_d / 'full_b1_key_plan.jsonl.gz')
        shutil.copy(plan_dir / 'full_b1_run_plan.jsonl.gz', plan_d / 'full_b1_run_plan.jsonl.gz')
        plan = json.loads(Path(SYNTH_PLAN).read_text())
        del plan['selection_rows']
        plan['key_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_key_plan.jsonl.gz').read_bytes()).hexdigest()
        plan['run_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_run_plan.jsonl.gz').read_bytes()).hexdigest()
        tmp_plan = plan_d / 'modified_plan.json'
        tmp_plan.write_text(json.dumps(plan))
        r = subprocess.run([sys.executable, ADMIT_CLI, '--plan-manifest', str(tmp_plan),
            '--shard-root', shard_root, '--synthetic'],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert 'STRICT_SHARD_SET_ADMISSION_FAIL' in r.stdout
        assert 'plan manifest missing required field: selection_rows' in r.stdout


def test_string_global_count_is_not_silently_skipped():
    """String count value fails with type error."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f'{td}/shards'
        _generate_all_shards(shard_root)
        plan_d = Path(td)
        plan_dir = Path(SYNTH_PLAN).parent
        shutil.copy(plan_dir / 'full_b1_key_plan.jsonl.gz', plan_d / 'full_b1_key_plan.jsonl.gz')
        shutil.copy(plan_dir / 'full_b1_run_plan.jsonl.gz', plan_d / 'full_b1_run_plan.jsonl.gz')
        plan = json.loads(Path(SYNTH_PLAN).read_text())
        plan['governed_rows'] = '1152'
        plan['key_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_key_plan.jsonl.gz').read_bytes()).hexdigest()
        plan['run_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_run_plan.jsonl.gz').read_bytes()).hexdigest()
        tmp_plan = plan_d / 'modified_plan.json'
        tmp_plan.write_text(json.dumps(plan))
        r = subprocess.run([sys.executable, ADMIT_CLI, '--plan-manifest', str(tmp_plan),
            '--shard-root', shard_root, '--synthetic'],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert 'STRICT_SHARD_SET_ADMISSION_FAIL' in r.stdout
        assert 'governed_rows' in r.stdout and 'integer' in r.stdout.lower()


def test_bool_global_count_is_not_accepted_as_integer():
    """Bool True fails with type error."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f'{td}/shards'
        _generate_all_shards(shard_root)
        plan_d = Path(td)
        plan_dir = Path(SYNTH_PLAN).parent
        shutil.copy(plan_dir / 'full_b1_key_plan.jsonl.gz', plan_d / 'full_b1_key_plan.jsonl.gz')
        shutil.copy(plan_dir / 'full_b1_run_plan.jsonl.gz', plan_d / 'full_b1_run_plan.jsonl.gz')
        plan = json.loads(Path(SYNTH_PLAN).read_text())
        plan['baseline_rows'] = True
        plan['key_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_key_plan.jsonl.gz').read_bytes()).hexdigest()
        plan['run_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_run_plan.jsonl.gz').read_bytes()).hexdigest()
        tmp_plan = plan_d / 'modified_plan.json'
        tmp_plan.write_text(json.dumps(plan))
        r = subprocess.run([sys.executable, ADMIT_CLI, '--plan-manifest', str(tmp_plan),
            '--shard-root', shard_root, '--synthetic'],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert 'STRICT_SHARD_SET_ADMISSION_FAIL' in r.stdout
        assert 'baseline_rows' in r.stdout and 'bool' in r.stdout.lower()


def test_bool_run_shard_id_does_not_equal_integer_zero():
    """False is not accepted as integer 0 for shard_id."""
    with tempfile.TemporaryDirectory() as td:
        plan_d = Path(td) / "plan"
        tmp_plan = _copy_complete_synthetic_plan_fixture(plan_d)
        # Load and modify run plan
        raw = gzip.decompress((plan_d / 'full_b1_run_plan.jsonl.gz').read_bytes()).decode('utf-8')
        lines = [l for l in raw.split("\n") if l != ""]
        # Replace a shard_id with False
        for i, line in enumerate(lines):
            obj = json.loads(line)
            if obj.get('shard_id') == 0:
                obj['shard_id'] = False
                lines[i] = json.dumps(obj)
                break
        new_raw = "\n".join(lines) + "\n"
        (plan_d / "full_b1_run_plan.jsonl.gz").write_bytes(gzip.compress(new_raw.encode("utf-8"), mtime=0))
        plan = json.loads(Path(SYNTH_PLAN).read_text())
        plan['key_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_key_plan.jsonl.gz').read_bytes()).hexdigest()
        plan['run_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_run_plan.jsonl.gz').read_bytes()).hexdigest()
        tmp_plan.write_text(json.dumps(plan))
        r = subprocess.run([sys.executable, ADMIT_CLI, '--plan-manifest', str(tmp_plan),
            '--shard-root', plan_d, '--synthetic'],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert 'STRICT_SHARD_SET_ADMISSION_FAIL' in r.stdout
        assert 'shard_id' in r.stdout and 'bool' in r.stdout.lower()


def test_corrupt_plan_manifest_is_structured_cli_failure():
    """Invalid plan manifest JSON produces structured failure."""
    with tempfile.TemporaryDirectory() as td:
        bad_plan = Path(td) / 'bad.json'
        bad_plan.write_text('{"mode":')
        r = subprocess.run([sys.executable, ADMIT_CLI, '--plan-manifest', str(bad_plan),
            '--shard-root', td, '--synthetic'],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert 'STRICT_SHARD_SET_ADMISSION_FAIL' in r.stdout
        assert 'corrupt JSON' in r.stdout


def test_corrupt_run_plan_jsonl_is_structured_failure():
    """Malformed JSONL in run plan produces structured failure with line number."""
    with tempfile.TemporaryDirectory() as td:
        plan_d = Path(td) / "plan"
        tmp_plan = _copy_complete_synthetic_plan_fixture(plan_d)
        # Create valid gzip but malformed JSONL
        corrupt_text = "valid_line\n" + '{"broken"\n'
        (plan_d / "full_b1_run_plan.jsonl.gz").write_bytes(gzip.compress(corrupt_text.encode("utf-8"), mtime=0))
        plan = json.loads(Path(SYNTH_PLAN).read_text())
        plan['key_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_key_plan.jsonl.gz').read_bytes()).hexdigest()
        plan['run_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_run_plan.jsonl.gz').read_bytes()).hexdigest()
        tmp_plan.write_text(json.dumps(plan))
        r = subprocess.run([sys.executable, ADMIT_CLI, '--plan-manifest', str(tmp_plan),
            '--shard-root', td, '--synthetic'],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert 'STRICT_SHARD_SET_ADMISSION_FAIL' in r.stdout
        assert 'JSON parse error at line' in r.stdout


def test_corrupt_shard_manifest_is_structured_admission_failure():
    """Corrupt shard manifest JSON produces structured admission failure."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f'{td}/shards'
        _generate_all_shards(shard_root)
        (Path(shard_root) / 'shard_0' / 'shard_manifest.json').write_text('{"shard_id":')
        r = subprocess.run([sys.executable, ADMIT_CLI, '--plan-manifest', SYNTH_PLAN,
            '--shard-root', shard_root, '--synthetic'],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert 'STRICT_SHARD_SET_ADMISSION_FAIL' in r.stdout
        assert 'corrupt JSON' in r.stdout


def test_failure_rows_nonzero_declaration_fails():
    """Plan declaring failure_rows=1 fails at schema stage."""
    with tempfile.TemporaryDirectory() as td:
        plan_d = Path(td)
        plan_dir = Path(SYNTH_PLAN).parent
        shutil.copy(plan_dir / 'full_b1_key_plan.jsonl.gz', plan_d / 'full_b1_key_plan.jsonl.gz')
        shutil.copy(plan_dir / 'full_b1_run_plan.jsonl.gz', plan_d / 'full_b1_run_plan.jsonl.gz')
        plan = json.loads(Path(SYNTH_PLAN).read_text())
        plan['failure_rows'] = 1
        plan['key_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_key_plan.jsonl.gz').read_bytes()).hexdigest()
        plan['run_plan_sha256'] = hashlib.sha256((plan_d / 'full_b1_run_plan.jsonl.gz').read_bytes()).hexdigest()
        tmp_plan = plan_d / 'modified_plan.json'
        tmp_plan.write_text(json.dumps(plan))
        r = subprocess.run([sys.executable, ADMIT_CLI, '--plan-manifest', str(tmp_plan),
            '--shard-root', td, '--synthetic'],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert 'STRICT_SHARD_SET_ADMISSION_FAIL' in r.stdout
        assert 'failure_rows must equal 0' in r.stdout


def test_bool_shard_manifest_id_does_not_equal_integer_one():
    """True does not pass as integer shard_id 1."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        sm = json.loads((Path(shard_root) / "shard_1" / "shard_manifest.json").read_text())
        sm["shard_id"] = True
        (Path(shard_root) / "shard_1" / "shard_manifest.json").write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "shard_id" in r.stdout and "bool" in r.stdout.lower()


def test_float_shard_key_count_is_not_accepted_as_integer():
    """4.0 does not pass as integer key_count."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        sm = json.loads((Path(shard_root) / "shard_0" / "shard_manifest.json").read_text())
        sm["key_count"] = 4.0
        (Path(shard_root) / "shard_0" / "shard_manifest.json").write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "key_count" in r.stdout and "float" in r.stdout.lower()


@pytest.mark.parametrize("field,value", [
    ("baseline_rows", True),
    ("governed_rows", 576.0),
    ("failure_rows", False),
])
def test_bool_or_float_shard_row_count_is_rejected(field, value):
    """Bool/float shard manifest row counts are rejected at strict schema."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        sm = json.loads((Path(shard_root) / "shard_0" / "shard_manifest.json").read_text())
        sm[field] = value
        (Path(shard_root) / "shard_0" / "shard_manifest.json").write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0, f"expected fail for {field}={value!r}, got pass"
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert field in r.stdout


def test_self_consistent_corrupt_shard_ledger_is_structured_validator_failure():
    """Corrupt gzip with self-consistent manifest SHA produces structured failure."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        # Replace governed ledger with non-gzip bytes
        gl_path = Path(shard_root) / "shard_0" / "governed_ledger.csv.gz"
        bad_bytes = b"not-a-gzip-file"
        gl_path.write_bytes(bad_bytes)
        # Update manifest governed_sha256 to match
        sm = json.loads((Path(shard_root) / "shard_0" / "shard_manifest.json").read_text())
        sm["governed_sha256"] = hashlib.sha256(bad_bytes).hexdigest()
        (Path(shard_root) / "shard_0" / "shard_manifest.json").write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "validator data error" in r.stdout
        assert "Traceback" not in r.stderr


@pytest.mark.parametrize("value", [
    True,
    1.0,
    "1",
    None,
])
def test_non_integer_shard_schema_version_is_rejected(value):
    """Bool, float, string, None schema_version values are rejected."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        sm = json.loads((Path(shard_root) / "shard_0" / "shard_manifest.json").read_text())
        sm["schema_version"] = value
        (Path(shard_root) / "shard_0" / "shard_manifest.json").write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0, f"expected fail for schema_version={value!r}"
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "schema_version" in r.stdout and "integer 1" in r.stdout


def test_wrong_integer_shard_schema_version_is_rejected():
    """schema_version=2 is rejected."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        sm = json.loads((Path(shard_root) / "shard_0" / "shard_manifest.json").read_text())
        sm["schema_version"] = 2
        (Path(shard_root) / "shard_0" / "shard_manifest.json").write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "schema_version must equal 1" in r.stdout


def test_self_consistent_missing_run_id_column_is_structured_validator_failure():
    """Valid gzip/CSV but missing run_id column → KeyError containment."""
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        # Read baseline ledger, remove run_id column
        bl_path = Path(shard_root) / "shard_0" / "baseline_ledger.csv.gz"
        raw = gzip.decompress(bl_path.read_bytes()).decode("utf-8")
        lines = raw.split("\n")
        header_parts = lines[0].split(",")
        # Find and remove run_id
        rid_idx = header_parts.index("run_id")
        new_header = [h for h in header_parts if h != "run_id"]
        new_data = []
        for line in lines[1:]:
            if line == "":
                continue
            parts = line.split(",")
            new_parts = [p for i, p in enumerate(parts) if i != rid_idx]
            new_data.append(",".join(new_parts))
        new_text = ",".join(new_header) + "\n" + "\n".join(new_data) + "\n"
        new_bytes = gzip.compress(new_text.encode("utf-8"), mtime=0)
        bl_path.write_bytes(new_bytes)
        # Update manifest SHA
        sm = json.loads((Path(shard_root) / "shard_0" / "shard_manifest.json").read_text())
        sm["baseline_sha256"] = hashlib.sha256(new_bytes).hexdigest()
        (Path(shard_root) / "shard_0" / "shard_manifest.json").write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, ADMIT_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_SHARD_SET_ADMISSION_FAIL" in r.stdout
        assert "validator data error" in r.stdout
        assert "KeyError" in r.stdout
        assert "Traceback" not in r.stderr
