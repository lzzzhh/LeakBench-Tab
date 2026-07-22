"""R10c-2 Strict global merge tests — deterministic atomic publication."""
import gzip, hashlib, io, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path; import pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
MERGE_CLI = str(ROOT/"scripts/t0_b_full_b1/merge_full_b1_shards.py")
SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")


def _generate_all_shards(shard_root):
    plan = json.loads(Path(SYNTH_PLAN).read_text())
    plan_dir = Path(SYNTH_PLAN).parent
    keys = [json.loads(l) for l in gzip.decompress(
        (plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
    for sid in sorted(set(k["shard_id"] for k in keys)):
        out = str(Path(shard_root) / f"shard_{sid}")
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", str(sid), "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"shard {sid} failed: {r.stderr[:300]}"


def _five_shas(d):
    d = Path(d)
    return {f"{n}_ledger.csv.gz": hashlib.sha256((d / f"{n}_ledger.csv.gz").read_bytes()).hexdigest()
            for n in ["baseline", "governed", "selection", "failure"]} | \
           {"merge_manifest.json": hashlib.sha256((d / "merge_manifest.json").read_bytes()).hexdigest()}


def test_exact_synthetic_merge_pass():
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        merged_out = f"{td}/merged"
        _generate_all_shards(shard_root)
        r = subprocess.run([sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--output-dir", merged_out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, f"stdout={r.stdout[:500]}\nstderr={r.stderr[:300]}"
        assert "STRICT_GLOBAL_MERGE_PASS" in r.stdout
        assert "canonical_keys=8" in r.stdout
        out = Path(merged_out)
        for f in ["baseline_ledger.csv.gz", "governed_ledger.csv.gz",
                   "selection_ledger.csv.gz", "failure_ledger.csv.gz",
                   "merge_manifest.json"]:
            assert (out / f).exists(), f"missing {f}"


def test_merge_is_deterministic():
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        _generate_all_shards(shard_root)
        out1 = f"{td}/merged1"
        out2 = f"{td}/merged2"
        subprocess.run([sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--output-dir", out1, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        subprocess.run([sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--output-dir", out2, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        shas1 = _five_shas(out1)
        shas2 = _five_shas(out2)
        for k in shas1:
            assert shas1[k] == shas2[k], f"{k} differs between merges"


def test_invalid_source_shard_blocks_publication():
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        merged_out = f"{td}/merged"
        _generate_all_shards(shard_root)
        # Tamper shard_0 governed ledger
        gpath = Path(shard_root) / "shard_0" / "governed_ledger.csv.gz"
        raw = gzip.decompress(gpath.read_bytes()).decode("utf-8")
        lines = raw.split("\n")
        data = [l for l in lines[1:] if l != ""]
        parts = data[0].split(",")
        parts[13] = "0.999" if parts[13] != "0.999" else "0.001"
        data[0] = ",".join(parts)
        new_text = lines[0] + "\n" + "\n".join(sorted(data)) + "\n"
        nb = gzip.compress(new_text.encode("utf-8"), mtime=0)
        gpath.write_bytes(nb)
        sm = json.loads((Path(shard_root) / "shard_0" / "shard_manifest.json").read_text())
        sm["governed_sha256"] = hashlib.sha256(nb).hexdigest()
        (Path(shard_root) / "shard_0" / "shard_manifest.json").write_text(json.dumps(sm))
        r = subprocess.run([sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--output-dir", merged_out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "STRICT_GLOBAL_MERGE_FAIL" in r.stdout
        assert not Path(merged_out).exists()


def test_existing_output_refuses_overwrite():
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        merged_out = f"{td}/merged"
        _generate_all_shards(shard_root)
        Path(merged_out).mkdir()
        (Path(merged_out) / "sentinel.txt").write_text("do not touch")
        r = subprocess.run([sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--output-dir", merged_out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode != 0
        assert "MERGE_OUTPUT_ALREADY_EXISTS" in r.stdout
        assert (Path(merged_out) / "sentinel.txt").read_text() == "do not touch"


def test_candidate_payload_tamper_rejected():
    """Helper-level: tampered candidate is rejected by source aggregate check."""
    from scripts.t0_b_full_b1.merge_contract import (
        validate_global_merge_candidate, build_source_shard_snapshot,
    )
    with tempfile.TemporaryDirectory() as td:
        shard_root = f"{td}/shards"
        merged_out = f"{td}/merged"
        _generate_all_shards(shard_root)
        subprocess.run([sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
            "--shard-root", shard_root, "--output-dir", merged_out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        plan = json.loads(Path(SYNTH_PLAN).read_text())
        plan_sha = hashlib.sha256(Path(SYNTH_PLAN).read_bytes()).hexdigest()
        plan_dir = Path(SYNTH_PLAN).parent
        keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
        runs = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode().strip().split("\n")]
        planned_ids = sorted(set(k["shard_id"] for k in keys))
        snapshot = build_source_shard_snapshot(Path(shard_root), planned_ids)
        # Tamper candidate governed
        gpath = Path(merged_out) / "governed_ledger.csv.gz"
        raw = gzip.decompress(gpath.read_bytes()).decode("utf-8")
        lines = raw.split("\n")
        data = [l for l in lines[1:] if l != ""]
        parts = data[0].split(",")
        parts[13] = "0.999" if parts[13] != "0.999" else "0.001"
        data[0] = ",".join(parts)
        nb = gzip.compress((lines[0] + "\n" + "\n".join(sorted(data)) + "\n").encode(), mtime=0)
        gpath.write_bytes(nb)
        mm = json.loads((Path(merged_out) / "merge_manifest.json").read_text())
        mm["governed_sha256"] = hashlib.sha256(nb).hexdigest()
        (Path(merged_out) / "merge_manifest.json").write_text(json.dumps(mm))
        result = validate_global_merge_candidate(
            merged_dir=Path(merged_out), plan_manifest=plan, plan_manifest_sha256=plan_sha,
            planned_shard_ids=planned_ids, snapshot=snapshot,
            shard_root=Path(shard_root), run_rows=runs, key_rows=keys,
        )
        assert not result.is_valid
        assert result.source_aggregate_valid is False


def test_header_only_ledger_is_exact_and_blank_row_is_rejected():
    """`header\n` is empty; `header\n\n` contains an illegal physical row."""
    from scripts.t0_b_full_b1.merge_contract import open_strict_sorted_ledger_rows
    with tempfile.TemporaryDirectory() as td:
        valid = Path(td) / "valid.csv.gz"
        invalid = Path(td) / "invalid.csv.gz"
        valid.write_bytes(gzip.compress(b"run_id\n", mtime=0))
        invalid.write_bytes(gzip.compress(b"run_id\n\n", mtime=0))
        with open_strict_sorted_ledger_rows(valid, "run_id", "valid") as rows:
            assert list(rows) == []
        with pytest.raises(ValueError, match="blank physical row"):
            with open_strict_sorted_ledger_rows(invalid, "run_id", "invalid") as rows:
                list(rows)


def test_missing_trailing_newline_is_rejected():
    from scripts.t0_b_full_b1.merge_contract import open_strict_sorted_ledger_rows
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bad.csv.gz"
        path.write_bytes(gzip.compress(b"run_id", mtime=0))
        with pytest.raises(ValueError, match="header missing trailing newline"):
            with open_strict_sorted_ledger_rows(path, "run_id", "bad") as rows:
                list(rows)


@pytest.mark.parametrize("seal", ["", "a" * 39, "a" * 41, "A" * 40, "g" * 40])
def test_tool_seal_requires_exact_lowercase_git_sha(seal):
    from scripts.t0_b_full_b1.merge_contract import validate_plan_schema
    plan = json.loads(Path(SYNTH_PLAN).read_text())
    plan["tool_seal_sha"] = seal
    errors = validate_plan_schema(plan, "synthetic")
    assert any("exact 40-char lowercase Git SHA" in error for error in errors)


def test_synthetic_plan_without_flag_is_rejected_as_nonproduction():
    with tempfile.TemporaryDirectory() as td:
        shard_root = Path(td) / "shards"
        shard_root.mkdir()
        output = Path(td) / "merged"
        r = subprocess.run(
            [sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
             "--shard-root", str(shard_root), "--output-dir", str(output)],
            capture_output=True, text=True, cwd=ROOT,
        )
        assert r.returncode != 0
        assert "incompatible with production plan" in r.stdout
        assert not output.exists()


def test_output_parent_symlink_is_rejected_before_publication():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard_root = root / "shards"
        shard_root.mkdir()
        real_parent = root / "real-parent"
        real_parent.mkdir()
        link_parent = root / "linked-parent"
        link_parent.symlink_to(real_parent, target_is_directory=True)
        output = link_parent / "merged"
        r = subprocess.run(
            [sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
             "--shard-root", str(shard_root), "--output-dir", str(output),
             "--synthetic"],
            capture_output=True, text=True, cwd=ROOT,
        )
        assert r.returncode != 0
        assert "MERGE_OUTPUT_PARENT_INVALID" in r.stdout
        assert not (real_parent / "merged").exists()


def test_candidate_source_comparison_does_not_read_ledger_bytes(monkeypatch):
    """Candidate validation must stream every CSV ledger, including hashing."""
    from scripts.t0_b_full_b1.merge_contract import (
        validate_global_merge_candidate, build_source_shard_snapshot,
    )
    with tempfile.TemporaryDirectory() as td:
        shard_root = Path(td) / "shards"
        merged_out = Path(td) / "merged"
        _generate_all_shards(shard_root)
        merge = subprocess.run(
            [sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
             "--shard-root", str(shard_root), "--output-dir", str(merged_out),
             "--synthetic"],
            capture_output=True, text=True, cwd=ROOT,
        )
        assert merge.returncode == 0, merge.stdout

        plan_path = Path(SYNTH_PLAN)
        plan = json.loads(plan_path.read_text())
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        plan_dir = plan_path.parent
        keys = [json.loads(line) for line in gzip.decompress(
            (plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()
        ).decode().strip().split("\n")]
        runs = [json.loads(line) for line in gzip.decompress(
            (plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()
        ).decode().strip().split("\n")]
        planned_ids = sorted({key["shard_id"] for key in keys})
        snapshot = build_source_shard_snapshot(shard_root, planned_ids)

        original_read_bytes = Path.read_bytes
        def guarded_read_bytes(path):
            if str(path).endswith(".csv.gz"):
                raise AssertionError(f"whole-ledger read forbidden: {path}")
            return original_read_bytes(path)
        monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

        result = validate_global_merge_candidate(
            merged_dir=merged_out, plan_manifest=plan,
            plan_manifest_sha256=plan_sha, planned_shard_ids=planned_ids,
            snapshot=snapshot, shard_root=shard_root, run_rows=runs,
            key_rows=keys,
        )
        assert result.is_valid, result.errors


def _copy_plan_fixture(destination):
    source = Path(SYNTH_PLAN).parent
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ["full_b1_plan_manifest.json", "full_b1_key_plan.jsonl.gz",
                 "full_b1_run_plan.jsonl.gz", "full_b1_shard_plan.json"]:
        shutil.copy2(source / name, destination / name)
    return destination / "full_b1_plan_manifest.json"


def test_key_plan_sha_mismatch_fails_before_output():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        plan_path = _copy_plan_fixture(root / "plan")
        plan = json.loads(plan_path.read_text())
        plan["key_plan_sha256"] = "0" * 64
        plan_path.write_text(json.dumps(plan))
        shard_root = root / "shards"
        shard_root.mkdir()
        output = root / "merged"
        result = subprocess.run(
            [sys.executable, MERGE_CLI, "--plan-manifest", str(plan_path),
             "--shard-root", str(shard_root), "--output-dir", str(output),
             "--synthetic"], capture_output=True, text=True, cwd=ROOT,
        )
        assert result.returncode != 0
        assert "key plan SHA mismatch" in result.stdout
        assert not output.exists()


def test_missing_run_plan_fails_before_output():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        plan_path = _copy_plan_fixture(root / "plan")
        (plan_path.parent / "full_b1_run_plan.jsonl.gz").unlink()
        shard_root = root / "shards"
        shard_root.mkdir()
        output = root / "merged"
        result = subprocess.run(
            [sys.executable, MERGE_CLI, "--plan-manifest", str(plan_path),
             "--shard-root", str(shard_root), "--output-dir", str(output),
             "--synthetic"], capture_output=True, text=True, cwd=ROOT,
        )
        assert result.returncode != 0
        assert "run plan file missing" in result.stdout
        assert not output.exists()


def test_source_shard_lock_blocks_merge_without_partial_output():
    from scripts.t0_b_full_b1.io_contract import exclusive_writer_lock
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard_root = root / "shards"
        output = root / "merged"
        _generate_all_shards(shard_root)
        with exclusive_writer_lock(shard_root / "shard_0", operation="test-holder"):
            result = subprocess.run(
                [sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
                 "--shard-root", str(shard_root), "--output-dir", str(output),
                 "--synthetic"], capture_output=True, text=True, cwd=ROOT,
            )
        assert result.returncode != 0
        assert "SOURCE_SHARD_LOCK_FAIL" in result.stdout
        assert not output.exists()


@pytest.mark.parametrize("mutation", ["unsorted", "duplicate", "failure_row", "digest"])
def test_corrupt_source_variants_fail_closed(mutation):
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard_root = root / "shards"
        output = root / "merged"
        _generate_all_shards(shard_root)
        shard = shard_root / "shard_0"
        manifest_path = shard / "shard_manifest.json"
        manifest = json.loads(manifest_path.read_text())

        if mutation in {"unsorted", "duplicate"}:
            ledger = shard / "baseline_ledger.csv.gz"
            lines = gzip.decompress(ledger.read_bytes()).decode().splitlines()
            rows = lines[1:]
            if mutation == "unsorted":
                rows[0], rows[-1] = rows[-1], rows[0]
            else:
                rows.append(rows[0])
            payload = (lines[0] + "\n" + "\n".join(rows) + "\n").encode()
            encoded = gzip.compress(payload, mtime=0)
            ledger.write_bytes(encoded)
            manifest["baseline_sha256"] = hashlib.sha256(encoded).hexdigest()
        elif mutation == "failure_row":
            ledger = shard / "failure_ledger.csv.gz"
            encoded = gzip.compress(b"run_id\nunexpected-failure\n", mtime=0)
            ledger.write_bytes(encoded)
            manifest["failure_sha256"] = hashlib.sha256(encoded).hexdigest()
            manifest["failure_rows"] = 1
        else:
            manifest["planned_run_ids_sha256"] = "0" * 64

        manifest_path.write_text(json.dumps(manifest))
        result = subprocess.run(
            [sys.executable, MERGE_CLI, "--plan-manifest", SYNTH_PLAN,
             "--shard-root", str(shard_root), "--output-dir", str(output),
             "--synthetic"], capture_output=True, text=True, cwd=ROOT,
        )
        assert result.returncode != 0, mutation
        assert "STRICT_GLOBAL_MERGE_FAIL" in result.stdout
        assert not output.exists()


def test_fsync_failure_cleans_staging_and_never_publishes(monkeypatch):
    import scripts.t0_b_full_b1.merge_full_b1_shards as merge_module
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard_root = root / "shards"
        output = root / "merged"
        _generate_all_shards(shard_root)

        def fail_fsync(_path):
            raise OSError("simulated publication fsync failure")

        monkeypatch.setattr(merge_module, "_fsync_path", fail_fsync)
        monkeypatch.setattr(sys, "argv", [
            "merge_full_b1_shards.py", "--plan-manifest", SYNTH_PLAN,
            "--shard-root", str(shard_root), "--output-dir", str(output),
            "--synthetic",
        ])
        with pytest.raises(SystemExit) as exc:
            merge_module.main()
        assert exc.value.code == 1
        assert not output.exists()
        assert not list(root.glob(".merged.staging.*"))
