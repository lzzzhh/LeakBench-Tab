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
