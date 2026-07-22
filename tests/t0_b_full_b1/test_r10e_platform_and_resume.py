"""Cross-platform durability and partial-shard resume regression tests."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/t0_b_full_b1/run_full_b1_shard.py"
SYNTH_PLAN = (
    ROOT
    / "results/edbt_t0_b_full_b1_preflight/synthetic_full_contract"
    / "full_b1_plan_manifest.json"
)


def test_windows_directory_fsync_is_an_explicit_noop(monkeypatch, tmp_path):
    from scripts.t0_b_full_b1 import io_contract, resume_contract

    def forbidden_open(*_args, **_kwargs):
        raise AssertionError("Windows directory fsync must not call os.open")

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(os, "open", forbidden_open)
    io_contract.fsync_parent_directory(tmp_path / "artifact.json")
    resume_contract._fsync_directory(tmp_path)


def test_dataframe_gzip_writer_pins_lf_line_endings(tmp_path):
    import gzip
    from scripts.t0_b_full_b1.io_contract import atomic_write_dataframe_gzip

    target = tmp_path / "rows.csv.gz"
    atomic_write_dataframe_gzip(
        target,
        pd.DataFrame([{"a": 1, "b": "x"}]),
        ["a", "b"],
    )
    payload = gzip.decompress(target.read_bytes())
    assert payload == b"a,b\n1,x\n"
    assert b"\r" not in payload


def test_declared_bundle_coverage_rejects_missing_and_hash_mismatch(tmp_path):
    from scripts.t0_b_full_b1 import run_full_b1_shard as runner

    path = tmp_path / "bundle.npz"
    relative = path
    path.write_bytes(b"frozen-bundle")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {"bundle_path": str(relative), "bundle_sha256": digest}

    runner.validate_declared_bundle_coverage([row], mode="production")
    path.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="declared bundle SHA mismatch"):
        runner.validate_declared_bundle_coverage([row], mode="production")
    path.unlink()
    with pytest.raises(RuntimeError, match="declared bundle missing or invalid"):
        runner.validate_declared_bundle_coverage([row], mode="production")


def test_partial_shard_resume_treats_absent_key_as_pending(tmp_path):
    output = tmp_path / "shard_0"
    initial = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--plan-manifest",
            str(SYNTH_PLAN),
            "--shard-id",
            "0",
            "--output-dir",
            str(output),
            "--synthetic",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert initial.returncode == 0, initial.stdout + initial.stderr

    fragment_dirs = sorted((output / "key_fragments").iterdir())
    assert len(fragment_dirs) == 4
    removed_id = fragment_dirs[0].name
    shutil.rmtree(fragment_dirs[0])

    resumed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--plan-manifest",
            str(SYNTH_PLAN),
            "--shard-id",
            "0",
            "--output-dir",
            str(output),
            "--synthetic",
            "--resume",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (output / "key_fragments" / removed_id / "completion_receipt.json").is_file()

    receipt = json.loads((output / "resume_receipt.json").read_text())
    assert receipt["recomputed"] == 1
    assert receipt["skipped"] == 3
    assert receipt["synthetic_call_counter_delta"]["lr_calls"] == 146
