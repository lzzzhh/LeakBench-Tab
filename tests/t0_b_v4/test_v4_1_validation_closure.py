"""T0-B V4.1 Regression Tests — provenance closure, receipt validation, tested-tree binding."""
from __future__ import annotations
import gzip, hashlib, json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_v4.validate_protocol_freeze_v4_1 import (
    validate_receipt, validate_tested_tree, SCIENTIFIC_FREEZE,
)

def test_receipt_repo_failed_triggers_error():
    r = {
        "repository_suite": {"passed": 353, "failed": 1, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "v4_targeted_suite": {"passed": 13, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "validation_scope": "LOCAL_VALIDATION_ONLY", "github_actions_configured": False,
        "scientific_design_modified": False, "scientific_freeze_commit": SCIENTIFIC_FREEZE,
        "tested_git_sha": "4cb54f01d11ef7250b7a300c0f7757abddead5a4", "timestamp_utc": "",
    }
    errs = validate_receipt(r)
    assert len(errs) > 0, "Should error on repo failed=1"
    assert any("failed=1" in e for e in errs)

def test_receipt_targeted_failed_triggers_error():
    r = {
        "repository_suite": {"passed": 354, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "v4_targeted_suite": {"passed": 12, "failed": 1, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "validation_scope": "LOCAL_VALIDATION_ONLY", "github_actions_configured": False,
        "scientific_design_modified": False, "scientific_freeze_commit": SCIENTIFIC_FREEZE,
        "tested_git_sha": "4cb54f01d11ef7250b7a300c0f7757abddead5a4", "timestamp_utc": "",
    }
    errs = validate_receipt(r)
    assert any("V4 failed=1" in e for e in errs)

def test_receipt_zero_passed_triggers_error():
    r = {
        "repository_suite": {"passed": 0, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "v4_targeted_suite": {"passed": 0, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "validation_scope": "LOCAL_VALIDATION_ONLY", "github_actions_configured": False,
        "scientific_design_modified": False, "scientific_freeze_commit": SCIENTIFIC_FREEZE,
        "tested_git_sha": "4cb54f01d11ef7250b7a300c0f7757abddead5a4", "timestamp_utc": "",
    }
    errs = validate_receipt(r)
    assert any("passed=0" in e for e in errs)

def test_receipt_wrong_scope_triggers_error():
    r = {
        "repository_suite": {"passed": 354, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "v4_targeted_suite": {"passed": 13, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "validation_scope": "CI", "github_actions_configured": False,
        "scientific_design_modified": False, "scientific_freeze_commit": SCIENTIFIC_FREEZE,
        "tested_git_sha": "4cb54f01d11ef7250b7a300c0f7757abddead5a4", "timestamp_utc": "",
    }
    errs = validate_receipt(r)
    assert any("scope" in e for e in errs)

def test_receipt_wrong_freeze_sha_triggers_error():
    r = {
        "repository_suite": {"passed": 354, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "v4_targeted_suite": {"passed": 13, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "validation_scope": "LOCAL_VALIDATION_ONLY", "github_actions_configured": False,
        "scientific_design_modified": False, "scientific_freeze_commit": "0000000000000000000000000000000000000000",
        "tested_git_sha": "4cb54f01d11ef7250b7a300c0f7757abddead5a4", "timestamp_utc": "",
    }
    errs = validate_receipt(r)
    assert any("scientific_freeze_commit" in e for e in errs)

def test_receipt_invalid_tested_sha_triggers_error():
    r = {
        "repository_suite": {"passed": 354, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "v4_targeted_suite": {"passed": 13, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1.0},
        "validation_scope": "LOCAL_VALIDATION_ONLY", "github_actions_configured": False,
        "scientific_design_modified": False, "scientific_freeze_commit": SCIENTIFIC_FREEZE,
        "tested_git_sha": "not-a-sha", "timestamp_utc": "",
    }
    errs = validate_receipt(r)
    assert any("tested_git_sha" in e for e in errs)

def test_current_receipt_passes():
    with open(ROOT / "results/edbt_t0_b/static_validation_receipt_v4_1.json") as f:
        rec = json.load(f)
    errs = validate_receipt(rec)
    assert len(errs) == 0, f"Current receipt should pass: {errs}"

def test_dryrun_bundle_disk_sha():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    for k in dr["keys"]:
        disk = hashlib.sha256((ROOT / k["bundle_path"]).read_bytes()).hexdigest()
        assert disk == k["bundle_sha256"]

def test_dryrun_split_hashes():
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    for k in dr["keys"]:
        b = np.load(ROOT / k["bundle_path"], allow_pickle=False)
        for sn in ["train_idx", "val_idx", "test_idx"]:
            actual = hashlib.sha256(b[sn].tobytes()).hexdigest()
            assert actual == k[f"{sn}_hash"]

def test_dryrun_keys_in_mapping_ledgers():
    for gz_name in ["policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gzip.decompress((ROOT / "results/edbt_t0_b" / gz_name).read_bytes()).decode("utf-8")
        keys = set()
        for line in data.strip().split("\n"):
            r = json.loads(line)
            keys.add((r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]))
        with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
            dr = json.load(f)
        for k in dr["keys"]:
            key = (k["dataset_index"], k["mechanism"], k["strength"], k["training_seed"])
            assert key in keys, f"Key {key} not in {gz_name}"

def test_mapping_sha_recompute():
    p = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
    data = gzip.decompress(p.read_bytes()).decode("utf-8")
    for line in data.strip().split("\n"):
        row = json.loads(line)
        rep = json.dumps([{"gid": g["opaque_group_id"], "members": g["member_encoded_indices"]} for g in row["groups"]], sort_keys=True)
        assert hashlib.sha256(rep.encode()).hexdigest() == row["mapping_sha256"]

def test_scientific_configs_no_diff():
    import subprocess
    r = subprocess.run(
        ["git", "diff", "--name-only", f"{SCIENTIFIC_FREEZE}...HEAD", "--",
         "configs/edbt_t0_b/policy_registry_v4.yaml", "configs/edbt_t0_b/dryrun_matrix_v4.json",
         "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.stdout.strip() == "", f"Scientific files modified: {r.stdout}"
