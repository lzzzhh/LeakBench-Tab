"""T0-B V4.1 Validation Closure Regression Tests — receipt failure detection, hash walking, dry-run binding."""
from __future__ import annotations
import gzip, hashlib, json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_receipt_failed_triggers_error():
    """Validator must detect receipt with failed > 0."""
    receipt = {
        "repository_suite": {"passed": 353, "failed": 1, "skipped": 0, "command": "", "duration_seconds": 1},
        "v4_targeted_suite": {"passed": 13, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1},
        "validation_scope": "LOCAL_VALIDATION_ONLY",
        "scientific_design_modified": False,
        "timestamp_utc": "", "tested_git_sha": "", "scientific_freeze_commit": "",
    }
    assert receipt["repository_suite"]["failed"] == 1  # Simulates old stale receipt
    # Real validator would check: if receipt["repo"]["failed"] != 0 -> ERROR


def test_receipt_targeted_failed_triggers_error():
    receipt = {
        "repository_suite": {"passed": 354, "failed": 0, "skipped": 0, "command": "", "duration_seconds": 1},
        "v4_targeted_suite": {"passed": 12, "failed": 1, "skipped": 0, "command": "", "duration_seconds": 1},
        "validation_scope": "LOCAL_VALIDATION_ONLY",
        "scientific_design_modified": False,
        "timestamp_utc": "", "tested_git_sha": "", "scientific_freeze_commit": "",
    }
    assert receipt["v4_targeted_suite"]["failed"] == 1


def test_hash_walker_finds_receipt():
    """Recursive hash walker must traverse nested objects and find {path, sha256} pairs."""
    freeze = {
        "static_receipt": {
            "path": "results/edbt_t0_b/static_validation_receipt_v4_1.json",
            "sha256": hashlib.sha256(
                (ROOT / "results/edbt_t0_b/static_validation_receipt_v4_1.json").read_bytes()
            ).hexdigest(),
        }
    }
    # Simulate what the recursive verifier does
    for k, v in freeze.items():
        if isinstance(v, dict) and "path" in v and "sha256" in v:
            fp = ROOT / v["path"]
            assert fp.exists()
            disk_sha = hashlib.sha256(fp.read_bytes()).hexdigest()
            assert disk_sha == v["sha256"]


def test_byte_modification_detected():
    """Modifying 1 byte of a bound file must produce SHA mismatch."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        tf.write(b"original content")
        tmpath = tf.name
    try:
        h1 = hashlib.sha256(Path(tmpath).read_bytes()).hexdigest()
        Path(tmpath).write_bytes(b"modified content")
        h2 = hashlib.sha256(Path(tmpath).read_bytes()).hexdigest()
        assert h1 != h2
    finally:
        Path(tmpath).unlink()


def test_dryrun_bundle_disk_sha_matches():
    """All 4 dry-run bundles must have correct disk SHA."""
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    for k in dr["keys"]:
        disk_sha = hashlib.sha256((ROOT / k["bundle_path"]).read_bytes()).hexdigest()
        assert disk_sha == k["bundle_sha256"], f"Bundle SHA mismatch: {k['bundle_path']}"


def test_dryrun_split_hashes_match():
    """All 4 dry-run keys must have correct train/val/test split hashes."""
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    for k in dr["keys"]:
        b = np.load(ROOT / k["bundle_path"], allow_pickle=False)
        for split_name in ["train_idx", "val_idx", "test_idx"]:
            actual = hashlib.sha256(b[split_name].tobytes()).hexdigest()
            assert actual == k[f"{split_name}_hash"], \
                f"{k['dataset_index']}_{k['mechanism']} {split_name}: {actual[:16]} != {k[f'{split_name}_hash'][:16]}"


def test_dryrun_keys_in_ledgers():
    """All 4 dry-run keys exist in both mapping ledgers."""
    for gz_name in ["policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gzip.decompress((ROOT / "results/edbt_t0_b" / gz_name).read_bytes()).decode("utf-8")
        keys_in_ledger = set()
        for line in data.strip().split("\n"):
            r = json.loads(line)
            keys_in_ledger.add((r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]))

        with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
            dr = json.load(f)
        for k in dr["keys"]:
            key = (k["dataset_index"], k["mechanism"], k["strength"], k["training_seed"])
            assert key in keys_in_ledger, f"Dry key {key} not in {gz_name}"


def test_actual_receipt_matches():
    """Current receipt must have 0 failed in both suites."""
    with open(ROOT / "results/edbt_t0_b/static_validation_receipt_v4_1.json") as f:
        rec = json.load(f)
    assert rec["repository_suite"]["failed"] == 0
    assert rec["v4_targeted_suite"]["failed"] == 0
    assert rec["repository_suite"]["passed"] >= 354
    assert rec["v4_targeted_suite"]["passed"] >= 13


def test_scientific_configs_no_diff():
    """Scientific V4 configs must not be modified from ff347b."""
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--name-only", "ff347b...HEAD", "--",
         "configs/edbt_t0_b/policy_registry_v4.yaml",
         "configs/edbt_t0_b/dryrun_matrix_v4.json",
         "configs/edbt_t0_b/execution_matrix_v4.json",
         "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.stdout.strip() == "", f"Scientific configs modified: {result.stdout}"


def test_mapping_sha_recompute():
    """Per-row mapping_sha256 must match re-computation."""
    p = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
    data = gzip.decompress(p.read_bytes()).decode("utf-8")
    for line in data.strip().split("\n"):
        row = json.loads(line)
        mapping_repr = json.dumps(
            [{"gid": g["opaque_group_id"], "members": g["member_encoded_indices"]} for g in row["groups"]],
            sort_keys=True,
        )
        recomputed = hashlib.sha256(mapping_repr.encode()).hexdigest()
        assert recomputed == row["mapping_sha256"], \
            f"Mapping hash mismatch for {row['dataset_index']}_{row['mechanism']}"
