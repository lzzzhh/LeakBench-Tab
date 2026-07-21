"""R10b-2 Selection payload and semantic validation tests."""
import gzip, hashlib, io, json, os, subprocess, sys, tempfile
from pathlib import Path; import pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

from scripts.t0_b_full_b1.fragment_contract import (
    parse_int_array_json, parse_group_id_array_json,
    normalize_selection_payload, SelectionContractError,
    validate_selection_payload_consistency,
    validate_selection_realized_cost,
    validate_semantic_group_atomicity, validate_m09_eight_columns,
    validate_completed_key,
)


def _write_gz(path, df):
    import io as _io
    buf = _io.StringIO(); df.to_csv(buf, index=False, header=True)
    path.write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))


# ─── Array parsing ──────────────────────────────────────────────

def test_parse_int_array_valid():
    assert parse_int_array_json("[1, 2, 3]") == [1, 2, 3]
    assert parse_int_array_json([1, 2, 3]) == [1, 2, 3]  # pre-parsed

def test_parse_int_array_single_element():
    assert parse_int_array_json("[5]") == [5]

def test_parse_int_array_empty():
    assert parse_int_array_json("[]") == []

def test_parse_int_array_duplicate_fails():
    with pytest.raises(SelectionContractError, match="duplicate"):
        parse_int_array_json("[1, 1, 2]")

def test_parse_int_array_negative_fails():
    with pytest.raises(SelectionContractError, match="negative"):
        parse_int_array_json("[0, -1]")

def test_parse_int_array_bool_fails():
    with pytest.raises(SelectionContractError):
        parse_int_array_json("[true, false]")

def test_parse_group_id_valid():
    assert parse_group_id_array_json('["a", "b"]') == ["a", "b"]

def test_parse_group_id_empty_string_fails():
    with pytest.raises(SelectionContractError):
        parse_group_id_array_json('[""]')


# ─── Payload normalization ─────────────────────────────────────

def test_normalize_valid_payload():
    p = normalize_selection_payload({"selection_hash": "abc", "policy": "P2", "contract": "semantic_group",
        "budget_bp": 500, "removed_encoded_indices": "[1,2]", "removed_group_ids": "[]", "realized_encoded_cost": 2})
    assert p["realized_encoded_cost"] == 2
    assert p["removed_encoded_indices"] == [1, 2]

def test_normalize_invalid_policy_fails():
    with pytest.raises(SelectionContractError):
        normalize_selection_payload({"selection_hash": "a", "policy": "P99", "contract": "x", "budget_bp": 0,
            "removed_encoded_indices": "[]", "removed_group_ids": "[]", "realized_encoded_cost": 0})


# ─── Payload consistency ───────────────────────────────────────

def test_same_hash_same_payload_passes():
    p = [{"selection_hash": "h1", "policy": "P2", "contract": "sg", "budget_bp": 500, "removed_encoded_indices": [1], "removed_group_ids": [], "realized_encoded_cost": 1},
         {"selection_hash": "h1", "policy": "P2", "contract": "sg", "budget_bp": 500, "removed_encoded_indices": [1], "removed_group_ids": [], "realized_encoded_cost": 1}]
    assert validate_selection_payload_consistency(p) == []

def test_same_hash_different_payload_fails():
    p = [{"selection_hash": "h1", "policy": "P2", "contract": "sg", "budget_bp": 500, "removed_encoded_indices": [1], "removed_group_ids": [], "realized_encoded_cost": 1},
         {"selection_hash": "h1", "policy": "P2", "contract": "sg", "budget_bp": 500, "removed_encoded_indices": [2], "removed_group_ids": [], "realized_encoded_cost": 1}]
    assert len(validate_selection_payload_consistency(p)) > 0


# ─── Realized cost ─────────────────────────────────────────────

def test_selection_cost_mismatch_fails():
    p = [{"selection_hash": "h1", "policy": "P2", "contract": "sg", "budget_bp": 500, "removed_encoded_indices": [1, 2, 3], "removed_group_ids": [], "realized_encoded_cost": 5}]
    assert len(validate_selection_realized_cost(p)) > 0

def test_selection_cost_matches_passes():
    p = [{"selection_hash": "h1", "policy": "P2", "contract": "sg", "budget_bp": 500, "removed_encoded_indices": [1, 2, 3], "removed_group_ids": [], "realized_encoded_cost": 3}]
    assert validate_selection_realized_cost(p) == []


# ─── Semantic atomicity ───────────────────────────────────────

def test_complete_semantic_group_passes():
    members = {"g001": {1, 2, 3}}
    p = {"contract": "semantic_group", "removed_group_ids": ["g001"], "removed_encoded_indices": [1, 2, 3]}
    assert validate_semantic_group_atomicity(p, members) == []

def test_partial_semantic_group_fails():
    members = {"g001": {1, 2, 3}}
    p = {"contract": "semantic_group", "removed_group_ids": ["g001"], "removed_encoded_indices": [1, 2]}
    assert len(validate_semantic_group_atomicity(p, members)) > 0

def test_unknown_group_fails():
    members = {"g001": {1, 2}}
    p = {"contract": "semantic_group", "removed_group_ids": ["g999"], "removed_encoded_indices": []}
    assert len(validate_semantic_group_atomicity(p, members)) > 0


# ─── M09 ───────────────────────────────────────────────────────

def test_m09_eight_columns_passes():
    """M09 with 8-column group passes when group is selected completely."""
    pol_map = {"groups": [{"opaque_group_id": "g012", "member_encoded_indices": list(range(12, 20)), "group_size": 8}]}
    kp = {"mechanism": "M09"}
    p = {"contract": "semantic_group", "removed_group_ids": ["g012"], "removed_encoded_indices": list(range(12, 20))}
    assert validate_m09_eight_columns(p, pol_map, kp) == []

def test_m09_seven_columns_fails():
    pol_map = {"groups": [{"opaque_group_id": "g012", "member_encoded_indices": list(range(12, 20)), "group_size": 8}]}
    kp = {"mechanism": "M09"}
    p = {"contract": "semantic_group", "removed_group_ids": ["g012"], "removed_encoded_indices": list(range(12, 19))}
    assert len(validate_m09_eight_columns(p, pol_map, kp)) > 0

def test_m09_skips_non_m09_keys():
    pol_map = {"groups": []}
    kp = {"mechanism": "M01"}
    assert validate_m09_eight_columns({"contract": "sg", "removed_group_ids": [], "removed_encoded_indices": []}, pol_map, kp) == []


# ─── CLI: first execution still works ──────────────────────────

def test_first_execution_still_passes():
    RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
    SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", f"{td}/shard_0", "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[:300]


def test_complete_resume_still_passes():
    RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
    SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        r2 = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode == 0
