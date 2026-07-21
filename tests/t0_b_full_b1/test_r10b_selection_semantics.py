"""R10b-2 Selection payload and semantic validation tests."""
import gzip, hashlib, io, json, os, subprocess, sys, tempfile
from pathlib import Path; import pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
SYNTH_PLAN = str(ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_plan_manifest.json")

from scripts.t0_b_full_b1.fragment_contract import (
    parse_int_array_json, parse_group_id_array_json,
    normalize_selection_payload, SelectionContractError,
    validate_selection_payload_consistency,
    validate_selection_realized_cost,
    validate_semantic_group_atomicity, validate_m09_eight_columns,
    validate_completed_key, validate_policy_mapping,
    validate_semantic_mapping, validate_encoded_column_contract,
    validate_governed_realized_cost, ValidatedPolicyMapping,
    canonical_selection_payload_json,
)


def _write_gz(path, df):
    import io as _io
    buf = _io.StringIO(); df.to_csv(buf, index=False, header=True)
    path.write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))


# ─── Array parsing ──────────────────────────────────────────────

def test_parse_int_array_valid():
    assert parse_int_array_json("[1, 2, 3]") == [1, 2, 3]
    # List input must now be rejected
    with pytest.raises(SelectionContractError, match="JSON string"):
        parse_int_array_json([1, 2, 3])

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
    """M09 with 8-column union passes via ValidatedSemanticMapping."""
    pol_map = {"groups": [{"opaque_group_id": "g012", "member_encoded_indices": list(range(12, 20)), "group_size": 8}]}
    sem_map = {"leak_group_ids": ["g012"]}
    kp = {"mechanism": "M09", "n_total_columns": 20}
    vm = validate_policy_mapping(pol_map, kp)
    vs = validate_semantic_mapping(sem_map, kp, vm)
    p = {"contract": "semantic_group", "removed_group_ids": ["g012"], "removed_encoded_indices": list(range(12, 20))}
    assert validate_m09_eight_columns(p, vs, kp) == []

def test_m09_seven_columns_fails():
    """M09 with 7-column union must fail via validate_semantic_mapping."""
    pol_map = {"groups": [{"opaque_group_id": "g012", "member_encoded_indices": list(range(12, 19)), "group_size": 7}]}
    kp = {"mechanism": "M09", "n_total_columns": 19}
    vm = validate_policy_mapping(pol_map, kp)
    with pytest.raises(SelectionContractError, match="size=7"):
        validate_semantic_mapping({"leak_group_ids": ["g012"]}, kp, vm)

def test_m09_skips_non_m09_keys():
    """Non-M09 keys skip M09 validation."""
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": [0], "group_size": 1}]}
    kp = {"mechanism": "M01", "n_total_columns": 1}
    vm = validate_policy_mapping(pol_map, kp)
    vs = validate_semantic_mapping({"leak_group_ids": ["g000"]}, kp, vm)
    p = {"contract": "sg", "removed_group_ids": [], "removed_encoded_indices": []}
    assert validate_m09_eight_columns(p, vs, kp) == []

def test_m09_missing_leak_group_ids_fails():
    """M09 without leak_group_ids must fail in validate_semantic_mapping."""
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": [0], "group_size": 1}]}
    kp = {"mechanism": "M09", "n_total_columns": 1}
    vm = validate_policy_mapping(pol_map, kp)
    with pytest.raises(SelectionContractError, match="leak_group_ids"):
        validate_semantic_mapping({}, kp, vm)


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


# ═══════════════════════════════════════════════════════════════════
# POLICY MAPPING NEGATIVE TESTS
# ═══════════════════════════════════════════════════════════════════

def test_mapping_duplicate_group_id_fails():
    with pytest.raises(SelectionContractError, match="duplicate group ID"):
        validate_policy_mapping({"groups": [
            {"opaque_group_id": "g000", "member_encoded_indices": [0], "group_size": 1},
            {"opaque_group_id": "g000", "member_encoded_indices": [1], "group_size": 1},
        ]}, {"n_total_columns": 2})

def test_mapping_group_size_mismatch_fails():
    with pytest.raises(SelectionContractError, match="group_size"):
        validate_policy_mapping({"groups": [
            {"opaque_group_id": "g000", "member_encoded_indices": [0, 1], "group_size": 99},
        ]}, {"n_total_columns": 2})

def test_mapping_cross_group_overlap_fails():
    with pytest.raises(SelectionContractError, match="already claimed"):
        validate_policy_mapping({"groups": [
            {"opaque_group_id": "g000", "member_encoded_indices": [0, 1], "group_size": 2},
            {"opaque_group_id": "g001", "member_encoded_indices": [1, 2], "group_size": 2},
        ]}, {"n_total_columns": 3})

def test_mapping_index_equal_n_total_fails():
    with pytest.raises(SelectionContractError, match="out of bounds"):
        validate_policy_mapping({"groups": [
            {"opaque_group_id": "g000", "member_encoded_indices": [3], "group_size": 1},
        ]}, {"n_total_columns": 3})

def test_mapping_negative_index_fails():
    with pytest.raises(SelectionContractError, match="out of bounds"):
        validate_policy_mapping({"groups": [
            {"opaque_group_id": "g000", "member_encoded_indices": [-1], "group_size": 1},
        ]}, {"n_total_columns": 3})

def test_mapping_group_size_string_fails():
    with pytest.raises(SelectionContractError, match="integral"):
        validate_policy_mapping({"groups": [
            {"opaque_group_id": "g000", "member_encoded_indices": [0], "group_size": "1"},
        ]}, {"n_total_columns": 1})

def test_mapping_member_float_fails():
    with pytest.raises(SelectionContractError, match="integral"):
        validate_policy_mapping({"groups": [
            {"opaque_group_id": "g000", "member_encoded_indices": [0.5], "group_size": 1},
        ]}, {"n_total_columns": 1})


# ═══════════════════════════════════════════════════════════════════
# ENCODED-COLUMN TESTS
# ═══════════════════════════════════════════════════════════════════

def test_encoded_column_partial_group_passes():
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": [0, 1, 2], "group_size": 3}]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 3})
    p = {"contract": "encoded_column", "removed_encoded_indices": [1], "removed_group_ids": []}
    assert validate_encoded_column_contract(p, vm) == []

def test_encoded_column_unknown_group_fails():
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": [0], "group_size": 1}]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 1})
    p = {"contract": "encoded_column", "removed_encoded_indices": [0], "removed_group_ids": ["missing"]}
    assert len(validate_encoded_column_contract(p, vm)) > 0

def test_encoded_column_out_of_bounds_fails():
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": [0], "group_size": 1}]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 3})
    p = {"contract": "encoded_column", "removed_encoded_indices": [3], "removed_group_ids": []}
    assert len(validate_encoded_column_contract(p, vm)) > 0


# ═══════════════════════════════════════════════════════════════════
# SEMANTIC MAPPING TESTS
# ═══════════════════════════════════════════════════════════════════

def test_semantic_mapping_none_returns_structured_error():
    with pytest.raises(SelectionContractError, match="must be dict"):
        validate_semantic_mapping(None, {"mechanism": "M01"}, ValidatedPolicyMapping({}, frozenset(), 1))

def test_semantic_mapping_list_returns_structured_error():
    with pytest.raises(SelectionContractError, match="must be dict"):
        validate_semantic_mapping([], {"mechanism": "M01"}, ValidatedPolicyMapping({}, frozenset(), 1))

def test_semantic_mapping_non_string_group_fails():
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": [0], "group_size": 1}]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 1})
    with pytest.raises(SelectionContractError, match="expected str"):
        validate_semantic_mapping({"leak_group_ids": [123]}, {"mechanism": "M01"}, vm)

def test_semantic_mapping_duplicate_group_fails():
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": [0], "group_size": 1}]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 1})
    with pytest.raises(SelectionContractError, match="duplicate"):
        validate_semantic_mapping({"leak_group_ids": ["g000", "g000"]}, {"mechanism": "M01"}, vm)

def test_semantic_mapping_unknown_group_fails():
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": [0], "group_size": 1}]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 1})
    with pytest.raises(SelectionContractError, match="not in policy mapping"):
        validate_semantic_mapping({"leak_group_ids": ["unknown"]}, {"mechanism": "M01"}, vm)

def test_m09_explicit_union_eight_passes():
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": list(range(8)), "group_size": 8}]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 8})
    vs = validate_semantic_mapping({"leak_group_ids": ["g000"]}, {"mechanism": "M09"}, vm)
    assert vs.leak_encoded_indices == frozenset(range(8))

def test_m09_explicit_union_seven_fails():
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": list(range(7)), "group_size": 7}]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 7})
    with pytest.raises(SelectionContractError, match="size=7"):
        validate_semantic_mapping({"leak_group_ids": ["g000"]}, {"mechanism": "M09"}, vm)

def test_m09_explicit_union_nine_fails():
    pol_map = {"groups": [{"opaque_group_id": "g000", "member_encoded_indices": list(range(9)), "group_size": 9}]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 9})
    with pytest.raises(SelectionContractError, match="size=9"):
        validate_semantic_mapping({"leak_group_ids": ["g000"]}, {"mechanism": "M09"}, vm)

def test_m09_unrelated_eight_column_group_not_used():
    """Unrelated 8-column group must NOT satisfy M09 rule; only explicit leak_group_ids matter."""
    pol_map = {"groups": [
        {"opaque_group_id": "unrelated_8", "member_encoded_indices": list(range(8)), "group_size": 8},
        {"opaque_group_id": "actual_leak_7", "member_encoded_indices": list(range(8, 15)), "group_size": 7},
    ]}
    vm = validate_policy_mapping(pol_map, {"n_total_columns": 15})
    with pytest.raises(SelectionContractError, match="size=7"):
        validate_semantic_mapping({"leak_group_ids": ["actual_leak_7"]}, {"mechanism": "M09"}, vm)


# ═══════════════════════════════════════════════════════════════════
# STRICT GOVERNED SCALAR TESTS
# ═══════════════════════════════════════════════════════════════════

def _gdf(cost):
    import pandas as pd
    return pd.DataFrame({"selection_hash": ["h1"], "realized_cost": [cost]})

def test_governed_cost_string_fails():
    errors = validate_governed_realized_cost(_gdf("2"), {"h1": {"realized_encoded_cost": 2}})
    assert len(errors) > 0
    assert any("integral" in e for e in errors)

def test_governed_cost_float_fails():
    errors = validate_governed_realized_cost(_gdf(2.0), {"h1": {"realized_encoded_cost": 2}})
    assert len(errors) > 0

def test_governed_cost_bool_fails():
    errors = validate_governed_realized_cost(_gdf(True), {"h1": {"realized_encoded_cost": 1}})
    assert len(errors) > 0

def test_governed_cost_negative_fails():
    errors = validate_governed_realized_cost(_gdf(-1), {"h1": {"realized_encoded_cost": 1}})
    assert len(errors) > 0

def test_governed_hash_none_fails():
    df = pd.DataFrame({"selection_hash": [None], "realized_cost": [1]})
    errors = validate_governed_realized_cost(df, {"h1": {"realized_encoded_cost": 1}})
    assert len(errors) > 0


# ═══════════════════════════════════════════════════════════════════
# M09 COMPLETE VALIDATOR DEEP TEST
# ═══════════════════════════════════════════════════════════════════

def test_complete_validator_m09_uses_explicit_seven_column_leak():
    """Complete validator must detect M09 with 7-column explicit leak group as incomplete."""
    import tempfile
    from scripts.t0_b_full_b1.fragment_contract import validate_completed_key, CompletedKeyValidation
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        cid = "test_m09_deep"
        fdir = tdp / cid; fdir.mkdir()
        # Build valid fragments
        kp = {"canonical_key_id": cid, "mechanism": "M09", "dataset_index": 0, "strength": "S1", "training_seed": 13, "n_total_columns": 15}
        planned = ["bl_0", "bl_1"] + [f"gl_{i:06d}" for i in range(144)]
        bl_df = pd.DataFrame({"run_id": ["bl_0", "bl_1"]})
        gl_df = pd.DataFrame({"run_id": planned[2:146], "selection_hash": ["s"] * 144, "realized_cost": [0]*144})
        sl_df = pd.DataFrame({"selection_hash": ["s"]*144, "policy": ["P2"]*144, "contract": ["encoded_column"]*144, "budget_bp": [500]*144, "removed_encoded_indices": ["[]"]*144, "removed_group_ids": ["[]"]*144, "realized_encoded_cost": [0]*144})
        fl_df = pd.DataFrame(columns=["run_id"])
        for name, df in [("baseline", bl_df), ("governed", gl_df), ("selection", sl_df), ("failure", fl_df)]:
            buf = io.StringIO(); df.to_csv(buf, index=False, header=True)
            (fdir / f"{name}.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode(), mtime=0))
        # Build manifest and receipt
        from scripts.t0_b_full_b1.fragment_contract import build_fragment_manifest
        manifest = build_fragment_manifest(cid, kp, planned, planned, fdir/"baseline.csv.gz", fdir/"governed.csv.gz", fdir/"selection.csv.gz", fdir/"failure.csv.gz", "ps")
        with open(fdir / "fragment_manifest.json", "w") as f: json.dump(manifest, f)
        receipt = {"schema_version": 1, "canonical_key_id": cid, "status": "complete", "fragment_manifest_sha256": hashlib.sha256((fdir/"fragment_manifest.json").read_bytes()).hexdigest(), "baseline_rows": 2, "governed_rows": 144, "selection_rows": 144, "failure_rows": 0}
        with open(fdir / "completion_receipt.json", "w") as f: json.dump(receipt, f)
        # Policy: unrelated_8 + actual_leak_7
        pol_map = {"groups": [
            {"opaque_group_id": "g000", "member_encoded_indices": list(range(8)), "group_size": 8},
            {"opaque_group_id": "g001", "member_encoded_indices": list(range(8,15)), "group_size": 7},
        ]}
        sem_map = {"leak_group_ids": ["g001"]}
        result = validate_completed_key(kp, planned, fdir, "ps", pol_map, sem_map)
        assert not result.is_complete
        assert any("size=7" in e for e in result.errors)


# ═══════════════════════════════════════════════════════════════════
# DEEP CLI COST MUTATION
# ═══════════════════════════════════════════════════════════════════

def test_cli_deep_cost_mutation_detected_after_sha_and_digest_closure():
    """All SHA/manifest/digest checks pass, but cost mutation detected by resume."""
    import shutil
    with tempfile.TemporaryDirectory() as td:
        out = f"{td}/shard_0"
        r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic"],
            capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0
        # Find first key
        frags = Path(out) / "key_fragments"
        first_key = sorted(frags.iterdir())[0]
        # Mutate: increase first row realized_encoded_cost by 1
        from scripts.t0_b_full_b1.fragment_contract import (
            normalize_selection_payload, canonical_selection_payload_json, build_fragment_manifest
        )
        sl_path = first_key / "selection.csv.gz"
        sl_df = pd.read_csv(sl_path)
        sl_df.loc[0, "realized_encoded_cost"] = int(sl_df.loc[0, "realized_encoded_cost"]) + 1
        buf = io.StringIO(); sl_df.to_csv(buf, index=False, header=True)
        sl_path.write_bytes(gzip.compress(buf.getvalue().encode(), mtime=0))
        # Rebuild manifest with correct SHAs and digests for modified files
        bl_path = first_key / "baseline.csv.gz"
        gl_path = first_key / "governed.csv.gz"
        fl_path = first_key / "failure.csv.gz"
        cid = first_key.name
        # Load key plan info
        key_data = gzip.decompress((ROOT/"results/edbt_t0_b_full_b1_preflight/synthetic_full_contract/full_b1_key_plan.jsonl.gz").read_bytes()).decode()
        kp = None
        for line in key_data.strip().split("\n"):
            rk = json.loads(line)
            if rk["canonical_key_id"] == cid: kp = rk; break
        assert kp is not None
        planned = [f"bl_{i}" for i in range(2)] + [f"gl_{i:06d}" for i in range(144)]
        manifest = build_fragment_manifest(cid, kp, planned, planned, bl_path, gl_path, sl_path, fl_path, "ps")
        with open(first_key / "fragment_manifest.json", "w") as f: json.dump(manifest, f)
        # Update receipt
        with open(first_key / "completion_receipt.json") as f: rec = json.load(f)
        rec["fragment_manifest_sha256"] = hashlib.sha256((first_key / "fragment_manifest.json").read_bytes()).hexdigest()
        with open(first_key / "completion_receipt.json", "w") as f: json.dump(rec, f)
        # Verify SHA/digest closure
        assert hashlib.sha256(sl_path.read_bytes()).hexdigest() == manifest["selection_sha256"]
        assert rec["fragment_manifest_sha256"] == hashlib.sha256((first_key / "fragment_manifest.json").read_bytes()).hexdigest()
        # Actual multiset (compute inline)
        sel_hashes = sorted(sl_df["selection_hash"].tolist())
        actual_multiset = hashlib.sha256(("\n".join(sel_hashes) + "\n").encode()).hexdigest()
        assert actual_multiset == manifest["selection_hash_multiset_sha256"]
        payloads = [normalize_selection_payload(row) for _, row in sl_df.iterrows()]
        canon = [canonical_selection_payload_json(p) for p in payloads]
        actual_digest = hashlib.sha256(("\n".join(sorted(canon)) + "\n").encode()).hexdigest()
        assert actual_digest == manifest["selection_payload_digest_sha256"]
        # Resume: must fail closed
        r2 = subprocess.run([sys.executable, RUNNER, "--plan-manifest", SYNTH_PLAN,
            "--shard-id", "0", "--output-dir", out, "--synthetic", "--resume"],
            capture_output=True, text=True, cwd=ROOT)
        assert r2.returncode != 0
        assert "RESUME_VALIDATION_FAIL" in r2.stdout
        assert "cost" in (r2.stdout + r2.stderr).lower()
        assert "SHA mismatch" not in (r2.stdout + r2.stderr)
        assert "digest mismatch" not in (r2.stdout + r2.stderr)
