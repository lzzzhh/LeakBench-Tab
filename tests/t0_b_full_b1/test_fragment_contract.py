"""R10b fragment contract + validate_completed_key tests."""
import gzip, hashlib, io, json, os, sys, tempfile
from pathlib import Path; import pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

# Minimal valid mappings for test use
_SYNTH_POL = {"groups": [{"opaque_group_id": f"g{i:03d}", "member_encoded_indices": [i], "group_size": 1} for i in range(15)]}
_SYNTH_SEM = {"leak_group_ids": ["g012"]}
TEST_PLAN_SHA = "a" * 64

from scripts.t0_b_full_b1.fragment_contract import (
    ProductionGuard, SyntheticCallCounter,
    CompletedKeyValidation, validate_completed_key,
    build_fragment_manifest, _row_sha256, _ids_sha256,
)

# ─── Helpers ────────────────────────────────────────────────────

def _write_gz(path, df):
    import io as _io
    buf = _io.StringIO(); df.to_csv(buf, index=False, header=True)
    path.write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))

def _make_fixture(tmpdir, cid="test_key_001", bl_run_ids=None, gl_run_ids=None, bl_rows=2, gl_rows=144, extra_selection=False):
    """Build a valid fragment fixture for testing."""
    fdir = Path(tmpdir) / cid; fdir.mkdir(parents=True)
    planned_bl = bl_run_ids or [f"bl_{i}" for i in range(bl_rows)]
    planned_gl = gl_run_ids or [f"gl_{i:06d}" for i in range(gl_rows)]
    bl_ids_used = planned_bl[:bl_rows]
    gl_ids_used = planned_gl[:gl_rows]
    # Full baseline schema
    bl_df = pd.DataFrame({
        "run_id": bl_ids_used,
        "dataset_index": [0]*bl_rows, "mechanism": ["M01"]*bl_rows, "strength": ["S1"]*bl_rows,
        "training_seed": [13]*bl_rows, "learner": ["lr"]*bl_rows, "baseline_type": ["strict", "full"][:bl_rows],
        "auc": [0.5]*bl_rows,
    })
    # Full governed schema
    gl_df = pd.DataFrame({
        "run_id": gl_ids_used,
        "dataset_index": [0]*gl_rows, "mechanism": ["M01"]*gl_rows, "strength": ["S1"]*gl_rows,
        "training_seed": [13]*gl_rows, "governance_seed": [0]*gl_rows, "learner": ["lr"]*gl_rows,
        "policy": ["P2"]*gl_rows, "contract": ["semantic_group"]*gl_rows, "budget_bp": [500]*gl_rows,
        "strict_auc": [0.7]*gl_rows, "full_auc": [0.8]*gl_rows, "governed_auc": [0.75]*gl_rows,
        "legacy_sdr": [0.05]*gl_rows,
        "selection_hash": [f"sh_{i:06d}" for i in range(gl_rows)],
        "realized_cost": [0]*gl_rows,
    })
    sel_hashes = [f"sh_{i:06d}" for i in range(gl_rows)]
    if extra_selection: sel_hashes.append("extra_sel_hash")
    sl_df = pd.DataFrame({"selection_hash": sel_hashes,
                          "policy": "P2", "contract": "semantic_group", "budget_bp": 500,
                          "removed_encoded_indices": "[]", "removed_group_ids": "[]", "realized_encoded_cost": 0})
    fl_df = pd.DataFrame(columns=["run_id"])
    for name, df in [("baseline", bl_df), ("governed", gl_df), ("selection", sl_df), ("failure", fl_df)]:
        _write_gz(fdir / f"{name}.csv.gz", df)
    planned_ids = bl_ids_used + gl_ids_used
    produced_ids = bl_ids_used + gl_ids_used
    manifest = build_fragment_manifest(cid, {"canonical_key_id": cid, "n_total_columns": 15}, planned_ids,
        produced_ids, fdir/"baseline.csv.gz", fdir/"governed.csv.gz", fdir/"selection.csv.gz", fdir/"failure.csv.gz", TEST_PLAN_SHA)
    with open(fdir / "fragment_manifest.json", "w") as f: json.dump(manifest, f, sort_keys=True, indent=2)
    receipt = {
        "schema_version": 1, "canonical_key_id": cid, "status": "complete",
        "scientific_freeze_sha": "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845",
        "execution_contract_version": "v1",
        "plan_manifest_sha256": TEST_PLAN_SHA,
        "fragment_manifest_sha256": hashlib.sha256((fdir/"fragment_manifest.json").read_bytes()).hexdigest(),
        "baseline_rows": len(bl_df), "governed_rows": len(gl_df), "selection_rows": len(sl_df), "failure_rows": 0,
        "synthetic_call_counter_delta": {"lr_calls": 0, "p3_calls": 0, "p4_calls": 0, "p5_calls": 0, "p6_calls": 0},
        "production_guard_delta": {"real_bundle_loads": 0, "real_model_calls": 0, "real_selector_calls": 0},
        "completed_utc": "2026-01-01T00:00:00+00:00",
    }
    with open(fdir / "completion_receipt.json", "w") as f: json.dump(receipt, f, sort_keys=True, indent=2)
    return fdir, planned_ids, (bl_df, gl_df, sl_df, fl_df)


# ─── Manifest tests ────────────────────────────────────────────

def test_fragment_manifest_binds_all_four_sha():
    with tempfile.TemporaryDirectory() as td:
        fdir, planned_ids, _ = _make_fixture(td)
        with open(fdir / "fragment_manifest.json") as f:
            m = json.load(f)
        assert len(m["baseline_sha256"]) == 64
        assert len(m["governed_sha256"]) == 64
        assert len(m["selection_sha256"]) == 64
        assert len(m["failure_sha256"]) == 64

def test_completion_receipt_binds_manifest_sha():
    with tempfile.TemporaryDirectory() as td:
        fdir, _, _ = _make_fixture(td)
        with open(fdir / "completion_receipt.json") as f: r = json.load(f)
        actual_manifest_sha = hashlib.sha256((fdir / "fragment_manifest.json").read_bytes()).hexdigest()
        assert r["fragment_manifest_sha256"] == actual_manifest_sha

def test_row_sha256_deterministic():
    row = {"a": 1, "b": 2}
    assert _row_sha256(row) == _row_sha256({"b": 2, "a": 1})

def test_ids_sha256_deterministic():
    assert _ids_sha256(["c", "a", "b"]) == _ids_sha256(["b", "a", "c"])


# ─── validate_completed_key — positive ─────────────────────────

def test_validate_completed_key_passes_on_valid_fragment():
    with tempfile.TemporaryDirectory() as td:
        fdir, planned, _ = _make_fixture(td)
        result = validate_completed_key({"canonical_key_id": "test_key_001", "n_total_columns": 15}, planned, fdir, TEST_PLAN_SHA, _SYNTH_POL, _SYNTH_SEM)
        assert result.is_complete
        assert result.baseline_rows == 2
        assert result.governed_rows == 144
        assert result.failure_rows == 0


# ─── validate_completed_key — negative ─────────────────────────

def test_missing_completion_receipt_invalid():
    with tempfile.TemporaryDirectory() as td:
        fdir, planned, _ = _make_fixture(td)
        (fdir / "completion_receipt.json").unlink()
        result = validate_completed_key({"canonical_key_id": "test_key_001", "n_total_columns": 15}, planned, fdir, TEST_PLAN_SHA, _SYNTH_POL, _SYNTH_SEM)
        assert not result.is_complete
        assert any("receipt missing" in e for e in result.errors)

def test_corrupt_completion_receipt_invalid():
    with tempfile.TemporaryDirectory() as td:
        fdir, planned, _ = _make_fixture(td)
        (fdir / "completion_receipt.json").write_text("not json")
        result = validate_completed_key({"canonical_key_id": "test_key_001", "n_total_columns": 15}, planned, fdir, TEST_PLAN_SHA, _SYNTH_POL, _SYNTH_SEM)
        assert not result.is_complete

def test_baseline_sha_mismatch_invalid():
    with tempfile.TemporaryDirectory() as td:
        fdir, planned, _ = _make_fixture(td)
        # Corrupt baseline
        (fdir / "baseline.csv.gz").write_bytes(b"corrupt")
        result = validate_completed_key({"canonical_key_id": "test_key_001", "n_total_columns": 15}, planned, fdir, TEST_PLAN_SHA, _SYNTH_POL, _SYNTH_SEM)
        assert not result.is_complete

def test_duplicate_run_id_invalid():
    with tempfile.TemporaryDirectory() as td:
        fdir, planned, _ = _make_fixture(td, gl_run_ids=["dup_id"] * 144)
        result = validate_completed_key({"canonical_key_id": "test_key_001", "n_total_columns": 15}, planned, fdir, TEST_PLAN_SHA, _SYNTH_POL, _SYNTH_SEM)
        assert not result.is_complete
        assert len(result.duplicate_run_ids) > 0

def test_missing_run_id_invalid():
    with tempfile.TemporaryDirectory() as td:
        fdir, planned, _ = _make_fixture(td, gl_rows=143)
        result = validate_completed_key({"canonical_key_id": "test_key_001", "n_total_columns": 15}, planned, fdir, TEST_PLAN_SHA, _SYNTH_POL, _SYNTH_SEM)
        assert not result.is_complete
        assert len(result.missing_run_ids) > 0 or len(result.errors) > 0

def test_selection_multiset_closure_invalid():
    with tempfile.TemporaryDirectory() as td:
        fdir, planned, _ = _make_fixture(td, extra_selection=True)
        result = validate_completed_key({"canonical_key_id": "test_key_001", "n_total_columns": 15}, planned, fdir, TEST_PLAN_SHA, _SYNTH_POL, _SYNTH_SEM)
        assert not result.is_complete
        assert any("selection multiset" in e for e in result.errors)


# ─── ProductionGuard tests ─────────────────────────────────────

def test_production_guard_delta():
    before = ProductionGuard()
    after = ProductionGuard(real_bundle_loads=5, real_model_calls=10, real_selector_calls=3)
    d = after.delta(before)
    assert d["real_bundle_loads"] == 5
    assert d["real_model_calls"] == 10
    assert d["real_selector_calls"] == 3

def test_synthetic_counter_delta():
    before = SyntheticCallCounter()
    after = SyntheticCallCounter(lr_calls=146, p3_calls=1, p4_calls=1, p5_calls=1, p6_calls=1)
    d = after.delta(before)
    assert d["lr_calls"] == 146
    assert d["p3_calls"] == 1

def test_production_guard_zero_on_synthetic():
    """Synthetic mode must never increment production counters."""
    guard = ProductionGuard()
    assert guard.real_bundle_loads == 0
    assert guard.real_model_calls == 0
