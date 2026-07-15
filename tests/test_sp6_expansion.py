"""SP6 model-expansion tests — ModernNCA formal integrity + SP5 immutability."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np, pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SP6 = ROOT / "artifacts/sp6"
KEY = ["dataset_index", "mechanism", "strength", "model", "seed"]


@pytest.fixture(scope="module")
def mnca():
    return pd.read_csv(SP6 / "modernnca/formal/model_cells.csv").drop_duplicates("run_id", keep="last")


def test_modernnca_5500_success(mnca):
    assert len(mnca) == 5500
    assert int((mnca["status"] == "SUCCESS").sum()) == 5500


def test_modernnca_no_dup_missing_nonfinite(mnca):
    assert int(mnca.duplicated(KEY).sum()) == 0
    assert np.isfinite(mnca["paired_harm"]).all()
    cov = mnca.groupby("mechanism").size()
    assert (cov == 500).all() and len(cov) == 11


def test_modernnca_cuda_only(mnca):
    assert set(mnca["device"].unique()) == {"cuda"}


def test_modernnca_train_only_candidates(mnca):
    # candidate row-id hash identical across strict/full (both train rows only)
    assert (mnca["strict_candidate_row_ids_hash"] == mnca["full_candidate_row_ids_hash"]).all()


def test_modernnca_strict_full_isolated(mnca):
    assert (mnca["strict_preprocessor_hash"] != mnca["full_preprocessor_hash"]).all()


def test_modernnca_upstream_hash_byte_identical():
    h = hashlib.sha256((ROOT / "third_party/modernnca/modernNCA.py").read_bytes()).hexdigest()
    assert h == "02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f"


def test_sp5_core_unchanged_in_extended():
    ext = pd.read_csv(SP6 / "claim_ledger_v3_extended.csv")
    sp5 = ext[ext["evidence_tier"] == "SP5_FROZEN_CORE"]
    assert len(sp5) == 27500
    orig = pd.read_csv(ROOT / "artifacts/sp5/claim_ledger_v2.csv")
    # paired_harm values for SP5 rows must match original exactly
    m = sp5.merge(orig[KEY + ["paired_harm"]], on=KEY, suffixes=("_ext", "_orig"))
    assert len(m) == 27500
    assert np.allclose(m["paired_harm_ext"], m["paired_harm_orig"])


def test_extended_ledger_38500():
    ext = pd.read_csv(SP6 / "claim_ledger_v3_extended.csv")
    assert len(ext) == 38500
    assert set(ext["evidence_tier"].unique()) == {"SP5_FROZEN_CORE", "SP6_MODEL_EXPANSION"}


def test_evidence_tier_separation():
    ext = pd.read_csv(SP6 / "claim_ledger_v3_extended.csv")
    assert int((ext["evidence_tier"] == "SP6_MODEL_EXPANSION").sum()) == 11000
    assert set(ext[ext["evidence_tier"] == "SP6_MODEL_EXPANSION"]["model"].unique()) == {"modernnca", "tabr"}


def test_legacy_modern_models_excluded():
    import json
    audit = json.loads((SP6 / "archive/legacy_model_implementations_audit.json").read_text())
    for entry in audit:
        assert "SUPERSEDED_FOR_SP6" in entry["status"]
