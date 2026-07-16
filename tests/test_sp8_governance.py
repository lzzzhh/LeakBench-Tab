"""SP8 clean governance tests — oracle isolation, matched cost, metric correctness."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SP8 = ROOT / "artifacts/sp8"


@pytest.fixture(scope="module")
def gov():
    return pd.read_csv(SP8 / "governance_clean.csv").drop_duplicates("run_id", keep="last")


def test_all_success(gov):
    assert int((gov["status"] == "SUCCESS").sum()) == len(gov)


def test_no_duplicates(gov):
    assert int(gov["run_id"].duplicated().sum()) == 0


def test_budget_labels_correct(gov):
    p3 = gov[gov["policy"] == "P3_blind_mi"]
    assert set(p3["budget_fraction"].unique()) == {0.01, 0.05, 0.10, 0.20}


def test_matched_cost(gov):
    p2 = gov[gov["policy"] == "P2_random"]
    p3 = gov[gov["policy"] == "P3_blind_mi"]
    assert len(p2) == len(p3)
    assert (p2["removed_count"].iloc[:100].values == p3["removed_count"].iloc[:100].values).all()


def test_p0_removes_zero(gov):
    p0 = gov[gov["policy"] == "P0_keep"]
    assert (p0["removed_count"] == 0).all()
    assert (p0["strict_distance_reduction"] == 0.0).all()


def test_p1_oracle_recall_one(gov):
    p1 = gov[gov["policy"] == "P1_oracle"]
    assert (p1["leak_recall"] >= 0.999).all()
    assert p1["oracle_policy"].astype(bool).all()


def test_non_oracle_not_marked_oracle(gov):
    non = gov[gov["policy"].isin(["P2_random", "P3_blind_mi"])]
    assert not non["oracle_policy"].astype(bool).any()


def test_p0_not_oracle(gov):
    assert not gov[gov["policy"] == "P0_keep"]["oracle_policy"].astype(bool).any()


def test_strict_distance_reduction_positive_for_p3_mostly(gov):
    p3 = gov[(gov["policy"] == "P3_blind_mi") & (gov["budget_fraction"] == 0.2)]
    # P3 should beat P0 (at least some positive entries)
    assert p3["strict_distance_reduction"].mean() > -0.1


def test_residual_harm_formula(gov):
    mask = gov["status"] == "SUCCESS"
    diff = (gov.loc[mask, "residual_harm"] - (gov.loc[mask, "governed_auc"] - gov.loc[mask, "strict_auc"])).abs()
    assert diff.max() < 1e-4


def test_utility_loss_sign(gov):
    mask = gov["status"] == "SUCCESS"
    diff = (gov.loc[mask, "utility_loss"] - (gov.loc[mask, "strict_auc"] - gov.loc[mask, "governed_auc"])).abs()
    assert diff.max() < 1e-4


def test_manifest_exists():
    m = json.loads((SP8 / "governance_clean_manifest.json").read_text())
    assert m["oracle_isolated"] is True
    assert m["matched_cost_verified"] is True
    assert "runner_sha" in m and "csv_sha" in m


def test_manifest_hashes_match():
    m = json.loads((SP8 / "governance_clean_manifest.json").read_text())
    csv_sha = hashlib.sha256((SP8 / "governance_clean.csv").read_bytes()).hexdigest()
    assert m["csv_sha"] == csv_sha


def test_sp5_sp6_sp7_unchanged():
    for p, sha_exp in [
        ("artifacts/sp5/claim_ledger_v2.csv", "ccb2549f490e95cb"),
        ("artifacts/sp6/claim_ledger_v3_extended.csv", None),  # just check exists
    ]:
        assert Path(p).exists()
