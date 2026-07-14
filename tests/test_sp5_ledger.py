"""SP5-G validation tests — ledger integrity, coverage, exclusions, reproducibility."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SP5 = ROOT / "artifacts/sp5"
KEY = ["dataset_index", "mechanism", "strength", "model", "seed"]
MODELS = ["lr", "rf", "lightgbm", "catboost", "tabm"]
MECHS = [f"M{i:02d}" for i in range(1, 12)]


@pytest.fixture(scope="module")
def led():
    return pd.read_csv(SP5 / "claim_ledger_v2.csv")


def test_ledger_27500_coverage(led):
    assert len(led) == 27500


def test_primary_key_uniqueness(led):
    assert int(led.duplicated(KEY).sum()) == 0


def test_five_model_eleven_mechanism(led):
    assert sorted(led["model"].unique()) == sorted(MODELS)
    assert sorted(led["mechanism"].unique()) == sorted(MECHS)
    cov = led.groupby(["mechanism", "model"]).size()
    assert (cov == 500).all()


def test_no_non_finite(led):
    for c in ["strict_auc", "full_auc", "paired_harm", "detectability_value"]:
        assert np.isfinite(led[c]).all()


def test_no_modern_models(led):
    forbidden = {"modernnca", "tabr", "tabicl", "tabpfnv2", "tabpfn"}
    assert not (set(led["model"].str.lower()) & forbidden)


def test_csv_parquet_equality(led):
    b = pd.read_parquet(SP5 / "claim_ledger_v2.parquet")
    assert len(led) == len(b) and list(led.columns) == list(b.columns)
    assert led[KEY].reset_index(drop=True).equals(b[KEY].reset_index(drop=True))


def test_detectability_provenance_m08_corrected(led):
    """M08 detectability must be the corrected SP4 value (~0.43), not old ~0.048."""
    m08 = led[led["mechanism"] == "M08"]["detectability_value"].mean()
    assert 0.35 < m08 < 0.55, f"M08 detectability {m08} not corrected"


def test_category_taxonomy(led):
    cat = led.drop_duplicates("mechanism").set_index("mechanism")["mechanism_category"]
    assert set(cat[cat == "simple"].index) == {"M01", "M02", "M06", "M10"}
    assert set(cat[cat == "structured"].index) == {"M04", "M05", "M08", "M09"}
    assert set(cat[cat == "boundary"].index) == {"M03", "M07", "M11"}


def test_no_code_drift_or_interim_in_ledger(led):
    # code-drift checkpoint had 5228 rows; interim M08 had synthetic. Ledger is exactly 27500.
    assert len(led) == 27500
    # source column must only contain known formal sources
    inp = pd.read_csv(SP5 / "claim_ledger_inputs_v2.csv")
    assert set(inp["source"].unique()) <= {"core_cpu", "base7_tabm", "sp4_frozen", "m10_amendment"}


def test_cl4_paired_keys_align(led):
    piv = led.pivot_table(index=["dataset_index", "mechanism", "strength", "seed"],
                          columns="model", values="paired_harm")
    assert piv.shape[0] == 5500  # 11 mech x 5 str x 5 seed x 20 ds
    assert piv.notna().all().all()  # every model present for every task


def test_cl10_55_profiles(led):
    prof = led.groupby(["mechanism", "model"]).size()
    assert len(prof) == 55


def test_claim_matrix_exists():
    m = pd.read_csv(SP5 / "claim_evidence_matrix_v2.csv")
    assert set(["CL2", "CL3", "CL4", "CL10"]).issubset(set(m["claim_id"]))


def test_no_track_b_formal_results():
    # Track B inventory must exist but no governance cell result files under sp5
    assert (SP5 / "track_b_inventory.csv").exists()
    assert not list(SP5.glob("**/governance_cells*.csv"))


def test_bootstrap_reproducible(led):
    """Re-running cluster bootstrap with fixed seed gives identical CI."""
    def boot(df, col, seed=20260714, n=500):
        rng = np.random.RandomState(seed)
        units = df["dataset_index"].unique()
        g = {u: df[df["dataset_index"] == u][col].values for u in units}
        means = [np.concatenate([g[u] for u in rng.choice(units, len(units), True)]).mean()
                 for _ in range(n)]
        return np.percentile(means, [2.5, 97.5])
    sub = led[led["model"] == "lr"]
    a = boot(sub, "paired_harm")
    b = boot(sub, "paired_harm")
    assert np.allclose(a, b)
