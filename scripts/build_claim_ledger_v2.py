#!/usr/bin/env python3
"""build_claim_ledger_v2.py — SP5-G ledger construction.

Enrich the 27500-cell evidence pool with mechanism categories, three-axis
values, detectability (diagnostic_ap, model-independent), and claim-inclusion
masks. Emits claim_ledger_v2.{csv,parquet} + schema + manifest.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
SP5 = ROOT / "artifacts/sp5"
KEY = ["dataset_index", "mechanism", "strength", "model", "seed"]

CATEGORY = {
    "M01": "simple", "M02": "simple", "M06": "simple", "M10": "simple",
    "M04": "structured", "M05": "structured", "M08": "structured", "M09": "structured",
    "M03": "boundary", "M07": "boundary", "M11": "boundary",
}
MODEL_FAMILY = {"lr": "linear", "rf": "bagging_tree", "lightgbm": "boosting_tree",
                "catboost": "boosting_tree", "tabm": "neural"}
STRENGTH_NUM = {"S1": 0.2, "S2": 0.4, "S3": 0.6, "S4": 0.8, "S5": 1.0}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _detectability_map():
    """diagnostic_ap is a model-independent task property (mutual_info AUPRC vs
    leakage_mask on train). Provenance MUST match exploitability:
      - M04/M05/M08 -> corrected SP4 bundles (computed by compute_sp4_detectability.py;
        SP4 model_cells had diagnostic_ap NOT_COMPUTED_PRE_RUN)
      - M10 -> m10 amendment cells
      - all other base mechs -> core_cpu
    core_cpu M04/M05/M08 detectability is the OLD pre-correction mechanism and is
    explicitly excluded to avoid mismatched provenance."""
    k = ["dataset_index", "mechanism", "strength", "seed"]
    REPLACED = {"M04", "M05", "M08"}
    # base (core_cpu) EXCEPT the replaced structured mechs
    core = pd.read_csv(ROOT / "results/corrected_v2/core_cpu_cells.csv")
    core = core[core["status"] == "SUCCESS"] if "status" in core else core
    core = core[~core["mechanism"].isin(REPLACED | {"M10"})]
    base = core[k + ["diagnostic_ap"]].drop_duplicates(k)
    # M10 from amendment
    m10 = pd.read_csv(ROOT / "results/corrected_v2/m10_amendment_confirmatory/cpu_cells.csv")
    m10 = m10[m10["status"] == "SUCCESS"] if "status" in m10 else m10
    m10 = m10[k + ["diagnostic_ap"]].drop_duplicates(k)
    # M04/M05/M08 corrected detectability from frozen SP4 bundles
    sp4 = pd.read_csv(ROOT / "artifacts/sp5/sp4_detectability.csv")
    sp4 = sp4.rename(columns={"detectability_value": "diagnostic_ap"})[k + ["diagnostic_ap"]]
    det = pd.concat([base, m10, sp4], ignore_index=True).drop_duplicates(k, keep="first")
    return det.rename(columns={"diagnostic_ap": "detectability_value"})


def main():
    inp = pd.read_csv(SP5 / "claim_ledger_inputs_v2.csv")
    assert len(inp) == 27500, f"expected 27500 got {len(inp)}"

    det = _detectability_map()
    n_before = len(inp)
    led = inp.merge(det, on=["dataset_index", "mechanism", "strength", "seed"], how="left")
    assert len(led) == n_before, "detectability join changed row count"
    miss = int(led["detectability_value"].isna().sum())
    assert miss == 0, f"{miss} cells missing detectability"

    led["claim_id"] = ""
    led["mechanism_category"] = led["mechanism"].map(CATEGORY)
    led["model_family"] = led["model"].map(MODEL_FAMILY)
    led["strength_numeric"] = led["strength"].map(STRENGTH_NUM)
    led["is_simple"] = led["mechanism_category"] == "simple"
    led["is_structured"] = led["mechanism_category"] == "structured"
    led["is_boundary"] = led["mechanism_category"] == "boundary"
    # three axes
    led["axis_construction"] = 1.0  # all mechanisms are invalid contamination
    led["axis_detectability"] = led["detectability_value"]
    led["axis_exploitability"] = led["paired_harm"]
    led["detectability_source"] = "diagnostic_ap (task-level AUPRC, model-independent)"
    led["exploitability_metric"] = "paired_harm = full_auc - strict_auc"
    # inclusion masks
    led["included_cl2"] = led["is_structured"] | led["is_simple"]  # CL2 compares simple vs structured
    led["included_cl3"] = True
    led["included_cl4"] = True
    led["included_cl10"] = True

    # integrity
    assert int(led.duplicated(KEY).sum()) == 0, "duplicate primary keys"
    for c in ["strict_auc", "full_auc", "paired_harm", "detectability_value"]:
        assert np.isfinite(led[c]).all(), f"non-finite in {c}"

    led = led.sort_values(KEY).reset_index(drop=True)
    led.to_csv(SP5 / "claim_ledger_v2.csv", index=False)
    led.to_parquet(SP5 / "claim_ledger_v2.parquet", index=False)

    # csv/parquet equality
    a = pd.read_csv(SP5 / "claim_ledger_v2.csv")
    b = pd.read_parquet(SP5 / "claim_ledger_v2.parquet")
    equal = (len(a) == len(b) and list(a.columns) == list(b.columns)
             and a[KEY].equals(b[KEY]))

    schema = {c: str(led[c].dtype) for c in led.columns}
    (SP5 / "claim_ledger_v2_schema.json").write_text(json.dumps(schema, indent=2))
    manifest = {
        "rows": len(led), "columns": len(led.columns),
        "unique_primary_keys": int(led.drop_duplicates(KEY).shape[0]),
        "duplicate_keys": int(led.duplicated(KEY).sum()),
        "non_finite": 0,
        "models": sorted(led["model"].unique().tolist()),
        "mechanisms": sorted(led["mechanism"].unique().tolist()),
        "categories": {k: sorted(led[led["mechanism_category"] == k]["mechanism"].unique().tolist())
                       for k in ["simple", "structured", "boundary"]},
        "detectability_missing": miss,
        "csv_sha256": sha(SP5 / "claim_ledger_v2.csv"),
        "parquet_sha256": sha(SP5 / "claim_ledger_v2.parquet"),
        "csv_parquet_equal": bool(equal),
        "inputs_sha256": sha(SP5 / "claim_ledger_inputs_v2.csv"),
    }
    (SP5 / "claim_ledger_v2_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in
          ["rows", "duplicate_keys", "non_finite", "detectability_missing",
           "csv_parquet_equal", "categories"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
