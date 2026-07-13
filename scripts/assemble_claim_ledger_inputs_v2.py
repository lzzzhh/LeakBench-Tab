#!/usr/bin/env python3
"""assemble_claim_ledger_inputs_v2.py — Gate SP5-F.

Assemble the complete 5-model x 11-mechanism formal evidence pool from:
  - core_cpu_cells (base evidence, 4 CPU models, all 11 mechanisms)
  - base-7 TabM confirmatory (TabM, 7 non-amended mechanisms)
with exact replacements applied by primary key
  [dataset_index, mechanism, strength, model, seed]:
  - M04/M05/M08 -> SP4 frozen (all 5 models)
  - M10        -> M10 amendment (CPU 4 models + TabM)

Replacement precedence (highest first):
  1. M10 amendment
  2. SP4 M04/M05/M08
  3. base evidence (core_cpu + base7 TabM)

Never mutates source files. Emits canonical inputs + replacement audit.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/sp5"
KEY = ["dataset_index", "mechanism", "strength", "model", "seed"]
MODELS = ["lr", "rf", "lightgbm", "catboost", "tabm"]
MECHS = [f"M{i:02d}" for i in range(1, 12)]
REPLACED = {"M04", "M05", "M08", "M10"}

SRC = {
    "core_cpu": "results/corrected_v2/core_cpu_cells.csv",
    "base7_tabm": "results/corrected_v2/tabm_confirmatory_base7_v2/formal/tabm_base7_cells.csv",
    "sp4": "results/structured_prior_replacement_v1/model_cells.csv",
    "m10_cpu": "results/corrected_v2/m10_amendment_confirmatory/cpu_cells.csv",
    "m10_tabm": "results/corrected_v2/m10_amendment_confirmatory_tabm/formal/m10_tabm_cells.csv",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _norm(df, source, exploit_from):
    """Normalize a source to the canonical column set. exploit_from is the
    column that holds the strict/clean baseline AUC (== mask-strict view)."""
    df = df.copy()
    df = df[df["status"] == "SUCCESS"] if "status" in df else df
    df = df.drop_duplicates("run_id", keep="last") if "run_id" in df else df
    out = pd.DataFrame()
    for k in KEY:
        out[k] = df[k]
    out["strict_auc"] = df[exploit_from].astype(float)
    out["full_auc"] = df["full_auc"].astype(float)
    out["paired_harm"] = df["full_auc"].astype(float) - df[exploit_from].astype(float)
    out["source"] = source
    out["run_id"] = df["run_id"] if "run_id" in df else ""
    out["config_hash"] = df["config_hash"] if "config_hash" in df else ""
    out["code_hash"] = df["code_hash"] if "code_hash" in df else ""
    return out


def main():
    frames = {}
    frames["core_cpu"] = _norm(pd.read_csv(ROOT / SRC["core_cpu"]), "core_cpu", "clean_auc")
    frames["base7_tabm"] = _norm(pd.read_csv(ROOT / SRC["base7_tabm"]), "base7_tabm", "clean_auc")
    frames["sp4"] = _norm(pd.read_csv(ROOT / SRC["sp4"]), "sp4_frozen", "strict_auc")
    frames["m10_cpu"] = _norm(pd.read_csv(ROOT / SRC["m10_cpu"]), "m10_amendment", "strict_auc")
    frames["m10_tabm"] = _norm(pd.read_csv(ROOT / SRC["m10_tabm"]), "m10_amendment", "strict_auc")

    # base = core_cpu (drop mechanisms that are exact-replaced) + base7 tabm
    base = frames["core_cpu"][~frames["core_cpu"]["mechanism"].isin(REPLACED)]
    base = pd.concat([base, frames["base7_tabm"]], ignore_index=True)

    # replacements
    sp4 = frames["sp4"]  # M04/M05/M08 all models
    m10 = pd.concat([frames["m10_cpu"], frames["m10_tabm"]], ignore_index=True)  # M10 all models

    assert not (set(base["mechanism"]) & REPLACED), "base contains replaced mechanisms"
    assert set(sp4["mechanism"]) == {"M04", "M05", "M08"}, "sp4 mechanism scope wrong"
    assert set(m10["mechanism"]) == {"M10"}, "m10 mechanism scope wrong"

    ledger = pd.concat([base, sp4, m10], ignore_index=True)

    # integrity
    dup = int(ledger.duplicated(KEY).sum())
    audit = {
        "gate": "SP5-F_evidence_pool",
        "total_cells": len(ledger),
        "expected": len(MODELS) * len(MECHS) * 20 * 5 * 5,
        "models": sorted(ledger["model"].unique().tolist()),
        "mechanisms": sorted(ledger["mechanism"].unique().tolist()),
        "duplicate_primary_keys": dup,
        "by_source": {k: int(v) for k, v in ledger["source"].value_counts().items()},
        "replacement": {
            "M04_M05_M08": int((ledger["source"] == "sp4_frozen").sum()),
            "M10": int((ledger["source"] == "m10_amendment").sum()),
            "base_core_cpu": int((ledger["source"] == "core_cpu").sum()),
            "base7_tabm": int((ledger["source"] == "base7_tabm").sum()),
        },
    }
    # coverage matrix
    cov = ledger.groupby(["mechanism", "model"]).size().unstack(fill_value=0)
    missing = []
    for m in MECHS:
        for mdl in MODELS:
            n = int(cov.loc[m, mdl]) if (m in cov.index and mdl in cov.columns) else 0
            if n != 500:
                missing.append((m, mdl, n))
    audit["cells_not_500"] = missing
    audit["complete"] = (len(ledger) == 27500 and dup == 0 and not missing)

    OUT.mkdir(parents=True, exist_ok=True)
    ledger = ledger.sort_values(KEY).reset_index(drop=True)
    ledger.to_csv(OUT / "claim_ledger_inputs_v2.csv", index=False)
    ledger.to_parquet(OUT / "claim_ledger_inputs_v2.parquet", index=False)
    audit["csv_sha256"] = sha(OUT / "claim_ledger_inputs_v2.csv")
    audit["parquet_sha256"] = sha(OUT / "claim_ledger_inputs_v2.parquet")
    audit["source_hashes"] = {k: sha(ROOT / v) for k, v in SRC.items()}
    (OUT / "claim_ledger_inputs_v2_manifest.json").write_text(json.dumps(audit, indent=2))

    # replacement audit table
    ra = ledger.groupby(["mechanism", "source"]).size().reset_index(name="cells")
    ra.to_csv(OUT / "replacement_audit.csv", index=False)

    print(json.dumps({k: audit[k] for k in
          ["total_cells", "expected", "duplicate_primary_keys", "complete",
           "by_source", "cells_not_500"]}, indent=2))
    print("coverage matrix (mechanism x model):")
    print(cov.to_string())
    return 0 if audit["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
