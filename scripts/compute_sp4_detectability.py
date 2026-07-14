#!/usr/bin/env python3
"""compute_sp4_detectability.py — corrected M04/M05/M08 detectability.

SP4 bundles were exported with diagnostic_ap NOT_COMPUTED_PRE_RUN. The core
detectability metric (mutual_info_classif AUPRC vs leakage_mask on train split)
is deterministic and model-independent. Reproduce it EXACTLY on the frozen SP4
bundles so corrected detectability matches corrected exploitability provenance.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "results/structured_prior_replacement_v1/task_bundles"
OUT = ROOT / "artifacts/sp5/sp4_detectability.csv"


def diagnostic_ap(X, y, train_idx, leakage_mask):
    """Exact replica of run_corrected_core.diagnostic_metrics AP."""
    scores = mutual_info_classif(X[train_idx], y[train_idx], random_state=42)
    scores = np.nan_to_num(scores, nan=0.0)
    truth = leakage_mask.astype(int)
    return float(average_precision_score(truth, scores))


def main():
    manifest = pd.read_csv(BUNDLE_DIR / "task_manifest.csv")
    rows = []
    for dsi in sorted(manifest["dataset_index"].unique()):
        sub = manifest[manifest["dataset_index"] == dsi]
        bundle_path = ROOT / sub["bundle_path"].iloc[0]
        with np.load(bundle_path, allow_pickle=False) as b:
            base_X = np.asarray(b["base_X"]); y = np.asarray(b["y"])
            train_idx = np.asarray(b["train_idx"])
            for _, r in sub.iterrows():
                key = str(r["bundle_key"])
                block = np.asarray(b[f"block__{key}"])
                mask = np.asarray(b[f"leak_mask__{key}"])
                X = np.concatenate((base_X, block), axis=1)
                ap = diagnostic_ap(X, y, train_idx, mask)
                rows.append({"dataset_index": int(dsi), "mechanism": r["mechanism"],
                             "strength": r["strength"], "seed": int(r["seed"]),
                             "detectability_value": ap})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    h = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {len(df)} detectability rows -> {OUT.name} sha256 {h[:16]}")
    print("per-mechanism corrected detectability (SP4):")
    print(df.groupby("mechanism")["detectability_value"].agg(["mean", "min", "max"]).round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
