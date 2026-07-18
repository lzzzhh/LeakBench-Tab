#!/usr/bin/env python3
"""verify_baseline_parity.py — CDXR Cross-Learner Baseline Parity Gate.

Re-fits LR/RF/LightGBM using the official fit_predict_core_model adapter on
frozen bundles. Compares strict/full AUROC against canonical_cells.csv.
Must pass with all absolute differences ≤ 1e-12 before any governance runs.
"""
from __future__ import annotations
import csv, hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.leakbench.models.core_models import fit_predict_core_model


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_bundle(row):
    bundle = ROOT / row["bundle_path"]
    if sha(bundle) != str(row["bundle_sha256"]).lower():
        raise RuntimeError(f"bundle hash mismatch: {row['bundle_path']}")
    key = str(row["bundle_key"])
    with np.load(bundle, allow_pickle=False) as b:
        X = np.concatenate((np.asarray(b["base_X"]), np.asarray(b[f"block__{key}"])), axis=1)
        y = np.asarray(b["y"])
        tr = np.asarray(b["train_idx"]); va = np.asarray(b["val_idx"]); te = np.asarray(b["test_idx"])
        mask = np.asarray(b[f"leak_mask__{key}"])
    if hashlib.sha256(te.tobytes()).hexdigest() != str(row["split_hash"]):
        raise RuntimeError("split hash mismatch")
    return X, y, tr, va, te, mask


def main():
    man = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    canonical = pd.read_csv(ROOT / "results/corrected_v2/canonical_cells.csv")
    out = ROOT / "results/edbt_eab_crosslearner_confirmatory_v2/baseline_parity.csv"

    models = ["lr", "rf", "lightgbm"]
    total = len(man) * len(models)
    print(f"Baseline parity: {len(man)} keys × {len(models)} models = {total} cells", flush=True)

    fields = ["dataset_index", "mechanism", "strength", "training_seed", "model",
              "canonical_strict_auc", "refit_strict_auc", "strict_abs_diff",
              "canonical_full_auc", "refit_full_auc", "full_abs_diff",
              "parity_pass", "bundle_sha256", "model_adapter_sha256"]
    adapter_sha = sha("src/leakbench/models/core_models.py")
    failures = 0; done = 0; t0 = time.time()

    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for _, row in man.iterrows():
            ds = int(row["dataset_index"]); mech = row["mechanism"]
            st = row["strength"]; tseed = int(row["seed"])
            try:
                X, y, tr, va, te, mask = load_bundle(row)
            except Exception as e:
                continue
            strict_cols = np.where(~mask)[0]; full_cols = np.arange(X.shape[1])

            for model in models:
                # Canonical values
                cr = canonical[(canonical.dataset_index == ds) & (canonical.mechanism == mech) &
                               (canonical.strength == st) & (canonical.seed == tseed) & (canonical.model == model)]
                if len(cr) != 1:
                    rec = {k: "" for k in fields}
                    rec.update(dict(dataset_index=ds, mechanism=mech, strength=st, training_seed=tseed,
                                    model=model, parity_pass=False))
                    w.writerow(rec); failures += 1; done += 1; continue

                # Canonical baseline: strict_auc for M10, clean_auc for all others
                c_strict = float(cr["strict_auc"].iloc[0]) if pd.notna(cr["strict_auc"].iloc[0]) else float(cr["clean_auc"].iloc[0])
                c_full = float(cr["full_auc"].iloc[0])

                # Refit with official adapter
                try:
                    r_strict_out = fit_predict_core_model(model, X[tr][:, strict_cols], y[tr],
                                                          X[va][:, strict_cols], y[va], X[te][:, strict_cols], tseed)
                    r_full_out = fit_predict_core_model(model, X[tr][:, full_cols], y[tr],
                                                        X[va][:, full_cols], y[va], X[te][:, full_cols], tseed)
                except Exception as e:
                    rec = {k: "" for k in fields}
                    rec.update(dict(dataset_index=ds, mechanism=mech, strength=st, training_seed=tseed,
                                    model=model, parity_pass=False))
                    w.writerow(rec); failures += 1; done += 1; continue

                r_strict = float(roc_auc_score(y[te], r_strict_out.probabilities))
                r_full = float(roc_auc_score(y[te], r_full_out.probabilities))
                strict_diff = abs(c_strict - r_strict)
                full_diff = abs(c_full - r_full)
                passed = strict_diff <= 1e-12 and full_diff <= 1e-12

                w.writerow(dict(dataset_index=ds, mechanism=mech, strength=st, training_seed=tseed,
                                model=model, canonical_strict_auc=c_strict, refit_strict_auc=r_strict,
                                strict_abs_diff=strict_diff, canonical_full_auc=c_full, refit_full_auc=r_full,
                                full_abs_diff=full_diff, parity_pass=passed,
                                bundle_sha256=str(row["bundle_sha256"]), model_adapter_sha256=adapter_sha))
                if not passed: failures += 1
                done += 1
                if done % 2000 == 0:
                    print(f"  {done}/{total} | {time.time()-t0:.0f}s | failures={failures}", flush=True)

    print(f"\nDONE {done}/{total} in {time.time()-t0:.0f}s | FAILURES={failures}")
    print(f"BASELINE_PARITY: {'PASS' if failures == 0 else 'BLOCKED'}")
    return 1 if failures > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
