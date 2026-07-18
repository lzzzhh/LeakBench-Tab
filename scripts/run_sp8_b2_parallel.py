#!/usr/bin/env python3
"""run_sp8_b2_parallel.py — B2 RF + LightGBM, parallel over keys via multiprocessing.

Each worker processes a subset of keys. Output CSV per worker merged at end.
20% budget, 20 gov seeds. Same strict/full from frozen bundles.
"""
from __future__ import annotations
import argparse, csv, hashlib, os, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

GOV_SEEDS = [2026071700 + i for i in range(20)]
BUDGET = 0.20
FIELDS = [
    "run_id","dataset_index","mechanism","strength","training_seed","governance_seed",
    "model","policy","budget_k","budget_fraction",
    "status","failure_reason",
    "strict_auc","full_auc","governed_auc",
    "strict_distance_reduction","initial_gap",
    "removed_count",
]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_cell(row):
    bundle = ROOT / row["bundle_path"]
    if sha_file(bundle) != str(row["bundle_sha256"]).lower():
        raise RuntimeError("bundle hash mismatch")
    key = str(row["bundle_key"])
    with np.load(bundle, allow_pickle=False) as b:
        X = np.concatenate((np.asarray(b["base_X"]), np.asarray(b[f"block__{key}"])), axis=1)
        y = np.asarray(b["y"])
        tr = np.asarray(b["train_idx"]); te = np.asarray(b["test_idx"])
        mask = np.asarray(b[f"leak_mask__{key}"])
    return X, y, tr, te, mask


def fit_model(model, Xtr, ytr, Xte, yte, seed):
    if model == "rf":
        m = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=1).fit(Xtr, ytr)
    elif model == "lightgbm":
        from lightgbm import LGBMClassifier
        m = LGBMClassifier(n_estimators=100, max_depth=10, random_state=seed, verbose=-1, device="gpu").fit(Xtr, ytr)
    else:
        raise ValueError(model)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def select_fields_mi(scores, k):
    return np.argsort(scores)[::-1][:k]


def select_fields_random(n, k, gs, ds_i, ts):
    rng = np.random.RandomState((gs * 100 + ds_i * 7 + ts * 13) % (2**31 - 1))
    return rng.choice(n, k, replace=False)


def process_key(args):
    """Process a single bundle-manifest row for all models. Returns list of result dicts."""
    row, models = args
    ds_i = int(row["dataset_index"]); mech = row["mechanism"]
    strength = row["strength"]; tseed = int(row["seed"])
    try:
        X, y, tr, te, mask = load_cell(row)
    except Exception as e:
        return [{"status": "FAILURE", "failure_reason": str(e)[:200], "dataset_index": ds_i}]

    n_features = X.shape[1]; strict_cols = np.where(~mask)[0]
    k = max(1, round(n_features * BUDGET))
    if k >= n_features:
        return []

    results = []
    for model in models:
        try:
            strict_auc = fit_model(model, X[tr][:, strict_cols], y[tr], X[te][:, strict_cols], y[te], tseed)
            full_auc = fit_model(model, X[tr], y[tr], X[te], y[te], tseed)
            gap = abs(full_auc - strict_auc)

            # P0
            rid0 = hashlib.sha256(f"b2p0|{ds_i}|{mech}|{strength}|{tseed}|{model}".encode()).hexdigest()[:20]
            results.append(dict(run_id=rid0, dataset_index=ds_i, mechanism=mech, strength=strength,
                training_seed=tseed, governance_seed=-1, model=model, policy="P0_keep", budget_k=0, budget_fraction=0.0,
                status="SUCCESS", strict_auc=round(strict_auc, 6), full_auc=round(full_auc, 6),
                governed_auc=round(strict_auc, 6), strict_distance_reduction=0.0, initial_gap=round(gap, 6),
                removed_count=0))

            # P3
            mi_scores = mutual_info_classif(X[tr], y[tr], random_state=42)
            mi_scores = np.nan_to_num(mi_scores, nan=0.0)
            mi_fields = select_fields_mi(mi_scores, k)
            keep_p3 = np.ones(n_features, dtype=bool); keep_p3[mi_fields] = False
            gov_auc = fit_model(model, X[tr][:, keep_p3], y[tr], X[te][:, keep_p3], y[te], tseed)
            sdr = abs(full_auc - strict_auc) - abs(gov_auc - strict_auc)
            rid3 = hashlib.sha256(f"b2p3|{ds_i}|{mech}|{strength}|{tseed}|{model}|{k}".encode()).hexdigest()[:20]
            results.append(dict(run_id=rid3, dataset_index=ds_i, mechanism=mech, strength=strength,
                training_seed=tseed, governance_seed=-1, model=model, policy="P3_blind_mi", budget_k=k, budget_fraction=BUDGET,
                status="SUCCESS", strict_auc=round(strict_auc, 6), full_auc=round(full_auc, 6),
                governed_auc=round(gov_auc, 6), strict_distance_reduction=round(sdr, 6),
                initial_gap=round(gap, 6), removed_count=k))

            # P2 multi-seed
            for gs in GOV_SEEDS:
                rm_fields = select_fields_random(n_features, k, gs, ds_i, tseed)
                keep_p2 = np.ones(n_features, dtype=bool); keep_p2[rm_fields] = False
                gov_auc = fit_model(model, X[tr][:, keep_p2], y[tr], X[te][:, keep_p2], y[te], tseed)
                sdr = abs(full_auc - strict_auc) - abs(gov_auc - strict_auc)
                rid2 = hashlib.sha256(f"b2p2|{ds_i}|{mech}|{strength}|{tseed}|{model}|{gs}".encode()).hexdigest()[:20]
                results.append(dict(run_id=rid2, dataset_index=ds_i, mechanism=mech, strength=strength,
                    training_seed=tseed, governance_seed=gs, model=model, policy="P2_random", budget_k=k, budget_fraction=BUDGET,
                    status="SUCCESS", strict_auc=round(strict_auc, 6), full_auc=round(full_auc, 6),
                    governed_auc=round(gov_auc, 6), strict_distance_reduction=round(sdr, 6),
                    initial_gap=round(gap, 6), removed_count=k))
        except Exception as e:
            results.append({"run_id": f"err_{ds_i}_{mech}_{model}", "status": "FAILURE",
                "failure_reason": str(e)[:200], "dataset_index": ds_i})
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--models", default="rf,lightgbm")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("locked; pass --allow-run")

    man = pd.read_csv(ROOT / args.bundle_manifest)
    models = args.models.split(",")
    keys = [row for _, row in man.iterrows()]
    est = len(keys) * (2 + len(GOV_SEEDS)) * len(models)
    print(f"B2 parallel: {len(keys)} keys x {(2+len(GOV_SEEDS))} fits x {len(models)} models = est {est} rows, {args.workers} workers", flush=True)

    t0 = time.time()
    task_args = [(row, models) for row in keys]
    with Pool(args.workers) as pool:
        all_results = []
        for i, res in enumerate(pool.imap_unordered(process_key, task_args, chunksize=10)):
            all_results.extend(res)
            if len(all_results) % 10000 == 0:
                print(f"  {len(all_results)} rows | {time.time()-t0:.0f}s", flush=True)

    # Write output
    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"DONE {len(all_results)} rows in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
