#!/usr/bin/env python3
"""run_sp8_b1_multiseed.py — B1 multi-governance-seed P2 for LR.

Adds 20 independent P2 random seeds per key. P0/P1/P3 computed once.
Strict/full baselines from frozen SP8 (reused, not re-fit).
Primary metric: strict_distance_reduction = |full-strict| - |governed-strict|.
Outputs long-table CSV with governance_seed column.
"""
from __future__ import annotations
import argparse, csv, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))

GOV_SEEDS = [2026071700 + i for i in range(20)]
BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.20]

FIELDS = [
    "run_id","dataset_index","mechanism","strength","training_seed","governance_seed",
    "policy","budget_k","budget_fraction",
    "status","failure_reason",
    "strict_auc","full_auc","governed_auc",
    "strict_distance_reduction","initial_gap",
    "removed_count","selection_mask_hash",
]

def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selection_hash(indices):
    values = np.sort(np.asarray(indices, dtype='<i8'))
    return hashlib.sha256(b"encoded_column_indices_v1\0" + values.tobytes()).hexdigest()


def load_cell(row):
    bundle = ROOT / row["bundle_path"]
    if sha_file(bundle) != str(row["bundle_sha256"]).lower():
        raise RuntimeError(f"bundle hash mismatch: {row['bundle_path']}")
    key = str(row["bundle_key"])
    with np.load(bundle, allow_pickle=False) as b:
        X = np.concatenate((np.asarray(b["base_X"]), np.asarray(b[f"block__{key}"])), axis=1)
        y = np.asarray(b["y"])
        tr = np.asarray(b["train_idx"]); te = np.asarray(b["test_idx"])
        mask = np.asarray(b[f"leak_mask__{key}"])
    return X, y, tr, te, mask


def _train_lr(Xtr, ytr, Xte, yte, seed):
    m = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def select_fields_mi(scores, k):
    return np.argsort(scores)[::-1][:k]


def select_fields_random(n_total, k, gov_seed, ds_i, training_seed):
    rng = np.random.RandomState((gov_seed * 100 + ds_i * 7 + training_seed * 13) % (2**31 - 1))
    return rng.choice(n_total, k, replace=False)


def unique_budgets(n_features):
    k_map = {f: max(1, round(n_features * f)) for f in BUDGET_FRACTIONS}
    return k_map, sorted(set(k_map.values()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--allow-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("locked; pass --allow-run")

    man = pd.read_csv(ROOT / args.bundle_manifest)
    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        if not args.resume: raise FileExistsError(f"{out} exists; pass --resume")
        completed = set(pd.read_csv(out)["run_id"].astype(str))

    n_keys = len(man)
    est = n_keys * (1 + len(BUDGET_FRACTIONS) + len(GOV_SEEDS) * len(BUDGET_FRACTIONS))
    print(f"B1 multi-seed P2: {n_keys} keys, est {est} rows", flush=True)
    started = time.time(); done = 0

    for _, row in man.iterrows():
        ds_i = int(row["dataset_index"]); mech = row["mechanism"]
        strength = row["strength"]; tseed = int(row["seed"])
        try:
            X, y, tr, te, mask = load_cell(row)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load bundle for dataset={ds_i} mechanism={mech} "
                f"strength={strength} seed={tseed}"
            ) from exc

        n_features = X.shape[1]; strict_cols = np.where(~mask)[0]
        k_map, unique_ks = unique_budgets(n_features)
        strict_auc = _train_lr(X[tr][:, strict_cols], y[tr], X[te][:, strict_cols], y[te], tseed)
        full_auc = _train_lr(X[tr], y[tr], X[te], y[te], tseed)
        gap = abs(full_auc - strict_auc)

        # P0: no removal
        rid0 = hashlib.sha256(f"b1p0|{ds_i}|{mech}|{strength}|{tseed}".encode()).hexdigest()[:20]
        done += 1
        if rid0 not in completed:
            _append(out, dict(run_id=rid0, dataset_index=ds_i, mechanism=mech, strength=strength,
                training_seed=tseed, governance_seed=-1, policy="P0_keep", budget_k=0, budget_fraction=0.0,
                status="SUCCESS", strict_auc=round(strict_auc,6), full_auc=round(full_auc,6),
                governed_auc=round(full_auc,6), strict_distance_reduction=0.0, initial_gap=round(gap,6),
                removed_count=0, selection_mask_hash=selection_hash([])))

        # P3: blind MI (one fit per unique k)
        mi_scores = mutual_info_classif(X[tr], y[tr], random_state=42)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)
        for k in unique_ks:
            if k <= 0 or k >= n_features: continue
            mi_fields = select_fields_mi(mi_scores, k)
            keep_p3 = np.ones(n_features, dtype=bool); keep_p3[mi_fields] = False
            gov_auc = _train_lr(X[tr][:, keep_p3], y[tr], X[te][:, keep_p3], y[te], tseed)
            sdr = abs(full_auc - strict_auc) - abs(gov_auc - strict_auc)
            for frac, kval in k_map.items():
                if kval != k: continue
                rid3 = hashlib.sha256(f"b1p3|{ds_i}|{mech}|{strength}|{tseed}|{k}|{frac:.3f}".encode()).hexdigest()[:20]
                done += 1
                if rid3 not in completed:
                    _append(out, dict(run_id=rid3, dataset_index=ds_i, mechanism=mech, strength=strength,
                        training_seed=tseed, governance_seed=-1, policy="P3_blind_mi", budget_k=k, budget_fraction=frac,
                        status="SUCCESS", strict_auc=round(strict_auc,6), full_auc=round(full_auc,6),
                        governed_auc=round(gov_auc,6), strict_distance_reduction=round(sdr,6),
                        initial_gap=round(gap,6), removed_count=k, selection_mask_hash=selection_hash(mi_fields)))

        # P2: multi-governance-seed random
        for k in unique_ks:
            if k <= 0 or k >= n_features: continue
            for gs in GOV_SEEDS:
                rm_fields = select_fields_random(n_features, k, gs, ds_i, tseed)
                keep_p2 = np.ones(n_features, dtype=bool); keep_p2[rm_fields] = False
                gov_auc = _train_lr(X[tr][:, keep_p2], y[tr], X[te][:, keep_p2], y[te], tseed)
                sdr = abs(full_auc - strict_auc) - abs(gov_auc - strict_auc)
                for frac, kval in k_map.items():
                    if kval != k: continue
                    rid2 = hashlib.sha256(f"b1p2|{ds_i}|{mech}|{strength}|{tseed}|{k}|{gs}|{frac:.3f}".encode()).hexdigest()[:20]
                    done += 1
                    if rid2 not in completed:
                        _append(out, dict(run_id=rid2, dataset_index=ds_i, mechanism=mech, strength=strength,
                            training_seed=tseed, governance_seed=gs, policy="P2_random", budget_k=k, budget_fraction=frac,
                            status="SUCCESS", strict_auc=round(strict_auc,6), full_auc=round(full_auc,6),
                            governed_auc=round(gov_auc,6), strict_distance_reduction=round(sdr,6),
                            initial_gap=round(gap,6), removed_count=k, selection_mask_hash=selection_hash(rm_fields)))

        if done % 5000 == 0:
            print(f"  {done}/{est} | {time.time()-started:.0f}s", flush=True)

    print(f"DONE {done}/{est} in {time.time()-started:.0f}s", flush=True)
    return 0


def _append(out, rec):
    write_hdr = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_hdr: w.writeheader()
        w.writerow(rec)


if __name__ == "__main__":
    raise SystemExit(main())
