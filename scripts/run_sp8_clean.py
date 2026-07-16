#!/usr/bin/env python3
"""run_sp8_clean.py — SP8 matched-cost governance, clean protocol.

Oracle-isolated: non-oracle policies NEVER access leakage_mask.
Policies:
  P0 — KEEP ALL (baseline, 0 fields removed)
  P1 — ORACLE_REMOVE_ALL (upper bound, oracle-only, NOT deployable)
  P2 — RANDOM_MATCHED (k fields at random, frozen seed)
  P3 — BLIND_MI (k fields with highest train-side mutual-information)

All non-oracle policies at the same budget_k remove exactly k fields.
Primary metric: strict_distance_reduction = |full-strict| - |governed-strict|
Record: strict_auc, full_auc, governed_auc, strict_distance_reduction, utility_loss.

Strict: train+eval on X[:, ~leakage_mask].  Full: train+eval on all X.
Governed: train+eval on X masked by policy (k columns removed).

Bundle-only, read-only.  Never injects, splits, or mutates.
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

GOV_SEED = 20260718
BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.20]

FIELDS = [
    "run_id","dataset_index","mechanism","strength","seed","policy","budget_k","budget_fraction",
    "status","failure_reason",
    "strict_auc","full_auc","governed_auc",
    "strict_distance_reduction","utility_loss","residual_harm",
    "removed_count","removed_leak_count","removed_legit_count",
    "leak_recall","legit_retention","oracle_policy",
]

def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def load_cell(row):
    bundle = ROOT / row["bundle_path"]
    if sha_file(bundle) != str(row["bundle_sha256"]).lower():
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


def _unique_budgets(n_features, fractions):
    """Compute unique k from fractions, deduplicating collapsed budgets."""
    k_map = {f: max(1, round(n_features * f)) for f in fractions}
    unique_ks = sorted(set(k_map.values()))
    return k_map, unique_ks


def _train_lr(Xtr, ytr, Xte, yte, seed):
    m = LogisticRegression(max_iter=1000, random_state=seed).fit(Xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def select_fields_random(n_total, k, seed_offset):
    rng = np.random.RandomState(GOV_SEED + seed_offset)
    return rng.choice(n_total, k, replace=False)

def select_fields_mi(scores, k):
    return np.argsort(scores)[::-1][:k]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--mechanisms", default="all")
    ap.add_argument("--strengths", default="all")
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--allow-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("locked; pass --allow-run")

    man = pd.read_csv(ROOT / args.bundle_manifest)
    for col, val in [("dataset_index", args.datasets), ("mechanism", args.mechanisms),
                     ("strength", args.strengths), ("seed", args.seeds)]:
        if val != "all":
            man = man[man[col].astype(str).isin(val.split(","))]

    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        if not args.resume: raise FileExistsError(f"{out} exists; pass --resume")
        completed = set(pd.read_csv(out)["run_id"].astype(str))

    started = time.time(); cells_done = 0; cells_est = len(man) * (2 + 2 * 4)
    print(f"SP8 clean governance: ~{len(man)} keys, ~{cells_est} cells", flush=True)

    for _, row in man.iterrows():
        ds_i = int(row["dataset_index"]); mech = row["mechanism"]
        strength = row["strength"]; seed = int(row["seed"])
        try: X, y, tr, va, te, mask = load_cell(row)
        except Exception as e: continue

        n_features = X.shape[1]; n_injected = int(row["n_injected"])
        leak_indices = set(np.where(mask)[0])
        legit_indices = set(range(n_features)) - leak_indices
        k_map, unique_ks = _unique_budgets(n_features, BUDGET_FRACTIONS)

        # --- Train strict and full baselines (model-independent, shared across policies) ---
        strict_cols = np.where(~mask)[0]
        strict_auc = _train_lr(X[tr][:, strict_cols], y[tr], X[te][:, strict_cols], y[te], seed)
        full_auc = _train_lr(X[tr], y[tr], X[te], y[te], seed)

        leak_cols = np.where(mask)[0]
        for policy, k, oracle_flag, frac_label, removed in [
            ("P0_keep", 0, False, 0.0, np.array([], dtype=int)),
            ("P1_oracle", int(mask.sum()), True, 0.0, leak_cols),
         ]:
            rec = _run_policy(completed, out, ds_i, mech, strength, seed, policy, k, frac_label,
                              strict_auc, full_auc, X, y, tr, te, mask, leak_indices, legit_indices,
                              seed, oracle_flag, removed)
            cells_done += 1
            if cells_done % 500 == 0:
                print(f"  {cells_done} cells | {time.time()-started:.0f}s", flush=True)

        for k in unique_ks:
            if k <= 0 or k >= n_features: continue
            for frac, kval in k_map.items():
                if kval != k: continue  # only process each unique k once

            # Compute BLIND train-side MI scores (model-independent, no leakage_mask)
            mi_scores = mutual_info_classif(X[tr], y[tr], random_state=42)
            mi_scores = np.nan_to_num(mi_scores, nan=0.0)

            # P2: Random matched-cost (negative control)
            rm_fields = select_fields_random(n_features, k, ds_i * 1000 + seed * 13)
            keep_mask = np.ones(n_features, dtype=bool)
            keep_mask[rm_fields] = False
            _run_policy(completed, out, ds_i, mech, strength, seed, "P2_random", k, frac,
                        strict_auc, full_auc, X, y, tr, te, mask, leak_indices, legit_indices,
                        seed, False, rm_fields)
            cells_done += 1

            # P3: Blind MI (deployable, no leakage_mask)
            mi_fields = select_fields_mi(mi_scores, k)
            keep_mask_p3 = np.ones(n_features, dtype=bool)
            keep_mask_p3[mi_fields] = False
            _run_policy(completed, out, ds_i, mech, strength, seed, "P3_blind_mi", k, frac,
                        strict_auc, full_auc, X, y, tr, te, mask, leak_indices, legit_indices,
                        seed, False, mi_fields)
            cells_done += 1

    print(f"DONE {cells_done}/{cells_est} in {time.time()-started:.0f}s", flush=True)
    return 0


def _run_policy(completed, out, ds_i, mech, strength, seed, policy, budget_k, budget_frac,
                strict_auc, full_auc, X, y, tr, te, mask, leak_indices, legit_indices,
                lr_seed, oracle_policy, removed_fields):
    rid_key = f"sp8clean|{ds_i}|{mech}|{strength}|{seed}|{policy}|{budget_k}|{GOV_SEED}"
    run_id = hashlib.sha256(rid_key.encode()).hexdigest()[:20]
    rec = {k: "" for k in FIELDS}
    rec.update(dict(run_id=run_id, dataset_index=ds_i, mechanism=mech, strength=strength,
                    seed=seed, policy=policy, budget_k=budget_k, budget_fraction=budget_frac,
                    status="FAILURE", strict_auc=round(strict_auc,6), full_auc=round(full_auc,6),
                    oracle_policy=str(oracle_policy).lower()))
    if run_id in completed: return
    try:
        keep_cols = np.setdiff1d(np.arange(X.shape[1]), removed_fields)
        if len(keep_cols) >= 2:
            gov_auc = _train_lr(X[tr][:, keep_cols], y[tr], X[te][:, keep_cols], y[te], lr_seed)
        else:
            gov_auc = strict_auc

        dist_full = abs(full_auc - strict_auc)
        dist_gov = abs(gov_auc - strict_auc)
        sdr = dist_full - dist_gov       # positive = closer to strict
        ul = strict_auc - gov_auc         # positive = over-removal
        rh = gov_auc - strict_auc

        removed_leak = int(len(set(removed_fields) & leak_indices))
        removed_legit = int(len(set(removed_fields) & legit_indices))
        n_leak = max(1, len(leak_indices))
        n_legit = max(1, len(legit_indices))

        rec.update(dict(status="SUCCESS", governed_auc=round(gov_auc,6),
                        strict_distance_reduction=round(sdr,6), utility_loss=round(ul,6),
                        residual_harm=round(rh,6),
                        removed_count=len(removed_fields),
                        removed_leak_count=removed_leak, removed_legit_count=removed_legit,
                        leak_recall=round(removed_leak/n_leak,4),
                        legit_retention=round(1.0-removed_legit/max(1,n_legit),4)))
    except Exception as e:
        rec["failure_reason"] = f"{type(e).__name__}: {str(e)[:200]}"
    _append(out, rec)


def _append(out, rec):
    write_hdr = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_hdr: w.writeheader()
        w.writerow(rec)


if __name__ == "__main__":
    raise SystemExit(main())
