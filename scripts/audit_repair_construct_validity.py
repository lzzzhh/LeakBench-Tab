#!/usr/bin/env python3
"""T0-A1: Deterministic selection reconstruction for all 709,500 rows.

Reconstructs P3 (MI) and P2 (random) selections from frozen bundle manifest.
Compares recomputed selection_hash with recorded selection_mask_hash.
Uses one-pass bundle loading with per-key caching.
"""
from __future__ import annotations
import csv, hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOV_SEEDS = [2026071700 + i for i in range(20)]
MI_SEED = 42
BUDGET = 0.20
HASH_PREFIX = b'encoded_column_indices_v1\0'

def selection_hash(indices: np.ndarray) -> str:
    arr = np.sort(indices).astype(np.int64)
    return hashlib.sha256(HASH_PREFIX + arr.tobytes()).hexdigest()

def p2_seed_formula(gov_seed: int, ds: int, ts: int) -> int:
    return (gov_seed * 100 + ds * 7 + ts * 13) % (2**31 - 1)

def load_bundle(row) -> tuple:
    b = np.load(ROOT / row.bundle_path, allow_pickle=False)
    k = row.bundle_key
    X = np.concatenate((b['base_X'], b[f'block__{k}']), axis=1)
    y = b['y']
    tr = b['train_idx']
    mask = b[f'leak_mask__{k}']
    return X, y, tr, mask

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--input-csvs", nargs="+", required=True)
    ap.add_argument("--output-dir", default="results/edbt_t0_r2")
    ap.add_argument("--allow-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress", type=int, default=500)
    args = ap.parse_args()
    if not args.allow_run:
        raise RuntimeError("locked; pass --allow-run")

    man = pd.read_csv(ROOT / args.bundle_manifest)

    # Index by key: load and compute MI once per key
    bundle_cache = {}
    mi_cache = {}
    k_cache = {}

    for _, r in man.iterrows():
        key = (int(r.dataset_index), r.mechanism, r.strength, int(r.seed))
        try:
            X, y, tr, mask = load_bundle(r)
            nf = X.shape[1]
            k = max(1, round(nf * BUDGET))
            mi = mutual_info_classif(X[tr], y[tr], random_state=MI_SEED)
            mi = np.nan_to_num(mi, nan=0.0)
            bundle_cache[key] = (nf, k, mask, np.argsort(mi)[::-1][:k])
            mi_cache[key] = mi
            k_cache[key] = k
        except Exception as e:
            print(f"WARN: bundle load failed for {key}: {e}", flush=True)

    print(f"Loaded {len(bundle_cache)}/{len(man)} bundles", flush=True)

    total_checked = 0
    total_mismatch = 0

    for csv_path in args.input_csvs:
        print(f"\n=== Checking {csv_path} ===", flush=True)
        df = pd.read_csv(ROOT / csv_path)
        # Filter to 20% budget only
        df = df[df.budget_fraction == 0.20]
        print(f"  {len(df)} rows at 20% budget", flush=True)

        mismatches = 0
        for i, row in df.iterrows():
            key = (int(row.dataset_index), row.mechanism, row.strength, int(row.training_seed))
            if key not in bundle_cache:
                print(f"  MISSING BUNDLE: {key}", flush=True)
                mismatches += 1
                continue

            nf, k, mask, mi_indices = bundle_cache[key]

            if row.policy == 'P3_blind_mi':
                rec_hash = selection_hash(mi_indices)
            elif row.policy == 'P2_random':
                gs = int(row.governance_seed)
                ds = int(row.dataset_index)
                ts = int(row.training_seed)
                seed = p2_seed_formula(gs, ds, ts)
                rng = np.random.RandomState(seed)
                rf = rng.choice(nf, k, replace=False)
                rec_hash = selection_hash(rf)
            else:
                continue  # skip P0

            if rec_hash != row.selection_mask_hash:
                mismatches += 1
                if mismatches <= 5:
                    print(f"  MISMATCH: key={key} policy={row.policy} "
                          f"recorded={row.selection_mask_hash[:16]}... "
                          f"reconstructed={rec_hash[:16]}...", flush=True)

            total_checked += 1
            if total_checked % args.progress == 0:
                print(f"  ... {total_checked} checked, {mismatches} mismatches so far", flush=True)

            if args.limit and total_checked >= args.limit:
                break

        print(f"  {csv_path}: {mismatches} mismatches out of {len(df)} rows", flush=True)
        total_mismatch += mismatches

    print(f"\n=== T0-A1 VERDICT ===")
    print(f"Total rows checked: {total_checked}")
    print(f"Total mismatches: {total_mismatch}")
    if total_mismatch == 0:
        print("PASS: All selection hashes reconstructed correctly")
    else:
        print(f"BLOCKED: {total_mismatch} selection hash mismatches")

if __name__ == "__main__":
    raise SystemExit(main())
