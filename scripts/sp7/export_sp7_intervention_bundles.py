#!/usr/bin/env python3
"""export_sp7_intervention_bundles.py — SP7-D sentinel intervention bundles.

Creates immutable .npz bundles for each intervention variant from frozen SP4
M04/M05 bundles. Handles:
- I2 (permuted contamination): group-permuted leak columns, within split
- I4 (contamination-only): only the leak columns, no strict features
- O1 (fixed-epoch): same as full view, epoch enforced by runner
- S1 (training fraction): nested subsets at 25/50/75/100%

Uses only sentinel datasets (6). Never uses test labels.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[2]
SP4_DIR = ROOT / "results/structured_prior_replacement_v1/task_bundles"
MAN = ROOT / "artifacts/sp6/sp6_bundle_manifest.csv"
OUT = ROOT / "artifacts/sp7/bundles"
SENTINEL_DS = [0, 2, 4, 7, 9, 14]
STRENGTHS = ["S1", "S2", "S3", "S4", "S5"]
SEEDS_ALL = [13, 42, 2026, 3407, 7777]
S1_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
S1_SEED = 20260715
FIXED_EPOCH_FRACTION = 0.5  # 50% of max epochs (preregistered)
PERM_SEED_BASE = 2026071500


def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def write_bundle(path, arrays, meta):
    np.savez_compressed(path, **arrays)
    meta["bundle_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return meta


def main():
    man = pd.read_csv(MAN)
    man = man[man["mechanism"].isin(["M04", "M05"]) & man["dataset_index"].isin(SENTINEL_DS)]
    config = yaml.safe_load((ROOT / "configs/sp6/tabr_v1.yaml").read_text())
    max_epochs = config["model"]["max_epochs"]  # 100
    fixed_epoch = max(1, int(max_epochs * FIXED_EPOCH_FRACTION))  # 50

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    bundles_saved = 0

    for _, r in man.iterrows():
        ds_i = int(r["dataset_index"]); mech = r["mechanism"]
        bid = f"{mech}_{ds_i}_{r['strength']}_s{int(r['seed'])}"
        # Only process sentinel datasets; skip non-sentinel
        if ds_i not in SENTINEL_DS: continue

        bp = r["bundle_path"]; key = r["bundle_key"]
        b = np.load(SP4_DIR / Path(bp).name, allow_pickle=False)
        tr, va, te = np.asarray(b["train_idx"]), np.asarray(b["val_idx"]), np.asarray(b["test_idx"])
        block = np.asarray(b[f"block__{key}"]); mask = np.asarray(b[f"leak_mask__{key}"])
        Xraw = np.concatenate((np.asarray(b["base_X"]), block), axis=1); yraw = np.asarray(b["y"])
        strict_cols = np.where(~mask)[0]; full_cols = np.arange(Xraw.shape[1])
        leak_cols = np.where(mask)[0]

        # ---- I2: Permuted Contamination ----
        perm_seed = PERM_SEED_BASE + ds_i * 100 + int(r["seed"])
        rng = np.random.RandomState(perm_seed)
        XP = Xraw.copy()
        for tag, idx in [("train", tr), ("val", va), ("test", te)]:
            perm_idx = rng.permutation(len(idx))
            XP[idx[:, None], leak_cols] = Xraw[idx[perm_idx][:, None], leak_cols]
        pid = f"{mech}_{ds_i}_{r['strength']}_s{int(r['seed'])}_I2"
        arrs = {"X_train": XP[tr], "y_train": yraw[tr], "X_valid": XP[va], "y_valid": yraw[va],
                "X_test": XP[te], "y_test": yraw[te], "strict_cols": strict_cols, "full_cols_permuted": full_cols,
                "intervention": "I2_permuted", "perm_seed": perm_seed, "fixed_epoch": fixed_epoch}
        write_bundle(OUT / f"{pid}.npz", arrs, {"id": pid})
        rows.append({"id": pid, "intervention": "I2_permuted", "dataset_index": ds_i, "mechanism": mech,
                     "strength": r["strength"], "seed": int(r["seed"])})
        bundles_saved += 1

        # ---- I4: Contamination-Only ----
        pid4 = f"{mech}_{ds_i}_{r['strength']}_s{int(r['seed'])}_I4"
        arrs4 = {"X_train": Xraw[tr][:, leak_cols], "y_train": yraw[tr], "X_valid": Xraw[va][:, leak_cols], "y_valid": yraw[va],
                 "X_test": Xraw[te][:, leak_cols], "y_test": yraw[te], "strict_cols": np.arange(len(leak_cols)),
                 "full_cols_permuted": np.arange(len(leak_cols)), "intervention": "I4_contam_only", "fixed_epoch": fixed_epoch}
        write_bundle(OUT / f"{pid4}.npz", arrs4, {"id": pid4})
        rows.append({"id": pid4, "intervention": "I4_contam_only", "dataset_index": ds_i, "mechanism": mech,
                     "strength": r["strength"], "seed": int(r["seed"])})
        bundles_saved += 1

        # ---- O1: Fixed Epoch ----
        pid1 = f"{mech}_{ds_i}_{r['strength']}_s{int(r['seed'])}_O1"
        arrs1 = {"X_train": Xraw[tr], "y_train": yraw[tr], "X_valid": Xraw[va], "y_valid": yraw[va],
                 "X_test": Xraw[te], "y_test": yraw[te], "strict_cols": strict_cols, "full_cols_permuted": full_cols,
                 "intervention": "O1_fixed_epoch", "fixed_epoch": fixed_epoch}
        write_bundle(OUT / f"{pid1}.npz", arrs1, {"id": pid1})
        rows.append({"id": pid1, "intervention": "O1_fixed_epoch", "dataset_index": ds_i, "mechanism": mech,
                     "strength": r["strength"], "seed": int(r["seed"])})
        bundles_saved += 1

    # ---- S1: Training Fraction Study (S3, seed 13 ONLY) ----
    s1_man = man[(man["strength"] == "S3") & (man["seed"] == 13) & (man["dataset_index"].isin(SENTINEL_DS))]
    for _, r in s1_man.iterrows():
        ds_i = int(r["dataset_index"]); mech = r["mechanism"]
        bp = r["bundle_path"]; key = r["bundle_key"]
        b = np.load(SP4_DIR / Path(bp).name, allow_pickle=False)
        tr, va, te = np.asarray(b["train_idx"]), np.asarray(b["val_idx"]), np.asarray(b["test_idx"])
        block = np.asarray(b[f"block__{key}"]); mask = np.asarray(b[f"leak_mask__{key}"])
        Xraw = np.concatenate((np.asarray(b["base_X"]), block), axis=1); yraw = np.asarray(b["y"])
        strict_cols = np.where(~mask)[0]; full_cols = np.arange(Xraw.shape[1])
        for frac in S1_FRACTIONS:
            rng2 = np.random.RandomState(S1_SEED)
            n_samp = max(4, int(len(tr) * frac))
            tr_sub = rng2.choice(tr, n_samp, replace=False); tr_sub.sort()
            pidf = f"{mech}_{ds_i}_S3_s13_S1_{int(frac*100)}"
            arrsf = {"X_train": Xraw[tr_sub], "y_train": yraw[tr_sub], "X_valid": Xraw[va], "y_valid": yraw[va],
                     "X_test": Xraw[te], "y_test": yraw[te], "strict_cols": strict_cols, "full_cols_permuted": full_cols,
                     "intervention": f"S1_fraction_{int(frac*100)}", "train_fraction": frac,
                     "train_subset": tr_sub, "fixed_epoch": fixed_epoch}
            write_bundle(OUT / f"{pidf}.npz", arrsf, {"id": pidf})
            rows.append({"id": pidf, "intervention": f"S1_fraction_{int(frac*100)}", "dataset_index": ds_i,
                         "mechanism": mech, "strength": "S3", "seed": 13, "train_fraction": frac})
            bundles_saved += 1

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sp7_bundle_manifest.csv", index=False)
    print(f"exported {bundles_saved} bundles -> {OUT}")
    print(f"I2: {df.intervention.eq('I2_permuted').sum()} cells")
    print(f"I4: {df.intervention.eq('I4_contam_only').sum()} cells")
    print(f"O1: {df.intervention.eq('O1_fixed_epoch').sum()} cells")
    print(f"S1: {df.intervention.str.startswith('S1').sum()} cells (4 fractions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
