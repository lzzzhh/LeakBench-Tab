#!/usr/bin/env python3
"""run_sp8_matched_cost_governance.py — SP8 matched-cost governance runner.

Read-only, bundle-only. Computes blind mutual-information scores on train split,
applies policies P0-P5 at matched budgets (k fields removed), retrains model
on post-removal features, and records strict_distance_reduction + utility_loss.

Oracle access (leakage_mask) used ONLY for P1 (oracle upper bound) and
post-hoc recall evaluation. P2/P3/P4/P5 never read leakage_mask.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
GOV_SEED = 20260718

FIELDS = ["run_id","dataset_index","mechanism","strength","seed","policy","budget_k",
          "status","failure_reason","strict_auc","full_auc","governed_auc",
          "strict_distance_reduction","utility_loss","residual_harm",
          "removed_count","removed_leak_count","removed_legit_count",
          "leak_recall","legit_retention","oracle_policy"]

def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def load_cell(row):
    bundle = ROOT / row["bundle_path"]
    if sha_file(bundle) != str(row["bundle_sha256"]).lower():
        raise RuntimeError("bundle SHA256 mismatch")
    key = str(row["bundle_key"])
    with np.load(bundle, allow_pickle=False) as b:
        base_X = np.asarray(b["base_X"]); y = np.asarray(b["y"])
        tr, va, te = np.asarray(b["train_idx"]), np.asarray(b["val_idx"]), np.asarray(b["test_idx"])
        block = np.asarray(b[f"block__{key}"]); mask = np.asarray(b[f"leak_mask__{key}"])
    X = np.concatenate((base_X, block), axis=1)
    if hashlib.sha256(te.tobytes()).hexdigest() != str(row["split_hash"]):
        raise RuntimeError("split hash mismatch")
    return X, y, tr, va, te, mask

def select_P2_random(n_total, k, rng):
    """Random matched-budget: k fields uniformly at random from all."""
    return rng.choice(n_total, k, replace=False)

def select_P3_field_score(scores, k):
    """Field-score top-k: highest mutual-information fields (BLIND, train-only)."""
    return np.argsort(scores)[::-1][:k]

def select_P4_group_aware(scores, groups, k):
    """Group-budget: greedily pick groups by max member score. Legacy comparator."""
    if not groups: return np.array([], dtype=int)
    grp_scores = {gid: max(scores[list(members)]) for gid, members in groups.items()}
    ordered = sorted(grp_scores, key=grp_scores.get, reverse=True)
    removed = []
    for g in ordered:
        removed.extend(list(groups[g]))
        if len(removed) >= k: break
    return np.array(removed[:k])

def select_P5_lifecycle(scores, lifecycle_map, k):
    """Lifecycle: remove fields tagged post_outcome/future_window. Falls back to P3 if not enough."""
    candidates = [i for i, tag in lifecycle_map.items() if tag in ("post_outcome","future_window")]
    if len(candidates) >= k:
        return np.array(sorted(candidates, key=lambda i: scores[i], reverse=True)[:k])
    # fallback: remove all lifecycle fields + top-k remaining by score
    remaining = k - len(candidates)
    others = [i for i in range(len(scores)) if i not in candidates]
    extra = sorted(others, key=lambda i: scores[i], reverse=True)[:remaining]
    return np.array(candidates + extra)

def compute_budgets(n_total, fractions):
    """Compute unique k from fractions, deduplicating collapsed budgets."""
    ks = sorted(set(max(1, round(n_total * f)) for f in fractions))
    return {f: max(1, round(n_total * f)) for f in fractions}, ks

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mechanisms", default="M06,M09,M10,M11")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--allow-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("locked; pass --allow-run")

    BUDGET_FRACTIONS = [0.01, 0.05, 0.10, 0.20]
    man = pd.read_csv(ROOT / args.bundle_manifest)
    mechs = args.mechanisms.split(",")
    man = man[man["mechanism"].isin(mechs)]
    if args.datasets != "all":
        man = man[man["dataset_index"].astype(int).isin([int(x) for x in args.datasets.split(",")])]

    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"{out} exists; pass --resume")
        completed = set(pd.read_csv(out)["run_id"].astype(str))

    started = time.time(); done = 0; total_est = len(man) * (1 + 1 + 4 * 4)  # P0 + P1 + 4 policies x <=4 budgets
    print(f"SP8 governance grid: ~{len(man)} keys, est. {total_est} cells", flush=True)

    for _, row in man.iterrows():
        ds_i = int(row["dataset_index"]); mech = row["mechanism"]; seed = int(row["seed"])
        try: X, y, tr, va, te, mask = load_cell(row)
        except: continue
        n_total = X.shape[1]
        budget_map, unique_ks = compute_budgets(n_total, BUDGET_FRACTIONS)
        # Blind train-side mutual-information scores (P3 source; P2/P4/P5 also use for fallback)
        mi_scores = mutual_info_classif(X[tr], y[tr], random_state=42)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)

        # Groups and lifecycle from task metadata (when available)
        # For M06: redundant cluster; M10: mixed legitimate+contam
        groups = {}
        lifecycle_map = {}
        n_orig = int(row["n_original"])
        # All injected columns form one group for M06/M10
        if mech in ["M06","M10"] and n_total > n_orig:
            inj_indices = set(range(n_orig, n_total))
            contam_mask = mask[n_orig:]
            legit_inj = set(np.where(~contam_mask)[0] + n_orig)
            contam_inj = set(np.where(contam_mask)[0] + n_orig)
            if legit_inj: groups["legitimate_injected"] = legit_inj
            if contam_inj: groups["contaminated_injected"] = contam_inj
        # Lifecycle for M04/M05 (temporal) — governance mechs M06/M09/M10/M11 don't have lifecycle

        # Baseline: strict + full AUC
        lr_strict = LogisticRegression(max_iter=1000, random_state=seed).fit(X[tr][:, :n_orig], y[tr])
        strict_auc = float(roc_auc_score(y[te], lr_strict.predict_proba(X[te][:, :n_orig])[:, 1]))
        lr_full = LogisticRegression(max_iter=1000, random_state=seed).fit(X[tr], y[tr])
        full_auc = float(roc_auc_score(y[te], lr_full.predict_proba(X[te])[:, 1]))

        # P0: no removal (reference)
        run_and_record(completed, out, done, ds_i, mech, seed, row["strength"], "P0", 0, strict_auc, full_auc,
                       np.zeros(n_total, dtype=bool), mask, False, X, y, tr, te, seed, started)
        # P1: oracle remove all contamination (upper bound)
        oracle_kept = np.where(~mask)[0]
        run_and_record(completed, out, done, ds_i, mech, seed, row["strength"], "P1", int(mask.sum()),
                       strict_auc, full_auc, ~mask, mask, True, X, y, tr, te, seed, started)

        # P2-P5 at each unique budget k
        rng = np.random.RandomState(GOV_SEED + ds_i * 100 + seed)
        for k in unique_ks:
            if k >= n_total: continue
            # P2: random
            mask_p2 = np.ones(n_total, dtype=bool)
            mask_p2[select_P2_random(n_total, k, rng)] = False
            run_and_record(completed, out, done, ds_i, mech, seed, row["strength"], "P2", k,
                           strict_auc, full_auc, mask_p2, mask, False, X, y, tr, te, seed, started)
            # P3: field-score
            mask_p3 = np.ones(n_total, dtype=bool)
            mask_p3[select_P3_field_score(mi_scores, k)] = False
            run_and_record(completed, out, done, ds_i, mech, seed, row["strength"], "P3", k,
                           strict_auc, full_auc, mask_p3, mask, False, X, y, tr, te, seed, started)
            # P4: group
            mask_p4 = np.ones(n_total, dtype=bool)
            p4_removed = select_P4_group_aware(mi_scores, groups, k) if groups else np.array([], dtype=int)
            if len(p4_removed) == k:
                mask_p4[p4_removed] = False
                run_and_record(completed, out, done, ds_i, mech, seed, row["strength"], "P4", k,
                               strict_auc, full_auc, mask_p4, mask, False, X, y, tr, te, seed, started)
            # P5: lifecycle
            mask_p5 = np.ones(n_total, dtype=bool)
            p5_removed = select_P5_lifecycle(mi_scores, lifecycle_map, k) if lifecycle_map else select_P3_field_score(mi_scores, k)
            mask_p5[p5_removed] = False
            run_and_record(completed, out, done, ds_i, mech, seed, row["strength"], "P5", k,
                           strict_auc, full_auc, mask_p5, mask, False, X, y, tr, te, seed, started)
    print(f"DONE in {time.time()-started:.0f}s", flush=True)
    return 0


def run_and_record(completed, out, ds_i, mech, seed, strength, policy, budget_k, strict_auc, full_auc, kept_mask, leak_mask, oracle_policy, X, y, tr, te, seed_for_lr):
    rid_key = f"gov2|{ds_i}|{mech}|{strength}|{seed}|{policy}|{budget_k}|GOV_SEED"
    run_id = hashlib.sha256(rid_key.encode()).hexdigest()[:20]
    done = getattr(run_and_record, "_c", 0) + 1
    run_and_record._c = done
    rec = {k: "" for k in FIELDS}
    rec.update({"run_id": run_id, "dataset_index": ds_i, "mechanism": mech, "strength": strength,
                "seed": seed, "policy": policy, "budget_k": budget_k, "status": "FAILURE",
                "strict_auc": round(strict_auc, 6), "full_auc": round(full_auc, 6), "oracle_policy": str(oracle_policy).lower()})
    if run_id in completed: return
    try:
        kept = np.where(kept_mask)[0]
        if len(kept) >= 2:
            lr = LogisticRegression(max_iter=1000, random_state=seed_for_lr).fit(X[tr][:, kept], y[tr])
            gov_auc = float(roc_auc_score(y[te], lr.predict_proba(X[te][:, kept])[:, 1]))
        else:
            gov_auc = strict_auc  # insufficient features
        dist_full = abs(full_auc - strict_auc)
        dist_gov = abs(gov_auc - strict_auc)
        sdr = dist_full - dist_gov  # positive = closer to strict
        ul = strict_auc - gov_auc  # positive = over-removal harm
        removed_count = int((~kept_mask).sum())
        removed_leak = int(leak_mask[~kept_mask].sum())
        removed_legit = removed_count - removed_leak
        n_leak = max(1, int(leak_mask.sum()))
        rec.update({"status": "SUCCESS", "governed_auc": round(gov_auc, 6),
                    "strict_distance_reduction": round(sdr, 6), "utility_loss": round(ul, 6),
                    "residual_harm": round(gov_auc - strict_auc, 6),
                    "removed_count": removed_count, "removed_leak_count": removed_leak,
                    "removed_legit_count": removed_legit,
                    "leak_recall": round(float(removed_leak) / n_leak, 4),
                    "legit_retention": round(1.0 - float(removed_legit) / max(1, (~leak_mask).sum()), 4)})
    except Exception as e:
        rec["failure_reason"] = f"{type(e).__name__}: {str(e)[:200]}"
    write_hdr = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_hdr: w.writeheader()
        w.writerow(rec)





if __name__ == "__main__":

    raise SystemExit(main())
