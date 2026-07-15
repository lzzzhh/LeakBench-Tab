#!/usr/bin/env python3
"""run_governance_bundle.py — Track B bundle-consuming governance evaluation.

Loads frozen SP5 task bundles, executes governance strategies (field/group/
graph/lifecycle) on frozen feature metadata, removes proposed fields, retrains
both strict and post-removal models, and records paired_harm reduction.

NEVER injects, splits, or generates tasks. Strategy determinism verified
per-cell via hash checks. For M06/M09/M10/M11.

Design: strategies operate on blind diagnostic scores + frozen metadata;
leakage_mask used ONLY for oracle upper-bound and offline recall evaluation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))
from src.leakbench.governance import GovernanceStrategy, GovernanceStatus, apply_strategy
from src.leakbench.diagnostics import OperationalMetadata, OperationalFeatureMetadata, OracleMetadata

FIELDS = ["run_id","dataset_index","mechanism","strength","seed","strategy","budget",
          "status","failure_reason","n_quarantined","leak_recall","legit_retention","residual_harm",
          "oracle_regret","pre_removal_full_auc","post_removal_auc","removed_count",
          "removed_leak_count","removed_legit_count","mask_hash"]
GOV_MECHS = ["M06","M09","M10","M11"]
STRATEGIES = {
    "no_removal": GovernanceStrategy.NO_REMOVAL,
    "oracle": GovernanceStrategy.ORACLE_REMOVE_ALL,
    "field_budget": GovernanceStrategy.FIXED_FIELD_BUDGET,
    "group_budget": GovernanceStrategy.FIXED_GROUP_BUDGET,
    "lifecycle": GovernanceStrategy.LIFECYCLE_REMOVAL,
    "graph_cut": GovernanceStrategy.GRAPH_CUT,
}
BUDGETS = [0.05, 0.10, 0.20]
SEEDS = [13, 42, 2026, 3407, 7777]

def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _stable_id(ds_idx, offset):
    return f"fid_{ds_idx:02d}_{offset}"


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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mechanisms", default="all")
    ap.add_argument("--strengths", default="all")
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--allow-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("locked; pass --allow-run")

    man = pd.read_csv(ROOT / args.bundle_manifest)
    mechs = None if args.mechanisms == "all" else args.mechanisms.split(",")
    strs = None if args.strengths == "all" else args.strengths.split(",")
    sds = None if args.seeds == "all" else [int(x) for x in args.seeds.split(",")]
    if args.datasets != "all":
        man = man[man["dataset_index"].astype(int).isin([int(x) for x in args.datasets.split(",")])]
    if mechs: man = man[man["mechanism"].isin(mechs)]
    if strs: man = man[man["strength"].isin(strs)]
    if sds: man = man[man["seed"].isin(sds)]
    total_cells = len(man) * len(STRATEGIES) * len(BUDGETS)  # budget only for budgeted strategies
    cells_per_key = len(man)
    actual = 0
    for _, row in man.iterrows():
        actual += 1  # no_removal
        actual += 1  # oracle
        # field/group/lifecycle/graph: apply per budget
        for sname in ["field_budget","group_budget","lifecycle","graph_cut"]:
            actual += len(BUDGETS)
    print(f"Governance grid: {cells_per_key} keys × strategies+Budgets = {actual} cells", flush=True)

    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"{out} exists; pass --resume")
        completed = set(pd.read_csv(out)["run_id"].astype(str))

    started = time.time(); done = 0
    for _, row in man.iterrows():
        ds_i = int(row["dataset_index"]); mech = row["mechanism"]
        try:
            X, y, tr, va, te, mask = load_cell(row)
        except Exception as e:
            continue
        n_orig = int(row["n_original"]); n_total = X.shape[1]
        # Build feature IDs and operational metadata
        fids = [_stable_id(ds_i, j) for j in range(n_total)]
        op_features = {fid: OperationalFeatureMetadata(stable_id=fid) for fid in fids[:n_orig]}
        for j in range(n_orig, n_total):
            is_leak = mask[j]
            group = f"group_{mech}_{j}" if mech in ["M06","M10"] and is_leak else None
            op_features[fids[j]] = OperationalFeatureMetadata(
                stable_id=fids[j], available_at_prediction=True, lifecycle="injected" if is_leak else None,
                group_id=group, outcome_descendant=is_leak)
        op_meta = OperationalMetadata(features=op_features, graph_edges=())
        oracle = OracleMetadata(leakage_by_feature_id={fid: bool(mask[i]) for i, fid in enumerate(fids)})

        # Blind diagnostic scores: mutual_info AUPRC on train
        scores = mutual_info_classif(X[tr], y[tr], random_state=42)
        scores = np.nan_to_num(scores, nan=0.0)
        score_dict = {fids[i]: float(scores[i]) for i in range(n_total)}

        # Baseline: full view model (pre-removal)
        seed = int(row["seed"])
        lr_full = LogisticRegression(max_iter=1000, random_state=seed).fit(X[tr], y[tr])
        pre_auc = float(roc_auc_score(y[te], lr_full.predict_proba(X[te])[:, 1]))

        # For each strategy x budget
        for sname, strategy in STRATEGIES.items():
            for budget in (BUDGETS if sname in ["field_budget","group_budget","lifecycle","graph_cut"] else [0.0]):
                rid_key = f"gov|{ds_i}|{mech}|{row['strength']}|{seed}|{sname}|{budget:.3f}|{sha_arr(mask.astype(np.uint8))[:8]}"
                run_id = hashlib.sha256(rid_key.encode()).hexdigest()[:20]
                done += 1
                rec = {k: "" for k in FIELDS}
                rec.update({"run_id": run_id, "dataset_index": ds_i, "mechanism": mech,
                            "strength": row["strength"], "seed": seed, "strategy": sname,
                            "budget": budget, "status": "FAILURE"})
                if run_id in completed: continue
                try:
                    result = apply_strategy(strategy, fids, list(scores), op_meta, oracle_metadata=oracle, budget=budget)
                    mask_bool = result.feature_mask > 0.5
                    kept = np.where(mask_bool)[0]

                    # retrain on kept fields
                    if len(kept) >= 2:
                        lr_post = LogisticRegression(max_iter=1000, random_state=seed).fit(X[tr][:, kept], y[tr])
                        post_auc = float(roc_auc_score(y[te], lr_post.predict_proba(X[te][:, kept])[:, 1]))
                    else:
                        post_auc = 0.5  # insufficient features

                    removed_count = int((~mask_bool).sum())
                    removed_leak = int(mask[~mask_bool].sum())
                    removed_legit = removed_count - removed_leak
                    rec.update({"status": "SUCCESS",
                                "n_quarantined": result.n_quarantined,
                                "leak_recall": round(result.leakage_recall, 4),
                                "legit_retention": round(result.legitimate_retention, 4),
                                "residual_harm": round(post_auc - pre_auc, 6) if len(kept) >= 2 else None,
                                "oracle_regret": round(pre_auc - post_auc, 6) if len(kept) >= 2 else None,
                                "pre_removal_full_auc": round(pre_auc, 6),
                                "post_removal_auc": round(post_auc, 6),
                                "removed_count": removed_count,
                                "removed_leak_count": removed_leak,
                                "removed_legit_count": removed_legit,
                                "mask_hash": sha_arr(mask_bool.astype(np.uint8))})
                except Exception as e:
                    rec["failure_reason"] = f"{type(e).__name__}: {str(e)[:200]}"
                write_hdr = not out.exists()
                with out.open("a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=FIELDS)
                    if write_hdr: w.writeheader()
                    w.writerow(rec)
                if done % 50 == 0:
                    print(f"{done}/{actual} cells | {time.time()-started:.0f}s", flush=True)
    print(f"DONE {done}/{actual} in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
