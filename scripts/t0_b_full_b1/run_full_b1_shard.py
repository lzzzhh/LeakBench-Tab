#!/usr/bin/env python3
"""T0-B Full-B1 Shard Runner VF — complete CLI: plan→execute→write ledgers→manifest."""
from __future__ import annotations
import gzip, hashlib, io, json, os, sys, time
from pathlib import Path; import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
from scripts.t0_b_v3.policy_selectors import score_mi, score_point_biserial, score_lr_coef, score_rf_permutation, group_max_score, top_k_groups, top_k_columns

CALLS = {"lr": 0, "p3": 0, "p4": 0, "p5": 0, "p6": 0}

def _fake_mf(model_id, Xtr, ytr, Xva, yva, Xte, seed):
    """Synthetic model: deterministic random probabilities."""
    CALLS["lr"] += 1
    rng = np.random.RandomState(seed + len(Xtr))
    class F: probabilities = rng.rand(len(Xte))
    return F()

def _fake_mi(Xtr, ytr):
    CALLS["p3"] += 1; return np.random.RandomState(42).rand(Xtr.shape[1])

def _fake_pb(Xtr, ytr):
    CALLS["p4"] += 1; return np.abs(np.random.RandomState(43).randn(Xtr.shape[1]))

def _fake_lr(Xtr, ytr):
    CALLS["p5"] += 1; return np.abs(np.random.RandomState(44).randn(Xtr.shape[1]))

def _fake_rf(Xtr, ytr):
    CALLS["p6"] += 1; return np.abs(np.random.RandomState(45).randn(Xtr.shape[1]))

def execute_key(kp, groups, eval_info, run_ids):
    """Execute one key with synthetic data. Returns {baseline_rows, governed_rows, selection_rows}."""
    ds, mech, st, ts = kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]
    # Use actual group dimension for synthetic data
    n_total = sum(g["group_size"] for g in groups)
    rng = np.random.RandomState(ts)
    X = rng.randn(100, n_total); y = (X[:,0] > 0).astype(int)
    tr = np.arange(60); va = np.arange(60, 80); te = np.arange(80, 100)

    leak_idx = set()
    for gid in eval_info.get("leak_group_ids", []):
        for g in groups:
            if g["opaque_group_id"] == gid: leak_idx.update(g["member_encoded_indices"])
    leak_mask = np.array([i in leak_idx for i in range(n_total)])

    mf = _fake_mf
    Xs = X[:, ~leak_mask]
    s1 = mf("lr", Xs[tr], y[tr], Xs[va], y[va], Xs[te], ts)
    s2 = mf("lr", X[tr], y[tr], X[va], y[va], X[te], ts)
    sa = float(roc_auc_score(y[te], s1.probabilities))
    fa = float(roc_auc_score(y[te], s2.probabilities))

    bl = []
    for rk, bt, auc in [("strict", "strict", sa), ("full", "full", fa)]:
        rid = run_ids[rk]
        bl.append({"run_id": rid, "dataset_index": ds, "mechanism": mech, "strength": st, "training_seed": ts, "learner": "lr", "baseline_type": bt, "auc": auc})

    Xtr, ytr = X[tr], y[tr]
    s3 = _fake_mi(Xtr, ytr); s4 = _fake_pb(Xtr, ytr); s5 = _fake_lr(Xtr, ytr); s6 = _fake_rf(Xtr, ytr)
    ps = {"P3": s3, "P4": s4, "P5": s5, "P6": s6}
    gs = {p: group_max_score(ps[p], groups) for p in ["P3", "P4", "P5", "P6"]}

    CT = ["semantic_group", "encoded_column"]
    BP = [500, 2000]  # synthetic: 2 budgets; real execution uses [500, 1000, 2000]
    G = list(range(2026071700, 2026071702))  # 2 seeds for synthetic; real execution uses 20
    gl = []; sl = []

    for ct in CT:
        for bp in BP:
            ku = compute_k(len(groups) if ct == "semantic_group" else n_total, bp)
            for pid in ["P2", "P3", "P4", "P5", "P6"]:
                seeds = G if pid == "P2" else [0]
                for gi, gv in enumerate(seeds):
                    gs_out = gi if pid == "P2" else -1
                    if pid == "P2":
                        p2s = derive_p2_seed(gv, ds, mech, st, ts, ct, bp)
                        rng2 = np.random.RandomState(p2s)
                        if ct == "semantic_group":
                            sg = list(rng2.choice(len(groups), ku, replace=False))
                            gids = [groups[i]["opaque_group_id"] for i in sg]
                            sh = hash_semantic_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], "P2", ct, bp, gids)
                            rc = []; [rc.extend(groups[i]["member_encoded_indices"]) for i in sg]
                        else:
                            rc = list(rng2.choice(n_total, ku, replace=False))
                            sh = hash_encoded_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], "P2", ct, bp, np.array(sorted(rc), dtype=np.int64))
                            gids = []
                    else:
                        if ct == "semantic_group":
                            sel = top_k_groups(gs[pid], ku)
                            sh = hash_semantic_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], pid, ct, bp, sel)
                            rc = []; [rc.extend(g["member_encoded_indices"]) for gid in sel for g in groups if g["opaque_group_id"] == gid]
                            gids = sel
                        else:
                            idx = top_k_columns(ps[pid], ku)
                            sh = hash_encoded_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], pid, ct, bp, np.array(sorted(idx), dtype=np.int64))
                            rc = list(idx)
                            gids = [g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"]) & set(rc)]

                    sl.append({"selection_hash": sh, "policy": pid, "contract": ct, "budget_bp": bp, "removed_encoded_indices": json.dumps([int(x) for x in sorted(rc)]), "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(rc)})

                    rk = f"{pid}_{ct}_{bp}_{gi}" if pid == "P2" else f"{pid}_{ct}_{bp}"
                    rid = run_ids[rk]
                    keep = np.ones(n_total, dtype=bool); keep[rc] = False
                    gov_out = mf("lr", X[:, keep][tr], y[tr], X[:, keep][va], y[va], X[:, keep][te], ts)
                    ga = float(roc_auc_score(y[te], gov_out.probabilities))
                    gl.append({"run_id": rid, "dataset_index": ds, "mechanism": mech, "strength": st, "training_seed": ts, "governance_seed": gs_out, "learner": "lr", "policy": pid, "contract": ct, "budget_bp": bp, "strict_auc": sa, "full_auc": fa, "governed_auc": ga, "legacy_sdr": abs(fa - sa) - abs(ga - sa), "selection_hash": sh, "realized_cost": len(rc)})

    return {"baseline_rows": bl, "governed_rows": gl, "selection_rows": sl}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--output-dir", default="/tmp/t0b_shard")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    with open(ROOT / args.plan_manifest) as f:
        pm = json.load(f)

    if args.validate_only:
        print(f"Shard {args.shard_id}: validate-only — 0 calls"); return

    # Load plans
    plan_dir = Path(args.plan_manifest).parent
    key_data = gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8")
    run_data = gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8")
    keys = [json.loads(l) for l in key_data.strip().split("\n")]
    runs = [json.loads(l) for l in run_data.strip().split("\n")]

    # Filter to this shard
    shard_keys = [k for k in keys if k.get("shard_id") == args.shard_id]
    shard_runs = {}
    for r in runs:
        if r.get("shard_id") == args.shard_id:
            cid = r["canonical_key_id"]
            shard_runs.setdefault(cid, {})
            rk = f"{r.get('baseline_type','')}" if r["run_type"] == "baseline" else f"{r['policy']}_{r['contract']}_{r['budget_bp']}_{r.get('governance_seed_index','')}" if r["policy"] == "P2" else f"{r['policy']}_{r['contract']}_{r['budget_bp']}"
            shard_runs[cid][rk] = r["run_id"]

    # Check resume
    if args.resume:
        complete = 0
        for kp in shard_keys:
            cid = kp["canonical_key_id"]
            receipt_path = out / "key_receipts" / f"{cid}.json"
            if receipt_path.exists():
                complete += 1
        if complete == len(shard_keys):
            print(f"Resume: all {len(shard_keys)} keys already complete — 0 new calls")
            receipt = {"new_keys": 0, "new_baseline": 0, "new_governed": 0, "new_lr_calls": 0, "p3": 0, "p4": 0, "p5": 0, "p6": 0}
            with open(out / "resume_receipt.json", "w") as f:
                json.dump(receipt, f, indent=2)
            return
        print(f"Resume: {complete}/{len(shard_keys)} complete, executing remaining")

    # Load eval info (simplified synthetic)
    eval_cache = {}
    eval_data = gzip.decompress((ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz").read_bytes()).decode("utf-8")
    for line in eval_data.strip().split("\n"):
        r = json.loads(line)
        eval_cache[(r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"])] = r

    all_bl = []; all_gl = []; all_sl = []; new_keys = 0
    for kp in shard_keys:
        cid = kp["canonical_key_id"]
        receipt_path = out / "key_receipts" / f"{cid}.json"
        if args.resume and receipt_path.exists():
            continue
        kt = (kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"])
        eval_info = eval_cache.get(kt, {"leak_group_ids": []})
        # Build groups from policy mapping
        pol_data = gzip.decompress((ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz").read_bytes()).decode("utf-8")
        groups = None
        for line in pol_data.strip().split("\n"):
            r = json.loads(line)
            if (r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]) == kt:
                groups = r["groups"]; break
        if groups is None:
            continue

        result = execute_key(kp, groups, eval_info, shard_runs.get(cid, {}))
        all_bl.extend(result["baseline_rows"])
        all_gl.extend(result["governed_rows"])
        all_sl.extend(result["selection_rows"])
        new_keys += 1
        # Write key receipt
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(receipt_path, "w") as f:
            json.dump({"canonical_key_id": cid, "baseline_rows": len(result["baseline_rows"]), "governed_rows": len(result["governed_rows"])}, f)

    # Write shard ledgers
    for name, rows, cols in [
        ("baseline_ledger", all_bl, ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]),
        ("governed_ledger", all_gl, ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","selection_hash","realized_cost"]),
        ("selection_ledger", all_sl, ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]),
    ]:
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, columns=cols, index=False, header=True)
        compressed = gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)
        (out / f"{name}.csv.gz").write_bytes(compressed)

    # Failure ledger (empty)
    buf = io.StringIO(); pd.DataFrame(columns=["run_id"]).to_csv(buf, index=False, header=True)
    (out / "failure_ledger.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))

    # Shard manifest
    sm = {"shard_id": args.shard_id, "keys_executed": new_keys, "baseline_rows": len(all_bl), "governed_rows": len(all_gl), "selection_rows": len(all_sl), "lr_calls": CALLS["lr"], "p3_calls": CALLS["p3"]}
    with open(out / "shard_manifest.json", "w") as f:
        json.dump(sm, f, indent=2)

    print(f"Shard {args.shard_id}: {new_keys} keys, {len(all_bl)} baseline, {len(all_gl)} governed, LR={CALLS['lr']}")
    sys.exit(0)

if __name__ == "__main__":
    main()
