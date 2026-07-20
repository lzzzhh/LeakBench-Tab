#!/usr/bin/env python3
"""T0-B Full-B1 Shard Runner VF — unified kernel: production/synthetic via dependency injection."""
from __future__ import annotations
import gzip, hashlib, io, json, os, sys, time
from pathlib import Path; import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
from scripts.t0_b_v3.policy_selectors import score_mi, score_point_biserial, score_lr_coef, score_rf_permutation, group_max_score, top_k_groups, top_k_columns

CALLS = {"lr": 0, "p3": 0, "p4": 0, "p5": 0, "p6": 0}

# --- Dependency injection ---
_real_bundle = True  # default: production

def _load_bundle(kp):
    if _real_bundle:
        bundle = np.load(ROOT / kp["bundle_path"], allow_pickle=False)
        X = np.concatenate((bundle["base_X"], bundle[f"block__{kp['bundle_key']}"]), axis=1)
        return X, bundle["y"], bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]
    else:
        n_total = kp["n_original"] + kp["n_injected"]
        rng = np.random.RandomState(kp["training_seed"])
        X = rng.randn(100, n_total); y = (X[:, 0] > 0).astype(int)
        return X, y, np.arange(60), np.arange(60, 80), np.arange(80, 100)

def _mf(model_id, Xtr, ytr, Xva, yva, Xte, seed):
    if _real_bundle:
        from src.leakbench.models.core_models import fit_predict_core_model
        CALLS["lr"] += 1; return fit_predict_core_model(model_id, Xtr, ytr, Xva, yva, Xte, seed)
    CALLS["lr"] += 1; rng = np.random.RandomState(seed + len(Xtr))
    class F: probabilities = rng.rand(len(Xte))
    return F()

def _mi(Xtr, ytr):
    if _real_bundle: CALLS["p3"] += 1; return score_mi(Xtr, ytr)
    CALLS["p3"] += 1; return np.random.RandomState(42).rand(Xtr.shape[1])

def _pb(Xtr, ytr):
    if _real_bundle: CALLS["p4"] += 1; return score_point_biserial(Xtr, ytr)
    CALLS["p4"] += 1; return np.abs(np.random.RandomState(43).randn(Xtr.shape[1]))

def _lr(Xtr, ytr):
    if _real_bundle: CALLS["p5"] += 1; return score_lr_coef(Xtr, ytr)
    CALLS["p5"] += 1; return np.abs(np.random.RandomState(44).randn(Xtr.shape[1]))

def _rf(Xtr, ytr):
    if _real_bundle: CALLS["p6"] += 1; return score_rf_permutation(Xtr, ytr)
    CALLS["p6"] += 1; return np.abs(np.random.RandomState(45).randn(Xtr.shape[1]))


def execute_key(kp, groups, eval_info, run_ids):
    """Unified kernel: only executes policies with run IDs in the plan."""
    ds, mech, st, ts = kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]
    X, y, tr, va, te = _load_bundle(kp)
    n_total = X.shape[1]

    # Determine required policies from run_ids keys
    required_policies = set()
    for rk in run_ids:
        if rk.startswith("P"):
            required_policies.add(rk.split("_")[0])
    # If no explicit policy keys, execute all (backward compat)
    if not required_policies:
        required_policies = {"P2", "P3", "P4", "P5", "P6"}

    leak_idx = set()
    for gid in eval_info.get("leak_group_ids", []):
        for g in groups:
            if g["opaque_group_id"] == gid: leak_idx.update(g["member_encoded_indices"])
    leak_mask = np.array([i in leak_idx for i in range(n_total)])

    # Baseline
    Xs = X[:, ~leak_mask]
    s1 = _mf("lr", Xs[tr], y[tr], Xs[va], y[va], Xs[te], ts)
    s2 = _mf("lr", X[tr], y[tr], X[va], y[va], X[te], ts)
    sa = float(roc_auc_score(y[te], s1.probabilities)); fa = float(roc_auc_score(y[te], s2.probabilities))

    bl = [{"run_id": run_ids[fk], "dataset_index": ds, "mechanism": mech, "strength": st, "training_seed": ts, "learner": "lr", "baseline_type": fk, "auc": v}
          for fk, v in [("strict", sa), ("full", fa)]]

    # Ranking
    Xtr, ytr = X[tr], y[tr]
    scores = {}; gscores = {}
    if "P3" in required_policies: scores["P3"] = _mi(Xtr, ytr)
    if "P4" in required_policies: scores["P4"] = _pb(Xtr, ytr)
    if "P5" in required_policies: scores["P5"] = _lr(Xtr, ytr)
    if "P6" in required_policies: scores["P6"] = _rf(Xtr, ytr)
    for p in scores: gscores[p] = group_max_score(scores[p], groups)

    CT = ["semantic_group", "encoded_column"]; BP = [500, 1000, 2000]
    G = list(range(2026071700, 2026071720))
    gl = []; sl = []

    for ct in CT:
        for bp in BP:
            ku = compute_k(len(groups) if ct == "semantic_group" else n_total, bp)
            for pid in sorted(required_policies):
                seeds = G if pid == "P2" else [0]
                for gi, gv in enumerate(seeds):
                    gs_out = gi if pid == "P2" else -1
                    if pid == "P2":
                        p2s = derive_p2_seed(gv, ds, mech, st, ts, ct, bp)
                        rng2 = np.random.RandomState(p2s)
                        if ct == "semantic_group":
                            sg = list(rng2.choice(len(groups), ku, replace=False))
                            gids = [groups[i]["opaque_group_id"] for i in sg]; rc = []
                            [rc.extend(groups[i]["member_encoded_indices"]) for i in sg]
                            sh = hash_semantic_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], "P2", ct, bp, gids)
                        else:
                            rc = list(rng2.choice(n_total, ku, replace=False))
                            sh = hash_encoded_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], "P2", ct, bp, np.array(sorted(rc), dtype=np.int64))
                            gids = []
                    else:
                        if ct == "semantic_group":
                            sel = top_k_groups(gscores[pid], ku); rc = []
                            [rc.extend(g["member_encoded_indices"]) for gid in sel for g in groups if g["opaque_group_id"] == gid]
                            gids = sel; sh = hash_semantic_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], pid, ct, bp, sel)
                        else:
                            idx = top_k_columns(scores[pid], ku); rc = list(idx)
                            gids = [g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"]) & set(rc)]
                            sh = hash_encoded_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], pid, ct, bp, np.array(sorted(idx), dtype=np.int64))

                    sl.append({"selection_hash": sh, "policy": pid, "contract": ct, "budget_bp": bp, "removed_encoded_indices": json.dumps([int(x) for x in sorted(rc)]), "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(rc)})

                    rk = f"{pid}_{ct}_{bp}_{gi}" if pid == "P2" else f"{pid}_{ct}_{bp}_0"
                    keep = np.ones(n_total, dtype=bool); keep[rc] = False
                    gov = _mf("lr", X[:, keep][tr], y[tr], X[:, keep][va], y[va], X[:, keep][te], ts)
                    ga = float(roc_auc_score(y[te], gov.probabilities))
                    gl.append({"run_id": run_ids[rk], "dataset_index": ds, "mechanism": mech, "strength": st, "training_seed": ts, "governance_seed": gs_out, "learner": "lr", "policy": pid, "contract": ct, "budget_bp": bp, "strict_auc": sa, "full_auc": fa, "governed_auc": ga, "legacy_sdr": abs(fa - sa) - abs(ga - sa), "selection_hash": sh, "realized_cost": len(rc)})

    return {"baseline_rows": bl, "governed_rows": gl, "selection_rows": sl}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True); ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--output-dir", default="/tmp/t0b_shard"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    global _real_bundle; _real_bundle = not args.synthetic

    with open(ROOT / args.plan_manifest) as f: pm = json.load(f)
    if args.validate_only:
        print("validate-only — 0 calls"); return

    plan_dir = Path(args.plan_manifest).parent
    keys = [json.loads(l) for l in gzip.decompress((plan_dir/"full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
    runs = [json.loads(l) for l in gzip.decompress((plan_dir/"full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]

    # Build run-ID map
    rid_map = {}; planned_ids = set()
    for r in runs:
        if r.get("shard_id") == args.shard_id:
            cid = r["canonical_key_id"]; rid_map.setdefault(cid, {})
            rk = r.get("baseline_type", "") or f"{r['policy']}_{r['contract']}_{r['budget_bp']}_{r.get('governance_seed_index','')}"
            rid_map[cid][rk] = r["run_id"]; planned_ids.add(r["run_id"])

    # Load eval + groups
    eval_map = {}; group_map = {}
    for gz_name, target in [("semantic_evaluation_mapping_v3.jsonl.gz", eval_map), ("policy_group_mapping_v3.jsonl.gz", group_map)]:
        for line in gzip.decompress((ROOT/"results/edbt_t0_b"/gz_name).read_bytes()).decode("utf-8").strip().split("\n"):
            r = json.loads(line); target[(r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"])] = r

    # Check resume
    if args.resume:
        complete = 0
        for kp in keys:
            if kp.get("shard_id") == args.shard_id:
                rp = out / "key_receipts" / f"{kp['canonical_key_id']}.json"
                if rp.exists(): complete += 1
        total = sum(1 for k in keys if k.get("shard_id") == args.shard_id)
        if complete == total:
            print(f"Resume: all {total} keys complete — 0 new calls"); return
        print(f"Resume: {complete}/{total} complete")

    all_bl = []; all_gl = []; all_sl = []; new_keys = 0; produced_ids = set()
    for kp in keys:
        if kp.get("shard_id") != args.shard_id: continue
        cid = kp["canonical_key_id"]
        rp = out / "key_receipts" / f"{cid}.json"
        if args.resume and rp.exists(): continue
        kt = (kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"])
        result = execute_key(kp, group_map.get(kt, {}).get("groups", []), eval_map.get(kt, {"leak_group_ids": []}), rid_map.get(cid, {}))
        all_bl.extend(result["baseline_rows"]); all_gl.extend(result["governed_rows"]); all_sl.extend(result["selection_rows"])
        new_keys += 1
        for row in result["baseline_rows"] + result["governed_rows"]: produced_ids.add(row["run_id"])
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w") as f: json.dump({"cid": cid, "bl": len(result["baseline_rows"]), "gl": len(result["governed_rows"])}, f)

    # Verify ID parity
    missing = planned_ids - produced_ids; extra = produced_ids - planned_ids
    if missing: print(f"WARNING: {len(missing)} planned IDs not produced"); sys.exit(1)
    if extra: print(f"WARNING: {len(extra)} extra IDs produced")

    # Write ledgers
    for name, rows, cols in [
        ("baseline_ledger", all_bl, ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]),
        ("governed_ledger", all_gl, ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","selection_hash","realized_cost"]),
        ("selection_ledger", all_sl, ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]),
        ("failure_ledger", [{"run_id": "none"}], ["run_id"]),
    ]:
        df = pd.DataFrame(rows, columns=cols)
        buf = io.StringIO(); df.to_csv(buf, columns=cols, index=False, header=True)
        (out / f"{name}.csv.gz").write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))

    sm = {"shard_id": args.shard_id, "keys": new_keys, "bl": len(all_bl), "gl": len(all_gl), "sl": len(all_sl), "lr": CALLS["lr"]}
    with open(out / "shard_manifest.json", "w") as f: json.dump(sm, f)
    print(f"Shard {args.shard_id}: {new_keys} keys, {len(all_bl)} bl, {len(all_gl)} gl, LR={CALLS['lr']}")

if __name__ == "__main__": main()
