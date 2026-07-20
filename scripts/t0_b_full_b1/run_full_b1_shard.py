#!/usr/bin/env python3
"""T0-B Full-B1 Shard Runner VF — full contract, key fragments, atomic I/O, resume."""
from __future__ import annotations
from dataclasses import dataclass, field
import gzip, hashlib, io, json, os, sys, time
from pathlib import Path; import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
from scripts.t0_b_v3.policy_selectors import score_mi, score_point_biserial, score_lr_coef, score_rf_permutation, group_max_score, top_k_groups, top_k_columns
from scripts.t0_b_full_b1.run_key_contract import baseline_lookup_key, governed_lookup_key, build_run_id_lookup, validate_lookup_complete

# ======================================================================
# Dependency injection
# ======================================================================
@dataclass
class ExecutionDependencies:
    mode: str = "production"  # "production" | "synthetic"
    call_counter: dict = field(default_factory=lambda: {"lr": 0, "p3": 0, "p4": 0, "p5": 0, "p6": 0})

    def bundle_loader(self, kp):
        if self.mode == "synthetic":
            # Derive dimensions from groups, not key plan (groups match real mapping)
            n_tot = kp.get("n_total_columns", kp["n_original"] + kp["n_injected"])
            rng = np.random.RandomState(kp["training_seed"])
            X = rng.randn(100, n_tot); y = (X[:, 0] > 0).astype(int)
            return X, y, np.arange(60), np.arange(60, 80), np.arange(80, 100)
        bundle = np.load(ROOT / kp["bundle_path"], allow_pickle=False)
        X = np.concatenate((bundle["base_X"], bundle[f"block__{kp['bundle_key']}"]), axis=1)
        return X, bundle["y"], bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]

    def model_factory(self, model_id, Xtr, ytr, Xva, yva, Xte, seed):
        self.call_counter["lr"] += 1
        if self.mode == "synthetic":
            rng = np.random.RandomState(seed + len(Xtr))
            class F: probabilities = rng.rand(len(Xte))
            return F()
        from src.leakbench.models.core_models import fit_predict_core_model
        return fit_predict_core_model(model_id, Xtr, ytr, Xva, yva, Xte, seed)

    def mi_scorer(self, Xtr, ytr):
        self.call_counter["p3"] += 1
        if self.mode == "synthetic": return np.random.RandomState(42).rand(Xtr.shape[1])
        return score_mi(Xtr, ytr)

    def pb_scorer(self, Xtr, ytr):
        self.call_counter["p4"] += 1
        if self.mode == "synthetic": return np.abs(np.random.RandomState(43).randn(Xtr.shape[1]))
        return score_point_biserial(Xtr, ytr)

    def lr_scorer(self, Xtr, ytr):
        self.call_counter["p5"] += 1
        if self.mode == "synthetic": return np.abs(np.random.RandomState(44).randn(Xtr.shape[1]))
        return score_lr_coef(Xtr, ytr)

    def rf_scorer(self, Xtr, ytr):
        self.call_counter["p6"] += 1
        if self.mode == "synthetic": return np.abs(np.random.RandomState(45).randn(Xtr.shape[1]))
        return score_rf_permutation(Xtr, ytr)

    def mapping_loader(self, gz_name, key_tuple):
        data = gzip.decompress((ROOT / "results/edbt_t0_b" / gz_name).read_bytes()).decode("utf-8")
        for line in data.strip().split("\n"):
            r = json.loads(line)
            if (r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]) == key_tuple:
                return r
        raise KeyError(f"Mapping not found for {key_tuple} in {gz_name}")


def execute_key(kp, deps, run_ids):
    """Unified kernel: one canonical key."""
    ds, mech, st, ts = kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]

    eval_info = deps.mapping_loader("semantic_evaluation_mapping_v3.jsonl.gz", (ds, mech, st, ts))
    pol_info = deps.mapping_loader("policy_group_mapping_v3.jsonl.gz", (ds, mech, st, ts))
    groups = pol_info["groups"]
    # Patch n_total from actual groups for synthetic mode
    kp["n_total_columns"] = sum(g["group_size"] for g in groups)

    X, y, tr, va, te = deps.bundle_loader(kp)
    n_total = X.shape[1]

    leak_idx = set()
    for gid in eval_info.get("leak_group_ids", []):
        for g in groups:
            if g["opaque_group_id"] == gid: leak_idx.update(g["member_encoded_indices"])
    leak_mask = np.array([i in leak_idx for i in range(n_total)])

    mf = deps.model_factory
    Xs = X[:, ~leak_mask]
    s1 = mf("lr", Xs[tr], y[tr], Xs[va], y[va], Xs[te], ts)
    s2 = mf("lr", X[tr], y[tr], X[va], y[va], X[te], ts)
    sa = float(roc_auc_score(y[te], s1.probabilities)); fa = float(roc_auc_score(y[te], s2.probabilities))

    bl = [{"run_id": run_ids[baseline_lookup_key(bt)], "dataset_index": ds, "mechanism": mech, "strength": st,
           "training_seed": ts, "learner": "lr", "baseline_type": bt, "auc": auc}
          for bt, auc in [("strict", sa), ("full", fa)]]

    Xtr, ytr = X[tr], y[tr]
    scores = {p: f(Xtr, ytr) for p, f in [("P3", deps.mi_scorer), ("P4", deps.pb_scorer), ("P5", deps.lr_scorer), ("P6", deps.rf_scorer)]}
    gscores = {p: group_max_score(scores[p], groups) for p in scores}

    CT = ["semantic_group", "encoded_column"]; BP = [500, 1000, 2000]
    G = list(range(2026071700, 2026071720))
    gl = []; sl = []

    for ct in CT:
        for bp in BP:
            ku = compute_k(len(groups) if ct == "semantic_group" else n_total, bp)
            for pid in ["P2", "P3", "P4", "P5", "P6"]:
                seeds = range(20) if pid == "P2" else [-1]
                for gi in seeds:
                    gs_out = gi if pid == "P2" else -1
                    if pid == "P2":
                        p2s = derive_p2_seed(G[gi], ds, mech, st, ts, ct, bp)
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
                            gids = sel
                            sh = hash_semantic_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], pid, ct, bp, sel)
                        else:
                            idx = top_k_columns(scores[pid], ku); rc = list(idx)
                            gids = [g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"]) & set(rc)]
                            sh = hash_encoded_selection(ds, mech, st, ts, kp["bundle_key"], kp["bundle_sha256"], pid, ct, bp, np.array(sorted(idx), dtype=np.int64))

                    sl.append({"selection_hash": sh, "policy": pid, "contract": ct, "budget_bp": bp,
                               "removed_encoded_indices": json.dumps([int(x) for x in sorted(rc)]),
                               "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(rc)})

                    lk = governed_lookup_key(pid, ct, bp, gi)
                    keep = np.ones(n_total, dtype=bool); keep[rc] = False
                    gov = mf("lr", X[:, keep][tr], y[tr], X[:, keep][va], y[va], X[:, keep][te], ts)
                    ga = float(roc_auc_score(y[te], gov.probabilities))
                    gl.append({"run_id": run_ids[lk], "dataset_index": ds, "mechanism": mech, "strength": st,
                               "training_seed": ts, "governance_seed": gs_out, "learner": "lr", "policy": pid,
                               "contract": ct, "budget_bp": bp, "strict_auc": sa, "full_auc": fa,
                               "governed_auc": ga, "legacy_sdr": abs(fa - sa) - abs(ga - sa),
                               "selection_hash": sh, "realized_cost": len(rc)})

    return {"baseline_rows": bl, "governed_rows": gl, "selection_rows": sl}


def _write_gz(path, df, cols):
    buf = io.StringIO(); df.to_csv(buf, columns=cols, index=False, header=True)
    path.write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True); ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--output-dir", default="/tmp/t0b_shard"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    deps = ExecutionDependencies(mode="synthetic" if args.synthetic else "production")

    with open(ROOT / args.plan_manifest) as f: pm = json.load(f)
    if args.validate_only:
        print("validate-only — 0 calls"); return

    plan_dir = Path(args.plan_manifest).parent
    keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
    runs = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]

    shard_keys = [k for k in keys if k.get("shard_id") == args.shard_id]
    shard_runs = [r for r in runs if r.get("shard_id") == args.shard_id]
    rid_lookup = build_run_id_lookup(shard_runs)

    # Resume check
    if args.resume:
        complete = 0
        for kp in shard_keys:
            cid = kp["canonical_key_id"]
            if (out / "key_fragments" / cid / "completion_receipt.json").exists():
                complete += 1
        if complete == len(shard_keys):
            print(f"Resume: all {len(shard_keys)} keys complete — 0 new calls"); return
        print(f"Resume: {complete}/{len(shard_keys)} complete")

    all_bl = []; all_gl = []; all_sl = []; new_keys = 0
    for kp in shard_keys:
        cid = kp["canonical_key_id"]
        fdir = out / "key_fragments" / cid; fdir.mkdir(parents=True, exist_ok=True)
        if args.resume and (fdir / "completion_receipt.json").exists():
            # Add existing rows
            for ledger, lst in [("baseline", all_bl), ("governed", all_gl), ("selection", all_sl)]:
                data = gzip.decompress((fdir / f"{ledger}.csv.gz").read_bytes()).decode("utf-8")
                for line in data.strip().split("\n")[1:]:
                    lst.append(line)
            continue

        result = execute_key(kp, deps, rid_lookup.get(cid, {}))
        new_keys += 1

        # Write key fragments
        for name, rows, cols in [
            ("baseline", result["baseline_rows"], ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]),
            ("governed", result["governed_rows"], ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","selection_hash","realized_cost"]),
            ("selection", result["selection_rows"], ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]),
            ("failure", [], ["run_id"]),
        ]:
            _write_gz(fdir / f"{name}.csv.gz", pd.DataFrame(rows, columns=cols), cols)

        with open(fdir / "completion_receipt.json", "w") as f:
            json.dump({"cid": cid, "bl": len(result["baseline_rows"]), "gl": len(result["governed_rows"]), "sl": len(result["selection_rows"]), "lr": deps.call_counter["lr"]}, f)

        for row in result["baseline_rows"] + result["governed_rows"]:
            all_bl.append(row) if "baseline_type" in row else all_gl.append(row)
        all_sl.extend(result["selection_rows"])

    # Rebuild shard ledgers from all fragments
    for name, rows, cols in [
        ("baseline_ledger", all_bl, ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]),
        ("governed_ledger", all_gl, ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","selection_hash","realized_cost"]),
        ("selection_ledger", all_sl, ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]),
    ]:
        if isinstance(rows[0], dict):
            _write_gz(out / f"{name}.csv.gz", pd.DataFrame(rows, columns=cols), cols)
        else:
            # CSV lines — merge header
            hdr = ",".join(cols)
            content = hdr + "\n" + "\n".join(sorted(set(rows))) + "\n"
            (out / f"{name}.csv.gz").write_bytes(gzip.compress(content.encode("utf-8"), mtime=0))

    _write_gz(out / "failure_ledger.csv.gz", pd.DataFrame(columns=["run_id"]), ["run_id"])

    sm = {"shard_id": args.shard_id, "keys": len(shard_keys), "new_keys": new_keys,
          "bl": len(set(row.get("run_id","") if isinstance(row,dict) else row.split(",")[0] for row in all_bl)),
          "gl": len(set(row.get("run_id","") if isinstance(row,dict) else row.split(",")[0] for row in all_gl)),
          "lr": deps.call_counter["lr"], "p3": deps.call_counter["p3"]}
    with open(out / "shard_manifest.json", "w") as f: json.dump(sm, f)
    print(f"Shard {args.shard_id}: {new_keys} new keys, bl={sm['bl']}, gl={sm['gl']}, lr={sm['lr']}")

if __name__ == "__main__": main()
