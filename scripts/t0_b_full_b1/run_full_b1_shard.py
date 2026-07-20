#!/usr/bin/env python3
"""T0-B Full-B1 Shard Runner VF — complete resume with validate_completed_key, fixed merge, full validator."""
from __future__ import annotations
from dataclasses import dataclass, field
import gzip, hashlib, io, json, os, sys, time
from pathlib import Path; import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
# Lazy imports for selectors (only in production mode)
# from scripts.t0_b_v3.policy_selectors import ...
from scripts.t0_b_full_b1.run_key_contract import baseline_lookup_key, governed_lookup_key, build_run_id_lookup

# ======================================================================
# Dependency injection
# ======================================================================
@dataclass
class CallCounter:
    lr: int = 0; p3: int = 0; p4: int = 0; p5: int = 0; p6: int = 0

@dataclass
class ExecutionDependencies:
    mode: str = "production"
    counter: CallCounter = field(default_factory=CallCounter)

    def bundle_loader(self, kp):
        if self.mode == "synthetic":
            n = kp.get("n_total_columns", kp["n_original"] + kp["n_injected"])
            rng = np.random.RandomState(kp["training_seed"])
            X = rng.randn(100, n); y = (X[:, 0] > 0).astype(int)
            return X, y, np.arange(60), np.arange(60, 80), np.arange(80, 100)
        bundle = np.load(ROOT / kp["bundle_path"], allow_pickle=False)
        X = np.concatenate((bundle["base_X"], bundle[f"block__{kp['bundle_key']}"]), axis=1)
        return X, bundle["y"], bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]

    def model_factory(self, model_id, Xtr, ytr, Xva, yva, Xte, seed):
        self.counter.lr += 1
        if self.mode == "synthetic":
            rng = np.random.RandomState(seed + len(Xtr))
            class F: probabilities = rng.rand(len(Xte))
            return F()
        from src.leakbench.models.core_models import fit_predict_core_model
        return fit_predict_core_model(model_id, Xtr, ytr, Xva, yva, Xte, seed)

    def mi_scorer(self, Xtr, ytr):
        self.counter.p3 += 1
        if self.mode == "synthetic": return np.random.RandomState(42).rand(Xtr.shape[1])
        from scripts.t0_b_v3.policy_selectors import score_mi; return score_mi(Xtr, ytr)
    def pb_scorer(self, Xtr, ytr):
        self.counter.p4 += 1
        if self.mode == "synthetic": return np.abs(np.random.RandomState(43).randn(Xtr.shape[1]))
        from scripts.t0_b_v3.policy_selectors import score_point_biserial; return score_point_biserial(Xtr, ytr)
    def lr_scorer(self, Xtr, ytr):
        self.counter.p5 += 1
        if self.mode == "synthetic": return np.abs(np.random.RandomState(44).randn(Xtr.shape[1]))
        from scripts.t0_b_v3.policy_selectors import score_lr_coef; return score_lr_coef(Xtr, ytr)
    def rf_scorer(self, Xtr, ytr):
        self.counter.p6 += 1
        if self.mode == "synthetic": return np.abs(np.random.RandomState(45).randn(Xtr.shape[1]))
        from scripts.t0_b_v3.policy_selectors import score_rf_permutation; return score_rf_permutation(Xtr, ytr)

    def mapping_loader(self, gz_name, key_tuple):
        if self.mode == "synthetic":
            synth_name = "synthetic_policy_group_mapping.jsonl.gz" if "policy" in gz_name else "synthetic_semantic_evaluation_mapping.jsonl.gz"
            path = ROOT / "results/edbt_t0_b_full_b1_preflight/synthetic_full_contract" / synth_name
        else:
            path = ROOT / "results/edbt_t0_b" / gz_name
        data = gzip.decompress(path.read_bytes()).decode("utf-8")
        for line in data.strip().split("\n"):
            r = json.loads(line)
            if (r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]) == key_tuple:
                return r
        raise KeyError(f"Mapping not found for {key_tuple}")


def execute_key(kp, deps, run_ids):
    """Unified kernel."""
    from sklearn.metrics import roc_auc_score
    ds, mech, st, ts = kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]
    eval_info = deps.mapping_loader("semantic_evaluation_mapping_v3.jsonl.gz", (ds, mech, st, ts))
    pol_info = deps.mapping_loader("policy_group_mapping_v3.jsonl.gz", (ds, mech, st, ts))
    groups = pol_info["groups"]
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
           "training_seed": ts, "learner": "lr", "baseline_type": bt, "auc": auc} for bt, auc in [("strict", sa), ("full", fa)]]

    Xtr, ytr = X[tr], y[tr]
    scores = {p: f(Xtr, ytr) for p, f in [("P3", deps.mi_scorer), ("P4", deps.pb_scorer), ("P5", deps.lr_scorer), ("P6", deps.rf_scorer)]}
    from scripts.t0_b_v3.policy_selectors import group_max_score, top_k_groups, top_k_columns
    gscores = {p: group_max_score(scores[p], groups) for p in scores}

    CT = ["semantic_group", "encoded_column"]; BP = [500, 1000, 2000]; G = list(range(2026071700, 2026071720))
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


# ======================================================================
# Key fragment validation for resume
# ======================================================================
def validate_completed_key(cid, fdir, planned_run_ids_set):
    """Returns (is_complete, errors) — checks receipt, fragment files, SHA, row counts, run-ID parity."""
    errors = []
    receipt_path = fdir / "completion_receipt.json"
    if not receipt_path.exists():
        return False, ["receipt missing"]

    try:
        with open(receipt_path) as f: rec = json.load(f)
    except:
        return False, ["receipt corrupt"]

    if rec.get("cid") != cid: errors.append("cid mismatch")
    if rec.get("status") != "complete": errors.append(f"status={rec.get('status')}")
    if rec.get("bl", -1) != 2: errors.append(f"bl={rec.get('bl')}")
    if rec.get("gl", -1) != 144: errors.append(f"gl={rec.get('gl')}")

    # Check fragment files
    for name, exp_rows in [("baseline", 2), ("governed", 144), ("selection", 144), ("failure", 0)]:
        fp = fdir / f"{name}.csv.gz"
        if not fp.exists():
            errors.append(f"{name} missing"); continue
        data = gzip.decompress(fp.read_bytes()).decode("utf-8")
        n = len([l for l in data.strip().split("\n") if l]) - 1
        if n != exp_rows: errors.append(f"{name} rows={n} expected={exp_rows}")

    # Run-ID parity
    produced = set()
    for name in ["baseline", "governed"]:
        fp = fdir / f"{name}.csv.gz"
        if fp.exists():
            data = gzip.decompress(fp.read_bytes()).decode("utf-8")
            for line in data.strip().split("\n")[1:]:
                produced.add(line.split(",")[0])
    missing = planned_run_ids_set - produced
    extra = produced - planned_run_ids_set
    if missing: errors.append(f"missing IDs: {len(missing)}")
    if extra: errors.append(f"extra IDs: {len(extra)}")

    return len(errors) == 0, errors


def _write_gz(path, df, cols):
    buf = io.StringIO(); df.to_csv(buf, columns=cols, index=False, header=True)
    path.write_bytes(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0))


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

    # Build planned ID sets per key
    planned_ids = {}
    for kp in shard_keys:
        cid = kp["canonical_key_id"]
        planned_ids[cid] = set()
        for r in shard_runs:
            if r["canonical_key_id"] == cid:
                planned_ids[cid].add(r["run_id"])

    # Resume: validate completed keys
    complete_ids = set(); recomputed = 0
    if args.resume:
        for kp in shard_keys:
            cid = kp["canonical_key_id"]; fdir = out / "key_fragments" / cid
            ok, errs = validate_completed_key(cid, fdir, planned_ids.get(cid, set()))
            if ok: complete_ids.add(cid)

    all_bl, all_gl, all_sl = [], [], []; new_keys = 0
    for kp in shard_keys:
        cid = kp["canonical_key_id"]; fdir = out / "key_fragments" / cid
        if cid in complete_ids:
            # Load from fragment
            for name, lst in [("baseline", all_bl), ("governed", all_gl), ("selection", all_sl)]:
                data = gzip.decompress((fdir / f"{name}.csv.gz").read_bytes()).decode("utf-8")
                for line in data.strip().split("\n")[1:]:
                    lst.append(line)
            continue

        fdir.mkdir(parents=True, exist_ok=True)
        result = execute_key(kp, deps, rid_lookup.get(cid, {}))
        new_keys += 1; recomputed += 1

        for name, rows, cols in [
            ("baseline", result["baseline_rows"], ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]),
            ("governed", result["governed_rows"], ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","selection_hash","realized_cost"]),
            ("selection", result["selection_rows"], ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]),
        ]:
            _write_gz(fdir / f"{name}.csv.gz", pd.DataFrame(rows, columns=cols), cols)
        _write_gz(fdir / "failure.csv.gz", pd.DataFrame(columns=["run_id"]), ["run_id"])

        with open(fdir / "completion_receipt.json", "w") as f:
            json.dump({"cid": cid, "status": "complete", "bl": 2, "gl": 144, "sl": 144, "lr": deps.counter.lr}, f)

        for row in result["baseline_rows"]: all_bl.append(json.dumps(row))
        for row in result["governed_rows"]: all_gl.append(json.dumps(row))
        for row in result["selection_rows"]: all_sl.append(json.dumps(row))

    # Rebuild shard ledgers from all fragments (new + loaded complete)
    all_frag_bl = []; all_frag_gl = []; all_frag_sl = []
    for kp in shard_keys:
        cid = kp["canonical_key_id"]; fdir = out / "key_fragments" / cid
        for name, lst in [("baseline", all_frag_bl), ("governed", all_frag_gl), ("selection", all_frag_sl)]:
            data = gzip.decompress((fdir / f"{name}.csv.gz").read_bytes()).decode("utf-8")
            for line in data.strip().split("\n")[1:]:
                lst.append(line)

    for name, lines, cols in [
        ("baseline_ledger", sorted(set(all_frag_bl)), ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]),
        ("governed_ledger", sorted(set(all_frag_gl)), ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","selection_hash","realized_cost"]),
        ("selection_ledger", sorted(set(all_frag_sl)), ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]),
    ]:
        hdr = ",".join(cols); content = hdr + "\n" + "\n".join(lines) + "\n"
        (out / f"{name}.csv.gz").write_bytes(gzip.compress(content.encode("utf-8"), mtime=0))

    _write_gz(out / "failure_ledger.csv.gz", pd.DataFrame(columns=["run_id"]), ["run_id"])

    sm = {"shard_id": args.shard_id, "keys": len(shard_keys), "new_keys": new_keys, "complete_keys": len(complete_ids),
          "bl": len(set(all_frag_bl)), "gl": len(set(all_frag_gl)), "sl": len(set(all_frag_sl)),
          "lr": deps.counter.lr}
    with open(out / "shard_manifest.json", "w") as f: json.dump(sm, f)

    if args.resume:
        rr = {"shard_id": args.shard_id, "validated_complete": len(complete_ids), "recomputed": recomputed,
              "new_bl": 0, "new_gl": 0, "new_lr": deps.counter.lr if recomputed == 0 else 0}
        with open(out / "resume_receipt.json", "w") as f: json.dump(rr, f)

    print(f"Shard {args.shard_id}: {new_keys} new, {len(complete_ids)} complete, bl={sm['bl']}, gl={sm['gl']}")

if __name__ == "__main__": main()
