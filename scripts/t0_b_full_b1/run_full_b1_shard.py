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
from scripts.t0_b_full_b1.io_contract import (
    atomic_write_gzip_text, atomic_write_json, atomic_write_dataframe_gzip,
    exclusive_writer_lock, WriterLockError, cleanup_stale_temp_files,
)
from scripts.t0_b_full_b1.fragment_contract import (
    ProductionGuard, SyntheticCallCounter,
    build_fragment_manifest, validate_completed_key,
)

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
    """Legacy helper — delegates to atomic_write_dataframe_gzip."""
    atomic_write_dataframe_gzip(path, df, cols)


def _find_duplicates(values):
    """Return list of values that appear more than once."""
    seen = {}; dups = []
    for v in values:
        if v in seen:
            if seen[v] == 1: dups.append(v)
            seen[v] += 1
        else:
            seen[v] = 1
    return dups


def shard_keys_for_validate(args):
    """Check that the shard has planned keys (for validate-only mode)."""
    plan_dir = Path(args.plan_manifest).parent
    keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
    return [k for k in keys if k.get("shard_id") == args.shard_id]


def _audit_run_plan_for_shard(shard_runs, shard_keys):
    """Full run-plan audit: lookup universe, seed validation, schema checks."""
    from scripts.t0_b_full_b1.run_key_contract import (
        baseline_lookup_key, governed_lookup_key, expected_lookup_keys_for_key,
    )
    errors = []
    # Check run IDs unique and non-null
    all_rids = [r.get("run_id") for r in shard_runs]
    if any(rid is None or rid == "" for rid in all_rids):
        errors.append("null or empty run_id found")
    rid_dups = _find_duplicates(all_rids)
    if rid_dups:
        errors.append(f"duplicate run IDs: {len(rid_dups)}")
    # Per-key audit
    expected_keys = expected_lookup_keys_for_key()
    for kp in shard_keys:
        cid = kp["canonical_key_id"]
        key_runs = [r for r in shard_runs if r["canonical_key_id"] == cid]
        if len(key_runs) != 146:
            errors.append(f"key {cid[:12]}: expected 146 runs, got {len(key_runs)}")
            continue
        # Build lookup keys
        actual_lk = set()
        for r in key_runs:
            if r["run_type"] == "baseline":
                lk = baseline_lookup_key(r["baseline_type"])
            else:
                # Validate schema
                if r["policy"] not in ("P2","P3","P4","P5","P6"):
                    errors.append(f"key {cid[:12]}: invalid policy {r['policy']}"); continue
                if r["contract"] not in ("semantic_group","encoded_column"):
                    errors.append(f"key {cid[:12]}: invalid contract {r['contract']}"); continue
                if r["budget_bp"] not in (500,1000,2000):
                    errors.append(f"key {cid[:12]}: invalid budget {r['budget_bp']}"); continue
                gi = r["governance_seed_index"]
                if r["policy"] == "P2" and not (0 <= gi <= 19):
                    errors.append(f"key {cid[:12]}: P2 seed {gi} out of range"); continue
                if r["policy"] != "P2" and gi != -1:
                    errors.append(f"key {cid[:12]}: {r['policy']} seed {gi} must be -1"); continue
                lk = governed_lookup_key(r["policy"], r["contract"], r["budget_bp"], gi)
            if lk in actual_lk:
                errors.append(f"key {cid[:12]}: duplicate lookup key {lk}")
            actual_lk.add(lk)
        missing = expected_keys - actual_lk
        extra = actual_lk - expected_keys
        if missing:
            errors.append(f"key {cid[:12]}: {len(missing)} missing lookup keys")
        if extra:
            errors.append(f"key {cid[:12]}: {len(extra)} extra lookup keys")
    return errors


def _read_gzip_csv(path):
    """Read gzip CSV into DataFrame."""
    import pandas as pd
    return pd.read_csv(pd.io.common.BytesIO(gzip.decompress(path.read_bytes())))


def _require_unique_column(df, column, label):
    """Raise if column has duplicates."""
    dups = df[column].duplicated()
    if dups.any():
        dup_count = dups.sum()
        raise RuntimeError(f"FAIL: duplicate {label} {column}: {dup_count}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True); ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--output-dir", default="/tmp/t0b_shard"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)

    with open(ROOT / args.plan_manifest) as f: pm = json.load(f)
    plan_dir = Path(args.plan_manifest).parent

    if args.validate_only:
        # Full run-plan audit — zero model calls
        errors = []
        for fname, sha_key in [("full_b1_key_plan.jsonl.gz", "key_plan_sha256"),
                                ("full_b1_run_plan.jsonl.gz", "run_plan_sha256")]:
            fp = plan_dir / fname
            if not fp.exists():
                errors.append(f"{fname} missing"); continue
            actual = hashlib.sha256(fp.read_bytes()).hexdigest()
            if actual != pm.get(sha_key, ""):
                errors.append(f"{fname} SHA mismatch")
        keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
        runs = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
        shard_keys = [k for k in keys if k.get("shard_id") == args.shard_id]
        shard_runs = [r for r in runs if r.get("shard_id") == args.shard_id]
        if not shard_keys:
            errors.append("no planned keys for shard")
        # Full lookup universe audit
        errors.extend(_audit_run_plan_for_shard(shard_runs, shard_keys))
        if errors:
            print("VALIDATION_FAIL: " + "; ".join(errors[:5])); sys.exit(1)
        print("VALIDATION_PASS"); return

    deps = ExecutionDependencies(mode="synthetic" if args.synthetic else "production")
    keys = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_key_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]
    runs = [json.loads(l) for l in gzip.decompress((plan_dir / "full_b1_run_plan.jsonl.gz").read_bytes()).decode("utf-8").strip().split("\n")]

    shard_keys = [k for k in keys if k.get("shard_id") == args.shard_id]
    shard_runs = [r for r in runs if r.get("shard_id") == args.shard_id]
    rid_lookup = build_run_id_lookup(shard_runs)

    planned_ids = {}
    for kp in shard_keys:
        cid = kp["canonical_key_id"]
        planned_ids[cid] = set()
        for r in shard_runs:
            if r["canonical_key_id"] == cid:
                planned_ids[cid].add(r["run_id"])

    # Execute within writer lock
    try:
        with exclusive_writer_lock(out, operation=f"run_shard_{args.shard_id}"):
            cleanup_stale_temp_files(out, min_age_seconds=3600)
            out.mkdir(parents=True, exist_ok=True)

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
                    for name, lst in [("baseline", all_bl), ("governed", all_gl), ("selection", all_sl)]:
                        data = gzip.decompress((fdir / f"{name}.csv.gz").read_bytes()).decode("utf-8")
                        for line in data.strip().split("\n")[1:]:
                            if line: lst.append(line)
                    continue

                fdir.mkdir(parents=True, exist_ok=True)
                result = execute_key(kp, deps, rid_lookup.get(cid, {}))
                new_keys += 1; recomputed += 1

                for name, rows, cols in [
                    ("baseline", result["baseline_rows"], ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]),
                    ("governed", result["governed_rows"], ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","selection_hash","realized_cost"]),
                    ("selection", result["selection_rows"], ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]),
                ]:
                    atomic_write_dataframe_gzip(fdir / f"{name}.csv.gz", pd.DataFrame(rows, columns=cols), cols)
                atomic_write_dataframe_gzip(fdir / "failure.csv.gz", pd.DataFrame(columns=["run_id"]), ["run_id"])
                atomic_write_json(fdir / "completion_receipt.json", {"cid": cid, "status": "complete", "bl": 2, "gl": 144, "sl": 144, "lr": deps.counter.lr})

                for row in result["baseline_rows"]: all_bl.append(json.dumps(row))
                for row in result["governed_rows"]: all_gl.append(json.dumps(row))
                for row in result["selection_rows"]: all_sl.append(json.dumps(row))

            # Rebuild shard ledgers — structured duplicate detection
            all_frag_bl = []; all_frag_gl = []; all_frag_sl = []; all_frag_fl = []
            for kp in shard_keys:
                cid = kp["canonical_key_id"]; fdir = out / "key_fragments" / cid
                for name, lst in [("baseline", all_frag_bl), ("governed", all_frag_gl), ("selection", all_frag_sl), ("failure", all_frag_fl)]:
                    fp = fdir / f"{name}.csv.gz"
                    if fp.exists():
                        data = gzip.decompress(fp.read_bytes()).decode("utf-8")
                        for line in data.strip().split("\n")[1:]:
                            if line: lst.append(line)

            # Check duplicates — baseline and governed run IDs only
            # (selection hashes can legitimately repeat across P2 seeds)
            bl_ids = [l.split(",")[0] for l in all_frag_bl]
            gl_ids = [l.split(",")[0] for l in all_frag_gl]
            bl_dups = _find_duplicates(bl_ids)
            gl_dups = _find_duplicates(gl_ids)
            if bl_dups:
                print(f"FAIL: duplicate baseline run IDs: {len(bl_dups)}"); sys.exit(1)
            if gl_dups:
                print(f"FAIL: duplicate governed run IDs: {len(gl_dups)}"); sys.exit(1)

            if all_frag_fl:
                print(f"FAIL: {len(all_frag_fl)} failure rows detected"); sys.exit(1)

            sorted_bl = sorted(all_frag_bl)
            sorted_gl = sorted(all_frag_gl)
            sorted_sl = sorted(all_frag_sl)

            bl_cols = "run_id,dataset_index,mechanism,strength,training_seed,learner,baseline_type,auc"
            gl_cols = "run_id,dataset_index,mechanism,strength,training_seed,governance_seed,learner,policy,contract,budget_bp,strict_auc,full_auc,governed_auc,legacy_sdr,selection_hash,realized_cost"
            sl_cols = "selection_hash,policy,contract,budget_bp,removed_encoded_indices,removed_group_ids,realized_encoded_cost"

            for name, lines, hdr in [
                ("baseline_ledger", sorted_bl, bl_cols),
                ("governed_ledger", sorted_gl, gl_cols),
                ("selection_ledger", sorted_sl, sl_cols),
            ]:
                content = hdr + "\n" + "\n".join(lines) + ("\n" if lines else "\n")
                atomic_write_gzip_text(out / f"{name}.csv.gz", content)

            atomic_write_gzip_text(out / "failure_ledger.csv.gz", "run_id\n")

            sm = {"shard_id": args.shard_id, "keys": len(shard_keys), "new_keys": new_keys, "complete_keys": len(complete_ids),
                  "bl": len(sorted_bl), "gl": len(sorted_gl), "sl": len(sorted_sl),
                  "lr": deps.counter.lr}
            atomic_write_json(out / "shard_manifest.json", sm)

            if args.resume:
                rr = {"shard_id": args.shard_id, "validated_complete": len(complete_ids), "recomputed": recomputed,
                      "new_bl": 0, "new_gl": 0, "new_lr": deps.counter.lr if recomputed == 0 else deps.counter.lr}
                atomic_write_json(out / "resume_receipt.json", rr)

            print(f"Shard {args.shard_id}: {new_keys} new, {len(complete_ids)} complete, bl={sm['bl']}, gl={sm['gl']}")
    except WriterLockError as exc:
        print(f"FAIL: {exc}"); sys.exit(1)

if __name__ == "__main__": main()
