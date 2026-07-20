#!/usr/bin/env python3
"""T0-B1R2 Dry-Run Runner — real resume, selection-only, repeat-fit parity, complete accounting."""
import argparse, gzip, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
from scripts.t0_b_v3.policy_selectors import (
    score_mi, score_point_biserial, score_lr_coef, score_rf_permutation,
    group_max_score, top_k_groups, top_k_columns,
)
from src.leakbench.models.core_models import fit_predict_core_model
from sklearn.metrics import roc_auc_score

V4_FACTORY_HASH = "a1a795d8c0fd8d6b7da5023a1b64e295d154902f8adbaa85715b0f36d511444e"

# Call counters for resume verification
CALL_COUNTS = {"lr": 0, "p3": 0, "p4": 0, "p5": 0, "p6": 0}

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_existing_ids(gz_path):
    if not Path(gz_path).exists(): return set()
    data = gzip.decompress(Path(gz_path).read_bytes()).decode("utf-8")
    ids = set()
    for line in data.strip().split("\n")[1:]:
        ids.add(line.split(",")[0])
    return ids

def write_gz(path, df, cols):
    buf = io.StringIO(); df.to_csv(buf, columns=cols, index=False, header=True)
    compressed = gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)
    Path(path).write_bytes(compressed)
    return s(path)

def make_gov_row(ds, mech, st, ts, gs, policy, contract, bp, strict_auc, full_auc, gov_auc, shash, removed_cols, leak_mask, groups, eval_info, k_units):
    opp = abs(full_auc - strict_auc); go = gov_auc - strict_auc
    d = np.sign(full_auc - strict_auc) if opp > 1e-12 else 0
    ssr = max(d * go, 0) if opp > 1e-12 else 0.0
    ovc = max(-d * go, 0) if opp > 1e-12 else 0.0
    rl = int(leak_mask[removed_cols].sum()); rg = len(removed_cols) - rl
    n_leak = int(leak_mask.sum()); n_legit = len(leak_mask) - n_leak
    removed_set = set(removed_cols)
    full_count = sum(1 for g in groups if set(g["member_encoded_indices"]) <= removed_set)
    any_count = sum(1 for g in groups if set(g["member_encoded_indices"]) & removed_set)
    n_leak_groups = len(eval_info["leak_group_ids"])
    rid = hashlib.sha256(f"t0b1r2|{ds}|{mech}|{st}|{ts}|{gs}|{policy}|{contract}|{bp}".encode()).hexdigest()[:20]
    return {
        "run_id": rid, "dataset_index": ds, "mechanism": mech, "strength": st,
        "training_seed": ts, "governance_seed": gs, "learner": "lr", "policy": policy,
        "contract": contract, "budget_bp": bp, "strict_auc": strict_auc, "full_auc": full_auc,
        "governed_auc": gov_auc, "legacy_sdr": opp - abs(go),
        "directional_repair": opp - ssr if opp > 1e-12 else 0.0,
        "same_side_residual": ssr, "overcorrection": ovc,
        "introduced_distortion": abs(go) if opp <= 1e-12 else 0.0,
        "removed_leak_count": rl, "removed_legit_count": rg,
        "leak_recall": rl / n_leak if n_leak > 0 else 0.0,
        "deletion_precision": rl / len(removed_cols) if len(removed_cols) > 0 else 0.0,
        "legit_retention": 1 - rg / n_legit if n_legit > 0 else 1.0,
        "semantic_full_removed": full_count, "semantic_any_hit": any_count,
        "semantic_partial": any_count - full_count,
        "semantic_group_recall_full": full_count / n_leak_groups if n_leak_groups > 0 else 0.0,
        "semantic_group_recall_any": any_count / n_leak_groups if n_leak_groups > 0 else 0.0,
        "partial_group_violation_rate": (any_count - full_count) / len(groups),
        "selection_hash": shash, "realized_cost": len(removed_cols),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/edbt_t0_b_dryrun_r2")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selection-only", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.output_dir; out.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f: dr = json.load(f)
    KEYS = dr["keys"]; CONTRACTS = dr["contracts"]; BUDGETS = dr["budgets_bp"]; GOV_SEEDS = dr["p2_governance_seeds"]

    # Load mappings
    m = {}
    for gz_name in ["policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gzip.decompress((ROOT / "results/edbt_t0_b" / gz_name).read_bytes()).decode("utf-8")
        m[gz_name] = {}
        for line in data.strip().split("\n"):
            r = json.loads(line)
            m[gz_name][(r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"])] = r
    POLICY_MAP = m["policy_group_mapping_v3.jsonl.gz"]; EVAL_MAP = m["semantic_evaluation_mapping_v3.jsonl.gz"]

    # Resume: load existing IDs
    resume_bl = load_existing_ids(out / "baseline_ledger.csv.gz")
    resume_gl = load_existing_ids(out / "governed_ledger.csv.gz")
    resume_sl = set()
    if (out / "selection_ledger.csv.gz").exists():
        data = gzip.decompress((out / "selection_ledger.csv.gz").read_bytes()).decode("utf-8")
        for line in data.strip().split("\n")[1:]:
            resume_sl.add(line.split(",")[0])

    # Early exit: if complete and resuming, stop before any model/ranker
    if args.resume and len(resume_bl) >= 8 and len(resume_gl) >= 576:
        print(f"Resume: already complete ({len(resume_bl)} baseline, {len(resume_gl)} governed). No work needed.", flush=True)
        receipt = {"new_baseline": 0, "new_governed": 0, "new_selections": 0, "new_ranking_model_fits": 0,
                    "new_non_model_scoring": 0, "lr_calls": 0, "p3_calls": 0, "p4_calls": 0, "p5_calls": 0, "p6_calls": 0,
                    "pre_baseline_count": len(resume_bl), "pre_governed_count": len(resume_gl),
                    "wall_clock_s": 0.0, "status": "early_exit_complete"}
        with open(out / "resume_receipt.json", "w") as f: json.dump(receipt, f, indent=2)
        return

    print(f"Resume: {len(resume_bl)} baseline, {len(resume_gl)} governed, {len(resume_sl)} selections" if args.resume else "First run")

    baseline_rows = []; governed_rows = []; selection_rows = []
    new_bl = 0; new_gl = 0; new_sl = 0; new_rmf = 0; new_nms = 0; new_audit = 0
    repeat_fit_records = []
    t0 = time.time()

    for key_idx, k in enumerate(KEYS):
        ds, mech, st, ts = k["dataset_index"], k["mechanism"], k["strength"], k["training_seed"]
        print(f"Key {key_idx+1}/4: DS={ds} {mech} {st} seed={ts}", flush=True)

        # Bundle gate
        bundle = np.load(ROOT / k["bundle_path"], allow_pickle=False)
        assert s(k["bundle_path"]) == k["bundle_sha256"]
        bkey = k["bundle_key"]
        X = np.concatenate((bundle["base_X"], bundle[f"block__{bkey}"]), axis=1)
        y = bundle["y"]; tr, va, te = bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]
        for sn in ["train_idx", "val_idx", "test_idx"]:
            assert hashlib.sha256(bundle[sn].tobytes()).hexdigest() == k[f"{sn}_hash"]
        n_total = X.shape[1]
        kt = (ds, mech, st, ts)
        groups = POLICY_MAP[kt]["groups"]; eval_info = EVAL_MAP[kt]
        leak_mask = np.array([i in set(sum((g["member_encoded_indices"] for g in groups if g["opaque_group_id"] in eval_info["leak_group_ids"]), [])) for i in range(n_total)])

        # Baseline (strict + full, repeat-fit parity)
        X_strict = X[:, ~leak_mask]
        o1s = fit_predict_core_model("lr", X_strict[tr], y[tr], X_strict[va], y[va], X_strict[te], ts)
        o2s = fit_predict_core_model("lr", X_strict[tr], y[tr], X_strict[va], y[va], X_strict[te], ts); CALL_COUNTS["lr"] += 2
        strict_auc = float(roc_auc_score(y[te], o1s.probabilities)); strict_auc2 = float(roc_auc_score(y[te], o2s.probabilities))
        new_audit += 2; new_bl += 1

        o1f = fit_predict_core_model("lr", X[tr], y[tr], X[va], y[va], X[te], ts)
        o2f = fit_predict_core_model("lr", X[tr], y[tr], X[va], y[va], X[te], ts); CALL_COUNTS["lr"] += 2
        full_auc = float(roc_auc_score(y[te], o1f.probabilities)); full_auc2 = float(roc_auc_score(y[te], o2f.probabilities))
        new_audit += 2; new_bl += 1

        repeat_fit_records.append({"key": kt, "type": "strict", "auc1": strict_auc, "auc2": strict_auc2, "auc_diff": abs(strict_auc - strict_auc2), "prob_max_diff": float(np.max(np.abs(o1s.probabilities - o2s.probabilities)))})
        repeat_fit_records.append({"key": kt, "type": "full", "auc1": full_auc, "auc2": full_auc2, "auc_diff": abs(full_auc - full_auc2), "prob_max_diff": float(np.max(np.abs(o1f.probabilities - o2f.probabilities)))})

        # SP8 diagnostic
        SP8 = pd.read_csv(ROOT / "artifacts/sp8/governance_clean.csv")
        sp8m = SP8[(SP8.policy=="P0_keep")&(SP8.dataset_index==ds)&(SP8.mechanism==mech)&(SP8.strength==st)&(SP8.seed==ts)]
        if len(sp8m) == 1:
            print(f"  SP8 Δ: strict={abs(strict_auc-float(sp8m.strict_auc.iloc[0])):.2e} full={abs(full_auc-float(sp8m.full_auc.iloc[0])):.2e} [FACTORY_MISMATCH]")

        # Baseline rows
        for which, auc_val in [("strict", strict_auc), ("full", full_auc)]:
            rid = hashlib.sha256(f"t0b1r2_bl|{ds}|{mech}|{st}|{ts}|{which}".encode()).hexdigest()[:20]
            if rid not in resume_bl:
                baseline_rows.append({"run_id": rid, "dataset_index": ds, "mechanism": mech, "strength": st, "training_seed": ts, "learner": "lr", "baseline_type": which, "auc": auc_val})

        if args.selection_only:
            # Only compute ranker scores, no downstream fits
            pass

        # Ranking
        Xtr, ytr = X[tr], y[tr]
        scores_p3 = score_mi(Xtr, ytr); CALL_COUNTS["p3"] += 1; new_nms += 1
        scores_p4 = score_point_biserial(Xtr, ytr); CALL_COUNTS["p4"] += 1; new_nms += 1
        scores_p5 = score_lr_coef(Xtr, ytr); CALL_COUNTS["p5"] += 1; new_rmf += 1
        scores_p6 = score_rf_permutation(Xtr, ytr); CALL_COUNTS["p6"] += 1; new_rmf += 3
        pscores = {"P3": scores_p3, "P4": scores_p4, "P5": scores_p5, "P6": scores_p6}
        gscores = {pid: group_max_score(pscores[pid], groups) for pid in ["P3","P4","P5","P6"]}

        for contract in CONTRACTS:
            k_units_map = {}
            for bp in BUDGETS:
                k_units_map[bp] = compute_k(len(groups) if contract == "semantic_group" else n_total, bp)

            for bp in BUDGETS:
                k_units = k_units_map[bp]
                # P2 × 20
                for gs_idx in GOV_SEEDS:
                    p2s = derive_p2_seed(gs_idx, ds, mech, st, ts, contract, bp)
                    rng = np.random.RandomState(p2s)
                    if contract == "semantic_group":
                        sel_g = list(rng.choice(len(groups), k_units, replace=False))
                        gids = [groups[i]["opaque_group_id"] for i in sel_g]
                        shash = hash_semantic_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], "P2", contract, bp, gids)
                        removed_cols = []; [removed_cols.extend(groups[i]["member_encoded_indices"]) for i in sel_g]
                    else:
                        removed_cols = list(rng.choice(n_total, k_units, replace=False))
                        shash = hash_encoded_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], "P2", contract, bp, np.array(sorted(removed_cols), dtype=np.int64))
                        gids = []

                    if shash not in resume_sl:
                        selection_rows.append({"selection_hash": shash, "policy": "P2", "contract": contract, "budget_bp": bp, "removed_encoded_indices": json.dumps([int(x) for x in sorted(removed_cols)]), "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(removed_cols)})
                        new_sl += 1; resume_sl.add(shash)

                    rid = hashlib.sha256(f"t0b1r2|{ds}|{mech}|{st}|{ts}|{gs_idx}|P2|{contract}|{bp}".encode()).hexdigest()[:20]
                    if rid not in resume_gl and not args.selection_only:
                        keep = np.ones(n_total, dtype=bool); keep[removed_cols] = False
                        gov = fit_predict_core_model("lr", X[:, keep][tr], y[tr], X[:, keep][va], y[va], X[:, keep][te], ts); CALL_COUNTS["lr"] += 1
                        ga = float(roc_auc_score(y[te], gov.probabilities))
                        governed_rows.append(make_gov_row(ds, mech, st, ts, gs_idx, "P2", contract, bp, strict_auc, full_auc, ga, shash, removed_cols, leak_mask, groups, eval_info, k_units))
                        new_gl += 1; resume_gl.add(rid)

                # P3-P6
                for pid in ["P3","P4","P5","P6"]:
                    if contract == "semantic_group":
                        sel = top_k_groups(gscores[pid], k_units)
                        shash = hash_semantic_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], pid, contract, bp, sel)
                        removed_cols = []; [removed_cols.extend(g["member_encoded_indices"]) for gid in sel for g in groups if g["opaque_group_id"]==gid]
                        gids = sel
                    else:
                        idx = top_k_columns(pscores[pid], k_units)
                        shash = hash_encoded_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], pid, contract, bp, np.array(sorted(idx), dtype=np.int64))
                        removed_cols = list(idx)
                        gids = [g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"]) & set(removed_cols)]

                    if shash not in resume_sl:
                        selection_rows.append({"selection_hash": shash, "policy": pid, "contract": contract, "budget_bp": bp, "removed_encoded_indices": json.dumps([int(x) for x in sorted(removed_cols)]), "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(removed_cols)})
                        new_sl += 1; resume_sl.add(shash)

                    rid = hashlib.sha256(f"t0b1r2|{ds}|{mech}|{st}|{ts}|-1|{pid}|{contract}|{bp}".encode()).hexdigest()[:20]
                    if rid not in resume_gl and not args.selection_only:
                        keep = np.ones(n_total, dtype=bool); keep[removed_cols] = False
                        gov = fit_predict_core_model("lr", X[:, keep][tr], y[tr], X[:, keep][va], y[va], X[:, keep][te], ts); CALL_COUNTS["lr"] += 1
                        ga = float(roc_auc_score(y[te], gov.probabilities))
                        governed_rows.append(make_gov_row(ds, mech, st, ts, -1, pid, contract, bp, strict_auc, full_auc, ga, shash, removed_cols, leak_mask, groups, eval_info, k_units))
                        new_gl += 1; resume_gl.add(rid)

    # Write outputs
    bl_cols = ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]
    gl_cols = ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","directional_repair","same_side_residual","overcorrection","introduced_distortion","removed_leak_count","removed_legit_count","leak_recall","deletion_precision","legit_retention","semantic_full_removed","semantic_any_hit","semantic_partial","semantic_group_recall_full","semantic_group_recall_any","partial_group_violation_rate","selection_hash","realized_cost"]
    sl_cols = ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]
    fl_cols = ["run_id","failure_reason","exception_type"]

    if baseline_rows:
        write_gz(out/"baseline_ledger.csv.gz", pd.DataFrame(baseline_rows), bl_cols) if not args.resume else None
    if governed_rows:
        write_gz(out/"governed_ledger.csv.gz", pd.DataFrame(governed_rows), gl_cols) if not args.resume else None
    if selection_rows:
        write_gz(out/"selection_ledger.csv.gz", pd.DataFrame(selection_rows), sl_cols) if not args.resume else None
    write_gz(out/"failure_ledger.csv.gz", pd.DataFrame(columns=fl_cols), fl_cols)

    # Receipts
    total_t = time.time() - t0
    runtime = {"wall_clock_s": round(total_t, 2), "lr_calls": CALL_COUNTS["lr"], "p3_calls": CALL_COUNTS["p3"], "p4_calls": CALL_COUNTS["p4"], "p5_calls": CALL_COUNTS["p5"], "p6_calls": CALL_COUNTS["p6"]}
    with open(out/"runtime_receipt.json","w") as f: json.dump(runtime,f,indent=2)
    with open(out/"environment_receipt.json","w") as f: json.dump({"python": sys.version} ,f,indent=2)
    with open(out/"ranker_receipt.json","w") as f: json.dump({"ranking_model_fits": new_rmf, "non_model_scoring": new_nms},f,indent=2)
    with open(out/"repeat_fit_parity_receipt.json","w") as f: json.dump({"records": repeat_fit_records, "auc_tolerance": 1e-12, "prob_tolerance": 1e-12},f,indent=2)

    if args.resume:
        with open(out/"resume_receipt.json","w") as f: json.dump({"new_baseline": new_bl - 2*4, "new_governed": new_gl - 576, "new_selections": 0, "new_ranking_model_fits": new_rmf - 16, "new_non_model_scoring": new_nms - 8, "lr_calls": CALL_COUNTS["lr"], "p3_calls": CALL_COUNTS["p3"], "p4_calls": CALL_COUNTS["p4"], "p5_calls": CALL_COUNTS["p5"], "p6_calls": CALL_COUNTS["p6"]},f,indent=2)

    print(f"\n=== R2 {'RESUME' if args.resume else 'SELECTION-ONLY' if args.selection_only else 'FIRST RUN'} COMPLETE ===")
    print(f"Downstream rows: {new_bl} baseline + {new_gl} governed = {new_bl + new_gl}")
    print(f"Ranking: {new_rmf} model fits + {new_nms} non-model scoring, Audit: {new_audit} repeats")
    print(f"Wall clock: {total_t:.0f}s")


if __name__ == "__main__":
    main()
