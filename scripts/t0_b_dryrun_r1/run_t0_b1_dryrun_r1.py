#!/usr/bin/env python3
"""T0-B1R Dry-Run Runner — corrected: factory-conditional parity, resume, selection-only."""
from __future__ import annotations
import argparse, gzip, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
from scripts.t0_b_v3.policy_selectors import (
    score_mi, score_point_biserial, score_lr_coef, score_rf_permutation,
    group_max_score, top_k_groups, top_k_columns,
)
from src.leakbench.models.core_models import fit_predict_core_model
from sklearn.metrics import roc_auc_score

# SP8 factory: LogisticRegression(max_iter=1000, random_state=seed) — NO StandardScaler, NO C=1.0
# V4 factory: StandardScaler + LogisticRegression(max_iter=2000, C=1.0, random_state=training_seed)
# → FACTORY MISMATCH: SP8_NUMERIC_PARITY_NOT_APPLICABLE_FACTORY_MISMATCH
V4_FACTORY_HASH = "a1a795d8c0fd8d6b7da5023a1b64e295d154902f8adbaa85715b0f36d511444e"

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_mappings():
    m = {}
    for gz_path in ["results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz", "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gzip.decompress((ROOT / gz_path).read_bytes()).decode("utf-8")
        m[gz_path.split("/")[-1]] = {}
        for line in data.strip().split("\n"):
            r = json.loads(line)
            m[gz_path.split("/")[-1]][(r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"])] = r
    return m["policy_group_mapping_v3.jsonl.gz"], m["semantic_evaluation_mapping_v3.jsonl.gz"]

def write_gz(path, df, cols):
    buf = io.StringIO(); df.to_csv(buf, columns=cols, index=False, header=True)
    compressed = gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)
    Path(path).write_bytes(compressed)
    return s(path)

def append_gz(path, df, cols):
    existing = gzip.decompress(Path(path).read_bytes()).decode("utf-8") if Path(path).exists() else ""
    buf = io.StringIO(); df.to_csv(buf, columns=cols, index=False, header=(not existing))
    combined = existing + buf.getvalue()
    compressed = gzip.compress(combined.encode("utf-8"), mtime=0)
    Path(path).write_bytes(compressed)
    return s(path)

def load_existing_run_ids(gz_path):
    if not Path(gz_path).exists(): return set()
    data = gzip.decompress(Path(gz_path).read_bytes()).decode("utf-8")
    ids = set()
    for line in data.strip().split("\n")[1:]:  # skip header
        ids.add(line.split(",")[0])
    return ids

def make_gov_row(ds, mech, st, ts, gs, policy, contract, bp, strict_auc, full_auc, gov_auc, shash, removed_cols, leak_mask, groups, eval_info, k_units):
    opp = abs(full_auc - strict_auc); go = gov_auc - strict_auc
    sg = full_auc - strict_auc; d = np.sign(sg) if opp > 1e-12 else 0
    ssr = max(d * go, 0) if opp > 1e-12 else 0.0
    ovc = max(-d * go, 0) if opp > 1e-12 else 0.0
    drep = opp - ssr if opp > 1e-12 else 0.0
    legacy_sdr = opp - abs(go); intr = abs(go) if opp <= 1e-12 else 0.0
    rl = int(leak_mask[removed_cols].sum()); rg = len(removed_cols) - rl
    n_leak = int(leak_mask.sum()); n_legit = len(leak_mask) - n_leak
    leak_rec = rl / n_leak if n_leak > 0 else 0.0
    del_prec = rl / len(removed_cols) if len(removed_cols) > 0 else 0.0
    leg_ret = 1 - rg / n_legit if n_legit > 0 else 1.0
    removed_set = set(removed_cols)
    full_count = sum(1 for g in groups if set(g["member_encoded_indices"]) <= removed_set)
    any_count = sum(1 for g in groups if set(g["member_encoded_indices"]) & removed_set)
    partial_count = any_count - full_count
    n_leak_groups = len(eval_info["leak_group_ids"])
    rid = hashlib.sha256(f"t0b1r|{ds}|{mech}|{st}|{ts}|{gs}|{policy}|{contract}|{bp}".encode()).hexdigest()[:20]
    return {
        "run_id": rid, "dataset_index": ds, "mechanism": mech, "strength": st,
        "training_seed": ts, "governance_seed": gs, "learner": "lr", "policy": policy,
        "contract": contract, "budget_bp": bp, "strict_auc": strict_auc, "full_auc": full_auc,
        "governed_auc": gov_auc, "legacy_sdr": legacy_sdr, "directional_repair": drep,
        "same_side_residual": ssr, "overcorrection": ovc, "introduced_distortion": intr,
        "removed_leak_count": rl, "removed_legit_count": rg,
        "leak_recall": leak_rec, "deletion_precision": del_prec, "legit_retention": leg_ret,
        "residual_leak_fraction": (n_leak - rl) / len(leak_mask),
        "semantic_full_removed": full_count, "semantic_any_hit": any_count,
        "semantic_partial": partial_count,
        "semantic_group_recall_full": full_count / n_leak_groups if n_leak_groups > 0 else 0.0,
        "semantic_group_recall_any": any_count / n_leak_groups if n_leak_groups > 0 else 0.0,
        "partial_group_violation_rate": partial_count / len(groups),
        "selection_hash": shash, "realized_cost": len(removed_cols),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="configs/edbt_t0_b/dryrun_matrix_v4.json")
    ap.add_argument("--output-dir", default="results/edbt_t0_b_dryrun_r1")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selection-only", action="store_true")
    args = ap.parse_args()

    out = ROOT / args.output_dir; out.mkdir(parents=True, exist_ok=True)
    with open(ROOT / args.matrix) as f: dr = json.load(f)
    KEYS = dr["keys"]; CONTRACTS = dr["contracts"]; BUDGETS = dr["budgets_bp"]
    GOV_SEEDS = dr["p2_governance_seeds"]
    POLICY_MAP, EVAL_MAP = load_mappings()

    # Resume: load existing run_ids
    resume_baseline_ids = load_existing_run_ids(out / "baseline_ledger.csv.gz")
    resume_gov_ids = load_existing_run_ids(out / "governed_ledger.csv.gz")
    resume_sel_hashes = set()
    if (out / "selection_ledger.csv.gz").exists():
        data = gzip.decompress((out / "selection_ledger.csv.gz").read_bytes()).decode("utf-8")
        for line in data.strip().split("\n")[1:]:
            resume_sel_hashes.add(line.split(",")[0])

    if args.resume:
        print(f"Resume: {len(resume_baseline_ids)} baseline, {len(resume_gov_ids)} governed, {len(resume_sel_hashes)} selections already present")

    baseline_rows = []; governed_rows = []; selection_rows = []; failure_rows = []
    new_baseline = 0; new_governed = 0; new_selections = 0; new_ranking = 0
    t0 = time.time()

    for key_idx, k in enumerate(KEYS):
        ds, mech, st, ts = k["dataset_index"], k["mechanism"], k["strength"], k["training_seed"]
        print(f"\nKey {key_idx+1}/4: DS={ds} {mech} {st} seed={ts}", flush=True)

        # Bundle gate
        bundle = np.load(ROOT / k["bundle_path"], allow_pickle=False)
        assert s(k["bundle_path"]) == k["bundle_sha256"], "Bundle SHA mismatch"
        bkey = k["bundle_key"]
        X = np.concatenate((bundle["base_X"], bundle[f"block__{bkey}"]), axis=1)
        y = bundle["y"]; tr, va, te = bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]
        for sn in ["train_idx", "val_idx", "test_idx"]:
            assert hashlib.sha256(bundle[sn].tobytes()).hexdigest() == k[f"{sn}_hash"]

        n_total = X.shape[1]; assert n_total == k["n_encoded_columns"]
        kt = (ds, mech, st, ts)
        groups = POLICY_MAP[kt]["groups"]; eval_info = EVAL_MAP[kt]
        leak_mask = np.array([i in set(sum((g["member_encoded_indices"] for g in groups if g["opaque_group_id"] in eval_info["leak_group_ids"]), [])) for i in range(n_total)])

        # Baseline: V4 repeat-fit parity (factory mismatch with SP8)
        X_strict = X[:, ~leak_mask]
        o1 = fit_predict_core_model("lr", X_strict[tr], y[tr], X_strict[va], y[va], X_strict[te], ts)
        o2 = fit_predict_core_model("lr", X_strict[tr], y[tr], X_strict[va], y[va], X_strict[te], ts)
        strict_auc = float(roc_auc_score(y[te], o1.probabilities))
        strict_auc2 = float(roc_auc_score(y[te], o2.probabilities))
        assert abs(strict_auc - strict_auc2) <= 1e-12, f"V4 repeat-fit parity fail: {strict_auc:.12f} vs {strict_auc2:.12f}"

        o1f = fit_predict_core_model("lr", X[tr], y[tr], X[va], y[va], X[te], ts)
        o2f = fit_predict_core_model("lr", X[tr], y[tr], X[va], y[va], X[te], ts)
        full_auc = float(roc_auc_score(y[te], o1f.probabilities))
        full_auc2 = float(roc_auc_score(y[te], o2f.probabilities))
        assert abs(full_auc - full_auc2) <= 1e-12, f"V4 repeat-fit parity fail"

        # SP8 comparison: record but don't assert equality
        SP8 = pd.read_csv(ROOT / "artifacts/sp8/governance_clean.csv")
        sp8m = SP8[(SP8.policy=="P0_keep")&(SP8.dataset_index==ds)&(SP8.mechanism==mech)&(SP8.strength==st)&(SP8.seed==ts)]
        if len(sp8m) == 1:
            sp8_sa, sp8_fa = float(sp8m.strict_auc.iloc[0]), float(sp8m.full_auc.iloc[0])
            print(f"  SP8: strict={sp8_sa:.10f} (Δ={abs(strict_auc-sp8_sa):.2e}) full={sp8_fa:.10f} (Δ={abs(full_auc-sp8_fa):.2e}) [FACTORY_MISMATCH: V4≠SP8]")

        # Baseline rows
        for which, auc_val in [("strict", strict_auc), ("full", full_auc)]:
            rid = hashlib.sha256(f"t0b1r_bl|{ds}|{mech}|{st}|{ts}|{which}".encode()).hexdigest()[:20]
            if rid not in resume_baseline_ids:
                baseline_rows.append({"run_id": rid, "dataset_index": ds, "mechanism": mech, "strength": st, "training_seed": ts, "learner": "lr", "baseline_type": which, "auc": auc_val})
                new_baseline += 1

        if args.selection_only:
            # Only compute selections, no model fitting
            pass

        # Ranking (once per key)
        Xtr, ytr = X[tr], y[tr]
        scores_p3 = score_mi(Xtr, ytr); new_ranking += 1
        scores_p4 = score_point_biserial(Xtr, ytr); new_ranking += 1
        scores_p5 = score_lr_coef(Xtr, ytr); new_ranking += 1
        scores_p6 = score_rf_permutation(Xtr, ytr); new_ranking += 3
        pscores = {"P3": scores_p3, "P4": scores_p4, "P5": scores_p5, "P6": scores_p6}
        gscores_cache = {pid: group_max_score(pscores[pid], groups) for pid in ["P3","P4","P5","P6"]}

        for contract in CONTRACTS:
            n_grp = len(groups)
            for bp in BUDGETS:
                if contract == "semantic_group": k_units = compute_k(n_grp, bp)
                else: k_units = compute_k(n_total, bp)

                # P2 × 20
                for gs_idx in GOV_SEEDS:
                    p2s = derive_p2_seed(gs_idx, ds, mech, st, ts, contract, bp)
                    rng = np.random.RandomState(p2s)
                    if contract == "semantic_group":
                        sel_g = list(rng.choice(n_grp, k_units, replace=False))
                        gids = [groups[i]["opaque_group_id"] for i in sel_g]
                        shash = hash_semantic_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], "P2", contract, bp, gids)
                        removed_cols = []; [removed_cols.extend(groups[i]["member_encoded_indices"]) for i in sel_g]
                    else:
                        removed_cols = list(rng.choice(n_total, k_units, replace=False))
                        shash = hash_encoded_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], "P2", contract, bp, np.array(sorted(removed_cols), dtype=np.int64))
                        gids = []

                    if shash not in resume_sel_hashes:
                        selection_rows.append({"selection_hash": shash, "policy": "P2", "contract": contract, "budget_bp": bp, "removed_encoded_indices": json.dumps([int(x) for x in sorted(removed_cols)]), "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(removed_cols)})
                        new_selections += 1; resume_sel_hashes.add(shash)

                    rid = hashlib.sha256(f"t0b1r|{ds}|{mech}|{st}|{ts}|{gs_idx}|lr|P2|{contract}|{bp}".encode()).hexdigest()[:20]
                    if rid not in resume_gov_ids and not args.selection_only:
                        keep = np.ones(n_total, dtype=bool); keep[removed_cols] = False
                        gov = fit_predict_core_model("lr", X[:, keep][tr], y[tr], X[:, keep][va], y[va], X[:, keep][te], ts)
                        ga = float(roc_auc_score(y[te], gov.probabilities))
                        governed_rows.append(make_gov_row(ds, mech, st, ts, gs_idx, "P2", contract, bp, strict_auc, full_auc, ga, shash, removed_cols, leak_mask, groups, eval_info, k_units))
                        new_governed += 1; resume_gov_ids.add(rid)

                # P3-P6
                for pid in ["P3","P4","P5","P6"]:
                    if contract == "semantic_group":
                        sel = top_k_groups(gscores_cache[pid], k_units)
                        shash = hash_semantic_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], pid, contract, bp, sel)
                        removed_cols = []; [removed_cols.extend(g["member_encoded_indices"]) for gid in sel for g in groups if g["opaque_group_id"]==gid]
                        gids = sel
                    else:
                        idx = top_k_columns(pscores[pid], k_units)
                        shash = hash_encoded_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], pid, contract, bp, np.array(sorted(idx), dtype=np.int64))
                        removed_cols = list(idx)
                        gids = [g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"]) & set(removed_cols)]

                    if shash not in resume_sel_hashes:
                        selection_rows.append({"selection_hash": shash, "policy": pid, "contract": contract, "budget_bp": bp, "removed_encoded_indices": json.dumps([int(x) for x in sorted(removed_cols)]), "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(removed_cols)})
                        new_selections += 1; resume_sel_hashes.add(shash)

                    rid = hashlib.sha256(f"t0b1r|{ds}|{mech}|{st}|{ts}|-1|lr|{pid}|{contract}|{bp}".encode()).hexdigest()[:20]
                    if rid not in resume_gov_ids and not args.selection_only:
                        keep = np.ones(n_total, dtype=bool); keep[removed_cols] = False
                        gov = fit_predict_core_model("lr", X[:, keep][tr], y[tr], X[:, keep][va], y[va], X[:, keep][te], ts)
                        ga = float(roc_auc_score(y[te], gov.probabilities))
                        governed_rows.append(make_gov_row(ds, mech, st, ts, -1, pid, contract, bp, strict_auc, full_auc, ga, shash, removed_cols, leak_mask, groups, eval_info, k_units))
                        new_governed += 1; resume_gov_ids.add(rid)

    # Write ledgers
    bl_cols = ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]
    gl_cols = ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","directional_repair","same_side_residual","overcorrection","introduced_distortion","removed_leak_count","removed_legit_count","leak_recall","deletion_precision","legit_retention","residual_leak_fraction","semantic_full_removed","semantic_any_hit","semantic_partial","semantic_group_recall_full","semantic_group_recall_any","partial_group_violation_rate","selection_hash","realized_cost"]
    sl_cols = ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]
    fl_cols = ["run_id","failure_reason","exception_type"]

    if baseline_rows:
        if args.resume: append_gz(out/"baseline_ledger.csv.gz", pd.DataFrame(baseline_rows), bl_cols)
        else: write_gz(out/"baseline_ledger.csv.gz", pd.DataFrame(baseline_rows), bl_cols)
    if governed_rows:
        if args.resume: append_gz(out/"governed_ledger.csv.gz", pd.DataFrame(governed_rows), gl_cols)
        else: write_gz(out/"governed_ledger.csv.gz", pd.DataFrame(governed_rows), gl_cols)
    if selection_rows:
        if args.resume: append_gz(out/"selection_ledger.csv.gz", pd.DataFrame(selection_rows), sl_cols)
        else: write_gz(out/"selection_ledger.csv.gz", pd.DataFrame(selection_rows), sl_cols)
    if not (out/"failure_ledger.csv.gz").exists():
        write_gz(out/"failure_ledger.csv.gz", pd.DataFrame(columns=fl_cols), fl_cols)

    # Resume receipt
    total_t = time.time() - t0
    receipt = {
        "new_baseline_rows": new_baseline, "new_governed_rows": new_governed,
        "new_selection_rows": new_selections, "new_ranking_fits": new_ranking,
        "wall_clock_s": round(total_t, 2),
        "pre_run_baseline_count": len(resume_baseline_ids) - new_baseline,
        "pre_run_governed_count": len(resume_gov_ids) - new_governed,
    }
    if args.resume or args.selection_only:
        with open(out/"resume_receipt.json", "w") as f: json.dump(receipt, f, indent=2)

    print(f"\n=== R1 {'RESUME' if args.resume else 'SELECTION-ONLY' if args.selection_only else 'FIRST RUN'} COMPLETE ===")
    print(f"Baseline: +{new_baseline}, Governed: +{new_governed}, Selections: +{new_selections}, Ranking: +{new_ranking}")
    print(f"Wall clock: {total_t:.0f}s")


if __name__ == "__main__":
    main()
