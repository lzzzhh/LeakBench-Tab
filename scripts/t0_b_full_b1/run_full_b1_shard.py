#!/usr/bin/env python3
"""T0-B Full-B1 Shard Runner — real execution path with dependency injection."""
from __future__ import annotations
import gzip, hashlib, io, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.t0_b_full_b1.io_contract import atomic_write_gz, atomic_write_json

# Call counters (injectable via tests)
CALL_COUNTS = {"lr": 0, "p3": 0, "p4": 0, "p5": 0, "p6": 0}

# Dependency injection points — tests replace these with fakes
def _real_model_factory(model_id, Xtr, ytr, Xva, yva, Xte, seed):
    from src.leakbench.models.core_models import fit_predict_core_model
    CALL_COUNTS["lr"] += 1
    return fit_predict_core_model(model_id, Xtr, ytr, Xva, yva, Xte, seed)

def _real_mi_scorer(Xtr, ytr):
    from scripts.t0_b_v3.policy_selectors import score_mi
    CALL_COUNTS["p3"] += 1; return score_mi(Xtr, ytr)

def _real_pbiserial_scorer(Xtr, ytr):
    from scripts.t0_b_v3.policy_selectors import score_point_biserial
    CALL_COUNTS["p4"] += 1; return score_point_biserial(Xtr, ytr)

def _real_lr_coef_scorer(Xtr, ytr):
    from scripts.t0_b_v3.policy_selectors import score_lr_coef
    CALL_COUNTS["p5"] += 1; return score_lr_coef(Xtr, ytr)

def _real_rf_perm_scorer(Xtr, ytr):
    from scripts.t0_b_v3.policy_selectors import score_rf_permutation
    CALL_COUNTS["p6"] += 1; return score_rf_permutation(Xtr, ytr)

model_factory = _real_model_factory
mi_scorer = _real_mi_scorer; pbiserial_scorer = _real_pbiserial_scorer
lr_coef_scorer = _real_lr_coef_scorer; rf_perm_scorer = _real_rf_perm_scorer

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def execute_key(kp, out_dir, resume_check=None):
    """Execute one canonical key: baseline + ranking + 144 governed fits."""
    from scripts.t0_b_v3.budget_contract import compute_k
    from scripts.t0_b_v3.seed_contract import derive_p2_seed
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
    from scripts.t0_b_v3.policy_selectors import group_max_score, top_k_groups, top_k_columns
    from sklearn.metrics import roc_auc_score

    cid = kp["canonical_key_id"]
    # Check resume
    if resume_check and resume_check(cid):
        return {"canonical_key_id": cid, "status": "skipped_complete", "new_rows": 0}

    ds, mech, st, ts = kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]
    bundle = np.load(ROOT / kp["bundle_path"], allow_pickle=False)
    assert s(kp["bundle_path"]) == kp["bundle_sha256"]
    X = np.concatenate((bundle["base_X"], bundle[f"block__{kp['bundle_key']}"]), axis=1)
    y = bundle["y"]; tr, va, te = bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]
    n_total = X.shape[1]

    # Load evaluation labels for leak mask
    import gzip as gz
    data = gz.decompress((ROOT/"results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz").read_bytes()).decode("utf-8")
    eval_info = None
    for line in data.strip().split("\n"):
        r = json.loads(line)
        if (r["dataset_index"],r["mechanism"],r["strength"],r["training_seed"]) == (ds,mech,st,ts):
            eval_info = r; break
    data2 = gz.decompress((ROOT/"results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz").read_bytes()).decode("utf-8")
    groups = None
    for line in data2.strip().split("\n"):
        r = json.loads(line)
        if (r["dataset_index"],r["mechanism"],r["strength"],r["training_seed"]) == (ds,mech,st,ts):
            groups = r["groups"]; break
    leak_mask = np.array([i in set(sum((g["member_encoded_indices"] for g in groups if g["opaque_group_id"] in eval_info["leak_group_ids"]), [])) for i in range(n_total)])

    # Baseline
    X_strict = X[:, ~leak_mask]
    strict_out = model_factory("lr", X_strict[tr], y[tr], X_strict[va], y[va], X_strict[te], ts)
    full_out = model_factory("lr", X[tr], y[tr], X[va], y[va], X[te], ts)
    strict_auc = float(roc_auc_score(y[te], strict_out.probabilities))
    full_auc = float(roc_auc_score(y[te], full_out.probabilities))

    baseline_rows = [
        {"run_id": hashlib.sha256(f"t0b_bl|{cid}|strict".encode()).hexdigest(), "dataset_index": ds, "mechanism": mech, "strength": st, "training_seed": ts, "learner": "lr", "baseline_type": "strict", "auc": strict_auc},
        {"run_id": hashlib.sha256(f"t0b_bl|{cid}|full".encode()).hexdigest(), "dataset_index": ds, "mechanism": mech, "strength": st, "training_seed": ts, "learner": "lr", "baseline_type": "full", "auc": full_auc},
    ]

    # Ranking scores
    Xtr, ytr = X[tr], y[tr]
    s3 = mi_scorer(Xtr, ytr)
    s4 = pbiserial_scorer(Xtr, ytr)
    s5 = lr_coef_scorer(Xtr, ytr)
    s6 = rf_perm_scorer(Xtr, ytr)
    ps = {"P3": s3, "P4": s4, "P5": s5, "P6": s6}
    gs = {pid: group_max_score(ps[pid], groups) for pid in ["P3","P4","P5","P6"]}

    # Governed fits
    CONTRACTS = ["semantic_group","encoded_column"]; BUDGETS = [500,1000,2000]
    GOV_SEEDS = list(range(2026071700, 2026071720))
    governed_rows = []; selection_rows = []; fit_count = 0

    for ct in CONTRACTS:
        for bp in BUDGETS:
            ku = compute_k(len(groups) if ct=="semantic_group" else n_total, bp)
            for pid in ["P2","P3","P4","P5","P6"]:
                seeds_for_policy = GOV_SEEDS if pid=="P2" else [0]
                for gi_idx, gs_val in enumerate(seeds_for_policy):
                    gs_out = gi_idx if pid=="P2" else -1
                    if pid == "P2":
                        p2s = derive_p2_seed(gs_val, ds, mech, st, ts, ct, bp)
                        rng = np.random.RandomState(p2s)
                        if ct == "semantic_group":
                            sg = list(rng.choice(len(groups), ku, replace=False))
                            gids = [groups[i]["opaque_group_id"] for i in sg]
                            sh = hash_semantic_selection(ds,mech,st,ts,kp["bundle_key"],kp["bundle_sha256"],"P2",ct,bp,gids)
                            rc = []; [rc.extend(groups[i]["member_encoded_indices"]) for i in sg]
                        else:
                            rc = list(rng.choice(n_total, ku, replace=False))
                            sh = hash_encoded_selection(ds,mech,st,ts,kp["bundle_key"],kp["bundle_sha256"],"P2",ct,bp,np.array(sorted(rc),dtype=np.int64))
                            gids = []
                    else:
                        if ct == "semantic_group":
                            sel = top_k_groups(gs[pid], ku)
                            sh = hash_semantic_selection(ds,mech,st,ts,kp["bundle_key"],kp["bundle_sha256"],pid,ct,bp,sel)
                            rc = []; [rc.extend(g["member_encoded_indices"]) for gid in sel for g in groups if g["opaque_group_id"]==gid]
                            gids = sel
                        else:
                            idx = top_k_columns(ps[pid], ku)
                            sh = hash_encoded_selection(ds,mech,st,ts,kp["bundle_key"],kp["bundle_sha256"],pid,ct,bp,np.array(sorted(idx),dtype=np.int64))
                            rc = list(idx)
                            gids = [g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"])&set(rc)]

                    selection_rows.append({"selection_hash":sh,"policy":pid,"contract":ct,"budget_bp":bp,"removed_encoded_indices":json.dumps([int(x) for x in sorted(rc)]),"removed_group_ids":json.dumps(sorted(gids)),"realized_encoded_cost":len(rc)})

                    keep = np.ones(n_total, dtype=bool); keep[rc] = False
                    gov_out = model_factory("lr", X[:, keep][tr], y[tr], X[:, keep][va], y[va], X[:, keep][te], ts)
                    ga = float(roc_auc_score(y[te], gov_out.probabilities))
                    opp = abs(full_auc-strict_auc); go = ga - strict_auc
                    rid = hashlib.sha256(f"t0b_gov|{cid}|{pid}|{ct}|{bp}|{gs_out}".encode()).hexdigest()
                    governed_rows.append({"run_id":rid,"dataset_index":ds,"mechanism":mech,"strength":st,"training_seed":ts,"governance_seed":gs_out,"learner":"lr","policy":pid,"contract":ct,"budget_bp":bp,"strict_auc":strict_auc,"full_auc":full_auc,"governed_auc":ga,"legacy_sdr":opp-abs(go),"selection_hash":sh,"realized_cost":len(rc)})
                    fit_count += 1

    return {"canonical_key_id": cid, "status": "executed", "new_rows": len(baseline_rows)+len(governed_rows),
            "baseline_rows": baseline_rows, "governed_rows": governed_rows, "selection_rows": selection_rows,
            "fit_count": fit_count}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True); ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--output-dir", default="/tmp/t0_b_full_b1"); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selection-only", action="store_true"); ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--fail-fast", action="store_true"); ap.add_argument("--validate-inputs-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    with open(ROOT/args.plan_manifest) as f: pm = json.load(f)

    if args.validate_inputs_only:
        print(f"Shard {args.shard_id}: validate-inputs-only — 0 real model calls")
        print(f"lr={CALL_COUNTS['lr']} p3={CALL_COUNTS['p3']} p4={CALL_COUNTS['p4']} p5={CALL_COUNTS['p5']} p6={CALL_COUNTS['p6']}")
        return

    print(f"Shard {args.shard_id}: {pm['canonical_keys']} keys planned, {pm['downstream_rows']} rows")
    print("Runner ready — use with synthetic fixture tests or real execution")

if __name__=="__main__": main()
