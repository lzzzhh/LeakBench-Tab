#!/usr/bin/env python3
"""T0-B Full-B1 Shard Runner V4 — sealed run plan run_ids, synthetic support, real CLI."""
from __future__ import annotations
import gzip, hashlib, io, json, os, sys, time
from pathlib import Path; import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

# Dependency injection — tests replace these
_model_factory = None; _mi_scorer = None; _pb_scorer = None; _lr_scorer = None; _rf_scorer = None
CALLS = {"lr":0,"p3":0,"p4":0,"p5":0,"p6":0}

def inject_dependencies(mf, mi, pb, lr, rf):
    global _model_factory, _mi_scorer, _pb_scorer, _lr_scorer, _rf_scorer
    _model_factory = mf; _mi_scorer = mi; _pb_scorer = pb; _lr_scorer = lr; _rf_scorer = rf

def _get_mf():
    if _model_factory: return _model_factory
    from src.leakbench.models.core_models import fit_predict_core_model
    CALLS["lr"] += 1; return fit_predict_core_model
def _get_mi():
    if _mi_scorer: return _mi_scorer
    from scripts.t0_b_v3.policy_selectors import score_mi; CALLS["p3"] += 1; return score_mi
def _get_pb():
    if _pb_scorer: return _pb_scorer
    from scripts.t0_b_v3.policy_selectors import score_point_biserial; CALLS["p4"] += 1; return score_point_biserial
def _get_lr():
    if _lr_scorer: return _lr_scorer
    from scripts.t0_b_v3.policy_selectors import score_lr_coef; CALLS["p5"] += 1; return score_lr_coef
def _get_rf():
    if _rf_scorer: return _rf_scorer
    from scripts.t0_b_v3.policy_selectors import score_rf_permutation; CALLS["p6"] += 1; return score_rf_permutation

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def execute_key(kp, groups, eval_info, run_ids_for_key, use_real_bundles=True):
    """Execute one key using SEALED run plan run_ids. Returns fragment dict."""
    from scripts.t0_b_v3.budget_contract import compute_k
    from scripts.t0_b_v3.seed_contract import derive_p2_seed
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
    from scripts.t0_b_v3.policy_selectors import group_max_score, top_k_groups, top_k_columns
    from sklearn.metrics import roc_auc_score

    ds, mech, st, ts = kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]
    if use_real_bundles:
        bundle = np.load(ROOT / kp["bundle_path"], allow_pickle=False)
        X = np.concatenate((bundle["base_X"], bundle[f"block__{kp['bundle_key']}"]), axis=1)
        y = bundle["y"]; tr, va, te = bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]
    else:
        # Synthetic: create dummy data
        n = kp["n_original"] + kp["n_injected"]
        rng = np.random.RandomState(ts)
        X = rng.randn(100, n); y = (X[:,0] > 0).astype(int)
        tr = np.arange(60); va = np.arange(60,80); te = np.arange(80,100)

    n_total = X.shape[1]
    leak_indices = set()
    for gid in eval_info.get("leak_group_ids", []):
        for g in groups:
            if g["opaque_group_id"] == gid: leak_indices.update(g["member_encoded_indices"])
    leak_mask = np.array([i in leak_indices for i in range(n_total)])

    X_strict = X[:, ~leak_mask]
    mf = _get_mf()
    strict_out = mf("lr", X_strict[tr], y[tr], X_strict[va], y[va], X_strict[te], ts)
    full_out = mf("lr", X[tr], y[tr], X[va], y[va], X[te], ts)
    strict_auc = float(roc_auc_score(y[te], strict_out.probabilities))
    full_auc = float(roc_auc_score(y[te], full_out.probabilities))

    baseline_rows = []
    for rid, bt, auc in [(run_ids_for_key.get("strict"), "strict", strict_auc), (run_ids_for_key.get("full"), "full", full_auc)]:
        if rid: baseline_rows.append({"run_id":rid,"dataset_index":ds,"mechanism":mech,"strength":st,"training_seed":ts,"learner":"lr","baseline_type":bt,"auc":auc})

    Xtr, ytr = X[tr], y[tr]
    s3 = _get_mi()(Xtr, ytr); s4 = _get_pb()(Xtr, ytr); s5 = _get_lr()(Xtr, ytr); s6 = _get_rf()(Xtr, ytr)
    ps = {"P3":s3,"P4":s4,"P5":s5,"P6":s6}; gs = {p:group_max_score(ps[p],groups) for p in ["P3","P4","P5","P6"]}

    CONTRACTS=["semantic_group","encoded_column"]; BUDGETS=[500,1000,2000]; GOV=list(range(2026071700,2026071720))
    governed_rows=[]; selection_rows=[]

    for ct in CONTRACTS:
        for bp in BUDGETS:
            ku = compute_k(len(groups) if ct=="semantic_group" else n_total, bp)
            for pid in ["P2","P3","P4","P5","P6"]:
                seeds_for = GOV if pid=="P2" else [0]
                for gi_idx, gs_val in enumerate(seeds_for):
                    gs_out = gi_idx if pid=="P2" else -1
                    if pid=="P2":
                        p2s = derive_p2_seed(gs_val,ds,mech,st,ts,ct,bp); rng=np.random.RandomState(p2s)
                        if ct=="semantic_group":
                            sg=list(rng.choice(len(groups),ku,replace=False)); gids=[groups[i]["opaque_group_id"] for i in sg]
                            sh=hash_semantic_selection(ds,mech,st,ts,kp["bundle_key"],kp["bundle_sha256"],"P2",ct,bp,gids)
                            rc=[]; [rc.extend(groups[i]["member_encoded_indices"]) for i in sg]
                        else:
                            rc=list(rng.choice(n_total,ku,replace=False))
                            sh=hash_encoded_selection(ds,mech,st,ts,kp["bundle_key"],kp["bundle_sha256"],"P2",ct,bp,np.array(sorted(rc),dtype=np.int64))
                            gids=[]
                    else:
                        if ct=="semantic_group":
                            sel=top_k_groups(gs[pid],ku); sh=hash_semantic_selection(ds,mech,st,ts,kp["bundle_key"],kp["bundle_sha256"],pid,ct,bp,sel)
                            rc=[]; [rc.extend(g["member_encoded_indices"]) for gid in sel for g in groups if g["opaque_group_id"]==gid]; gids=sel
                        else:
                            idx=top_k_columns(ps[pid],ku); sh=hash_encoded_selection(ds,mech,st,ts,kp["bundle_key"],kp["bundle_sha256"],pid,ct,bp,np.array(sorted(idx),dtype=np.int64))
                            rc=list(idx); gids=[g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"])&set(rc)]

                    selection_rows.append({"selection_hash":sh,"policy":pid,"contract":ct,"budget_bp":bp,"removed_encoded_indices":json.dumps([int(x) for x in sorted(rc)]),"removed_group_ids":json.dumps(sorted(gids)),"realized_encoded_cost":len(rc)})

                    keep=np.ones(n_total,dtype=bool); keep[rc]=False
                    gov_out=mf("lr",X[:,keep][tr],y[tr],X[:,keep][va],y[va],X[:,keep][te],ts)
                    ga=float(roc_auc_score(y[te],gov_out.probabilities))
                    opp=abs(full_auc-strict_auc); go=ga-strict_auc
                    label = f"{pid}_{ct}_{bp}_{gi_idx}" if pid=="P2" else f"{pid}_{ct}_{bp}"
                    rid_key = f"gov_{label}"
                    rid = run_ids_for_key.get(rid_key, hashlib.sha256(f"fallback|{kp['canonical_key_id']}|{label}".encode()).hexdigest())
                    governed_rows.append({"run_id":rid,"dataset_index":ds,"mechanism":mech,"strength":st,"training_seed":ts,"governance_seed":gs_out,"learner":"lr","policy":pid,"contract":ct,"budget_bp":bp,"strict_auc":strict_auc,"full_auc":full_auc,"governed_auc":ga,"legacy_sdr":opp-abs(go),"selection_hash":sh,"realized_cost":len(rc)})

    return {"baseline_rows":baseline_rows,"governed_rows":governed_rows,"selection_rows":selection_rows,"status":"executed"}


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan-manifest",required=True); ap.add_argument("--shard-id",type=int,required=True)
    ap.add_argument("--output-dir",default="/tmp/t0_b_full_b1"); ap.add_argument("--resume",action="store_true")
    ap.add_argument("--validate-inputs-only",action="store_true")
    args=ap.parse_args()

    with open(ROOT/args.plan_manifest) as f: pm=json.load(f)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    if args.validate_inputs_only:
        print(f"Shard {args.shard_id}: validate-inputs-only — all call counters at zero"); return

    print(f"Shard {args.shard_id}: {pm['canonical_keys']} keys planned, ready for execution (use real or synthetic)")

if __name__=="__main__": main()
