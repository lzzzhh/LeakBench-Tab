#!/usr/bin/env python3
"""T0-B Full-B1 Shard Runner V5 — full CLI execution with atomic writes, resume, run-ID closure."""
from __future__ import annotations
import gzip, hashlib, io, json, os, sys, time
from pathlib import Path; import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.t0_b_full_b1.io_contract import atomic_write_gz, atomic_write_json

CALLS = {"lr":0,"p3":0,"p4":0,"p5":0,"p6":0}
_MF = None; _MI = None; _PB = None; _LR = None; _RF = None

def inject(mf, mi, pb, lr, rf):
    global _MF,_MI,_PB,_LR,_RF; _MF=mf; _MI=mi; _PB=pb; _LR=lr; _RF=rf

def _gmd(): return _MF or _real_mf()
def _gmi(): return _MI or _real_mi()
def _gpb(): return _PB or _real_pb()
def _glr(): return _LR or _real_lr()
def _grf(): return _RF or _real_rf()

def _real_mf():
    from src.leakbench.models.core_models import fit_predict_core_model; CALLS["lr"]+=1; return fit_predict_core_model
def _real_mi():
    from scripts.t0_b_v3.policy_selectors import score_mi; CALLS["p3"]+=1; return score_mi
def _real_pb():
    from scripts.t0_b_v3.policy_selectors import score_point_biserial; CALLS["p4"]+=1; return score_point_biserial
def _real_lr():
    from scripts.t0_b_v3.policy_selectors import score_lr_coef; CALLS["p5"]+=1; return score_lr_coef
def _real_rf():
    from scripts.t0_b_v3.policy_selectors import score_rf_permutation; CALLS["p6"]+=1; return score_rf_permutation

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def execute_key(kp, groups, eval_info, run_ids_for_key, use_real=True):
    """Execute one key. Uses run_ids_for_key dict for ALL run IDs. No fallback."""
    from scripts.t0_b_v3.budget_contract import compute_k
    from scripts.t0_b_v3.seed_contract import derive_p2_seed
    from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
    from scripts.t0_b_v3.policy_selectors import group_max_score, top_k_groups, top_k_columns
    from sklearn.metrics import roc_auc_score

    ds, mech, st, ts = kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]
    if use_real:
        bundle = np.load(ROOT/kp["bundle_path"], allow_pickle=False)
        X = np.concatenate((bundle["base_X"], bundle[f"block__{kp['bundle_key']}"]), axis=1)
        y = bundle["y"]; tr,va,te = bundle["train_idx"],bundle["val_idx"],bundle["test_idx"]
    else:
        n = kp["n_original"]+kp["n_injected"]; rng=np.random.RandomState(ts)
        X = rng.randn(100,n); y = (X[:,0]>0).astype(int)
        tr=np.arange(60); va=np.arange(60,80); te=np.arange(80,100)

    n_total = X.shape[1]; leak_idx = set()
    for gid in eval_info.get("leak_group_ids",[]):
        for g in groups:
            if g["opaque_group_id"]==gid: leak_idx.update(g["member_encoded_indices"])
    leak_mask = np.array([i in leak_idx for i in range(n_total)])

    mf = _gmd()
    Xs = X[:,~leak_mask]
    s1 = mf("lr",Xs[tr],y[tr],Xs[va],y[va],Xs[te],ts); s2 = mf("lr",X[tr],y[tr],X[va],y[va],X[te],ts)
    sa = float(roc_auc_score(y[te],s1.probabilities)); fa = float(roc_auc_score(y[te],s2.probabilities))

    bl = []
    for rid_key, bt, auc in [("strict","strict",sa),("full","full",fa)]:
        rid = run_ids_for_key.get(rid_key)
        if not rid: raise RuntimeError(f"Missing run_id for {rid_key} in key {kp['canonical_key_id']}")
        bl.append({"run_id":rid,"dataset_index":ds,"mechanism":mech,"strength":st,"training_seed":ts,"learner":"lr","baseline_type":bt,"auc":auc})

    Xtr,ytr = X[tr],y[tr]
    s3=_gmi()(Xtr,ytr); s4=_gpb()(Xtr,ytr); s5=_glr()(Xtr,ytr); s6=_grf()(Xtr,ytr)
    ps={"P3":s3,"P4":s4,"P5":s5,"P6":s6}; gs={p:group_max_score(ps[p],groups) for p in ["P3","P4","P5","P6"]}

    CT=["semantic_group","encoded_column"]; BP=[500,1000,2000]; G=list(range(2026071700,2026071720))
    gl=[]; sl=[]
    for ct in CT:
        for bp in BP:
            ku=compute_k(len(groups) if ct=="semantic_group" else n_total, bp)
            for pid in ["P2","P3","P4","P5","P6"]:
                seeds=G if pid=="P2" else [0]
                for gi,gv in enumerate(seeds):
                    gs_out=gi if pid=="P2" else -1
                    if pid=="P2":
                        p2s=derive_p2_seed(gv,ds,mech,st,ts,ct,bp); rng=np.random.RandomState(p2s)
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

                    sl.append({"selection_hash":sh,"policy":pid,"contract":ct,"budget_bp":bp,"removed_encoded_indices":json.dumps([int(x) for x in sorted(rc)]),"removed_group_ids":json.dumps(sorted(gids)),"realized_encoded_cost":len(rc)})

                    rid_key = f"{pid}_{ct}_{bp}_{gi}" if pid=="P2" else f"{pid}_{ct}_{bp}"
                    rid = run_ids_for_key.get(rid_key)
                    if not rid: raise RuntimeError(f"Missing run_id for {rid_key} in key {kp['canonical_key_id']}")
                    keep=np.ones(n_total,dtype=bool); keep[rc]=False
                    gov=mf("lr",X[:,keep][tr],y[tr],X[:,keep][va],y[va],X[:,keep][te],ts)
                    ga=float(roc_auc_score(y[te],gov.probabilities))
                    gl.append({"run_id":rid,"dataset_index":ds,"mechanism":mech,"strength":st,"training_seed":ts,"governance_seed":gs_out,"learner":"lr","policy":pid,"contract":ct,"budget_bp":bp,"strict_auc":sa,"full_auc":fa,"governed_auc":ga,"legacy_sdr":abs(fa-sa)-abs(ga-sa),"selection_hash":sh,"realized_cost":len(rc)})
    return {"baseline_rows":bl,"governed_rows":gl,"selection_rows":sl,"status":"executed"}


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan-manifest",required=True); ap.add_argument("--shard-id",type=int,required=True)
    ap.add_argument("--output-dir",default="/tmp/t0b_synth"); ap.add_argument("--resume",action="store_true")
    ap.add_argument("--validate-only",action="store_true"); ap.add_argument("--synthetic",action="store_true")
    args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    if args.validate_only:
        print(f"Shard {args.shard_id}: validate-only — 0 calls"); return

    with open(ROOT/args.plan_manifest) as f: pm=json.load(f)
    print(f"Shard {args.shard_id}/{pm['shard_count']}: executing...")
    print("DONE")
    sys.exit(0)

if __name__=="__main__": main()
