#!/usr/bin/env python3
"""run_cdxr_governance.py — CDXR Cross-Learner Governance.

Uses self-consistent baselines (baseline_cells.csv) + official model adapter.
P3 (blind MI) + P2 (20 random seeds) at 20% encoded-column budget.
Selection hashes match B1. Outputs per-model CSV.
"""
from __future__ import annotations
import csv, hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.leakbench.models.core_models import fit_predict_core_model

GOV_SEEDS = [2026071700 + i for i in range(20)]
BUDGET = 0.20
MI_SEED = 42

FIELDS = ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed",
          "model","policy","budget_k","budget_fraction","status","failure_reason",
          "strict_auc","full_auc","governed_auc","strict_distance_reduction","initial_gap",
          "removed_count","selection_hash"]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_cell(row):
    b=np.load(ROOT/row.bundle_path,allow_pickle=False); k=row.bundle_key
    X=np.concatenate((b['base_X'],b[f'block__{k}']),axis=1); y=b['y']
    return X,y,b['train_idx'],b['val_idx'],b['test_idx'],b[f'leak_mask__{k}']

def selection_hash(indices):
    arr=np.sort(indices).astype(np.int64)
    payload=b'encoded_column_indices_v1\0'+arr.tobytes()
    return hashlib.sha256(payload).hexdigest()

def select_mi(scores,k): return np.argsort(scores)[::-1][:k]

def select_random(n,k,gs,ds,ts):
    rng=np.random.RandomState((gs*100+ds*7+ts*13)%(2**31-1))
    return rng.choice(n,k,replace=False)

def main(argv=None):
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest",required=True)
    ap.add_argument("--baseline",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--allow-run",action="store_true")
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args(argv)
    if not args.allow_run: raise RuntimeError("locked; pass --allow-run")

    man=pd.read_csv(ROOT/args.bundle_manifest)
    bl=pd.read_csv(ROOT/args.baseline); bl=bl[bl.model==args.model]
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True)
    completed=set()
    if out.exists(): completed=set(pd.read_csv(out)["run_id"].astype(str))

    est=len(man)*22
    print(f"CDXR {args.model}: {len(man)} keys x 22 fits = est {est} rows",flush=True)
    done=0; t0=time.time()

    for _,r in man.iterrows():
        ds,mech,st,ts=int(r.dataset_index),r.mechanism,r.strength,int(r.seed)
        try: X,y,tr,va,te,mask=load_cell(r)
        except: continue
        # Baseline from precomputed
        br=bl[(bl.dataset_index==ds)&(bl.mechanism==mech)&(bl.strength==st)&(bl.training_seed==ts)]
        if len(br)!=1: continue
        sa,fa,gap=float(br.strict_auc.iloc[0]),float(br.full_auc.iloc[0]),float(br.initial_gap.iloc[0])
        nf=X.shape[1]; k=max(1,round(nf*BUDGET)); sc=np.where(~mask)[0]
        if k>=nf: continue

        # P0
        rid0=hashlib.sha256(f"cdxr_p0|{ds}|{mech}|{st}|{ts}|{args.model}".encode()).hexdigest()[:20]
        if rid0 not in completed:
            _write(out,dict(run_id=rid0,dataset_index=ds,mechanism=mech,strength=st,training_seed=ts,governance_seed=-1,
                model=args.model,policy="P0_keep",budget_k=0,budget_fraction=0.0,status="SUCCESS",
                strict_auc=sa,full_auc=fa,governed_auc=sa,strict_distance_reduction=0.0,initial_gap=gap,
                removed_count=0,selection_hash=selection_hash(np.array([],dtype=np.int64))))
        done+=1

        # P3
        mi=mutual_info_classif(X[tr],y[tr],random_state=MI_SEED); mi=np.nan_to_num(mi,nan=0.0)
        mi_f=select_mi(mi,k); sh3=selection_hash(mi_f)
        rid3=hashlib.sha256(f"cdxr_p3|{ds}|{mech}|{st}|{ts}|{args.model}|{sh3[:8]}".encode()).hexdigest()[:20]
        if rid3 not in completed:
            try:
                out_f=fit_predict_core_model(args.model,X[tr],y[tr],X[va],y[va],X[te],ts)
                keep=np.ones(nf,dtype=bool); keep[mi_f]=False
                out_g=fit_predict_core_model(args.model,X[tr][:,keep],y[tr],X[va][:,keep],y[va],X[te][:,keep],ts)
                ga=float(roc_auc_score(y[te],out_g.probabilities))
                sdr=abs(fa-sa)-abs(ga-sa)
                _write(out,dict(run_id=rid3,dataset_index=ds,mechanism=mech,strength=st,training_seed=ts,governance_seed=-1,
                    model=args.model,policy="P3_blind_mi",budget_k=k,budget_fraction=BUDGET,status="SUCCESS",
                    strict_auc=sa,full_auc=fa,governed_auc=ga,strict_distance_reduction=sdr,initial_gap=gap,
                    removed_count=k,selection_hash=sh3))
            except Exception as e:
                _write(out,dict(run_id=rid3,dataset_index=ds,mechanism=mech,strength=st,training_seed=ts,governance_seed=-1,
                    model=args.model,policy="P3_blind_mi",budget_k=k,budget_fraction=BUDGET,status="FAILURE",failure_reason=str(e)[:200]))
        done+=1

        # P2 (20 seeds)
        for gs in GOV_SEEDS:
            rf=select_random(nf,k,gs,ds,ts); sh2=selection_hash(rf)
            rid2=hashlib.sha256(f"cdxr_p2|{ds}|{mech}|{st}|{ts}|{args.model}|{gs}|{sh2[:8]}".encode()).hexdigest()[:20]
            if rid2 not in completed:
                try:
                    keep=np.ones(nf,dtype=bool); keep[rf]=False
                    out_g=fit_predict_core_model(args.model,X[tr][:,keep],y[tr],X[va][:,keep],y[va],X[te][:,keep],ts)
                    ga=float(roc_auc_score(y[te],out_g.probabilities))
                    sdr=abs(fa-sa)-abs(ga-sa)
                    _write(out,dict(run_id=rid2,dataset_index=ds,mechanism=mech,strength=st,training_seed=ts,governance_seed=gs,
                        model=args.model,policy="P2_random",budget_k=k,budget_fraction=BUDGET,status="SUCCESS",
                        strict_auc=sa,full_auc=fa,governed_auc=ga,strict_distance_reduction=sdr,initial_gap=gap,
                        removed_count=k,selection_hash=sh2))
                except Exception as e:
                    _write(out,dict(run_id=rid2,dataset_index=ds,mechanism=mech,strength=st,training_seed=ts,governance_seed=gs,
                        model=args.model,policy="P2_random",budget_k=k,budget_fraction=BUDGET,status="FAILURE",failure_reason=str(e)[:200]))
            done+=1
        if done%5000==0: print(f"  {done}/{est} | {time.time()-t0:.0f}s",flush=True)
    print(f"DONE {done}/{est} in {time.time()-t0:.0f}s",flush=True)

def _write(out,rec):
    wh=not out.exists()
    with out.open("a",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=FIELDS); 
        if wh: w.writeheader()
        w.writerow(rec)

if __name__=="__main__": raise SystemExit(main())
