#!/usr/bin/env python3
"""Fit strict+full baseline for all 5500 keys × 3 models using official adapter.
Output: baseline_cells.csv — strict/full AUC per key/model, computed ONCE.
"""
from __future__ import annotations
import csv, hashlib, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.leakbench.models.core_models import fit_predict_core_model

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(row):
    b=np.load(ROOT/row.bundle_path,allow_pickle=False); k=row.bundle_key
    X=np.concatenate((b['base_X'],b[f'block__{k}']),axis=1); y=b['y']; mask=b[f'leak_mask__{k}']
    return X,y,b['train_idx'],b['val_idx'],b['test_idx'],mask

man=pd.read_csv(ROOT/'artifacts/sp6/sp6_bundle_manifest.csv')
out=ROOT/'results/edbt_eab_crosslearner_confirmatory_v2/baseline_cells.csv'
models=['lr','rf','lightgbm']
total=len(man)*len(models); done=0; t0=time.time()
print(f"Baseline: {len(man)} keys x {len(models)} models = {total} cells",flush=True)
with out.open('w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=['dataset_index','mechanism','strength','training_seed','model','strict_auc','full_auc','initial_gap','adapter_sha'])
    w.writeheader()
    for _,r in man.iterrows():
        ds,mech,st,ts=int(r.dataset_index),r.mechanism,r.strength,int(r.seed)
        X,y,tr,va,te,mask=load(r); sc=np.where(~mask)[0]
        for m in models:
            try:
                out_s=fit_predict_core_model(m,X[tr][:,sc],y[tr],X[va][:,sc],y[va],X[te][:,sc],ts)
                out_f=fit_predict_core_model(m,X[tr],y[tr],X[va],y[va],X[te],ts)
                sa=float(roc_auc_score(y[te],out_s.probabilities))
                fa=float(roc_auc_score(y[te],out_f.probabilities))
                w.writerow(dict(dataset_index=ds,mechanism=mech,strength=st,training_seed=ts,model=m,
                                strict_auc=sa,full_auc=fa,initial_gap=abs(fa-sa),
                                adapter_sha=sha('src/leakbench/models/core_models.py')))
            except: pass
            done+=1
            if done%2000==0: print(f"  {done}/{total} | {time.time()-t0:.0f}s",flush=True)
print(f"DONE {done}/{total} in {time.time()-t0:.0f}s")
