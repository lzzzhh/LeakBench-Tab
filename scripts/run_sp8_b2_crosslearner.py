#!/usr/bin/env python3
"""run_sp8_b2_crosslearner.py — B2 RF + LightGBM governance.

Multi-seed P2 (20 gov seeds) at 20% budget. P3 blind MI. P0 baseline.
Same strict/full references as SP8. Outputs to edbt_eab_revision CSV.
"""
from __future__ import annotations
import argparse, csv, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT))

GOV_SEEDS = [2026071700 + i for i in range(20)]
BUDGET = 0.20

FIELDS = [
    "run_id","dataset_index","mechanism","strength","training_seed","governance_seed",
    "model","policy","budget_k","budget_fraction",
    "status","failure_reason",
    "strict_auc","full_auc","governed_auc",
    "strict_distance_reduction","initial_gap",
    "removed_count","selection_mask_hash",
]

def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selection_hash(indices):
    values = np.sort(np.asarray(indices, dtype='<i8'))
    return hashlib.sha256(b"encoded_column_indices_v1\0" + values.tobytes()).hexdigest()

def load_cell(row):
    bundle = ROOT / row["bundle_path"]
    if sha_file(bundle) != str(row["bundle_sha256"]).lower():
        raise RuntimeError(f"bundle hash mismatch")
    key = str(row["bundle_key"])
    with np.load(bundle, allow_pickle=False) as b:
        X = np.concatenate((np.asarray(b["base_X"]), np.asarray(b[f"block__{key}"])), axis=1)
        y = np.asarray(b["y"])
        tr = np.asarray(b["train_idx"]); te = np.asarray(b["test_idx"])
        mask = np.asarray(b[f"leak_mask__{key}"])
    return X, y, tr, te, mask

def fit_model(model, Xtr, ytr, Xte, yte, seed):
    if model=="rf":
        m=RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Xtr,ytr)
    elif model=="lightgbm":
        from lightgbm import LGBMClassifier
        m=LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1, device='cpu').fit(Xtr,ytr)
    else:
        raise ValueError(f"unknown model {model}")
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:,1]))

def select_fields_mi(scores,k): return np.argsort(scores)[::-1][:k]
def select_fields_random(n,k,gs,ds_i,ts):
    rng=np.random.RandomState((gs*100+ds_i*7+ts*13)%(2**31-1))
    return rng.choice(n,k,replace=False)

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--models",default="rf,lightgbm")
    ap.add_argument("--allow-run",action="store_true")
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args(argv)
    if not args.allow_run: raise RuntimeError("locked")

    man=pd.read_csv(ROOT/args.bundle_manifest)
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True)
    completed=set()
    if out.exists():
        if not args.resume: raise FileExistsError(f"exists; pass --resume")
        completed=set(pd.read_csv(out)["run_id"].astype(str))

    models=args.models.split(",")
    n_keys=len(man); est=n_keys*(1+1+len(GOV_SEEDS))*len(models)
    print(f"B2: {n_keys} keys x {1+1+len(GOV_SEEDS)} fits x {len(models)} models = est {est} rows",flush=True)
    started=time.time(); done=0

    for _,row in man.iterrows():
        ds_i=int(row["dataset_index"]); mech=row["mechanism"]
        strength=row["strength"]; tseed=int(row["seed"])
        try:
            X,y,tr,te,mask=load_cell(row)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load bundle for dataset={ds_i} mechanism={mech} "
                f"strength={strength} seed={tseed}"
            ) from exc
        n_features=X.shape[1]; strict_cols=np.where(~mask)[0]
        k=max(1,round(n_features*BUDGET))
        if k>=n_features: continue

        for model in models:
            strict_auc=fit_model(model,X[tr][:,strict_cols],y[tr],X[te][:,strict_cols],y[te],tseed)
            full_auc=fit_model(model,X[tr],y[tr],X[te],y[te],tseed)
            gap=abs(full_auc-strict_auc)

            # P0
            rid0=hashlib.sha256(f"b2p0|{ds_i}|{mech}|{strength}|{tseed}|{model}".encode()).hexdigest()[:20]
            done+=1
            if rid0 not in completed:
                _append(out,dict(run_id=rid0,dataset_index=ds_i,mechanism=mech,strength=strength,
                    training_seed=tseed,governance_seed=-1,model=model,policy="P0_keep",budget_k=0,budget_fraction=0.0,
                    status="SUCCESS",strict_auc=round(strict_auc,6),full_auc=round(full_auc,6),
                    governed_auc=round(full_auc,6),strict_distance_reduction=0.0,initial_gap=round(gap,6),
                    removed_count=0,selection_mask_hash=selection_hash([])))

            # P3
            mi_scores=mutual_info_classif(X[tr],y[tr],random_state=42)
            mi_scores=np.nan_to_num(mi_scores,nan=0.0)
            mi_fields=select_fields_mi(mi_scores,k)
            keep_p3=np.ones(n_features,dtype=bool); keep_p3[mi_fields]=False
            gov_auc=fit_model(model,X[tr][:,keep_p3],y[tr],X[te][:,keep_p3],y[te],tseed)
            sdr=abs(full_auc-strict_auc)-abs(gov_auc-strict_auc)
            rid3=hashlib.sha256(f"b2p3|{ds_i}|{mech}|{strength}|{tseed}|{model}|{k}".encode()).hexdigest()[:20]
            done+=1
            if rid3 not in completed:
                _append(out,dict(run_id=rid3,dataset_index=ds_i,mechanism=mech,strength=strength,
                    training_seed=tseed,governance_seed=-1,model=model,policy="P3_blind_mi",budget_k=k,budget_fraction=BUDGET,
                    status="SUCCESS",strict_auc=round(strict_auc,6),full_auc=round(full_auc,6),
                    governed_auc=round(gov_auc,6),strict_distance_reduction=round(sdr,6),
                    initial_gap=round(gap,6),removed_count=k,selection_mask_hash=selection_hash(mi_fields)))

            # P2 multi-seed
            for gs in GOV_SEEDS:
                rm_fields=select_fields_random(n_features,k,gs,ds_i,tseed)
                keep_p2=np.ones(n_features,dtype=bool); keep_p2[rm_fields]=False
                gov_auc=fit_model(model,X[tr][:,keep_p2],y[tr],X[te][:,keep_p2],y[te],tseed)
                sdr=abs(full_auc-strict_auc)-abs(gov_auc-strict_auc)
                rid2=hashlib.sha256(f"b2p2|{ds_i}|{mech}|{strength}|{tseed}|{model}|{gs}".encode()).hexdigest()[:20]
                done+=1
                if rid2 not in completed:
                    _append(out,dict(run_id=rid2,dataset_index=ds_i,mechanism=mech,strength=strength,
                        training_seed=tseed,governance_seed=gs,model=model,policy="P2_random",budget_k=k,budget_fraction=BUDGET,
                        status="SUCCESS",strict_auc=round(strict_auc,6),full_auc=round(full_auc,6),
                        governed_auc=round(gov_auc,6),strict_distance_reduction=round(sdr,6),
                        initial_gap=round(gap,6),removed_count=k,selection_mask_hash=selection_hash(rm_fields)))
            if done%5000==0: print(f"  {done}/{est} | {time.time()-started:.0f}s",flush=True)
    print(f"DONE {done}/{est} in {time.time()-started:.0f}s",flush=True)
    return 0

def _append(out,rec):
    wh=not out.exists()
    with out.open("a",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=FIELDS)
        if wh: w.writeheader()
        w.writerow(rec)

if __name__=="__main__": raise SystemExit(main())
