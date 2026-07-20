#!/usr/bin/env python3
"""T0-B1R2.3 Selection Audit — payload-level comparison."""
import gzip, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
R2 = ROOT / "results/edbt_t0_b_dryrun_r2"
OUT = ROOT / "results/edbt_t0_b_dryrun_r2_3"; OUT.mkdir(parents=True, exist_ok=True)

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def forbidden(*a,**kw): raise AssertionError("LR called in selection audit")
import src.leakbench.models.core_models as cm; cm.fit_predict_core_model = forbidden

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
from scripts.t0_b_v3.policy_selectors import *

def main():
    with open(ROOT/"configs/edbt_t0_b/dryrun_matrix_v4.json") as f: dr = json.load(f)
    KEYS, CONTRACTS, BUDGETS, GOV = dr["keys"], dr["contracts"], dr["budgets_bp"], dr["p2_governance_seeds"]

    m = {}
    for gz in ["policy_group_mapping_v3.jsonl.gz","semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gzip.decompress((ROOT/"results/edbt_t0_b"/gz).read_bytes()).decode("utf-8")
        m[gz] = {}
        for line in data.strip().split("\n"):
            r = json.loads(line); m[gz][(r["dataset_index"],r["mechanism"],r["strength"],r["training_seed"])] = r
    PM, EM = m["policy_group_mapping_v3.jsonl.gz"], m["semantic_evaluation_mapping_v3.jsonl.gz"]

    events = []; ranking_fits = 0; non_model = 0

    for k in KEYS:
        ds, mech, st, ts = k["dataset_index"],k["mechanism"],k["strength"],k["training_seed"]
        bundle = np.load(ROOT/k["bundle_path"],allow_pickle=False)
        X = np.concatenate((bundle["base_X"],bundle[f"block__{k['bundle_key']}"]),axis=1)
        y = bundle["y"]; tr = bundle["train_idx"]; n_total = X.shape[1]
        groups = PM[(ds,mech,st,ts)]["groups"]
        Xtr, ytr = X[tr], y[tr]
        s3 = score_mi(Xtr,ytr); non_model+=1; s4 = score_point_biserial(Xtr,ytr); non_model+=1
        s5 = score_lr_coef(Xtr,ytr); ranking_fits+=1; s6 = score_rf_permutation(Xtr,ytr); ranking_fits+=3
        ps = {"P3":s3,"P4":s4,"P5":s5,"P6":s6}
        gs = {pid: group_max_score(ps[pid],groups) for pid in ["P3","P4","P5","P6"]}

        for ct in CONTRACTS:
            for bp in BUDGETS:
                ku = compute_k(len(groups) if ct=="semantic_group" else n_total, bp)
                for gs_idx in GOV:
                    p2s = derive_p2_seed(gs_idx,ds,mech,st,ts,ct,bp); rng = np.random.RandomState(p2s)
                    if ct=="semantic_group":
                        sg = list(rng.choice(len(groups),ku,replace=False))
                        gids = [groups[i]["opaque_group_id"] for i in sg]
                        sh = hash_semantic_selection(ds,mech,st,ts,k["bundle_key"],k["bundle_sha256"],"P2",ct,bp,gids)
                        rc = []; [rc.extend(groups[i]["member_encoded_indices"]) for i in sg]
                    else:
                        rc = list(rng.choice(n_total,ku,replace=False))
                        sh = hash_encoded_selection(ds,mech,st,ts,k["bundle_key"],k["bundle_sha256"],"P2",ct,bp,np.array(sorted(rc),dtype=np.int64))
                        gids = []
                    events.append({"selection_hash":sh,"policy":"P2","contract":ct,"budget_bp":bp,"removed_encoded_indices":json.dumps([int(x) for x in sorted(rc)]),"removed_group_ids":json.dumps(sorted(gids)),"realized_encoded_cost":len(rc)})
                for pid in ["P3","P4","P5","P6"]:
                    if ct=="semantic_group":
                        sel = top_k_groups(gs[pid],ku)
                        sh = hash_semantic_selection(ds,mech,st,ts,k["bundle_key"],k["bundle_sha256"],pid,ct,bp,sel)
                        rc = []; [rc.extend(g["member_encoded_indices"]) for gid in sel for g in groups if g["opaque_group_id"]==gid]
                        gids = sel
                    else:
                        idx = top_k_columns(ps[pid],ku)
                        sh = hash_encoded_selection(ds,mech,st,ts,k["bundle_key"],k["bundle_sha256"],pid,ct,bp,np.array(sorted(idx),dtype=np.int64))
                        rc = list(idx)
                        gids = [g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"])&set(rc)]
                    events.append({"selection_hash":sh,"policy":pid,"contract":ct,"budget_bp":bp,"removed_encoded_indices":json.dumps([int(x) for x in sorted(rc)]),"removed_group_ids":json.dumps(sorted(gids)),"realized_encoded_cost":len(rc)})

    # Load R2 selection ledger
    r2_data = gzip.decompress((R2/"selection_ledger.csv.gz").read_bytes()).decode("utf-8")
    r2_rows = {}
    for line in r2_data.strip().split("\n")[1:]:
        parts = line.split(",")
        r2_rows[parts[0]] = {"policy":parts[1],"contract":parts[2],"budget_bp":parts[3],"removed_encoded_indices":parts[4],"removed_group_ids":parts[5],"realized_encoded_cost":parts[6]}

    audit_hashes = {e["selection_hash"] for e in events}
    r2_hashes = set(r2_rows.keys())
    missing = r2_hashes - audit_hashes; extra = audit_hashes - r2_hashes
    conflicts = 0; payload_mismatches = 0

    # Build canonical: deduplicate events by hash, check consistency
    canonical = {}
    for e in events:
        h = e["selection_hash"]
        if h in canonical:
            prev = canonical[h]
            if (prev["policy"]!=e["policy"] or prev["contract"]!=e["contract"] or
                prev["budget_bp"]!=e["budget_bp"] or prev["removed_encoded_indices"]!=e["removed_encoded_indices"] or
                prev["removed_group_ids"]!=e["removed_group_ids"]):
                conflicts += 1
        else:
            canonical[h] = e

    # Compare canonical against R2
    for h, ce in canonical.items():
        if h in r2_rows:
            r = r2_rows[h]
            if (ce["policy"]!=r["policy"] or ce["contract"]!=r["contract"] or
                ce["removed_encoded_indices"]!=r["removed_encoded_indices"] or
                ce["removed_group_ids"]!=r["removed_group_ids"]):
                payload_mismatches += 1

    # Write ledgers
    cols = ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]
    for name, rows in [("selection_event_ledger",events),("selection_canonical_ledger",list(canonical.values()))]:
        buf = io.StringIO(); pd.DataFrame(rows).to_csv(buf,columns=cols,index=False,header=True)
        compressed = gzip.compress(buf.getvalue().encode("utf-8"),mtime=0)
        (OUT/f"{name}.csv.gz").write_bytes(compressed)

    receipt = {
        "generated_events": len(events), "canonical_unique": len(canonical),
        "duplicate_events": len(events)-len(canonical),
        "governed_references": len(r2_hashes), "downstream_lr_calls": 0,
        "missing_hashes": len(missing), "extra_hashes": len(extra),
        "conflicting_duplicate_payloads": conflicts, "canonical_payload_mismatches": payload_mismatches,
        "p2_seed_coverage": 20, "ranking_model_fits": ranking_fits, "non_model_scoring": non_model,
        "r2_ledger_sha256": s(str(R2/"selection_ledger.csv.gz")),
        "pass": len(missing)==0 and len(extra)==0 and conflicts==0 and payload_mismatches==0,
    }
    with open(OUT/"selection_determinism_receipt.json","w") as f: json.dump(receipt,f,indent=2)
    print(f"Selection: {receipt['generated_events']} events, {receipt['canonical_unique']} unique, missing={len(missing)} extra={len(extra)} conflicts={conflicts} payload_mismatches={payload_mismatches} PASS={receipt['pass']}")

if __name__=="__main__": main()
