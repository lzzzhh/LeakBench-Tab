#!/usr/bin/env python3
"""T0-B1R2.1 Selection Audit — verifies selection determinism without downstream LR."""
import gzip, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

from scripts.t0_b_v3.budget_contract import compute_k
from scripts.t0_b_v3.seed_contract import derive_p2_seed
from scripts.t0_b_v3.selection_hash import hash_encoded_selection, hash_semantic_selection
from scripts.t0_b_v3.policy_selectors import (
    score_mi, score_point_biserial, score_lr_coef, score_rf_permutation,
    group_max_score, top_k_groups, top_k_columns,
)

# Monkeypatch: fit_predict_core_model must NEVER be called
def _blocked_fit(*args, **kwargs):
    raise RuntimeError("fit_predict_core_model called in selection-only audit — HARD STOP")

# Apply monkeypatch
import src.leakbench.models.core_models as cm
cm.fit_predict_core_model = _blocked_fit

R2_SEL_CSV = ROOT / "results/edbt_t0_b_dryrun_r2/selection_ledger.csv.gz"

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out = ROOT / "results/edbt_t0_b_dryrun_r2_1"; out.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f: dr = json.load(f)
    KEYS = dr["keys"]; CONTRACTS = dr["contracts"]; BUDGETS = dr["budgets_bp"]; GOV_SEEDS = dr["p2_governance_seeds"]

    # Load mappings
    m = {}
    for gz_name in ["policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gzip.decompress((ROOT / "results/edbt_t0_b" / gz_name).read_bytes()).decode("utf-8")
        m[gz_name] = {}
        for line in data.strip().split("\n"):
            r = json.loads(line); m[gz_name][(r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"])] = r
    POLICY_MAP = m["policy_group_mapping_v3.jsonl.gz"]

    t0 = time.time()
    selection_rows = []
    ranking_model_fits = 0; non_model_scoring = 0

    for k in KEYS:
        ds, mech, st, ts = k["dataset_index"], k["mechanism"], k["strength"], k["training_seed"]
        bundle = np.load(ROOT / k["bundle_path"], allow_pickle=False)
        assert s(k["bundle_path"]) == k["bundle_sha256"]
        bkey = k["bundle_key"]
        X = np.concatenate((bundle["base_X"], bundle[f"block__{bkey}"]), axis=1)
        y = bundle["y"]; tr = bundle["train_idx"]
        n_total = X.shape[1]
        groups = POLICY_MAP[(ds, mech, st, ts)]["groups"]

        Xtr, ytr = X[tr], y[tr]
        scores_p3 = score_mi(Xtr, ytr); non_model_scoring += 1
        scores_p4 = score_point_biserial(Xtr, ytr); non_model_scoring += 1
        scores_p5 = score_lr_coef(Xtr, ytr); ranking_model_fits += 1
        scores_p6 = score_rf_permutation(Xtr, ytr); ranking_model_fits += 3
        pscores = {"P3": scores_p3, "P4": scores_p4, "P5": scores_p5, "P6": scores_p6}
        gscores = {pid: group_max_score(pscores[pid], groups) for pid in ["P3","P4","P5","P6"]}

        for contract in CONTRACTS:
            for bp in BUDGETS:
                k_units = compute_k(len(groups) if contract == "semantic_group" else n_total, bp)
                # P2
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
                    selection_rows.append({"selection_hash": shash, "policy": "P2", "contract": contract, "budget_bp": bp, "removed_encoded_indices": json.dumps([int(x) for x in sorted(removed_cols)]), "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(removed_cols)})

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
                    selection_rows.append({"selection_hash": shash, "policy": pid, "contract": contract, "budget_bp": bp, "removed_encoded_indices": json.dumps([int(x) for x in sorted(removed_cols)]), "removed_group_ids": json.dumps(sorted(gids)), "realized_encoded_cost": len(removed_cols)})

    # Compare with R2 selection ledger (hash-level comparison)
    r2_data = gzip.decompress(R2_SEL_CSV.read_bytes()).decode("utf-8")
    r2_hashes = set()
    for line in r2_data.strip().split("\n")[1:]:
        r2_hashes.add(line.split(",")[0])

    audit_hashes = {sr["selection_hash"] for sr in selection_rows}
    missing = audit_hashes - r2_hashes
    extra = r2_hashes - audit_hashes
    mismatches = len(missing) + len(extra)

    # Write audit ledger
    import pandas as pd
    cols = ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]
    buf = io.StringIO(); pd.DataFrame(selection_rows).to_csv(buf, columns=cols, index=False, header=True)
    compressed = gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)
    (out / "selection_audit_ledger.csv.gz").write_bytes(compressed)

    receipt = {
        "downstream_lr_calls": 0,
        "selection_count": len(selection_rows),
        "r2_selection_count": len(r2_hashes),
        "payload_mismatches": mismatches,
        "ranking_model_fits": ranking_model_fits,
        "non_model_scoring": non_model_scoring,
        "r2_ledger_sha256": s(str(R2_SEL_CSV)),
        "audit_ledger_sha256": hashlib.sha256((out/"selection_audit_ledger.csv.gz").read_bytes()).hexdigest(),
        "wall_clock_s": round(time.time() - t0, 2),
        "pass": mismatches == 0,
    }
    with open(out / "selection_determinism_receipt.json", "w") as f: json.dump(receipt, f, indent=2)
    print(f"Selection audit: {len(selection_rows)} generated, {len(r2_hashes)} R2, {mismatches} mismatches")
    print(f"PASS: {mismatches == 0}")
    if mismatches > 0: sys.exit(1)

if __name__ == "__main__":
    main()
