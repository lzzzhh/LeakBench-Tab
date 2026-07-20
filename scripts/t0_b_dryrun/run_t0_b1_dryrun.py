#!/usr/bin/env python3
"""T0-B1 Dry-Run Runner — frozen-protocol 4-key execution.

Reads dryrun_matrix_v4.json, executes exact 4 keys, writes deterministic gzip ledgers.
Imports from frozen V3/V4 modules — does NOT duplicate logic.
"""
from __future__ import annotations
import csv, gzip, hashlib, io, json, sys, time
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

# ============================================================
# Load frozen config
# ============================================================
with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
    DRYRUN = json.load(f)
KEYS = DRYRUN["keys"]
CONTRACTS = DRYRUN["contracts"]
BUDGETS_BP = DRYRUN["budgets_bp"]
POLICIES = DRYRUN["policies"]
GOV_SEEDS = DRYRUN["p2_governance_seeds"]
assert len(KEYS) == 4

# Load policy+eval mapping ledgers
def load_mapping(gz_path):
    data = gzip.decompress((ROOT / gz_path).read_bytes()).decode("utf-8")
    m = {}
    for line in data.strip().split("\n"):
        r = json.loads(line)
        m[(r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"])] = r
    return m

POLICY_MAP = load_mapping("results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz")
EVAL_MAP = load_mapping("results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz")

# Load SP8 baseline for parity
SP8 = pd.read_csv(ROOT / "artifacts/sp8/governance_clean.csv")
SP8_P0 = SP8[SP8.policy == "P0_keep"][["dataset_index","mechanism","strength","seed","strict_auc","full_auc"]].rename(columns={"seed":"training_seed"})

# ============================================================
# Gzip helpers
# ============================================================
def write_gz(path, df, cols):
    buf = io.StringIO()
    df.to_csv(buf, columns=cols, index=False, header=True)
    compressed = gzip.compress(buf.getvalue().encode("utf-8"), mtime=0)
    Path(path).write_bytes(compressed)
    return hashlib.sha256(compressed).hexdigest()

def append_gz(path, df, cols):
    if Path(path).exists():
        existing = gzip.decompress(Path(path).read_bytes()).decode("utf-8")
    else:
        existing = ""
    buf = io.StringIO()
    df.to_csv(buf, columns=cols, index=False, header=(not existing))
    combined = existing + buf.getvalue()
    compressed = gzip.compress(combined.encode("utf-8"), mtime=0)
    Path(path).write_bytes(compressed)
    return hashlib.sha256(compressed).hexdigest()

# ============================================================
# Main
# ============================================================
def main():
    out = ROOT / "results/edbt_t0_b_dryrun"
    out.mkdir(parents=True, exist_ok=True)

    baseline_rows = []
    governed_rows = []
    selection_rows = []
    failure_rows = []
    ranker_times = {}
    fit_times = {"strict": [], "full": [], "governed": []}
    t_total_start = time.time()

    for key_idx, k in enumerate(KEYS):
        ds = k["dataset_index"]; mech = k["mechanism"]; st = k["strength"]; ts = k["training_seed"]
        print(f"\n=== Key {key_idx+1}/4: DS={ds} {mech} {st} seed={ts} ===", flush=True)

        # --- Bundle Gate ---
        bundle = np.load(ROOT / k["bundle_path"], allow_pickle=False)
        disk_sha = hashlib.sha256((ROOT / k["bundle_path"]).read_bytes()).hexdigest()
        assert disk_sha == k["bundle_sha256"], f"Disk SHA mismatch"
        bkey = k["bundle_key"]
        X_block = np.concatenate((bundle["base_X"], bundle[f"block__{bkey}"]), axis=1)
        y = bundle["y"]
        tr, va, te = bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]
        for sn in ["train_idx", "val_idx", "test_idx"]:
            assert hashlib.sha256(bundle[sn].tobytes()).hexdigest() == k[f"{sn}_hash"], f"{sn} hash mismatch"

        n_total = X_block.shape[1]
        assert n_total == k["n_encoded_columns"]

        # Verify mapping ledgers
        kt = (ds, mech, st, ts)
        assert kt in POLICY_MAP, "Key not in policy mapping"
        assert kt in EVAL_MAP, "Key not in evaluation mapping"
        groups = POLICY_MAP[kt]["groups"]
        eval_info = EVAL_MAP[kt]
        leak_mask = np.array([i in set(sum((g["member_encoded_indices"] for g in groups if g["opaque_group_id"] in eval_info["leak_group_ids"]), [])) for i in range(n_total)])

        # --- Baseline LR Fits ---
        # Strict fit (remove leak columns)
        X_strict = X_block[:, ~leak_mask]
        t0 = time.time()
        strict_out = fit_predict_core_model("lr", X_strict[tr], y[tr], X_strict[va], y[va], X_strict[te], ts)
        fit_times["strict"].append(time.time() - t0)
        strict_auc = float(np.mean([float(a > b) for a in strict_out.probabilities for b in strict_out.probabilities]) if False else 0)
        from sklearn.metrics import roc_auc_score
        strict_auc = float(roc_auc_score(y[te], strict_out.probabilities))

        # Full fit (all columns)
        t0 = time.time()
        full_out = fit_predict_core_model("lr", X_block[tr], y[tr], X_block[va], y[va], X_block[te], ts)
        fit_times["full"].append(time.time() - t0)
        full_auc = float(roc_auc_score(y[te], full_out.probabilities))

        # Baseline parity with SP8
        sp8_match = SP8_P0[(SP8_P0.dataset_index == ds) & (SP8_P0.mechanism == mech) & (SP8_P0.strength == st) & (SP8_P0.training_seed == ts)]
        assert len(sp8_match) == 1, f"SP8 baseline not found for {kt}"
        assert abs(strict_auc - float(sp8_match.strict_auc.iloc[0])) <= 1e-6, f"Strict parity fail: {strict_auc:.10f} vs {sp8_match.strict_auc.iloc[0]:.10f}"
        assert abs(full_auc - float(sp8_match.full_auc.iloc[0])) <= 1e-6, f"Full parity fail"

        # Baseline rows
        for which, auc_val in [("strict", strict_auc), ("full", full_auc)]:
            rid = hashlib.sha256(f"t0b_baseline|{ds}|{mech}|{st}|{ts}|lr|{which}".encode()).hexdigest()[:20]
            baseline_rows.append({
                "run_id": rid, "dataset_index": ds, "mechanism": mech, "strength": st,
                "training_seed": ts, "learner": "lr", "baseline_type": which, "auc": auc_val,
            })

        # --- Ranking Scores (once per key, shared across contracts/budgets) ---
        Xtr, ytr = X_block[tr], y[tr]
        t0 = time.time(); scores_p3 = score_mi(Xtr, ytr); ranker_times["P3_MI"] = ranker_times.get("P3_MI", 0) + (time.time() - t0)
        t0 = time.time(); scores_p4 = score_point_biserial(Xtr, ytr); ranker_times["P4_PBISERIAL"] = ranker_times.get("P4_PBISERIAL", 0) + (time.time() - t0)
        t0 = time.time(); scores_p5 = score_lr_coef(Xtr, ytr); ranker_times["P5_LR_COEF"] = ranker_times.get("P5_LR_COEF", 0) + (time.time() - t0)
        t0 = time.time(); scores_p6 = score_rf_permutation(Xtr, ytr); ranker_times["P6_RF_PERM"] = ranker_times.get("P6_RF_PERM", 0) + (time.time() - t0)

        policy_scores = {"P3": scores_p3, "P4": scores_p4, "P5": scores_p5, "P6": scores_p6}
        group_scores_cache = {}
        for pid in ["P3", "P4", "P5", "P6"]:
            group_scores_cache[pid] = group_max_score(policy_scores[pid], groups)

        # --- For each contract × budget ---
        for contract in CONTRACTS:
            n_groups = len(groups)
            for bp in BUDGETS_BP:
                if contract == "semantic_group":
                    k_units = compute_k(n_groups, bp)
                    unit_type = "group"
                else:
                    k_units = compute_k(n_total, bp)
                    unit_type = "column"

                # --- P2 × 20 seeds ---
                for gs in GOV_SEEDS:
                    p2_seed = derive_p2_seed(gs, ds, mech, st, ts, contract, bp)
                    rng = np.random.RandomState(p2_seed)

                    if contract == "semantic_group":
                        selected_groups = list(rng.choice(n_groups, k_units, replace=False))
                        group_ids = [groups[i]["opaque_group_id"] for i in selected_groups]
                        shash = hash_semantic_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], "P2", contract, bp, group_ids)
                        # Expand groups to encoded columns
                        removed_cols = []
                        for i in selected_groups:
                            removed_cols.extend(groups[i]["member_encoded_indices"])
                    else:
                        removed_cols = list(rng.choice(n_total, k_units, replace=False))
                        shash = hash_encoded_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], "P2", contract, bp, np.array(sorted(removed_cols), dtype=np.int64))
                        group_ids = []

                    selection_rows.append({
                        "selection_hash": shash, "policy": "P2", "contract": contract,
                        "budget_bp": bp, "removed_encoded_indices": json.dumps(sorted(removed_cols)),
                        "removed_group_ids": json.dumps(sorted(group_ids)), "realized_encoded_cost": len(removed_cols),
                    })

                    # Governed LR fit
                    keep = np.ones(n_total, dtype=bool)
                    keep[removed_cols] = False
                    X_gov = X_block[:, keep]
                    t0 = time.time()
                    gov_out = fit_predict_core_model("lr", X_gov[tr], y[tr], X_gov[va], y[va], X_gov[te], ts)
                    fit_times["governed"].append(time.time() - t0)
                    gov_auc = float(roc_auc_score(y[te], gov_out.probabilities))
                    governed_rows.append(make_gov_row(ds, mech, st, ts, gs, "lr", "P2", contract, bp, strict_auc, full_auc, gov_auc, shash, removed_cols, leak_mask, groups, eval_info, k_units))

                # --- P3-P6 deterministic ---
                for pid in ["P3", "P4", "P5", "P6"]:
                    scores = policy_scores[pid]
                    gscores = group_scores_cache[pid]

                    if contract == "semantic_group":
                        selected = top_k_groups(gscores, k_units)
                        shash = hash_semantic_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], pid, contract, bp, selected)
                        removed_cols = []
                        for gid in selected:
                            for g in groups:
                                if g["opaque_group_id"] == gid:
                                    removed_cols.extend(g["member_encoded_indices"])
                        group_ids = selected
                    else:
                        removed_idx = top_k_columns(scores, k_units)
                        shash = hash_encoded_selection(ds, mech, st, ts, bkey, k["bundle_sha256"], pid, contract, bp, np.array(sorted(removed_idx), dtype=np.int64))
                        removed_cols = list(removed_idx)
                        group_ids = [g["opaque_group_id"] for g in groups if set(g["member_encoded_indices"]) & set(removed_cols)]

                    selection_rows.append({
                        "selection_hash": shash, "policy": pid, "contract": contract,
                        "budget_bp": bp, "removed_encoded_indices": json.dumps(sorted(removed_cols)),
                        "removed_group_ids": json.dumps(sorted(group_ids)), "realized_encoded_cost": len(removed_cols),
                    })

                    keep = np.ones(n_total, dtype=bool)
                    keep[removed_cols] = False
                    X_gov = X_block[:, keep]
                    t0 = time.time()
                    gov_out = fit_predict_core_model("lr", X_gov[tr], y[tr], X_gov[va], y[va], X_gov[te], ts)
                    fit_times["governed"].append(time.time() - t0)
                    gov_auc = float(roc_auc_score(y[te], gov_out.probabilities))
                    governed_rows.append(make_gov_row(ds, mech, st, ts, -1, "lr", pid, contract, bp, strict_auc, full_auc, gov_auc, shash, removed_cols, leak_mask, groups, eval_info, k_units))

    # --- Write outputs ---
    bl_cols = ["run_id","dataset_index","mechanism","strength","training_seed","learner","baseline_type","auc"]
    write_gz(out / "baseline_ledger.csv.gz", pd.DataFrame(baseline_rows), bl_cols)

    gl_cols = ["run_id","dataset_index","mechanism","strength","training_seed","governance_seed","learner","policy","contract","budget_bp","strict_auc","full_auc","governed_auc","legacy_sdr","directional_repair","same_side_residual","overcorrection","introduced_distortion","residual_harm","removed_leak_count","removed_legit_count","leak_recall","deletion_precision","legit_retention","residual_leak_fraction","semantic_full_removed_count","semantic_any_hit_count","semantic_partial_removed_count","semantic_group_recall_full","semantic_group_recall_any","partial_group_violation_rate","selection_hash","realized_cost"]
    write_gz(out / "governed_ledger.csv.gz", pd.DataFrame(governed_rows), gl_cols)

    sl_cols = ["selection_hash","policy","contract","budget_bp","removed_encoded_indices","removed_group_ids","realized_encoded_cost"]
    write_gz(out / "selection_ledger.csv.gz", pd.DataFrame(selection_rows), sl_cols)

    # Failure ledger (empty)
    fl_cols = ["run_id","failure_reason","exception_type"]
    write_gz(out / "failure_ledger.csv.gz", pd.DataFrame(columns=fl_cols), fl_cols)

    # Receipts
    total_t = time.time() - t_total_start
    runtime_receipt = {
        "total_wall_clock_s": round(total_t, 2),
        "ranker_times_s": {k: round(v, 2) for k, v in ranker_times.items()},
        "fit_times_s": {k: {"count": len(v), "total_s": round(sum(v), 2), "mean_s": round(np.mean(v), 4) if v else 0} for k, v in fit_times.items()},
    }
    with open(out / "runtime_receipt.json", "w") as f: json.dump(runtime_receipt, f, indent=2)
    env_receipt = {"python_version": sys.version, "numpy": np.__version__}
    with open(out / "environment_receipt.json", "w") as f: json.dump(env_receipt, f, indent=2)
    ranker_receipt = {"ranker_count": len(ranker_times)}
    with open(out / "ranker_receipt.json", "w") as f: json.dump(ranker_receipt, f, indent=2)

    print(f"\n=== DRY RUN COMPLETE ===")
    print(f"Baseline: {len(baseline_rows)} rows, Governed: {len(governed_rows)} rows, Selections: {len(selection_rows)}, Failures: 0")
    print(f"Wall clock: {total_t:.0f}s")


def make_gov_row(ds, mech, st, ts, gs, learner, policy, contract, bp, strict_auc, full_auc, governed_auc, shash, removed_cols, leak_mask, groups, eval_info, k_units):
    opp = abs(full_auc - strict_auc); go = governed_auc - strict_auc
    sg = full_auc - strict_auc; d = np.sign(sg) if opp > 1e-12 else 0
    ssr = max(d * go, 0) if opp > 1e-12 else 0.0
    ovc = max(-d * go, 0) if opp > 1e-12 else 0.0
    drep = opp - ssr if opp > 1e-12 else 0.0
    legacy_sdr = opp - abs(go)
    intr = abs(go) if opp <= 1e-12 else 0.0

    rl = int(leak_mask[removed_cols].sum()); rg = len(removed_cols) - rl
    n_leak = int(leak_mask.sum()); n_legit = len(leak_mask) - n_leak
    leak_rec = rl / n_leak if n_leak > 0 else 0.0
    del_prec = rl / len(removed_cols) if len(removed_cols) > 0 else 0.0
    leg_ret = 1 - rg / n_legit if n_legit > 0 else 1.0

    # Semantic metrics
    removed_set = set(removed_cols)
    full_count = 0; any_count = 0; partial_count = 0
    for g in groups:
        members = set(g["member_encoded_indices"])
        if members & removed_set:
            any_count += 1
            if members <= removed_set:
                full_count += 1
            else:
                partial_count += 1
    n_leak_groups = len(eval_info["leak_group_ids"])
    fg_recall = full_count / n_leak_groups if n_leak_groups > 0 else 0.0

    rid = hashlib.sha256(f"t0b_gov|{ds}|{mech}|{st}|{ts}|{gs}|{learner}|{policy}|{contract}|{bp}".encode()).hexdigest()[:20]
    return {
        "run_id": rid, "dataset_index": ds, "mechanism": mech, "strength": st,
        "training_seed": ts, "governance_seed": gs, "learner": learner, "policy": policy,
        "contract": contract, "budget_bp": bp, "strict_auc": strict_auc, "full_auc": full_auc,
        "governed_auc": governed_auc, "legacy_sdr": legacy_sdr, "directional_repair": drep,
        "same_side_residual": ssr, "overcorrection": ovc, "introduced_distortion": intr,
        "residual_harm": go, "removed_leak_count": rl, "removed_legit_count": rg,
        "leak_recall": leak_rec, "deletion_precision": del_prec, "legit_retention": leg_ret,
        "residual_leak_fraction": (n_leak - rl) / len(leak_mask) if len(leak_mask) > 0 else 0.0,
        "semantic_full_removed_count": full_count, "semantic_any_hit_count": any_count,
        "semantic_partial_removed_count": partial_count,
        "semantic_group_recall_full": fg_recall,
        "semantic_group_recall_any": any_count / n_leak_groups if n_leak_groups > 0 else 0.0,
        "partial_group_violation_rate": partial_count / len(groups) if len(groups) > 0 else 0.0,
        "selection_hash": shash, "realized_cost": len(removed_cols),
    }


if __name__ == "__main__":
    main()
