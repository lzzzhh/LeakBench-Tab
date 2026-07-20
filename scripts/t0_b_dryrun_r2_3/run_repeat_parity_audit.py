#!/usr/bin/env python3
"""T0-B1R2.2 Repeat-Fit Parity Audit — independent re-execution with full provenance."""
import gzip, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.leakbench.models.core_models import fit_predict_core_model
from sklearn.metrics import roc_auc_score

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SCI_FREEZE = "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845"

def main():
    out = ROOT / "results/edbt_t0_b_dryrun_r2_3"; out.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f: dr = json.load(f)

    records = []
    audit_lr_calls = 0

    for k in dr["keys"]:
        ds, mech, st, ts = k["dataset_index"], k["mechanism"], k["strength"], k["training_seed"]
        bundle = np.load(ROOT / k["bundle_path"], allow_pickle=False)
        assert s(k["bundle_path"]) == k["bundle_sha256"]
        X = np.concatenate((bundle["base_X"], bundle[f"block__{k['bundle_key']}"]), axis=1)
        y = bundle["y"]; tr, va, te = bundle["train_idx"], bundle["val_idx"], bundle["test_idx"]

        # Load evaluation mapping for leak mask
        data = gzip.decompress((ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz").read_bytes()).decode("utf-8")
        eval_info = None
        for line in data.strip().split("\n"):
            r = json.loads(line)
            if (r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]) == (ds, mech, st, ts):
                eval_info = r; break
        # Load policy mapping for groups
        data2 = gzip.decompress((ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz").read_bytes()).decode("utf-8")
        groups = None
        for line in data2.strip().split("\n"):
            r = json.loads(line)
            if (r["dataset_index"], r["mechanism"], r["strength"], r["training_seed"]) == (ds, mech, st, ts):
                groups = r["groups"]; break

        leak_mask = np.array([i in set(sum((g["member_encoded_indices"] for g in groups if g["opaque_group_id"] in eval_info["leak_group_ids"]), [])) for i in range(X.shape[1])])
        X_strict = X[:, ~leak_mask]

        for btype, X_data in [("strict", X_strict), ("full", X)]:
            o1 = fit_predict_core_model("lr", X_data[tr], y[tr], X_data[va], y[va], X_data[te], ts)
            o2 = fit_predict_core_model("lr", X_data[tr], y[tr], X_data[va], y[va], X_data[te], ts)
            audit_lr_calls += 2
            auc1 = float(roc_auc_score(y[te], o1.probabilities))
            auc2 = float(roc_auc_score(y[te], o2.probabilities))
            records.append({
                "key": f"{ds}_{mech}_{st}_{ts}", "type": btype,
                "auc1": round(auc1, 12), "auc2": round(auc2, 12),
                "auc_abs_diff": round(abs(auc1 - auc2), 15),
                "prob_max_diff": round(float(np.max(np.abs(o1.probabilities - o2.probabilities))), 15),
                "model_source_sha256": s("src/leakbench/models/core_models.py"),
                "model_config_sha256": s("configs/edbt_t0_b/model_factories_v4.json"),
                "bundle_sha256": k["bundle_sha256"],
                "train_idx_sha256": k["train_idx_hash"],
                "val_idx_sha256": k["val_idx_hash"],
                "test_idx_sha256": k["test_idx_hash"],
                "scientific_freeze_sha": SCI_FREEZE,
            })

    receipt = {
        "records": records,
        "auc_tolerance": 1e-12, "prob_tolerance": 1e-12,
        "r2_scientific_baseline_fits": 8,
        "r2_scientific_governed_fits": 576,
        "r2_scientific_downstream_fits": 584,
        "r2_original_additional_repeat_fits": 8,
        "r2_original_total_lr_calls": 592,
        "r2_3_evidence_audit_fits": audit_lr_calls,
        "all_pass": all(r["auc_abs_diff"] <= 1e-12 and r["prob_max_diff"] <= 1e-12 for r in records),
    }
    with open(out / "repeat_fit_provenance_receipt.json", "w") as f: json.dump(receipt, f, indent=2)
    print(f"Repeat parity: {len(records)} records, all_pass={receipt['all_pass']}, audit_lr_calls={audit_lr_calls}")

if __name__ == "__main__":
    main()
