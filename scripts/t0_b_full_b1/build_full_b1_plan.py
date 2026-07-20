#!/usr/bin/env python3
"""T0-B Full-B1 Plan Generator — derives 5,500-key canonical plan from frozen manifest."""
import gzip, hashlib, io, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
SCI_FREEZE = "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845"
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/edbt_t0_b_full_b1_preflight")
    ap.add_argument("--plan-only", action="store_true"); ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--shard-counts", type=int, nargs="+", default=[32,64,128,256])
    args = ap.parse_args()
    out = ROOT / args.output_dir; out.mkdir(parents=True, exist_ok=True)

    # Derive from frozen manifest
    man = pd.read_csv(ROOT/"artifacts/sp6/sp6_bundle_manifest.csv")
    assert len(man) == 5500, f"Expected 5500 keys, got {len(man)}"
    datasets = sorted(man.dataset_index.unique())
    mechanisms = sorted(man.mechanism.unique())
    strengths = sorted(man.strength.unique())
    seeds = sorted(man.seed.unique())
    assert len(datasets)==20 and len(mechanisms)==11 and len(strengths)==5 and len(seeds)==5

    # Load mappings
    for gz in ["policy_group_mapping_v3.jsonl.gz","semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gzip.decompress((ROOT/"results/edbt_t0_b"/gz).read_bytes()).decode("utf-8")
        m = {}
        for line in data.strip().split("\n"):
            r = json.loads(line)
            m[(r["dataset_index"],r["mechanism"],r["strength"],r["training_seed"])] = r
        assert len(m) == 5500, f"{gz} has {len(m)} keys"

    # Build key plan
    key_plan = []
    for _, r in man.iterrows():
        ds, mech, st, ts = int(r.dataset_index), r.mechanism, r.strength, int(r.seed)
        cid = hashlib.sha256(f"t0b_full_b1_key|{ds}|{mech}|{st}|{ts}".encode()).hexdigest()[:24]
        bundle = np.load(ROOT/r.bundle_path, allow_pickle=False)
        key_plan.append({
            "canonical_key_id": cid, "dataset_index": ds, "mechanism": mech,
            "strength": st, "training_seed": ts,
            "bundle_path": r.bundle_path, "bundle_key": r.bundle_key,
            "bundle_sha256": r.bundle_sha256,
            "train_idx_sha256": hashlib.sha256(bundle["train_idx"].tobytes()).hexdigest(),
            "val_idx_sha256": hashlib.sha256(bundle["val_idx"].tobytes()).hexdigest(),
            "test_idx_sha256": hashlib.sha256(bundle["test_idx"].tobytes()).hexdigest(),
            "n_original": int(r.n_original), "n_injected": int(r.n_injected),
            "expected_baseline_rows": 2, "expected_governed_rows": 144,
            "expected_ranking_model_fits": 4, "expected_non_model_scoring": 2,
        })

    assert len(key_plan) == 5500

    # Build run plan
    CONTRACTS = ["semantic_group","encoded_column"]; BUDGETS = [500,1000,2000]
    POLICIES = ["P2","P3","P4","P5","P6"]
    GOV_SEEDS = list(range(20))
    run_plan = []
    seen_rids = set()
    for kp in key_plan:
        cid, ds, mech, st, ts = kp["canonical_key_id"], kp["dataset_index"], kp["mechanism"], kp["strength"], kp["training_seed"]
        # Baseline
        for bt in ["strict","full"]:
            rid = hashlib.sha256(f"t0b_fb1_bl|{cid}|{bt}".encode()).hexdigest()[:24]
            assert rid not in seen_rids; seen_rids.add(rid)
            run_plan.append({"run_id":rid,"canonical_key_id":cid,"learner":"lr","run_type":"baseline","baseline_type":bt,"policy":"","contract":"","budget_bp":0,"governance_seed_index":-1,"selection_required":False,"bundle_sha256":kp["bundle_sha256"]})
        # Governed
        for ct in CONTRACTS:
            for bp in BUDGETS:
                for pid in POLICIES:
                    if pid == "P2":
                        for gi in GOV_SEEDS:
                            rid = hashlib.sha256(f"t0b_fb1_gov|{cid}|{ct}|{bp}|{pid}|{gi}".encode()).hexdigest()[:24]
                            assert rid not in seen_rids; seen_rids.add(rid)
                            run_plan.append({"run_id":rid,"canonical_key_id":cid,"learner":"lr","run_type":"governed","baseline_type":"","policy":pid,"contract":ct,"budget_bp":bp,"governance_seed_index":gi,"selection_required":True,"bundle_sha256":kp["bundle_sha256"]})
                    else:
                        rid = hashlib.sha256(f"t0b_fb1_gov|{cid}|{ct}|{bp}|{pid}".encode()).hexdigest()[:24]
                        assert rid not in seen_rids; seen_rids.add(rid)
                        run_plan.append({"run_id":rid,"canonical_key_id":cid,"learner":"lr","run_type":"governed","baseline_type":"","policy":pid,"contract":ct,"budget_bp":bp,"governance_seed_index":-1,"selection_required":True,"bundle_sha256":kp["bundle_sha256"]})

    assert len(run_plan) == 11000 + 792000  # baseline + governed

    # Shard plan: evaluate multiple shard counts
    shard_results = []
    for sc in args.shard_counts:
        shard_map = {}
        for kp in key_plan:
            sid = int(hashlib.sha256(kp["canonical_key_id"].encode()).hexdigest(), 16) % sc
            shard_map.setdefault(sid, []).append(kp)
        counts = [len(v) for v in shard_map.values()]
        load = [sum(k["expected_governed_rows"]+k["expected_baseline_rows"] for k in v) for v in shard_map.values()]
        shard_results.append({"shard_count":sc,"key_min":min(counts),"key_median":float(np.median(counts)),"key_max":max(counts),
                               "load_min":min(load),"load_median":float(np.median(load)),"load_max":max(load),
                               "key_imbalance_ratio":round(max(counts)/min(counts),2),
                               "load_imbalance_ratio":round(max(load)/min(load),2)})

    selected_sc = 64  # Good balance for 5500 keys
    for sr in shard_results:
        if sr["shard_count"] == selected_sc:
            print(f"Selected {selected_sc} shards: key {sr['key_min']}-{sr['key_max']} (median {sr['key_median']}), load {sr['load_min']}-{sr['load_max']} (median {sr['load_median']})")

    # Assign shards
    shard_assignments = []
    for kp in key_plan:
        sid = int(hashlib.sha256(kp["canonical_key_id"].encode()).hexdigest(), 16) % selected_sc
        shard_assignments.append({"canonical_key_id":kp["canonical_key_id"],"shard_id":sid})

    # Verify coverage
    assigned = set(a["canonical_key_id"] for a in shard_assignments)
    all_keys = set(kp["canonical_key_id"] for kp in key_plan)
    assert assigned == all_keys
    shard_counts = pd.DataFrame(shard_assignments).groupby("shard_id").size()
    key_imbalance = shard_counts.max() - shard_counts.min()
    print(f"Shard key distribution: min={shard_counts.min()}, max={shard_counts.max()}, imbalance={key_imbalance}")

    # Write deterministic gzip JSONL (one JSON object per line)
    for name, rows in [("full_b1_key_plan", key_plan), ("full_b1_run_plan", run_plan)]:
        jl = "\n".join(json.dumps(r) for r in rows) + "\n"
        compressed = gzip.compress(jl.encode("utf-8"), mtime=0)
        (out/f"{name}.jsonl.gz").write_bytes(compressed)

    # Shard plan
    shard_plan = {"selected_shard_count": selected_sc, "shard_evaluation": shard_results, "assignments": shard_assignments}
    with open(out/"full_b1_shard_plan.json","w") as f: json.dump(shard_plan, f, indent=2)

    # Plan manifest
    plan_manifest = {
        "scientific_freeze_sha": SCI_FREEZE, "canonical_keys": 5500,
        "baseline_rows": 11000, "governed_rows": 792000, "downstream_rows": 803000,
        "ranking_model_fits": 22000, "non_model_scoring": 11000,
        "selected_shard_count": selected_sc,
        "key_plan_sha256": s(str(out/"full_b1_key_plan.jsonl.gz")),
        "run_plan_sha256": s(str(out/"full_b1_run_plan.jsonl.gz")),
        "shard_plan_sha256": s(str(out/"full_b1_shard_plan.json")),
    }
    with open(out/"full_b1_plan_manifest.json","w") as f: json.dump(plan_manifest, f, indent=2)

    # Receipt
    receipt = {
        "canonical_keys": 5500, "baseline_rows": 11000, "governed_rows": 792000,
        "downstream_rows": 803000, "ranking_model_fits": 22000, "non_model_scoring": 11000,
        "run_ids_unique": len(seen_rids), "run_ids_expected": len(run_plan),
        "selected_shard_count": selected_sc, "shard_evaluation": shard_results,
        "pass": len(seen_rids)==len(run_plan),
    }
    with open(out/"full_b1_plan_receipt.json","w") as f: json.dump(receipt, f, indent=2)

    print(f"Plan: {receipt['canonical_keys']} keys, {receipt['downstream_rows']} downstream rows, {len(run_plan)} runs, {selected_sc} shards, PASS={receipt['pass']}")

if __name__=="__main__": main()
