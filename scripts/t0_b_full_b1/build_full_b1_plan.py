#!/usr/bin/env python3
"""T0-B Full-B1 Plan Generator V2 — full SHA IDs, balanced sharding, complete input closure."""
import gzip, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
SCI_FREEZE = "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845"
CONTRACT_VERSION = "t0_b_full_b1_v2"
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/edbt_t0_b_full_b1_preflight")
    ap.add_argument("--plan-only", action="store_true"); ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.output_dir; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # === INPUT CLOSURE ===
    man = pd.read_csv(ROOT/"artifacts/sp6/sp6_bundle_manifest.csv")
    assert len(man) == 5500
    datasets = sorted(man.dataset_index.unique())
    mechanisms = sorted(man.mechanism.unique())
    strengths = sorted(man.strength.unique())
    seeds = sorted(man.seed.unique())
    assert len(datasets)==20 and len(mechanisms)==11 and len(strengths)==5 and len(seeds)==5

    # Verify Cartesian completeness
    key_tuples = set()
    for _, r in man.iterrows():
        kt = (int(r.dataset_index), r.mechanism, r.strength, int(r.seed))
        assert kt not in key_tuples, f"Duplicate key: {kt}"
        key_tuples.add(kt)
    assert len(key_tuples) == 5500

    # Load mappings + verify key set equality
    for gz_name in ["policy_group_mapping_v3.jsonl.gz","semantic_evaluation_mapping_v3.jsonl.gz"]:
        data = gzip.decompress((ROOT/"results/edbt_t0_b"/gz_name).read_bytes()).decode("utf-8")
        m = {}
        for line in data.strip().split("\n"):
            r = json.loads(line)
            kt = (r["dataset_index"],r["mechanism"],r["strength"],r["training_seed"])
            assert kt not in m, f"Duplicate in {gz_name}: {kt}"
            m[kt] = r
        assert len(m) == 5500, f"{gz_name} has {len(m)} keys"
        assert set(m.keys()) == key_tuples, f"{gz_name} key mismatch"

    # Load V4 configs
    with open(ROOT/"configs/edbt_t0_b/dryrun_matrix_v4.json") as f: dr = json.load(f)
    CONTRACTS = dr["contracts"]; BUDGETS = dr["budgets_bp"]
    GOV_SEEDS = dr["p2_governance_seeds"]

    # === BUILD KEY PLAN ===
    key_plan = []
    for _, r in man.iterrows():
        ds, mech, st, ts = int(r.dataset_index), r.mechanism, r.strength, int(r.seed)
        cid = hashlib.sha256(f"t0b_fb1_key_v2|{ds}|{mech}|{st}|{ts}".encode()).hexdigest()
        # Verify bundle SHA
        disk_sha = s(r.bundle_path)
        assert disk_sha == r.bundle_sha256, f"Bundle SHA mismatch: {r.bundle_path}"
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
            "policy_mapping_key": f"{ds}|{mech}|{st}|{ts}",
            "semantic_mapping_key": f"{ds}|{mech}|{st}|{ts}",
            "expected_baseline_rows": 2, "expected_governed_rows": 144,
            "expected_ranking_model_fits": 4, "expected_non_model_scoring": 2,
            "estimated_workload": float(int(r.n_original)+int(r.n_injected)),
        })

    # === BALANCED SHARDING ===
    from scripts.t0_b_full_b1.execution_contract import balanced_shard_assignment
    SHARD_COUNT = 64
    assignments = balanced_shard_assignment(key_plan, SHARD_COUNT)
    assigned = set(a["canonical_key_id"] for a in assignments)
    all_keys = set(k["canonical_key_id"] for k in key_plan)
    assert assigned == all_keys
    dist = pd.DataFrame(assignments).groupby("shard_id").size()
    print(f"Shard distribution: min={dist.min()}, max={dist.max()}, diff={dist.max()-dist.min()}")

    # Add shard_id to key plan
    kp_map = {k["canonical_key_id"]: k for k in key_plan}
    for a in assignments:
        kp_map[a["canonical_key_id"]]["shard_id"] = a["shard_id"]

    # === BUILD RUN PLAN ===
    POLICIES = ["P2","P3","P4","P5","P6"]; DET_POLICIES = ["P3","P4","P5","P6"]
    run_plan = []
    seen_rids = set()
    for kp in key_plan:
        cid = kp["canonical_key_id"]
        # Baseline
        for bt in ["strict","full"]:
            rid = hashlib.sha256(f"t0b_fb1_bl_v2|{cid}|{bt}".encode()).hexdigest()
            assert rid not in seen_rids; seen_rids.add(rid)
            run_plan.append({"run_id":rid,"canonical_key_id":cid,"shard_id":kp["shard_id"],"learner":"lr","run_type":"baseline","baseline_type":bt,"policy":"","contract":"","budget_bp":0,"governance_seed_index":-1,"expected_selection_required":False,"scientific_freeze_sha":SCI_FREEZE,"bundle_sha256":kp["bundle_sha256"],"execution_contract_version":CONTRACT_VERSION})
        # Governed
        for ct in CONTRACTS:
            for bp in BUDGETS:
                for pid in POLICIES:
                    if pid == "P2":
                        for gi, gs_idx in enumerate(GOV_SEEDS):
                            rid = hashlib.sha256(f"t0b_fb1_gov_v2|{cid}|{ct}|{bp}|{pid}|{gi}".encode()).hexdigest()
                            assert rid not in seen_rids; seen_rids.add(rid)
                            run_plan.append({"run_id":rid,"canonical_key_id":cid,"shard_id":kp["shard_id"],"learner":"lr","run_type":"governed","baseline_type":"","policy":pid,"contract":ct,"budget_bp":bp,"governance_seed_index":gi,"expected_selection_required":True,"scientific_freeze_sha":SCI_FREEZE,"bundle_sha256":kp["bundle_sha256"],"execution_contract_version":CONTRACT_VERSION})
                    else:
                        rid = hashlib.sha256(f"t0b_fb1_gov_v2|{cid}|{ct}|{bp}|{pid}".encode()).hexdigest()
                        assert rid not in seen_rids; seen_rids.add(rid)
                        run_plan.append({"run_id":rid,"canonical_key_id":cid,"shard_id":kp["shard_id"],"learner":"lr","run_type":"governed","baseline_type":"","policy":pid,"contract":ct,"budget_bp":bp,"governance_seed_index":-1,"expected_selection_required":True,"scientific_freeze_sha":SCI_FREEZE,"bundle_sha256":kp["bundle_sha256"],"execution_contract_version":CONTRACT_VERSION})

    assert len(run_plan) == 803000

    # Write deterministic JSONL gzips
    for name, rows in [("full_b1_key_plan", key_plan), ("full_b1_run_plan", run_plan)]:
        jl = "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
        (out/f"{name}.jsonl.gz").write_bytes(gzip.compress(jl.encode("utf-8"), mtime=0))

    # Shard plan (assignments + evaluation)
    shard_loads = {}
    for a in assignments:
        sid = a["shard_id"]; kp = kp_map[a["canonical_key_id"]]
        shard_loads.setdefault(sid, []).append(kp["estimated_workload"])
    load_stats = {sid: {"count": len(v), "workload_sum": sum(v), "workload_mean": float(np.mean(v))} for sid, v in shard_loads.items()}
    shard_plan = {"shard_count": SHARD_COUNT, "shard_stats": load_stats}
    with open(out/"full_b1_shard_plan.json","w") as f: json.dump(shard_plan, f, indent=2)

    # Plan manifest
    plan_manifest = {
        "scientific_freeze_sha": SCI_FREEZE, "contract_version": CONTRACT_VERSION,
        "canonical_keys": 5500, "baseline_rows": 11000, "governed_rows": 792000, "downstream_rows": 803000,
        "ranking_model_fits": 22000, "non_model_scoring": 11000, "shard_count": SHARD_COUNT,
        "key_plan_sha256": s(str(out/"full_b1_key_plan.jsonl.gz")),
        "run_plan_sha256": s(str(out/"full_b1_run_plan.jsonl.gz")),
        "shard_plan_sha256": s(str(out/"full_b1_shard_plan.json")),
    }
    with open(out/"full_b1_plan_manifest.json","w") as f: json.dump(plan_manifest, f, indent=2)

    receipt = {
        "canonical_keys": 5500, "downstream_rows": 803000, "run_ids_unique": len(seen_rids),
        "shard_count": SHARD_COUNT, "shard_key_min": dist.min(), "shard_key_max": dist.max(),
        "pass": len(seen_rids)==len(run_plan) and dist.max()-dist.min()<=1,
        "wall_clock_s": round(time.time()-t0, 2),
    }
    with open(out/"full_b1_plan_receipt.json","w") as f: json.dump(receipt, f, indent=2)
    print(f"Plan: {receipt['canonical_keys']} keys, {receipt['downstream_rows']} rows, shards {receipt['shard_key_min']}-{receipt['shard_key_max']}, PASS={receipt['pass']}")

if __name__=="__main__": main()
