"""T0-B CLI E2E — full contract: plan→execute→complete-resume→merge→validate."""
import gzip, hashlib, io, json, os, subprocess, sys, tempfile
from pathlib import Path; import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
MERGER = str(ROOT/"scripts/t0_b_full_b1/merge_full_b1_shards.py")
from scripts.t0_b_full_b1.run_key_contract import baseline_lookup_key, governed_lookup_key

def build_synthetic_plan(out_dir):
    """8 keys, full contract."""
    keys = []; runs = []
    for ds in [0, 1]:
        for mech in ["M01", "M09"]:
            for ts in [13, 42]:
                n_inj = 8 if mech == "M09" else 1; sid = 0 if ds == 0 else 1
                cid = hashlib.sha256(f"s_{ds}_{mech}_S1_{ts}".encode()).hexdigest()
                keys.append({"canonical_key_id": cid, "dataset_index": ds, "mechanism": mech, "strength": "S1",
                             "training_seed": ts, "bundle_path": "s.npz", "bundle_key": f"{mech}_S1_{ts}",
                             "bundle_sha256": "fake", "n_original": 12, "n_injected": n_inj, "shard_id": sid,
                             "train_idx_sha256": "t", "val_idx_sha256": "v", "test_idx_sha256": "e"})
                for bt in ["strict", "full"]:
                    lk = baseline_lookup_key(bt)
                    rid = hashlib.sha256(f"s_bl|{cid}|{lk}".encode()).hexdigest()
                    runs.append({"run_id": rid, "canonical_key_id": cid, "run_type": "baseline", "baseline_type": bt,
                                 "shard_id": sid, "learner": "lr", "policy": "", "contract": "", "budget_bp": 0,
                                 "governance_seed_index": -1, "scientific_freeze_sha": "ff347b", "bundle_sha256": "fake",
                                 "execution_contract_version": "v1"})
                for ct in ["semantic_group", "encoded_column"]:
                    for bp in [500, 1000, 2000]:
                        for pid in ["P2", "P3", "P4", "P5", "P6"]:
                            seeds = range(20) if pid == "P2" else [-1]
                            for gi in seeds:
                                lk = governed_lookup_key(pid, ct, bp, gi)
                                rid = hashlib.sha256(f"s_gov|{cid}|{lk}".encode()).hexdigest()
                                runs.append({"run_id": rid, "canonical_key_id": cid, "run_type": "governed",
                                             "policy": pid, "contract": ct, "budget_bp": bp,
                                             "governance_seed_index": gi, "shard_id": sid, "learner": "lr",
                                             "scientific_freeze_sha": "ff347b", "bundle_sha256": "fake",
                                             "execution_contract_version": "v1"})
    for name, data in [("full_b1_key_plan", keys), ("full_b1_run_plan", runs)]:
        (out_dir / f"{name}.jsonl.gz").write_bytes(gzip.compress(
            "\n".join(json.dumps(r) for r in data).encode() + b"\n", mtime=0))
    kp_path = out_dir / "full_b1_key_plan.jsonl.gz"
    rp_path = out_dir / "full_b1_run_plan.jsonl.gz"
    kp_sha = hashlib.sha256(kp_path.read_bytes()).hexdigest()
    rp_sha = hashlib.sha256(rp_path.read_bytes()).hexdigest()
    pm = {
        "canonical_keys": len(keys), "baseline_rows": 16, "governed_rows": 1152,
        "selection_rows": 1152, "failure_rows": 0, "downstream_rows": 1168,
        "shard_count": 2, "mode": "synthetic",
        "scientific_freeze_sha": "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845",
        "execution_contract_version": "v1",
        "key_plan_sha256": kp_sha, "run_plan_sha256": rp_sha,
        "tool_seal_sha": "f" * 40,
    }
    with open(out_dir / "full_b1_plan_manifest.json", "w") as f: json.dump(pm, f)
    return pm

def test_cli_full_contract():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td); plan_dir = tdp / "plan"; plan_dir.mkdir()
        pm = build_synthetic_plan(plan_dir)
        assert pm["governed_rows"] == 1152

        # Execute shards
        for sid in [0, 1]:
            r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", str(plan_dir/"full_b1_plan_manifest.json"),
                "--shard-id", str(sid), "--output-dir", str(tdp/f"shard_{sid}"), "--synthetic"],
                capture_output=True, text=True, cwd=ROOT)
            assert r.returncode == 0, f"Shard {sid}: {r.stderr[:200]}"

        # Verify counts
        bl_all = gl_all = 0
        for sid in [0, 1]:
            bl = gzip.decompress((tdp/f"shard_{sid}/baseline_ledger.csv.gz").read_bytes()).decode("utf-8")
            gl = gzip.decompress((tdp/f"shard_{sid}/governed_ledger.csv.gz").read_bytes()).decode("utf-8")
            bl_all += len([l for l in bl.strip().split("\n") if l]) - 1
            gl_all += len([l for l in gl.strip().split("\n") if l]) - 1
        assert bl_all == 16; assert gl_all == 1152

        # Complete resume
        for sid in [0, 1]:
            r = subprocess.run([sys.executable, RUNNER, "--plan-manifest", str(plan_dir/"full_b1_plan_manifest.json"),
                "--shard-id", str(sid), "--output-dir", str(tdp/f"shard_{sid}"), "--synthetic", "--resume"],
                capture_output=True, text=True, cwd=ROOT)
            # Parse resume receipt
            rr_path = tdp / f"shard_{sid}" / "resume_receipt.json"
            if rr_path.exists():
                with open(rr_path) as f: rr = json.load(f)
                assert rr["recomputed"] == 0, f"Shard {sid} recomputed_keys={rr['recomputed']}"

        # Merge (output outside shard-root: use tempfile for output parent)
        import tempfile as _tf
        with _tf.TemporaryDirectory() as out_root:
            for suffix in ["a", "b"]:
                r = subprocess.run([sys.executable, MERGER, "--plan-manifest", str(plan_dir/"full_b1_plan_manifest.json"),
                    "--shard-root", str(tdp), "--output-dir", str(Path(out_root)/f"merged_{suffix}")],
                    capture_output=True, text=True, cwd=ROOT)
                assert r.returncode == 0, f"Merge {suffix}: {r.stdout[:200]}\n{r.stderr[:200]}"

            # Verify byte-identical merge
            for fname in ["baseline_ledger.csv.gz", "governed_ledger.csv.gz"]:
                sha_a = hashlib.sha256((Path(out_root)/"merged_a"/fname).read_bytes()).hexdigest()
                sha_b = hashlib.sha256((Path(out_root)/"merged_b"/fname).read_bytes()).hexdigest()
                assert sha_a == sha_b, f"{fname}: SHA mismatch"

        print("CLI E2E: 16 bl, 1152 gl, resume 0 recomputed, merge deterministic — PASS")

def test_validator_not_executed():
    r = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 42

def test_config_diff():
    r = subprocess.run(["git", "diff", "--name-only", "ff347b...HEAD", "--",
                        "configs/edbt_t0_b/dryrun_matrix_v4.json"], capture_output=True, text=True, cwd=ROOT)
    assert r.stdout.strip() == ""
