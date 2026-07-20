"""T0-B CLI E2E — unified kernel with synthetic mode."""
import gzip, hashlib, io, json, subprocess, sys, tempfile
from pathlib import Path; import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
MERGER = str(ROOT/"scripts/t0_b_full_b1/merge_full_b1_shards.py")

def build_mini_plan(out_dir):
    """2 keys, 1 shard, minimal contract."""
    keys=[]; runs=[]
    for ds in [0]:
        for mech in ["M01"]:
            for ts in [13]:
                cid=hashlib.sha256(f"s_{ds}_{mech}_S1_{ts}".encode()).hexdigest()
                keys.append({"canonical_key_id":cid,"dataset_index":ds,"mechanism":mech,"strength":"S1","training_seed":ts,"bundle_path":"s.npz","bundle_key":"k","bundle_sha256":"f","n_original":12,"n_injected":1,"shard_id":0})
                for bt in ["strict","full"]:
                    rid=hashlib.sha256(f"bl|{cid}|{bt}".encode()).hexdigest()
                    runs.append({"run_id":rid,"canonical_key_id":cid,"run_type":"baseline","baseline_type":bt,"shard_id":0})
                for ct in ["semantic_group"]:
                    for bp in [500]:
                        for pid in ["P2","P3"]:
                            seeds=list(range(2)) if pid=="P2" else [0]
                            for gi in seeds:
                                rid=hashlib.sha256(f"gov|{cid}|{pid}|{ct}|{bp}|{gi}".encode()).hexdigest()
                                runs.append({"run_id":rid,"canonical_key_id":cid,"run_type":"governed","policy":pid,"contract":ct,"budget_bp":bp,"governance_seed_index":gi,"shard_id":0})
    for name,data in [("full_b1_key_plan",keys),("full_b1_run_plan",runs)]:
        (out_dir/f"{name}.jsonl.gz").write_bytes(gzip.compress("\n".join(json.dumps(r) for r in data).encode()+b"\n",mtime=0))
    return {"canonical_keys":1,"baseline_rows":2,"governed_rows":len([r for r in runs if r["run_type"]=="governed"]),"shard_count":1}

def test_cli_mini():
    with tempfile.TemporaryDirectory() as td:
        tdp=Path(td); plan_dir=tdp/"plan"; plan_dir.mkdir()
        pm=build_mini_plan(plan_dir)
        with open(plan_dir/"full_b1_plan_manifest.json","w") as f: json.dump(pm,f)
        r=subprocess.run([sys.executable,RUNNER,"--plan-manifest",str(plan_dir/"full_b1_plan_manifest.json"),
            "--shard-id","0","--output-dir",str(tdp/"s0"),"--synthetic"],capture_output=True,text=True,cwd=ROOT)
        assert r.returncode==0,f"Failed: {r.stderr[:200]}"
        assert (tdp/"s0/baseline_ledger.csv.gz").exists()
        gl=gzip.decompress((tdp/"s0/governed_ledger.csv.gz").read_bytes()).decode("utf-8")
        assert len([l for l in gl.strip().split("\n") if l])==pm["governed_rows"]+1
        # Resume
        r=subprocess.run([sys.executable,RUNNER,"--plan-manifest",str(plan_dir/"full_b1_plan_manifest.json"),
            "--shard-id","0","--output-dir",str(tdp/"s0"),"--synthetic","--resume"],capture_output=True,text=True,cwd=ROOT)
        assert "0 new" in r.stdout or "all" in r.stdout.lower()
        print(f"CLI MINI: {pm['governed_rows']} governed — PASS")

def test_validator_not_executed():
    r=subprocess.run([sys.executable,str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py")],capture_output=True,text=True,cwd=ROOT)
    assert r.returncode==42

def test_config_diff():
    r=subprocess.run(["git","diff","--name-only","ff347b...HEAD","--","configs/edbt_t0_b/dryrun_matrix_v4.json"],capture_output=True,text=True,cwd=ROOT)
    assert r.stdout.strip()==""
