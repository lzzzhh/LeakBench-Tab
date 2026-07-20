"""T0-B CLI E2E — full pipeline via subprocess."""
import gzip, hashlib, io, json, subprocess, sys, tempfile
from pathlib import Path; import numpy as np, pytest
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
RUNNER = str(ROOT/"scripts/t0_b_full_b1/run_full_b1_shard.py")
MERGER = str(ROOT/"scripts/t0_b_full_b1/merge_full_b1_shards.py")

def build_plan(out_dir):
    keys=[]; runs=[]
    for ds in [0,1]:
        for mech in ["M01","M09"]:
            for ts in [13,42]:
                n_inj=8 if mech=="M09" else 1; sid=0 if ds==0 else 1
                cid=hashlib.sha256(f"s_{ds}_{mech}_S1_{ts}".encode()).hexdigest()
                keys.append({"canonical_key_id":cid,"dataset_index":ds,"mechanism":mech,"strength":"S1","training_seed":ts,"bundle_path":"s.npz","bundle_key":f"{mech}_S1_{ts}","bundle_sha256":"fake","n_original":12,"n_injected":n_inj,"shard_id":sid})
                for bt in ["strict","full"]:
                    rid=hashlib.sha256(f"bl|{cid}|{bt}".encode()).hexdigest()
                    runs.append({"run_id":rid,"canonical_key_id":cid,"run_type":"baseline","baseline_type":bt,"shard_id":sid})
                for ct in ["semantic_group","encoded_column"]:
                    for bp in [500,2000]:
                        for pid in ["P2","P3","P4","P5","P6"]:
                            seeds=[0,1] if pid=="P2" else [-1]
                            for gi in seeds:
                                rid=hashlib.sha256(f"gov|{cid}|{pid}|{ct}|{bp}|{gi}".encode()).hexdigest()
                                runs.append({"run_id":rid,"canonical_key_id":cid,"run_type":"governed","policy":pid,"contract":ct,"budget_bp":bp,"governance_seed_index":gi,"shard_id":sid})
    for name,data in [("full_b1_key_plan",keys),("full_b1_run_plan",runs)]:
        jl="\n".join(json.dumps(r) for r in data)+"\n"
        (out_dir/f"{name}.jsonl.gz").write_bytes(gzip.compress(jl.encode(),mtime=0))
    return {"canonical_keys":8,"baseline_rows":16,"governed_rows":192,"shard_count":2}

def test_cli_e2e():
    with tempfile.TemporaryDirectory() as td:
        tdp=Path(td); plan_dir=tdp/"plan"; plan_dir.mkdir()
        pm=build_plan(plan_dir)
        with open(plan_dir/"full_b1_plan_manifest.json","w") as f: json.dump(pm,f)
        # Execute both shards
        for sid in [0,1]:
            r=subprocess.run([sys.executable,RUNNER,"--plan-manifest",str(plan_dir/"full_b1_plan_manifest.json"),
                "--shard-id",str(sid),"--output-dir",str(tdp/f"shard_{sid}")],capture_output=True,text=True,cwd=ROOT)
            assert r.returncode==0,f"Shard {sid} failed: {r.stderr[:200]}"
            assert (tdp/f"shard_{sid}/shard_manifest.json").exists()
        # Resume shard 0
        r=subprocess.run([sys.executable,RUNNER,"--plan-manifest",str(plan_dir/"full_b1_plan_manifest.json"),
            "--shard-id","0","--output-dir",str(tdp/"shard_0"),"--resume"],capture_output=True,text=True,cwd=ROOT)
        assert r.returncode==0
        # Merge
        r=subprocess.run([sys.executable,MERGER,"--plan-manifest",str(plan_dir/"full_b1_plan_manifest.json"),
            "--shard-root",str(tdp),"--output-dir",str(tdp/"merged")],capture_output=True,text=True,cwd=ROOT)
        assert r.returncode==0
        assert (tdp/"merged/baseline_ledger.csv.gz").exists()
        bl=gzip.decompress((tdp/"merged/baseline_ledger.csv.gz").read_bytes()).decode("utf-8")
        gl=gzip.decompress((tdp/"merged/governed_ledger.csv.gz").read_bytes()).decode("utf-8")
        assert len([l for l in bl.strip().split("\n") if l])==17
        assert len([l for l in gl.strip().split("\n") if l])==193
        print(f"CLI E2E: {pm['canonical_keys']} keys, 16 bl, 192 gl — PASS")

def test_validator_not_executed():
    r=subprocess.run([sys.executable,str(ROOT/"scripts/t0_b_full_b1/validate_full_b1_results.py")],capture_output=True,text=True,cwd=ROOT)
    assert r.returncode==42

def test_config_diff():
    r=subprocess.run(["git","diff","--name-only","ff347b...HEAD","--","configs/edbt_t0_b/dryrun_matrix_v4.json"],capture_output=True,text=True,cwd=ROOT)
    assert r.stdout.strip()==""
