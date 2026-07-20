#!/usr/bin/env python3
"""T0-B1R2.3F Resume Hash Audit — writes to r2_3, checks returncode, binds R2 receipt."""
import gzip, hashlib, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out_dir = ROOT / "results/edbt_t0_b_dryrun_r2"
    out_r2_3 = ROOT / "results/edbt_t0_b_dryrun_r2_3"; out_r2_3.mkdir(parents=True, exist_ok=True)
    ledgers = ["baseline_ledger.csv.gz","governed_ledger.csv.gz","selection_ledger.csv.gz","failure_ledger.csv.gz"]

    pre = {}
    for l in ledgers:
        p = out_dir / l; pre[l] = {"sha256": s(str(p)), "rows": len(gzip.decompress(p.read_bytes()).decode("utf-8").strip().split("\n")) - 1}

    t0 = time.time()
    result = subprocess.run([sys.executable, str(ROOT/"scripts/t0_b_dryrun_r2/run_t0_b1_dryrun_r2.py"),"--output-dir",str(out_dir),"--resume"],
        capture_output=True, text=True, cwd=ROOT, env={**__import__('os').environ,"PYTHONPATH":"."})
    elapsed = time.time() - t0

    post = {}
    for l in ledgers:
        p = out_dir / l; post[l] = {"sha256": s(str(p)), "rows": len(gzip.decompress(p.read_bytes()).decode("utf-8").strip().split("\n")) - 1}

    sha_match = all(pre[l]["sha256"]==post[l]["sha256"] for l in ledgers)
    rows_match = all(pre[l]["rows"]==post[l]["rows"] for l in ledgers)

    gl_data = gzip.decompress((out_dir/"governed_ledger.csv.gz").read_bytes()).decode("utf-8")
    ids = set()
    for line in gl_data.strip().split("\n")[1:]: ids.add(line.split(",")[0])
    dup_count = len(gl_data.strip().split("\n"))-1-len(ids)

    # Read R2 resume receipt
    r2_rec = ROOT/"results/edbt_t0_b_dryrun_r2/resume_receipt.json"
    r2_rec_sha = s(str(r2_rec)) if r2_rec.exists() else "missing"
    r2_rec_data = json.load(open(r2_rec)) if r2_rec.exists() else {}

    receipt = {
        "subprocess_returncode": result.returncode,
        "pre_resume": pre, "post_resume": post,
        "sha_match": sha_match, "rows_match": rows_match,
        "duplicate_run_ids": dup_count,
        "resume_wall_clock_s": round(elapsed,2),
        "r2_resume_receipt_sha256": r2_rec_sha,
        "r2_resume_new_baseline": r2_rec_data.get("new_baseline",-1),
        "r2_resume_new_governed": r2_rec_data.get("new_governed",-1),
        "r2_resume_new_selections": r2_rec_data.get("new_selections",-1),
        "r2_resume_lr_calls": r2_rec_data.get("lr_calls",-1),
        "r2_resume_p3_calls": r2_rec_data.get("p3_calls",-1),
        "r2_resume_stdout_first_100": result.stdout[:100],
        "pass": result.returncode==0 and sha_match and rows_match and dup_count==0,
    }
    with open(out_r2_3/"resume_hash_receipt.json","w") as f: json.dump(receipt,f,indent=2)
    print(f"Resume: returncode={result.returncode} SHA={sha_match} rows={rows_match} dups={dup_count} PASS={receipt['pass']}")
    sys.exit(0 if receipt["pass"] else 1)

if __name__=="__main__": main()
