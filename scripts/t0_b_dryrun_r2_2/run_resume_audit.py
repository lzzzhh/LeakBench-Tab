#!/usr/bin/env python3
"""T0-B1R2.1 Resume Hash Audit — pre/post SHA comparison without modifying R2 ledgers."""
import gzip, hashlib, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out_dir = ROOT / "results/edbt_t0_b_dryrun_r2"
    out_r2_1 = ROOT / "results/edbt_t0_b_dryrun_r2_1"; out_r2_1.mkdir(parents=True, exist_ok=True)

    ledgers = ["baseline_ledger.csv.gz", "governed_ledger.csv.gz", "selection_ledger.csv.gz", "failure_ledger.csv.gz"]

    # Pre-resume hashes
    pre = {}
    for l in ledgers:
        p = out_dir / l
        pre[l] = {"sha256": s(str(p)), "rows": len(gzip.decompress(p.read_bytes()).decode("utf-8").strip().split("\n")) - 1}

    # Run R2 resume
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/t0_b_dryrun_r2/run_t0_b1_dryrun_r2.py"),
         "--output-dir", str(out_dir), "--resume"],
        capture_output=True, text=True, cwd=ROOT,
        env={**__import__('os').environ, "PYTHONPATH": "."},
    )
    elapsed = time.time() - t0

    # Post-resume hashes
    post = {}
    for l in ledgers:
        p = out_dir / l
        post[l] = {"sha256": s(str(p)), "rows": len(gzip.decompress(p.read_bytes()).decode("utf-8").strip().split("\n")) - 1}

    # Compare
    sha_match = all(pre[l]["sha256"] == post[l]["sha256"] for l in ledgers)
    rows_match = all(pre[l]["rows"] == post[l]["rows"] for l in ledgers)

    # Duplicate check
    gl_data = gzip.decompress((out_dir / "governed_ledger.csv.gz").read_bytes()).decode("utf-8")
    ids = set()
    for line in gl_data.strip().split("\n")[1:]:
        ids.add(line.split(",")[0])
    dup_count = len(gl_data.strip().split("\n")) - 1 - len(ids)

    receipt = {
        "pre_resume": pre,
        "post_resume": post,
        "sha_match": sha_match,
        "rows_match": rows_match,
        "duplicate_run_ids": dup_count,
        "resume_wall_clock_s": round(elapsed, 2),
        "resume_stdout_first_100": result.stdout[:100],
        "pass": sha_match and rows_match and dup_count == 0,
    }
    with open(out_r2_1 / "resume_hash_receipt.json", "w") as f: json.dump(receipt, f, indent=2)
    print(f"Resume hash audit: SHA match={sha_match}, rows match={rows_match}, duplicates={dup_count}")
    print(f"PASS: {receipt['pass']}")
    if not receipt["pass"]: sys.exit(1)

if __name__ == "__main__":
    main()
