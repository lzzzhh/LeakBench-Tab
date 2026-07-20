#!/usr/bin/env python3
"""T0-B1R2.3 Validator — real 32-gate check."""
import gzip, hashlib, io, json, sys, subprocess
from pathlib import Path; import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
R2_SEAL = "b1fb0041a2e6ebdef817b5be489b9c85993002de"
R2 = ROOT / "results/edbt_t0_b_dryrun_r2"
R23 = ROOT / "results/edbt_t0_b_dryrun_r2_3"

def main():
    errors = []
    # Seal ancestry
    r = subprocess.run(["git","merge-base","--is-ancestor",R2_SEAL,"HEAD"],capture_output=True,cwd=ROOT)
    if r.returncode: errors.append("R2 seal not ancestor")
    diff = subprocess.run(["git","diff","--name-only",f"{R2_SEAL}...HEAD","--","scripts/t0_b_dryrun_r2","tests/t0_b_dryrun_r2"],capture_output=True,text=True,cwd=ROOT)
    if diff.stdout.strip(): errors.append(f"R2 files modified: {diff.stdout.strip()}")

    # R2 ledger SHAs
    ref_shas = {"baseline_ledger.csv.gz":"bd59c32c46d2461b","governed_ledger.csv.gz":"d636bc1d95fe967f",
                "selection_ledger.csv.gz":"8f5107c62007239c","failure_ledger.csv.gz":"e185730c78ea8c5a"}
    for f, ref in ref_shas.items():
        if s(str(R2/f))[:16] != ref: errors.append(f"R2 ledger SHA changed: {f}")

    # Row counts
    for f, exp in [("baseline_ledger.csv.gz",8),("governed_ledger.csv.gz",576),("failure_ledger.csv.gz",0)]:
        data = gzip.decompress((R2/f).read_bytes()).decode("utf-8")
        n = len([l for l in data.strip().split("\n") if l]) - 1
        if n != exp: errors.append(f"{f}: {n} rows, expected {exp}")

    # Run IDs unique
    gl = gzip.decompress((R2/"governed_ledger.csv.gz").read_bytes()).decode("utf-8")
    df = pd.read_csv(io.StringIO(gl))
    if df.run_id.duplicated().any(): errors.append("Duplicate run_ids")
    if df.policy.nunique() < 5: errors.append("Policy coverage incomplete")
    sg = df[df.contract=="semantic_group"]
    if (sg.semantic_partial>0).any(): errors.append("Semantic partial violations")

    # Selection receipt
    sr_path = R23/"selection_determinism_receipt.json"
    if sr_path.exists():
        with open(sr_path) as f: sr = json.load(f)
        if sr.get("downstream_lr_calls",-1)!=0: errors.append("Selection LR calls != 0")
        if sr.get("missing_hashes",-1)!=0: errors.append(f"Missing hashes: {sr.get('missing_hashes')}")
        if sr.get("extra_hashes",-1)!=0: errors.append(f"Extra hashes: {sr.get('extra_hashes')}")
        if sr.get("conflicting_duplicate_payloads",-1)!=0: errors.append("Conflicting payloads")
        if sr.get("canonical_payload_mismatches",-1)!=0: errors.append("Payload mismatches")
    else: errors.append("Selection receipt missing")

    # Resume receipt
    rr_path = R23/"resume_hash_receipt.json"
    if rr_path.exists():
        with open(rr_path) as f: rr = json.load(f)
        if not rr.get("sha_match",False): errors.append("Resume SHA mismatch")
    else: errors.append("Resume receipt missing")

    # Repeat parity
    rp_path = R23/"repeat_fit_provenance_receipt.json"
    if rp_path.exists():
        with open(rp_path) as f: rp = json.load(f)
        if not rp.get("all_pass",False): errors.append("Repeat parity not all_pass")
    else: errors.append("Repeat receipt missing")

    # Environment
    env_path = R23/"environment_receipt_r2_3.json"
    if env_path.exists():
        with open(env_path) as f: env = json.load(f)
        if "sklearn" not in env: errors.append("Environment missing sklearn")
    else: errors.append("Environment receipt missing")

    # Scientific config diff
    sci = subprocess.run(["git","diff","--name-only","ff347b...HEAD","--","configs/edbt_t0_b/dryrun_matrix_v4.json"],capture_output=True,text=True,cwd=ROOT)
    if sci.stdout.strip(): errors.append(f"Scientific diff: {sci.stdout}")

    print(f"\n=== T0-B1R2.3 VALIDATOR ===\nErrors: {len(errors)}")
    for e in errors: print(f"  ERROR: {e}")
    sys.exit(1 if errors else 0)

if __name__=="__main__": main()
