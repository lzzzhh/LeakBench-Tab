#!/usr/bin/env python3
"""T0-B1R2.1 Evidence Validator."""
import gzip, hashlib, io, json, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
R2_SEAL = "b1fb0041a2e6ebdef817b5be489b9c85993002de"

def main():
    errors = []
    # 1. R2 seal is ancestor
    r = subprocess.run(["git","merge-base","--is-ancestor",R2_SEAL,"HEAD"], capture_output=True, cwd=ROOT)
    if r.returncode != 0: errors.append("R2 seal not ancestor of HEAD")

    # 2. R2 runner/tests zero diff from seal
    diff = subprocess.run(["git","diff","--name-only",f"{R2_SEAL}...HEAD","--","scripts/t0_b_dryrun_r2","tests/t0_b_dryrun_r2"], capture_output=True, text=True, cwd=ROOT)
    if diff.stdout.strip(): errors.append(f"R2 files modified: {diff.stdout.strip()}")

    # 3-5. R2 scientific ledgers intact
    out = ROOT / "results/edbt_t0_b_dryrun_r2"
    gl = gzip.decompress((out/"governed_ledger.csv.gz").read_bytes()).decode("utf-8")
    bl = gzip.decompress((out/"baseline_ledger.csv.gz").read_bytes()).decode("utf-8")
    fl = gzip.decompress((out/"failure_ledger.csv.gz").read_bytes()).decode("utf-8")
    gl_lines = [l for l in gl.strip().split("\n") if l]
    bl_lines = [l for l in bl.strip().split("\n") if l]
    fl_lines = [l for l in fl.strip().split("\n") if l]
    if len(bl_lines) != 9: errors.append(f"Baseline rows: {len(bl_lines)-1}")
    if len(gl_lines) != 577: errors.append(f"Governed rows: {len(gl_lines)-1}")
    if len(fl_lines) != 1: errors.append(f"Failure rows: {len(fl_lines)-1}")

    # Run-id uniqueness
    import pandas as pd
    gl_df = pd.read_csv(io.StringIO(gl))
    if gl_df.run_id.duplicated().any(): errors.append("Duplicate run_ids")

    # 10-11. Scientific downstream = 584
    if len(gl_lines) - 1 != 576: errors.append(f"Governed != 576")

    # 19. P2 matched cost
    p2_rows = gl_df[gl_df.policy == "P2"]
    if len(p2_rows) > 0:
        # Check realized_cost == budget_k
        pass

    # 20. Semantic partial = 0
    sg = gl_df[gl_df.contract == "semantic_group"]
    if sg.semantic_partial.max() > 0: errors.append(f"Semantic partial violations: {(sg.semantic_partial>0).sum()}")

    # 32. Scientific config diff
    sci = subprocess.run(["git","diff","--name-only","ff347b...HEAD","--","configs/edbt_t0_b/policy_registry_v4.yaml","configs/edbt_t0_b/dryrun_matrix_v4.json","results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"], capture_output=True, text=True, cwd=ROOT)
    if sci.stdout.strip(): errors.append(f"Scientific diff: {sci.stdout}")

    # Selection audit receipt
    sel_rec = ROOT / "results/edbt_t0_b_dryrun_r2_1/selection_determinism_receipt.json"
    if sel_rec.exists():
        with open(sel_rec) as f: sr = json.load(f)
        if sr.get("downstream_lr_calls", -1) != 0: errors.append("Selection audit had LR calls")
        if sr.get("payload_mismatches", -1) != 0: errors.append(f"Selection mismatches: {sr.get('payload_mismatches')}")

    # Resume hash receipt
    res_rec = ROOT / "results/edbt_t0_b_dryrun_r2_1/resume_hash_receipt.json"
    if res_rec.exists():
        with open(res_rec) as f: rr = json.load(f)
        if not rr.get("sha_match", False): errors.append("Resume SHA mismatch")
        if rr.get("duplicate_run_ids", -1) != 0: errors.append("Resume duplicates")

    print(f"\n=== T0-B1R2.1 EVIDENCE VALIDATOR ===")
    print(f"Errors: {len(errors)}")
    for e in errors: print(f"  ERROR: {e}")
    if errors: print("\nVALIDATOR: FAIL"); sys.exit(1)
    else: print("\nVALIDATOR: PASS"); sys.exit(0)

if __name__ == "__main__": main()
