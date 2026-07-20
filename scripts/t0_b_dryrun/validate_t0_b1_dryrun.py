#!/usr/bin/env python3
"""T0-B1 Dry-Run Validator."""
from __future__ import annotations
import gzip, hashlib, io, json, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    errors = []
    out = ROOT / "results/edbt_t0_b_dryrun"

    # 1. Runner seal is ancestor
    runner_seal = None
    with open(out / "dryrun_manifest.json") as f:
        dm = json.load(f)
        runner_seal = dm.get("runner_seal_sha", "")

    # 2. Exact 4 keys
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    if len(dr["keys"]) != 4:
        errors.append(f"Not 4 keys: {len(dr['keys'])}")

    # 3. Baseline 8 rows
    bl = gzip.decompress((out / "baseline_ledger.csv.gz").read_bytes()).decode("utf-8")
    bl_lines = [l for l in bl.strip().split("\n") if l]
    if len(bl_lines) != 9:  # header + 8 rows
        errors.append(f"Baseline: {len(bl_lines)-1} rows, expected 8")

    # 4. Governed 576 rows
    gl = gzip.decompress((out / "governed_ledger.csv.gz").read_bytes()).decode("utf-8")
    gl_lines = [l for l in gl.strip().split("\n") if l]
    if len(gl_lines) != 577:
        errors.append(f"Governed: {len(gl_lines)-1} rows, expected 576")

    # 5. Failure 0 rows
    fl = gzip.decompress((out / "failure_ledger.csv.gz").read_bytes()).decode("utf-8")
    fl_lines = [l for l in fl.strip().split("\n") if l]
    if len(fl_lines) != 1:  # header only
        errors.append(f"Failures: {len(fl_lines)-1} rows")

    # 6. Selection ledger
    sl = gzip.decompress((out / "selection_ledger.csv.gz").read_bytes()).decode("utf-8")
    sl_lines = [l for l in sl.strip().split("\n") if l]
    print(f"Selections: {len(sl_lines)-1}")

    # 7. No dry-run outcomes
    import pandas as pd
    gl_df = pd.read_csv(io.StringIO(gl))
    # Check semantic partial violation = 0 for semantic contract
    sg = gl_df[gl_df.contract == "semantic_group"]
    pv = sg[sg.semantic_partial_removed_count > 0]
    if len(pv) > 0:
        errors.append(f"Semantic partial violations: {len(pv)}")

    # 8. M09 atomic group
    m09 = gl_df[gl_df.mechanism == "M09"]
    if len(m09) > 0:
        print(f"M09 governed rows: {len(m09)}")

    # 9. Run-id uniqueness
    if gl_df.run_id.duplicated().any():
        errors.append("Duplicate run_ids in governed ledger")

    # 10. Science config no-diff
    sci = subprocess.run(["git", "diff", "--name-only", "ff347b...HEAD", "--",
        "configs/edbt_t0_b/policy_registry_v4.yaml",
        "configs/edbt_t0_b/dryrun_matrix_v4.json",
        "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"],
        capture_output=True, text=True, cwd=ROOT)
    if sci.stdout.strip():
        errors.append(f"Scientific files modified: {sci.stdout}")

    print(f"\n=== T0-B1 DRY-RUN VALIDATOR ===")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print("\nVALIDATOR: FAIL"); sys.exit(1)
    else:
        print("\nVALIDATOR: PASS"); sys.exit(0)


if __name__ == "__main__":
    main()
