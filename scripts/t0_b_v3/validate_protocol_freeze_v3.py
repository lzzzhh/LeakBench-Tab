#!/usr/bin/env python3
"""T0-B V3 Protocol Freeze Validator."""
from __future__ import annotations
import gzip, hashlib, json, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    errors = []

    # 1. V1/V2 superseded
    for path in ["results/edbt_t0_b/freeze_lineage.json", "results/edbt_t0_b/freeze_lineage_v3.json"]:
        with open(ROOT / path) as f:
            lin = json.load(f)
        if lin.get("v1_status") != "SUPERSEDED_PRE_OUTCOME":
            errors.append(f"{path}: v1_status not SUPERSEDED_PRE_OUTCOME")

    # 2. No outcome files
    result_dir = ROOT / "results/edbt_t0_b"
    allowed = {"protocol_freeze.json", "protocol_freeze_v2.json", "protocol_freeze_v3.json",
               "freeze_lineage.json", "freeze_lineage_v3.json",
               "policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz"}
    for f in result_dir.iterdir():
        if f.is_file() and f.name not in allowed:
            errors.append(f"Unexpected file in results/edbt_t0_b: {f.name}")

    dryrun = ROOT / "results/edbt_t0_b_dryrun"
    if dryrun.exists() and list(dryrun.iterdir()):
        errors.append("Dryrun directory not empty")

    # 3. Freeze hash closure
    with open(ROOT / "results/edbt_t0_b/protocol_freeze_v3.json") as f:
        fr = json.load(f)
    for name, info in fr.get("inputs", {}).items():
        sha = info.get("sha256", "")
        if not sha or len(sha) != 64:
            errors.append(f"Freeze input {name}: invalid SHA")

    # 4. Old P2 formula banned
    for py_file in Path("scripts/t0_b_v3").glob("*.py"):
        if "validate_protocol_freeze" in py_file.name:
            continue  # skip self-check
        content = py_file.read_text()
        if "(gov_seed * 100 + dataset_index * 7 + training_seed * 13)" in content:
            errors.append(f"Old P2 formula in {py_file}")

    # 5. P6 parameters explicit
    sel_src = (ROOT / "scripts/t0_b_v3/policy_selectors.py").read_text()
    for param in ["n_estimators=100", "n_repeats=5", "n_splits=3", "random_state=42"]:
        if param not in sel_src:
            errors.append(f"P6 missing explicit param: {param}")

    # 6. Policy mapping: 5500 rows, coverage
    pol_path = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
    data = gzip.decompress(pol_path.read_bytes()).decode("utf-8")
    lines = [l for l in data.strip().split("\n") if l]
    if len(lines) != 5500:
        errors.append(f"Policy mapping: {len(lines)} rows, expected 5500")

    for line in lines:
        row = json.loads(line)
        total_cols = sum(g["group_size"] for g in row["groups"])
        if total_cols != row["n_encoded_columns"]:
            errors.append(f"Key {row['dataset_index']}_{row['mechanism']}: group total {total_cols} != {row['n_encoded_columns']}")

    # 7. Evaluation mapping consistency
    eval_path = ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"
    eval_data = gzip.decompress(eval_path.read_bytes()).decode("utf-8")
    eval_lines = [l for l in eval_data.strip().split("\n") if l]
    if len(eval_lines) != 5500:
        errors.append(f"Evaluation mapping: {len(eval_lines)} rows")

    # 8. Execution counts
    if fr.get("execution_counts", {}).get("downstream_fits") != 1089000:
        errors.append("Execution counts incorrect")

    # 9. Dry-run matrix
    dr = ROOT / "configs/edbt_t0_b/dryrun_matrix_v3.json"
    if not dr.exists():
        errors.append("Dry-run matrix missing")

    # 10. R2 validator
    r2_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_release_r2_3.py")],
        capture_output=True, text=True,
    )
    if "VALIDATOR: PASS" not in r2_result.stdout:
        errors.append("R2 validator not PASS")

    # 11. Paper/frozen namespace
    diff = subprocess.run(
        ["git", "diff", "--name-only", "fbaa9f3...HEAD", "--",
         "artifacts/", "paper/", "results/edbt_eab_revision/",
         "results/corrected_v2/"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if diff.stdout.strip():
        errors.append(f"Frozen namespace modified: {diff.stdout.strip()}")

    # 12. No TODO/TBD
    for f in Path("configs/edbt_t0_b").glob("*.json"):
        for marker in ["TODO", "TBD", "PLACEHOLDER"]:
            if marker in f.read_text():
                errors.append(f"{marker} in {f}")

    # Summary
    print(f"\n=== T0-B V3 VALIDATOR ===")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print("\nVALIDATOR: FAIL")
        sys.exit(1)
    else:
        print("\nVALIDATOR: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
