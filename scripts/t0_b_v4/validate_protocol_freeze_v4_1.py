#!/usr/bin/env python3
"""T0-B V4.1 Validator — recursive hash closure + receipt + dry-run binding."""
from __future__ import annotations
import gzip, hashlib, json, sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def recursive_verify(obj, prefix="", errors=None):
    """Recursively verify all {path, sha256} bindings."""
    if errors is None: errors = []
    if isinstance(obj, dict):
        if "path" in obj and "sha256" in obj and isinstance(obj["sha256"], str) and len(obj["sha256"]) == 64:
            fp = ROOT / obj["path"]
            if not fp.exists():
                errors.append(f"{prefix}: path missing: {obj['path']}")
            else:
                actual = s(fp)
                if obj["sha256"] != actual:
                    errors.append(f"{prefix}: SHA mismatch {obj['path']}: {obj['sha256'][:16]} != {actual[:16]}")
        for k, v in obj.items():
            if k not in ("path", "sha256"):
                recursive_verify(v, f"{prefix}.{k}", errors)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            recursive_verify(v, f"{prefix}[{i}]", errors)
    return errors

def main():
    errors = []

    # 1. Allowlist result directory
    allowed = {
        "protocol_freeze.json", "protocol_freeze_v2.json", "protocol_freeze_v3.json",
        "protocol_freeze_v4.json",
        "freeze_lineage.json", "freeze_lineage_v3.json", "freeze_lineage_v4.json",
        "policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz",
        "static_validation_receipt_v4.json",
        "static_validation_receipt_v4_1.json",
        "validation_lineage_v4_1.json",
        "validation_closure_manifest_v4_1.json",
    }
    for f in (ROOT / "results/edbt_t0_b").iterdir():
        if f.is_file() and f.name not in allowed:
            errors.append(f"Unexpected file: {f.name}")

    dryrun = ROOT / "results/edbt_t0_b_dryrun"
    if dryrun.exists() and list(dryrun.iterdir()):
        errors.append("Dryrun not empty")

    # 2. V1/V2/V3 SUPERSEDED
    for lf in ["freeze_lineage.json", "freeze_lineage_v3.json", "freeze_lineage_v4.json"]:
        with open(ROOT / "results/edbt_t0_b" / lf) as f:
            lin = json.load(f)
        if lin.get("v1_status") != "SUPERSEDED_PRE_OUTCOME":
            errors.append(f"{lf}: wrong v1_status")

    # 3. V4.1 lineage
    with open(ROOT / "results/edbt_t0_b/validation_lineage_v4_1.json") as f:
        vl = json.load(f)
    if vl.get("scientific_freeze_commit") != "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845":
        errors.append("V4.1 lineage: wrong scientific_freeze_commit")
    if vl.get("scientific_design_modified") != False:
        errors.append("V4.1 lineage: scientific_design_modified != false")
    if vl.get("outcomes_observed_before_v4_1") != False:
        errors.append("V4.1 lineage: outcomes_observed != false")

    # 4. Receipt validation
    receipt_path = ROOT / "results/edbt_t0_b/static_validation_receipt_v4_1.json"
    if not receipt_path.exists():
        errors.append("V4.1 receipt missing")
    else:
        with open(receipt_path) as f:
            rec = json.load(f)
        if rec["repository_suite"]["failed"] != 0:
            errors.append(f"Receipt repo failed={rec['repository_suite']['failed']}")
        if rec["repository_suite"]["passed"] <= 0:
            errors.append("Receipt repo passed=0")
        if rec["v4_targeted_suite"]["failed"] != 0:
            errors.append(f"Receipt V4 failed={rec['v4_targeted_suite']['failed']}")
        if rec["v4_targeted_suite"]["passed"] <= 0:
            errors.append("Receipt V4 passed=0")
        if rec.get("validation_scope") != "LOCAL_VALIDATION_ONLY":
            errors.append(f"Receipt scope: {rec.get('validation_scope')}")
        if rec.get("scientific_design_modified") != False:
            errors.append("Receipt: scientific_design_modified != false")

    # 5. Recursive hash closure on freeze V4
    with open(ROOT / "results/edbt_t0_b/protocol_freeze_v4.json") as f:
        fr = json.load(f)
    recursive_verify(fr, "freeze_v4", errors)

    # 6. Closure manifest
    cm_path = ROOT / "results/edbt_t0_b/validation_closure_manifest_v4_1.json"
    if cm_path.exists():
        with open(cm_path) as f:
            cm = json.load(f)
        recursive_verify(cm, "closure_manifest", errors)
        if cm.get("scientific_freeze_commit") != "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845":
            errors.append("Closure manifest: wrong freeze commit")
        if cm.get("scientific_design_modified") != False:
            errors.append("Closure manifest: design modified")
    else:
        errors.append("Closure manifest missing")

    # 7. Mapping gzips: 5500 rows
    for gz_name, path in [
        ("policy", "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"),
        ("eval", "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"),
    ]:
        data = gzip.decompress((ROOT / path).read_bytes()).decode("utf-8")
        lines = [l for l in data.strip().split("\n") if l]
        if len(lines) != 5500:
            errors.append(f"{gz_name} mapping: {len(lines)} rows")

    # 8. Dry-run bundle binding (exact 4 keys)
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    if len(dr["keys"]) != 4:
        errors.append(f"Dry-run: {len(dr['keys'])} keys")
    if dr["expected_counts"]["total_downstream_fits"] != 584:
        errors.append(f"Dry-run downstream: {dr['expected_counts']['total_downstream_fits']}")

    man = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    for k in dr["keys"]:
        match = man[(man.dataset_index == k["dataset_index"]) &
                     (man.mechanism == k["mechanism"]) &
                     (man.strength == k["strength"]) &
                     (man.seed == k["training_seed"])]
        if len(match) != 1:
            errors.append(f"Dry key not in manifest: {k['dataset_index']}_{k['mechanism']}")
            continue
        mr = match.iloc[0]
        if mr["bundle_sha256"] != k["bundle_sha256"]:
            errors.append(f"Bundle SHA mismatch: {k['dataset_index']}_{k['mechanism']}")
        if mr["bundle_path"] != k["bundle_path"]:
            errors.append(f"Bundle path mismatch: {k['dataset_index']}_{k['mechanism']}")

        # Verify disk bundle SHA
        disk_sha = s(mr["bundle_path"])
        if disk_sha != k["bundle_sha256"]:
            errors.append(f"Disk bundle SHA mismatch: {mr['bundle_path']}")

        # Verify split hashes
        b = np.load(ROOT / mr["bundle_path"], allow_pickle=False)
        for split_name in ["train_idx", "val_idx", "test_idx"]:
            actual_hash = hashlib.sha256(b[split_name].tobytes()).hexdigest()
            if actual_hash != k[f"{split_name}_hash"]:
                errors.append(f"Split hash mismatch {k['dataset_index']}_{k['mechanism']} {split_name}")

        # Check mapping ledger has this key
        for gz_name, gz_path in [
            ("policy", "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"),
            ("eval", "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"),
        ]:
            data = gzip.decompress((ROOT / gz_path).read_bytes()).decode("utf-8")
            found = False
            for line in data.strip().split("\n"):
                r = json.loads(line)
                if (r["dataset_index"] == k["dataset_index"] and r["mechanism"] == k["mechanism"]
                    and r["strength"] == k["strength"] and r["training_seed"] == k["training_seed"]):
                    found = True
                    break
            if not found:
                errors.append(f"Key not in {gz_name} ledger: {k['dataset_index']}_{k['mechanism']}")

    # 9. R2 validator
    r2_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_release_r2_3.py")],
        capture_output=True, text=True,
    )
    if "VALIDATOR: PASS" not in r2_result.stdout:
        errors.append("R2 validator not PASS")

    # 10. Scientific config no-diff
    diff = subprocess.run(
        ["git", "diff", "--name-only", "ff347b...HEAD", "--",
         "configs/edbt_t0_b/policy_registry_v4.yaml",
         "configs/edbt_t0_b/cost_contracts_v4.json",
         "configs/edbt_t0_b/model_factories_v4.json",
         "configs/edbt_t0_b/claim_gates_v4.json",
         "configs/edbt_t0_b/execution_matrix_v4.json",
         "configs/edbt_t0_b/dryrun_matrix_v4.json",
         "reports/edbt_t0_b/protocol_v4.md",
         "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz",
         "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz",
         "artifacts/", "paper/"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if diff.stdout.strip():
        errors.append(f"Scientific/frozen files modified: {diff.stdout.strip()}")

    # 11. Frozen namespace
    ns_diff = subprocess.run(
        ["git", "diff", "--name-only", "fbaa9f3...HEAD", "--",
         "artifacts/", "paper/", "results/edbt_eab_revision/", "results/corrected_v2/"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if ns_diff.stdout.strip():
        errors.append(f"Frozen namespace modified: {ns_diff.stdout.strip()}")

    print(f"\n=== T0-B V4.1 VALIDATOR ===")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print("\nVALIDATOR: FAIL"); sys.exit(1)
    else:
        print("\nVALIDATOR: PASS"); sys.exit(0)

if __name__ == "__main__":
    main()
