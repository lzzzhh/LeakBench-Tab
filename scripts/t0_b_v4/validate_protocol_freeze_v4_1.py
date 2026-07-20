#!/usr/bin/env python3
"""T0-B V4.1 Validator — recursive hash closure + receipt + dry-run binding + tested-tree provenance."""
from __future__ import annotations
import gzip, hashlib, json, sys, subprocess
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SCIENTIFIC_FREEZE = "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845"

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def recursive_verify(obj, prefix="", errors=None):
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


def validate_receipt(receipt: dict) -> list[str]:
    """Pure function: validate a receipt dict. Returns list of error strings."""
    errors = []
    rs = receipt.get("repository_suite", {})
    if rs.get("failed", -1) != 0:
        errors.append(f"Receipt repo failed={rs.get('failed')}")
    if rs.get("passed", 0) <= 0:
        errors.append("Receipt repo passed=0")
    if "skipped" not in rs:
        errors.append("Receipt repo missing skipped field")
    if rs.get("duration_seconds", 0) <= 0:
        errors.append("Receipt repo duration_seconds <= 0")

    vs = receipt.get("v4_targeted_suite", {})
    if vs.get("failed", -1) != 0:
        errors.append(f"Receipt V4 failed={vs.get('failed')}")
    if vs.get("passed", 0) <= 0:
        errors.append("Receipt V4 passed=0")
    if "skipped" not in vs:
        errors.append("Receipt V4 missing skipped field")
    if vs.get("duration_seconds", 0) <= 0:
        errors.append("Receipt V4 duration_seconds <= 0")

    if receipt.get("validation_scope") != "LOCAL_VALIDATION_ONLY":
        errors.append(f"Receipt scope: {receipt.get('validation_scope')}")
    if receipt.get("github_actions_configured") != False:
        errors.append("Receipt github_actions_configured != false")
    if receipt.get("scientific_design_modified") != False:
        errors.append("Receipt scientific_design_modified != false")
    if receipt.get("scientific_freeze_commit") != SCIENTIFIC_FREEZE:
        errors.append(f"Receipt wrong scientific_freeze_commit: {receipt.get('scientific_freeze_commit')}")

    tested = receipt.get("tested_git_sha", "")
    if not tested or len(tested) != 40:
        errors.append(f"Receipt tested_git_sha invalid: {tested}")
    else:
        result = subprocess.run(["git", "cat-file", "-t", tested], capture_output=True, text=True, cwd=ROOT)
        if result.returncode != 0 or result.stdout.strip() != "commit":
            errors.append(f"Receipt tested_git_sha not a commit: {tested}")
        result = subprocess.run(["git", "merge-base", "--is-ancestor", tested, "HEAD"], capture_output=True, cwd=ROOT)
        if result.returncode != 0:
            errors.append(f"Receipt tested_git_sha {tested[:12]} not ancestor of HEAD")

    if "tested_artifacts" in receipt:
        for name, info in receipt["tested_artifacts"].items():
            if "path" in info and "sha256" in info:
                fp = ROOT / info["path"]
                if fp.exists():
                    actual = hashlib.sha256(fp.read_bytes()).hexdigest()
                    if info["sha256"] != actual:
                        errors.append(f"Receipt tested_artifact {name} SHA mismatch")
    return errors


def validate_tested_tree(tested_git_sha: str) -> list[str]:
    """Check tested-tree to HEAD diff for protected file changes."""
    errors = []
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{tested_git_sha}...HEAD"],
        capture_output=True, text=True, cwd=ROOT,
    )
    changed = set(result.stdout.strip().split("\n")) - {""}
    protected_prefixes = [
        "scripts/t0_b_v4/", "tests/t0_b_v4/", "configs/edbt_t0_b/",
        "results/edbt_t0_b/policy_group_mapping", "results/edbt_t0_b/semantic_evaluation_mapping",
        "reports/edbt_t0_b/protocol_v4.md",
    ]
    allowed = {
        "results/edbt_t0_b/static_validation_receipt_v4_1.json",
        "results/edbt_t0_b/validation_closure_manifest_v4_1.json",
        "results/edbt_t0_b/validation_lineage_v4_1r.json",
        "reports/edbt_t0_b/protocol_amendment_005_tested_tree_seal.md",
    }
    for f in changed:
        is_protected = any(f.startswith(p) for p in protected_prefixes)
        if is_protected and f not in allowed:
            errors.append(f"Protected file changed after tested tree: {f}")
    return errors


def main():
    errors = []
    allowed = {
        "protocol_freeze.json", "protocol_freeze_v2.json", "protocol_freeze_v3.json",
        "protocol_freeze_v4.json",
        "freeze_lineage.json", "freeze_lineage_v3.json", "freeze_lineage_v4.json",
        "policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz",
        "static_validation_receipt_v4.json", "static_validation_receipt_v4_1.json",
        "validation_lineage_v4_1.json", "validation_lineage_v4_1r.json",
        "validation_closure_manifest_v4_1.json",
    }
    for f in (ROOT / "results/edbt_t0_b").iterdir():
        if f.is_file() and f.name not in allowed:
            errors.append(f"Unexpected file: {f.name}")
    dryrun = ROOT / "results/edbt_t0_b_dryrun"
    if dryrun.exists() and list(dryrun.iterdir()):
        errors.append("Dryrun not empty")

    for lf in ["freeze_lineage.json", "freeze_lineage_v3.json", "freeze_lineage_v4.json"]:
        with open(ROOT / "results/edbt_t0_b" / lf) as f:
            lin = json.load(f)
        if lin.get("v1_status") != "SUPERSEDED_PRE_OUTCOME":
            errors.append(f"{lf}: wrong v1_status")

    # Receipt validation (pure function)
    receipt_path = ROOT / "results/edbt_t0_b/static_validation_receipt_v4_1.json"
    if not receipt_path.exists():
        errors.append("V4.1 receipt missing")
    else:
        with open(receipt_path) as f:
            rec = json.load(f)
        errors.extend(validate_receipt(rec))
        # Tested-tree provenance
        tested = rec.get("tested_git_sha", "")
        if tested and len(tested) == 40:
            errors.extend(validate_tested_tree(tested))

    # Recursive hash closure
    with open(ROOT / "results/edbt_t0_b/protocol_freeze_v4.json") as f:
        fr = json.load(f)
    recursive_verify(fr, "freeze_v4", errors)
    cm_path = ROOT / "results/edbt_t0_b/validation_closure_manifest_v4_1.json"
    if cm_path.exists():
        with open(cm_path) as f:
            cm = json.load(f)
        recursive_verify(cm, "closure_manifest", errors)
        if cm.get("scientific_freeze_commit") != SCIENTIFIC_FREEZE:
            errors.append("Closure manifest: wrong freeze commit")
        if cm.get("scientific_design_modified") != False:
            errors.append("Closure manifest: design modified")

    # Mapping gzips
    for gz_name, gp in [("policy", "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"),
                         ("eval", "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz")]:
        data = gzip.decompress((ROOT / gp).read_bytes()).decode("utf-8")
        lines = [l for l in data.strip().split("\n") if l]
        if len(lines) != 5500:
            errors.append(f"{gz_name} mapping: {len(lines)} rows")

    # Dry-run bundle binding
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    if len(dr["keys"]) != 4:
        errors.append(f"Dry-run: {len(dr['keys'])} keys")
    man = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    for k in dr["keys"]:
        match = man[(man.dataset_index == k["dataset_index"]) & (man.mechanism == k["mechanism"]) &
                     (man.strength == k["strength"]) & (man.seed == k["training_seed"])]
        if len(match) != 1:
            errors.append(f"Dry key not in manifest: {k['dataset_index']}_{k['mechanism']}")
            continue
        mr = match.iloc[0]
        if mr["bundle_sha256"] != k["bundle_sha256"]:
            errors.append(f"Bundle SHA mismatch: {k['dataset_index']}_{k['mechanism']}")
        disk_sha = s(mr["bundle_path"])
        if disk_sha != k["bundle_sha256"]:
            errors.append(f"Disk bundle SHA mismatch: {mr['bundle_path']}")
        b = np.load(ROOT / mr["bundle_path"], allow_pickle=False)
        for sn in ["train_idx", "val_idx", "test_idx"]:
            ah = hashlib.sha256(b[sn].tobytes()).hexdigest()
            if ah != k[f"{sn}_hash"]:
                errors.append(f"Split hash mismatch {k['dataset_index']}_{k['mechanism']} {sn}")

    # R2 validator
    r2 = subprocess.run([sys.executable, str(ROOT / "scripts/validate_release_r2_3.py")],
                        capture_output=True, text=True)
    if "VALIDATOR: PASS" not in r2.stdout:
        errors.append("R2 validator not PASS")

    # Frozen namespace
    ns = subprocess.run(["git", "diff", "--name-only", "fbaa9f3...HEAD", "--",
                         "artifacts/", "paper/", "results/edbt_eab_revision/", "results/corrected_v2/"],
                        capture_output=True, text=True, cwd=ROOT)
    if ns.stdout.strip():
        errors.append(f"Frozen namespace modified: {ns.stdout.strip()}")

    # Scientific config no-diff from freeze
    sci = subprocess.run(["git", "diff", "--name-only", f"{SCIENTIFIC_FREEZE}...HEAD", "--",
                          "configs/edbt_t0_b/policy_registry_v4.yaml",
                          "configs/edbt_t0_b/dryrun_matrix_v4.json",
                          "configs/edbt_t0_b/execution_matrix_v4.json",
                          "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz",
                          "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"],
                         capture_output=True, text=True, cwd=ROOT)
    if sci.stdout.strip():
        errors.append(f"Scientific files modified: {sci.stdout.strip()}")

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
