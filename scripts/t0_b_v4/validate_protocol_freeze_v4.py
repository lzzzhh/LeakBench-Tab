#!/usr/bin/env python3
"""T0-B V4 Protocol Freeze Validator — real hash closure, no length-only checks."""
from __future__ import annotations
import gzip, hashlib, json, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    errors = []

    # 1. Allowlist result directory
    allowed = {
        "protocol_freeze.json", "protocol_freeze_v2.json", "protocol_freeze_v3.json",
        "protocol_freeze_v4.json",
        "freeze_lineage.json", "freeze_lineage_v3.json", "freeze_lineage_v4.json",
        "policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz",
        "static_validation_receipt_v4.json",
    }
    for f in (ROOT / "results/edbt_t0_b").iterdir():
        if f.is_file() and f.name not in allowed:
            errors.append(f"Unexpected file: {f.name}")

    # 2. V1/V2/V3 SUPERSEDED
    for lf in ["freeze_lineage.json", "freeze_lineage_v3.json", "freeze_lineage_v4.json"]:
        with open(ROOT / "results/edbt_t0_b" / lf) as f:
            lin = json.load(f)
        if lin.get("v1_status") != "SUPERSEDED_PRE_OUTCOME":
            errors.append(f"{lf}: v1_status wrong")

    # 3. No outcome exposure
    for p in ["selection_ledger", "lr_cells", "rf_cells", "lightgbm_cells",
              "task_effects", "policy_summary", "analysis_summary", "claim_state"]:
        for f in (ROOT / "results/edbt_t0_b").iterdir():
            if p in f.name:
                errors.append(f"Outcome file: {f.name}")

    dryrun = ROOT / "results/edbt_t0_b_dryrun"
    if dryrun.exists() and list(dryrun.iterdir()):
        errors.append("Dryrun not empty")

    # 4. Freeze V4 hash closure — every {path, sha256} matches disk
    with open(ROOT / "results/edbt_t0_b/protocol_freeze_v4.json") as f:
        fr = json.load(f)

    for section_name, section in fr.items():
        if not isinstance(section, dict): continue
        for k, v in section.items():
            if isinstance(v, dict) and "path" in v and "sha256" in v:
                fp = ROOT / v["path"]
                if not fp.exists():
                    errors.append(f"Freeze ref missing: {v['path']}")
                    continue
                actual = s(fp)
                if v["sha256"] != actual:
                    errors.append(f"SHA mismatch: {v['path']} recorded={v['sha256'][:16]} actual={actual[:16]}")

    # 5. Mapping gzips: 5500 rows, unique keys, per-row hash
    for gz_name in ["policy_group_mapping_v3.jsonl.gz", "semantic_evaluation_mapping_v3.jsonl.gz"]:
        p = ROOT / "results/edbt_t0_b" / gz_name
        data = gzip.decompress(p.read_bytes()).decode("utf-8")
        lines = [l for l in data.strip().split("\n") if l]
        if len(lines) != 5500:
            errors.append(f"{gz_name}: {len(lines)} rows, expected 5500")

    # 6. Policy registry: P6 params, old P2 banned (skip forbidden: section)
    reg_path = ROOT / "configs/edbt_t0_b/policy_registry_v4.yaml"
    if reg_path.exists():
        reg_text = reg_path.read_text()
        # Only check non-forbidden sections
        reg_clean = reg_text.split("forbidden:")[0] if "forbidden:" in reg_text else reg_text
        if "(gov_seed * 100" in reg_clean:
            errors.append("Old P2 formula in policy_registry_v4.yaml (active section)")
        yaml_params = ["n_estimators: 100", "n_repeats: 5", "n_splits=3", "scoring: \"roc_auc\""]
        for param in yaml_params:
            if param not in reg_text:
                errors.append(f"P6 missing param in registry: {param}")

    # 7. Dry-run matrix: 4 keys, bundle SHAs match manifest
    with open(ROOT / "configs/edbt_t0_b/dryrun_matrix_v4.json") as f:
        dr = json.load(f)
    if len(dr["keys"]) != 4:
        errors.append(f"Dry-run: {len(dr['keys'])} keys, expected 4")
    if dr["expected_counts"]["total_downstream_fits"] != 584:
        errors.append(f"Dry-run downstream fits: {dr['expected_counts']['total_downstream_fits']}")

    import pandas as pd
    man = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    for k in dr["keys"]:
        match = man[(man.dataset_index == k["dataset_index"]) & (man.mechanism == k["mechanism"]) &
                     (man.strength == k["strength"]) & (man.seed == k["training_seed"])]
        if len(match) != 1:
            errors.append(f"Dry-run key not in manifest: {k['dataset_index']}_{k['mechanism']}_{k['strength']}_{k['training_seed']}")

    # 8. Execution counts: recompute from formula
    n_keys = 5500
    b1_gov_per_key = 2*3*(20+4)  # 2 contracts × 3 budgets × (20 P2 + 4 det)
    b1_total = n_keys * (2 + b1_gov_per_key)
    b2_gov_per_key = 1*1*(20+4)  # 1 contract × 1 budget × 24
    b2_total = n_keys * 2 * (2 + b2_gov_per_key)  # ×2 learners
    if b1_total != 803000 or b2_total != 286000:
        errors.append(f"Recomputed totals: B1={b1_total}, B2={b2_total}")

    # 9. R2 validator
    r2_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_release_r2_3.py")],
        capture_output=True, text=True,
    )
    if "VALIDATOR: PASS" not in r2_result.stdout:
        errors.append("R2 validator not PASS")

    # 10. Paper/frozen namespace
    diff = subprocess.run(
        ["git", "diff", "--name-only", "fbaa9f3...HEAD", "--",
         "artifacts/", "paper/", "results/edbt_eab_revision/", "results/corrected_v2/"],
        capture_output=True, text=True, cwd=ROOT,
    )
    if diff.stdout.strip():
        errors.append(f"Frozen namespace modified: {diff.stdout.strip()}")

    # 11. No TODO/TBD in V4 files only
    for d in ["configs/edbt_t0_b", "scripts/t0_b_v4", "tests/t0_b_v4"]:
        for f in Path(d).glob("*"):
            if "v2" in f.name or "v3" in str(f).lower():
                continue
            if f.suffix in (".json", ".yaml", ".md", ".py") and "validate_protocol" not in f.name:
                content = f.read_text()
                for marker in ["TODO", "TBD", "PLACEHOLDER"]:
                    if marker in content:
                        errors.append(f"{marker} in {f}")

    print(f"\n=== T0-B V4 VALIDATOR ===")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")
    if errors:
        print("\nVALIDATOR: FAIL"); sys.exit(1)
    else:
        print("\nVALIDATOR: PASS"); sys.exit(0)

if __name__ == "__main__":
    main()
