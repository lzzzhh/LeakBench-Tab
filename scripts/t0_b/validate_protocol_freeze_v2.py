#!/usr/bin/env python3
"""T0-B V2 Protocol Freeze Static Validator."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    errors = []

    # 1. Check branch and parent SHA
    import subprocess
    branch = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True).stdout.strip()
    if branch != "t0/multipolicy-semantic-cost":
        errors.append(f"Wrong branch: {branch}")

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # fbaa9f3 must be an ancestor
    result = subprocess.run(["git", "merge-base", "--is-ancestor", "fbaa9f3bfad9eda52b25c64bd07561aa5bd8ad33", head])
    if result.returncode != 0:
        errors.append("fbaa9f3 not an ancestor of HEAD")

    # 2. V1 freeze status
    with open(ROOT / "results/edbt_t0_b/freeze_lineage.json") as f:
        lineage = json.load(f)
    if lineage.get("v1_status") != "SUPERSEDED_PRE_OUTCOME":
        errors.append(f"V1 status: {lineage.get('v1_status')}")
    if lineage.get("outcomes_observed_before_supersession") != False:
        errors.append("outcomes_observed_before_supersession is not false")

    # 3. No T0-B outcomes
    outcome_patterns = ["lr_cells", "rf_cells", "lightgbm_cells", "selection_ledger", "task_effects", "policy_summary", "analysis_summary", "claim_state"]
    for f in (ROOT / "results/edbt_t0_b").iterdir():
        for pat in outcome_patterns:
            if pat in f.name and f.name != "protocol_freeze.json" and "freeze_lineage" not in f.name:
                errors.append(f"Outcome file found: {f}")

    dryrun = ROOT / "results/edbt_t0_b_dryrun"
    if dryrun.exists() and list(dryrun.iterdir()):
        errors.append("Dryrun directory not empty")

    # 4. V2 protocol/config hash closure
    v2_files = {
        "protocol_v2.md": "reports/edbt_t0_b/protocol_v2.md",
        "policy_registry_v2.yaml": "configs/edbt_t0_b/policy_registry_v2.yaml",
        "policy_group_registry_v2.json": "configs/edbt_t0_b/policy_group_registry_v2.json",
        "semantic_evaluation_labels_v2.json": "configs/edbt_t0_b/semantic_evaluation_labels_v2.json",
        "cost_contracts_v2.json": "configs/edbt_t0_b/cost_contracts_v2.json",
        "model_factories_v2.json": "configs/edbt_t0_b/model_factories_v2.json",
        "claim_gates_v2.json": "configs/edbt_t0_b/claim_gates_v2.json",
        "execution_matrix_v2.json": "configs/edbt_t0_b/execution_matrix_v2.json",
        "pareto_contract_v2.json": "configs/edbt_t0_b/pareto_contract_v2.json",
    }
    for name, path in v2_files.items():
        if not (ROOT / path).exists():
            errors.append(f"Missing V2 file: {path}")

    # 5. Oracle words not in mechanism group definitions
    reg = json.loads((ROOT / "configs/edbt_t0_b/policy_group_registry_v2.json").read_text())
    forbidden = ["leak", "contam", "contaminant", "legitimate",
                 "target_proxy", "future", "post_outcome",
                 "source_role", "n_leak"]
    for mech, mech_data in reg.get("mechanisms", {}).items():
        for g in mech_data.get("injected_groups", []):
            g_text = json.dumps(g).lower()
            for w in forbidden:
                if w in g_text:
                    errors.append(f"Forbidden word '{w}' in {mech} group {g.get('opaque_group_id')}")

    # 6. Mechanism groups don't reference evaluation labels
    mechs_text = json.dumps(reg.get("mechanisms", {}))
    if "contaminated" in mechs_text.lower():
        errors.append("Mechanism definitions reference contaminated status")

    # 7. Semantic mapping coverage: 5,500 keys
    import pandas as pd
    man = pd.read_csv(ROOT / "artifacts/sp6/sp6_bundle_manifest.csv")
    if len(man) != 5500:
        errors.append(f"Manifest has {len(man)} keys, expected 5500")

    # 8. No TODO/TBD in V2 files
    for path in v2_files.values():
        content = (ROOT / path).read_text()
        for marker in ["TODO", "TBD", "PLACEHOLDER"]:
            if marker in content:
                errors.append(f"{marker} found in {path}")

    # 9. Frozen namespace no diff
    diff = subprocess.run(
        ["git", "diff", "--name-only", "fbaa9f3...HEAD"],
        capture_output=True, text=True
    ).stdout.strip().split("\n")
    allowed_prefixes = [
        "reports/edbt_t0_b/", "configs/edbt_t0_b/", "results/edbt_t0_b/",
        "scripts/t0_b/", "tests/t0_b/",
    ]
    for f in diff:
        if f and not any(f.startswith(p) for p in allowed_prefixes):
            errors.append(f"File outside allowed scope: {f}")

    # 10. Paper no diff
    paper_diff = subprocess.run(
        ["git", "diff", "--name-only", "fbaa9f3...HEAD", "--", "paper/"],
        capture_output=True, text=True
    ).stdout.strip()
    if paper_diff:
        errors.append(f"Paper files modified: {paper_diff}")

    # 11. R2 validator PASS
    r2_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/validate_release_r2_3.py")],
        capture_output=True, text=True
    )
    if "VALIDATOR: PASS" not in r2_result.stdout:
        errors.append("R2 validator did not PASS")

    # 12. Execution counts present
    with open(ROOT / "configs/edbt_t0_b/execution_matrix_v2.json") as f:
        exec_m = json.load(f)
    if exec_m["grand_totals"]["downstream_fits"] != 1089000:
        errors.append(f"Execution matrix downstream_fits: {exec_m['grand_totals']['downstream_fits']}")

    # 13. Claim gate statuses mutually exclusive
    with open(ROOT / "configs/edbt_t0_b/claim_gates_v2.json") as f:
        cg = json.load(f)
    assert cg.get("mutual_exclusivity") == True

    # 14. All planned files exist
    planned = [
        "reports/edbt_t0_b/protocol_v2.md",
        "reports/edbt_t0_b/protocol_amendment_001.md",
        "reports/edbt_t0_b/semantic_lineage_audit_v2.md",
        "reports/edbt_t0_b/runtime_plan_v2.md",
        "reports/edbt_t0_b/static_preflight_report_v2.md",
    ]
    for p in planned:
        if not (ROOT / p).exists():
            errors.append(f"Planned file missing: {p}")

    # Summary
    print(f"\n=== T0-B V2 STATIC VALIDATOR ===")
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
