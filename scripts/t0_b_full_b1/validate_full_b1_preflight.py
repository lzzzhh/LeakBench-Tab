#!/usr/bin/env python3
"""T0-B Full-B1 Preflight Validator."""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
SCI_FREEZE = "ff347b0657e8faf5d0ec1a4ca283185ffe2f5845"

from scripts.t0_b_full_b1.merge_contract import (
    validate_plan_schema, validate_plan, validate_global_scope,
)

def main():
    errors = []
    # Lineage check
    lin = ROOT/"results/edbt_t0_b/full_b1_execution_lineage_v1.json"
    if lin.exists():
        with open(lin) as f: l = json.load(f)
        if l.get("canonical_keys")!=5500: errors.append(f"Lineage keys: {l.get('canonical_keys')}")
        if l.get("scientific_freeze")!=SCI_FREEZE: errors.append("Lineage freeze SHA wrong")
    else: errors.append("Lineage missing")

    # Plan check
    pref = ROOT/"results/edbt_t0_b_full_b1_preflight"
    for f in ["full_b1_key_plan.jsonl.gz","full_b1_run_plan.jsonl.gz","full_b1_shard_plan.json","full_b1_plan_manifest.json","full_b1_plan_receipt.json"]:
        if not (pref/f).exists(): errors.append(f"Plan file missing: {f}")

    manifest = {}
    manifest_path = pref / "full_b1_plan_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"Plan manifest unreadable: {exc}")
        if manifest:
            errors.extend(validate_plan_schema(manifest, "production"))
            if not errors:
                plan_errors, keys, runs = validate_plan(manifest, pref)
                errors.extend(plan_errors)
                if not plan_errors:
                    errors.extend(validate_global_scope(manifest, keys, runs))

    if (pref/"full_b1_plan_receipt.json").exists():
        with open(pref/"full_b1_plan_receipt.json") as f: rec = json.load(f)
        if rec.get("canonical_keys")!=5500: errors.append(f"Receipt keys: {rec.get('canonical_keys')}")
        if rec.get("downstream_rows")!=803000: errors.append(f"Receipt rows: {rec.get('downstream_rows')}")
        if not rec.get("pass",False): errors.append("Receipt not pass")
        if manifest_path.exists():
            actual_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if rec.get("plan_manifest_sha256") != actual_manifest_sha:
                errors.append("Receipt plan_manifest_sha256 mismatch")
        if rec.get("tool_seal_sha") != manifest.get("tool_seal_sha"):
            errors.append("Receipt tool_seal_sha mismatch")

    # No full-B1 outcomes
    fb1 = ROOT/"results/edbt_t0_b_full_b1"
    for pat in ["baseline_ledger","governed_ledger","selection_ledger"]:
        for f in fb1.glob(f"**/{pat}*") if fb1.exists() else []:
            errors.append(f"Full-B1 outcome found: {f}")

    print(f"\n=== T0-B FULL-B1 PREFLIGHT VALIDATOR ===\nErrors: {len(errors)}")
    for e in errors: print(f"  ERROR: {e}")
    sys.exit(1 if errors else 0)

if __name__=="__main__": main()
