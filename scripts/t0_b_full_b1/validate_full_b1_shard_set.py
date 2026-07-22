#!/usr/bin/env python3
"""T0-B R10c-1 — Strict shard-set admission CLI."""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_full_b1.merge_contract import (
    validate_shard_set,
)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True)
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    # ── Structured plan manifest loading ──
    plan_path = Path(args.plan_manifest)
    if not plan_path.exists():
        print("STRICT_SHARD_SET_ADMISSION_FAIL")
        print(f"  plan manifest not found: {plan_path}")
        sys.exit(1)
    if plan_path.is_symlink():
        print("STRICT_SHARD_SET_ADMISSION_FAIL")
        print(f"  plan manifest is a symlink: {plan_path}")
        sys.exit(1)
    try:
        raw_plan = plan_path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        print("STRICT_SHARD_SET_ADMISSION_FAIL")
        print(f"  plan manifest read error: {exc}")
        sys.exit(1)
    try:
        plan_manifest = json.loads(raw_plan)
    except json.JSONDecodeError as exc:
        print("STRICT_SHARD_SET_ADMISSION_FAIL")
        print(f"  plan manifest corrupt JSON: {exc}")
        sys.exit(1)
    if not isinstance(plan_manifest, dict):
        print("STRICT_SHARD_SET_ADMISSION_FAIL")
        print("  plan manifest not a JSON object")
        sys.exit(1)

    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_dir = plan_path.parent
    shard_root = Path(args.shard_root)

    expected_mode = "synthetic" if args.synthetic else "production"

    result = validate_shard_set(
        plan_manifest=plan_manifest,
        plan_manifest_sha256=plan_sha,
        plan_dir=plan_dir,
        shard_root=shard_root,
        expected_mode=expected_mode,
    )

    if result.is_valid:
        print("STRICT_SHARD_SET_ADMISSION_PASS")
        print(f"planned_shards={len(result.planned_shard_ids)}")
        print(f"validated_shards={len(result.validated_shard_ids)}")
        print(f"canonical_keys={result.canonical_keys}")
        print(f"baseline_rows={result.baseline_rows}")
        print(f"governed_rows={result.governed_rows}")
        print(f"selection_rows={result.selection_rows}")
        print(f"failure_rows={result.failure_rows}")
        print(f"downstream_rows={result.downstream_rows}")
        sys.exit(0)
    else:
        print("STRICT_SHARD_SET_ADMISSION_FAIL")
        for e in result.errors:
            print(f"  {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
