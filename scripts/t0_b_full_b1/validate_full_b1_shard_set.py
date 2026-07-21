#!/usr/bin/env python3
"""T0-B R10c-1 — Strict shard-set admission CLI."""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.t0_b_full_b1.merge_contract import (
    validate_shard_set, ShardSetAdmissionResult,
)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True)
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    plan_path = Path(args.plan_manifest)
    if not plan_path.exists():
        print(f"plan manifest not found: {plan_path}")
        sys.exit(1)

    plan_manifest = json.loads(plan_path.read_text())
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan_dir = plan_path.parent
    shard_root = Path(args.shard_root)

    result = validate_shard_set(
        plan_manifest=plan_manifest,
        plan_manifest_sha256=plan_sha,
        plan_dir=plan_dir,
        key_rows=[],
        run_rows=[],
        shard_root=shard_root,
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
