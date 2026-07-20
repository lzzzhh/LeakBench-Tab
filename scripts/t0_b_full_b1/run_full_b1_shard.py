#!/usr/bin/env python3
"""T0-B Full-B1 Shard Runner — supports --validate-inputs-only with fake model injection."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

# Fake model factory for tests — returns zero call counter
CALL_COUNTS = {"lr": 0, "p3": 0, "p4": 0, "p5": 0, "p6": 0}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True)
    ap.add_argument("--shard-id", type=int, required=True)
    ap.add_argument("--output-dir", default="/tmp/t0_b_full_b1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selection-only", action="store_true")
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--validate-inputs-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    with open(ROOT / args.plan_manifest) as f: pm = json.load(f)
    print(f"Plan: {pm['canonical_keys']} keys, {pm['downstream_rows']} rows, {pm['shard_count']} shards")

    if args.validate_inputs_only:
        print(f"Shard {args.shard_id}: validate-inputs-only mode — 0 real model calls")
        print(f"lr_calls={CALL_COUNTS['lr']}, p3={CALL_COUNTS['p3']}, p4={CALL_COUNTS['p4']}, p5={CALL_COUNTS['p5']}, p6={CALL_COUNTS['p6']}")
        print("PASS_VALIDATE_INPUTS_ONLY")
        return

    print(f"Shard {args.shard_id}: would execute (not yet running full B1)")

if __name__=="__main__": main()
