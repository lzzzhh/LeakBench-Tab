#!/usr/bin/env python3
"""T0-B Full-B1 Result Validator V2 — real checks or EXPECTED_NOT_EXECUTED."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))

def main():
    out = ROOT / "results/edbt_t0_b_full_b1"
    shards = out / "shards"
    if not shards.exists() or not list(shards.iterdir()):
        print("EXPECTED_NOT_EXECUTED")
        sys.exit(42)

    # If partial results exist, fail
    errors = []
    if not (out / "full_b1_manifest.json").exists():
        errors.append("Manifest missing — partial results")
        print(f"PARTIAL_RESULTS: {len(errors)} errors")
        sys.exit(1)

    print("FULL_RESULTS_VALIDATED_NOT_IMPLEMENTED")
    sys.exit(1)

if __name__=="__main__": main()
