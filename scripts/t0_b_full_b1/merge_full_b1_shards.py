#!/usr/bin/env python3
"""T0-B Full-B1 Merge VF — reads shards, validates, writes deterministic merged ledgers."""
from __future__ import annotations
import gzip, hashlib, io, json, sys
from pathlib import Path; import pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True)
    ap.add_argument("--shard-root", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    shard_root = Path(args.shard_root)
    with open(ROOT / args.plan_manifest) as f:
        pm = json.load(f)
    sc = pm["shard_count"]

    shard_dirs = [shard_root / f"shard_{sid}" for sid in range(sc)]
    # If no shard dirs have data, return not-executed
    if not any(d.exists() for d in shard_dirs):
        print("EXPECTED_NOT_EXECUTED")
        sys.exit(42)

    # Validate all shards present
    for sid, sd in enumerate(shard_dirs):
        if not (sd / "shard_manifest.json").exists():
            print(f"ERROR: Shard {sid} manifest missing")
            sys.exit(1)

    # Read all shard data
    all_bl = []; all_gl = []; all_sl = []
    for sid, sd in enumerate(shard_dirs):
        for ledger in ["baseline_ledger", "governed_ledger", "selection_ledger"]:
            data = gzip.decompress((sd / f"{ledger}.csv.gz").read_bytes()).decode("utf-8")
            lines = [l for l in data.strip().split("\n") if l]
            if sid == 0:
                header = lines[0]
            for line in lines[1:]:
                if ledger == "baseline_ledger":
                    all_bl.append(line)
                elif ledger == "governed_ledger":
                    all_gl.append(line)
                else:
                    all_sl.append(line)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # Write merged ledgers
    for name, rows, header_line in [
        ("baseline_ledger", all_bl, "run_id,dataset_index,mechanism,strength,training_seed,learner,baseline_type,auc"),
        ("governed_ledger", all_gl, "run_id,dataset_index,mechanism,strength,training_seed,governance_seed,learner,policy,contract,budget_bp,strict_auc,full_auc,governed_auc,legacy_sdr,selection_hash,realized_cost"),
        ("selection_ledger", all_sl, "selection_hash,policy,contract,budget_bp,removed_encoded_indices,removed_group_ids,realized_encoded_cost"),
    ]:
        content = header_line + "\n" + "\n".join(sorted(rows)) + "\n"
        compressed = gzip.compress(content.encode("utf-8"), mtime=0)
        (out / f"{name}.csv.gz").write_bytes(compressed)

    # Failure ledger (empty)
    fl = "run_id\n"
    (out / "failure_ledger.csv.gz").write_bytes(gzip.compress(fl.encode("utf-8"), mtime=0))

    # Manifest
    merged_manifest = {"merged_shards": sc, "baseline_rows": len(all_bl), "governed_rows": len(all_gl), "selection_rows": len(all_sl)}
    with open(out / "full_b1_manifest.json", "w") as f:
        json.dump(merged_manifest, f, indent=2)

    print(f"Merged: {len(all_bl)} baseline, {len(all_gl)} governed, {len(all_sl)} selections")
    sys.exit(0)

if __name__ == "__main__":
    main()
