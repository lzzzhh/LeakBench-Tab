#!/usr/bin/env python3
"""T0-B Full-B1 Merge VF — reads shards, validates, writes deterministic merged ledgers."""
import gzip, hashlib, io, json, sys
from pathlib import Path; import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
def s(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True); ap.add_argument("--shard-root", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    shard_root = Path(args.shard_root)

    with open(ROOT / args.plan_manifest) as f: pm = json.load(f)
    sc = pm["shard_count"]

    # Validate all shards
    for sid in range(sc):
        sd = shard_root / f"shard_{sid}"
        if not (sd / "shard_manifest.json").exists():
            print(f"ERROR: Shard {sid} manifest missing"); sys.exit(1)

    all_bl = []; all_gl = []; all_sl = []; all_fl = []
    for sid in range(sc):
        sd = shard_root / f"shard_{sid}"
        for ledger, lst in [("baseline_ledger", all_bl), ("governed_ledger", all_gl), ("selection_ledger", all_sl), ("failure_ledger", all_fl)]:
            data = gzip.decompress((sd / f"{ledger}.csv.gz").read_bytes()).decode("utf-8")
            for line in data.strip().split("\n")[1:]:
                lst.append(line)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    for name, rows, header in [
        ("baseline_ledger", sorted(set(all_bl)), "run_id,dataset_index,mechanism,strength,training_seed,learner,baseline_type,auc"),
        ("governed_ledger", sorted(set(all_gl)), "run_id,dataset_index,mechanism,strength,training_seed,governance_seed,learner,policy,contract,budget_bp,strict_auc,full_auc,governed_auc,legacy_sdr,selection_hash,realized_cost"),
        ("selection_ledger", sorted(set(all_sl)), "selection_hash,policy,contract,budget_bp,removed_encoded_indices,removed_group_ids,realized_encoded_cost"),
        ("failure_ledger", sorted(set(all_fl)) if all_fl else [], "run_id"),
    ]:
        content = header + "\n" + "\n".join(rows) + "\n"
        (out / f"{name}.csv.gz").write_bytes(gzip.compress(content.encode("utf-8"), mtime=0))

    manifest = {"merged_shards": sc, "baseline_rows": len(set(all_bl)), "governed_rows": len(set(all_gl)),
                "selection_rows": len(set(all_sl))}
    with open(out / "full_b1_manifest.json", "w") as f: json.dump(manifest, f)
    print(f"Merged: {manifest['baseline_rows']} bl, {manifest['governed_rows']} gl, {manifest['selection_rows']} sl")
    sys.exit(0)

if __name__ == "__main__": main()
