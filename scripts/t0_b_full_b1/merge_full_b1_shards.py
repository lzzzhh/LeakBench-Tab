#!/usr/bin/env python3
"""T0-B Full-B1 Merge — strict: no silent dedup, no failure masking, atomic writes."""
import gzip, hashlib, io, json, sys
from pathlib import Path; import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.t0_b_full_b1.io_contract import (
    atomic_write_gzip_text, atomic_write_json, exclusive_writer_lock, WriterLockError,
)

def _find_duplicates(values):
    seen = {}; dups = []
    for v in values:
        if v in seen:
            if seen[v] == 1: dups.append(v)
            seen[v] += 1
        else:
            seen[v] = 1
    return dups

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-manifest", required=True); ap.add_argument("--shard-root", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    shard_root = Path(args.shard_root)

    with open(ROOT / args.plan_manifest) as f: pm = json.load(f)
    sc = pm["shard_count"]

    # Validate all shards present
    for sid in range(sc):
        sd = shard_root / f"shard_{sid}"
        if not (sd / "shard_manifest.json").exists():
            print(f"FAIL: Shard {sid} manifest missing"); sys.exit(1)

    # Read all shard data
    all_bl = []; all_gl = []; all_sl = []; all_fl = []
    for sid in range(sc):
        sd = shard_root / f"shard_{sid}"
        for ledger, lst in [("baseline_ledger", all_bl), ("governed_ledger", all_gl),
                            ("selection_ledger", all_sl), ("failure_ledger", all_fl)]:
            fp = sd / f"{ledger}.csv.gz"
            if fp.exists():
                data = gzip.decompress(fp.read_bytes()).decode("utf-8")
                for line in data.strip().split("\n")[1:]:
                    if line: lst.append(line)

    # Check for duplicates BEFORE sorting — NO silent dedup
    bl_ids = [l.split(",")[0] for l in all_bl]
    gl_ids = [l.split(",")[0] for l in all_gl]
    bl_dups = _find_duplicates(bl_ids)
    gl_dups = _find_duplicates(gl_ids)
    if bl_dups:
        print(f"FAIL: duplicate baseline run IDs: {len(bl_dups)}"); sys.exit(1)
    if gl_dups:
        print(f"FAIL: duplicate governed run IDs: {len(gl_dups)}"); sys.exit(1)

    # Check failure rows — NO masking
    if all_fl:
        print(f"FAIL: {len(all_fl)} failure rows detected"); sys.exit(1)

    out = Path(args.output_dir)
    try:
        with exclusive_writer_lock(out, "merge"):
            # Sort by structured fields (not set)
            sorted_bl = sorted(all_bl)
            sorted_gl = sorted(all_gl)
            sorted_sl = sorted(all_sl)

            bl_hdr = "run_id,dataset_index,mechanism,strength,training_seed,learner,baseline_type,auc"
            gl_hdr = "run_id,dataset_index,mechanism,strength,training_seed,governance_seed,learner,policy,contract,budget_bp,strict_auc,full_auc,governed_auc,legacy_sdr,selection_hash,realized_cost"
            sl_hdr = "selection_hash,policy,contract,budget_bp,removed_encoded_indices,removed_group_ids,realized_encoded_cost"

            for name, lines, hdr in [
                ("baseline_ledger", sorted_bl, bl_hdr),
                ("governed_ledger", sorted_gl, gl_hdr),
                ("selection_ledger", sorted_sl, sl_hdr),
                ("failure_ledger", [], "run_id"),
            ]:
                content = hdr + "\n" + "\n".join(lines) + ("\n" if lines else "\n")
                atomic_write_gzip_text(out / f"{name}.csv.gz", content)

            manifest = {
                "merged_shards": sc,
                "baseline_rows": len(sorted_bl),
                "governed_rows": len(sorted_gl),
                "selection_rows": len(sorted_sl),
                "failure_rows": 0,
            }
            atomic_write_json(out / "full_b1_manifest.json", manifest)

            print(f"Merged: {manifest['baseline_rows']} bl, {manifest['governed_rows']} gl, {manifest['selection_rows']} sl")
            sys.exit(0)
    except WriterLockError as e:
        print(f"FAIL: {e}"); sys.exit(1)

if __name__ == "__main__": main()
