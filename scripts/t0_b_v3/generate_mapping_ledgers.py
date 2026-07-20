#!/usr/bin/env python3
"""Generate T0-B V3 policy-group mapping ledger and evaluation mapping ledger.

Produces:
- results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz (deterministic, mtime=0)
- results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz (deterministic, mtime=0)

Neutral group IDs: g000, g001, g002, ... — no oracle information.
"""
from __future__ import annotations
import gzip, hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MANIFEST_PATH = ROOT / "artifacts/sp6/sp6_bundle_manifest.csv"
OUT_POLICY = ROOT / "results/edbt_t0_b/policy_group_mapping_v3.jsonl.gz"
OUT_EVAL = ROOT / "results/edbt_t0_b/semantic_evaluation_mapping_v3.jsonl.gz"


def deterministic_gzip_lines(path: Path, lines: list[str]) -> str:
    """Write JSONL with deterministic gzip (mtime=0)."""
    data = "\n".join(lines) + "\n"
    compressed = gzip.compress(data.encode("utf-8"), mtime=0)
    path.write_bytes(compressed)
    return hashlib.sha256(compressed).hexdigest()


def main():
    man = pd.read_csv(MANIFEST_PATH)
    man = man.sort_values(["dataset_index", "mechanism", "strength", "seed"])
    assert len(man) == 5500, f"Expected 5500 keys, got {len(man)}"

    policy_lines: list[str] = []
    eval_lines: list[str] = []
    seen_keys = set()

    for _, row in man.iterrows():
        ds = int(row.dataset_index)
        mech = row.mechanism
        st = row.strength
        ts = int(row.seed)
        bkey = row.bundle_key
        bsha = row.bundle_sha256
        n_orig = int(row.n_original)
        n_inj = int(row.n_injected)
        n_leak = int(row.n_leak)
        n_total = n_orig + n_inj

        key_tuple = (ds, mech, st, ts)
        if key_tuple in seen_keys:
            raise ValueError(f"Duplicate key: {key_tuple}")
        seen_keys.add(key_tuple)

        # Load bundle to get leak mask
        b = np.load(ROOT / row.bundle_path, allow_pickle=False)
        leak_mask = b[f"leak_mask__{bkey}"]
        assert len(leak_mask) == n_total

        # Build neutral groups
        groups = []
        gid_counter = 0

        # Original columns: singletons g000..g(n_orig-1)
        for i in range(n_orig):
            groups.append({
                "opaque_group_id": f"g{gid_counter:03d}",
                "member_encoded_indices": [i],
                "group_size": 1,
            })
            gid_counter += 1

        # Injected columns: mechanism-specific grouping
        if mech in ("M06", "M09", "M11"):
            # One atomic group for all injected columns
            member_indices = list(range(n_orig, n_total))
            groups.append({
                "opaque_group_id": f"g{gid_counter:03d}",
                "member_encoded_indices": member_indices,
                "group_size": len(member_indices),
                "atomic": True,
            })
            gid_counter += 1
        elif mech == "M10":
            # Two singleton groups
            for offset in [0, 1]:
                groups.append({
                    "opaque_group_id": f"g{gid_counter:03d}",
                    "member_encoded_indices": [n_orig + offset],
                    "group_size": 1,
                })
                gid_counter += 1
        else:
            # M01-M05, M07, M08: one singleton per injected column
            for i in range(n_inj):
                groups.append({
                    "opaque_group_id": f"g{gid_counter:03d}",
                    "member_encoded_indices": [n_orig + i],
                    "group_size": 1,
                })
                gid_counter += 1

        # Verify coverage: every column in exactly one group
        covered = set()
        for g in groups:
            for idx in g["member_encoded_indices"]:
                if idx in covered:
                    raise ValueError(f"Column {idx} assigned to multiple groups in {key_tuple}")
                covered.add(idx)
        assert covered == set(range(n_total)), f"Coverage gap: {covered} vs {set(range(n_total))}"

        # Compute mapping hash
        mapping_repr = json.dumps(
            [{"gid": g["opaque_group_id"], "members": g["member_encoded_indices"]} for g in groups],
            sort_keys=True,
        )
        mapping_hash = hashlib.sha256(mapping_repr.encode()).hexdigest()

        # Policy mapping line (NO evaluation labels)
        policy_row = {
            "dataset_index": ds,
            "mechanism": mech,
            "strength": st,
            "training_seed": ts,
            "bundle_key": bkey,
            "bundle_sha256": bsha,
            "n_encoded_columns": n_total,
            "n_groups": len(groups),
            "groups": groups,
            "mapping_sha256": mapping_hash,
        }
        policy_lines.append(json.dumps(policy_row, sort_keys=True))

        # Evaluation mapping: classify groups by leak status
        leak_group_ids = []
        legit_group_ids = []
        for g in groups:
            members = set(g["member_encoded_indices"])
            n_leak_in_group = sum(1 for idx in members if leak_mask[idx])
            if n_leak_in_group == g["group_size"]:
                leak_group_ids.append(g["opaque_group_id"])
            elif n_leak_in_group == 0:
                legit_group_ids.append(g["opaque_group_id"])
            else:
                raise ValueError(f"Mixed-status group: {g['opaque_group_id']} has {n_leak_in_group} leak / {g['group_size']} total")

        eval_row = {
            "dataset_index": ds,
            "mechanism": mech,
            "strength": st,
            "training_seed": ts,
            "bundle_key": bkey,
            "bundle_sha256": bsha,
            "n_encoded_columns": n_total,
            "n_leak_columns": n_leak,
            "leak_group_ids": sorted(leak_group_ids),
            "legitimate_group_ids": sorted(legit_group_ids),
            "leak_mask_hash": hashlib.sha256(np.packbits(leak_mask.astype(bool)).tobytes()).hexdigest(),
        }
        eval_lines.append(json.dumps(eval_row, sort_keys=True))

    # Write deterministic gzips
    policy_sha = deterministic_gzip_lines(OUT_POLICY, policy_lines)
    eval_sha = deterministic_gzip_lines(OUT_EVAL, eval_lines)

    print(f"Policy mapping: {len(policy_lines)} rows, SHA={policy_sha}")
    print(f"Evaluation mapping: {len(eval_lines)} rows, SHA={eval_sha}")

    # Verify round-trip: decompress and check row count
    decompressed = gzip.decompress(OUT_POLICY.read_bytes()).decode("utf-8")
    assert len(decompressed.strip().split("\n")) == 5500
    decompressed = gzip.decompress(OUT_EVAL.read_bytes()).decode("utf-8")
    assert len(decompressed.strip().split("\n")) == 5500
    print("Round-trip verification: PASS")


if __name__ == "__main__":
    main()
