#!/usr/bin/env python3
"""Derive a standalone M10 pilot bundle from an immutable pilot bundle.

This is a byte-verified subset operation.  It never imports the dataset or
mechanism generators and it preserves each source task hash exactly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.leakbench.run_m10_amendment import (  # noqa: E402
    derive_strict_contract,
    file_sha256,
    load_amendment_config,
    load_bundle_contract,
    load_verified_task,
)


BASE_ARRAY_KEYS = (
    "base_X", "y", "train_idx", "val_idx", "test_idx", "timestamps",
    "base_entity_ids",
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/m10_amendment_v1.yaml")
    parser.add_argument(
        "--source-manifest",
        default="results/corrected_v2/diagnostic_pilot_tasks/task_manifest.csv",
    )
    parser.add_argument(
        "--output-dir", default="results/corrected_v2/m10_amendment_pilot_tasks"
    )
    args = parser.parse_args(argv)

    config_path = (ROOT / args.config).resolve()
    _, amendment, _, _, base_config_hash = load_amendment_config(config_path)
    source_manifest_path = (ROOT / args.source_manifest).resolve()
    source_manifest, source_manifest_hash, source_summary_path, source_summary = (
        load_bundle_contract(source_manifest_path, base_config_hash)
    )
    if source_manifest_hash != amendment["source_pilot_task_manifest_sha256"]:
        raise RuntimeError("Source pilot task manifest differs from the amendment binding")
    if file_sha256(source_summary_path) != amendment["source_pilot_bundle_summary_sha256"]:
        raise RuntimeError("Source pilot bundle summary differs from the amendment binding")
    if str(source_summary["dataset_namespace"]) != amendment["pilot_namespace"]:
        raise RuntimeError("Source pilot bundle has the wrong namespace")

    selected = source_manifest.loc[
        source_manifest["mechanism"].astype(str) == "M10"
    ].copy()
    if len(selected) != int(amendment["expected_pilot_tasks"]):
        raise RuntimeError(
            f"Expected {amendment['expected_pilot_tasks']} M10 pilot tasks, got {len(selected)}"
        )
    output_dir = (ROOT / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    output_rows = []
    verified_contracts = 0
    for source_bundle_raw, rows in selected.groupby("bundle_path", sort=False):
        source_bundle = (ROOT / str(source_bundle_raw)).resolve()
        expected_hashes = set(rows["bundle_sha256"].astype(str))
        if expected_hashes != {file_sha256(source_bundle)}:
            raise RuntimeError(f"Source bundle SHA256 mismatch: {source_bundle}")

        arrays = {}
        with np.load(source_bundle, allow_pickle=False) as source:
            for key in BASE_ARRAY_KEYS:
                if key in source.files:
                    arrays[key] = np.asarray(source[key])
            required_base = {"base_X", "y", "train_idx", "val_idx", "test_idx"}
            if not required_base.issubset(arrays):
                raise RuntimeError(f"Source bundle is missing base arrays: {source_bundle}")
            for _, row in rows.iterrows():
                task, _ = load_verified_task(row, ROOT)
                derive_strict_contract(task, row, amendment)
                verified_contracts += 1
                key = str(row["bundle_key"])
                for prefix in ("block", "leak_mask", "entity_ids", "source_ids"):
                    array_key = f"{prefix}__{key}"
                    if array_key not in source.files:
                        raise RuntimeError(f"Source bundle is missing {array_key}")
                    arrays[array_key] = np.asarray(source[array_key])

        dataset_id = str(rows.iloc[0]["dataset_id"])
        output_bundle = output_dir / f"{dataset_id}.npz"
        np.savez_compressed(output_bundle, **arrays)
        output_bundle_hash = file_sha256(output_bundle)
        for _, source_row in rows.iterrows():
            output_row = source_row.to_dict()
            output_row["bundle_path"] = str(output_bundle.relative_to(ROOT))
            output_row["bundle_sha256"] = output_bundle_hash
            output_rows.append(output_row)
        print(
            f"derived {dataset_id}: {len(rows)} M10 tasks, "
            f"{output_bundle.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )

    manifest = pd.DataFrame(output_rows)[list(source_manifest.columns)]
    manifest_path = output_dir / "task_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "schema_version": 1,
        "dataset_namespace": amendment["pilot_namespace"],
        "config_sha256": base_config_hash,
        "task_count": len(manifest),
        "datasets": sorted(int(value) for value in manifest["dataset_index"].unique()),
        "mechanisms": ["M10"],
        "strengths": list(amendment["strengths"]),
        "seeds": sorted(int(value) for value in manifest["seed"].unique()),
        "manifest_sha256": file_sha256(manifest_path),
        "derivation": "verified_M10_subset_without_regeneration",
        "source_manifest_path": str(source_manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": source_manifest_hash,
        "source_bundle_summary_path": str(source_summary_path.relative_to(ROOT)),
        "source_bundle_summary_sha256": file_sha256(source_summary_path),
        "strict_contract_verified_tasks": verified_contracts,
    }
    summary_path = output_dir / "bundle_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    derived, _, _, _ = load_bundle_contract(manifest_path, base_config_hash)
    for _, row in derived.iterrows():
        task, _ = load_verified_task(row, ROOT)
        derive_strict_contract(task, row, amendment)
    print(json.dumps({
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": file_sha256(manifest_path),
        "tasks": len(derived),
        "strict_contract_verified_tasks": verified_contracts,
    }, indent=2))


if __name__ == "__main__":
    main()
