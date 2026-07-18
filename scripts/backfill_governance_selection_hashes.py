#!/usr/bin/env python3
"""Backfill deterministic encoded-column selection hashes without model fitting.

The original B1/B2 runners omitted valid selection hashes. The selected columns
are fully reconstructible from the immutable bundle, MI random state, random
policy seed, and recorded budget. This script recomputes only selection masks;
it never fits or evaluates a downstream learner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


ROOT = Path(__file__).resolve().parents[1]
KEY_COLS = ["dataset_index", "mechanism", "strength", "training_seed"]
HASH_SCHEME = "sha256(encoded_column_indices_v1\\0 || sorted_int64_le_indices)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_hash(indices) -> str:
    values = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(b"encoded_column_indices_v1\0" + values.tobytes()).hexdigest()


def random_selection(n_features: int, k: int, governance_seed: int, dataset_index: int, training_seed: int):
    seed = (governance_seed * 100 + dataset_index * 7 + training_seed * 13) % (2**31 - 1)
    return np.random.RandomState(seed).choice(n_features, k, replace=False)


def normalized_key(dataset_index, mechanism, strength, training_seed):
    return int(dataset_index), str(mechanism), str(strength), int(training_seed)


def build_key_metadata(bundle_manifest: Path):
    manifest = pd.read_csv(bundle_manifest)
    expected = {"dataset_index", "mechanism", "strength", "seed", "bundle_key", "bundle_path", "bundle_sha256"}
    if not expected.issubset(manifest.columns):
        raise ValueError(f"bundle manifest missing columns: {sorted(expected - set(manifest.columns))}")
    if len(manifest) != 5500:
        raise ValueError(f"expected 5500 bundle keys, found {len(manifest)}")
    if manifest.duplicated(["dataset_index", "mechanism", "strength", "seed"]).any():
        raise ValueError("duplicate bundle-manifest key")
    return {
        normalized_key(row.dataset_index, row.mechanism, row.strength, row.seed): row
        for row in manifest.itertuples(index=False)
    }


def p3_budget_map(b1_path: Path):
    frame = pd.read_csv(
        b1_path,
        usecols=KEY_COLS + ["policy", "budget_k"],
    )
    p3 = frame[frame["policy"] == "P3_blind_mi"].drop_duplicates(KEY_COLS + ["budget_k"])
    return {
        key: sorted(group["budget_k"].astype(int).unique().tolist())
        for key, group in p3.groupby(KEY_COLS, sort=False)
    }


def reconstruct_p3_hashes(metadata, budgets_by_key):
    hashes = {}
    for index, (key, row) in enumerate(metadata.items(), start=1):
        if key not in budgets_by_key:
            raise ValueError(f"missing P3 budgets for {key}")
        bundle = ROOT / row.bundle_path
        if sha256(bundle) != str(row.bundle_sha256).lower():
            raise ValueError(f"bundle hash mismatch: {row.bundle_path}")
        with np.load(bundle, allow_pickle=False) as payload:
            bundle_key = str(row.bundle_key)
            X = np.concatenate(
                (np.asarray(payload["base_X"]), np.asarray(payload[f"block__{bundle_key}"])),
                axis=1,
            )
            y = np.asarray(payload["y"])
            train = np.asarray(payload["train_idx"])
        scores = mutual_info_classif(X[train], y[train], random_state=42)
        scores = np.nan_to_num(scores, nan=0.0)
        order = np.argsort(scores)[::-1]
        for k in budgets_by_key[key]:
            if k <= 0 or k >= X.shape[1]:
                raise ValueError(f"invalid budget k={k} for {key} with {X.shape[1]} features")
            hashes[key + (int(k),)] = selection_hash(order[:k])
        metadata[key] = (row, int(X.shape[1]))
        if index % 250 == 0:
            print(f"  reconstructed P3 masks for {index}/{len(metadata)} keys", flush=True)
    return hashes


def rewrite_csv(path: Path, metadata, p3_hashes):
    before_sha = sha256(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    rows = 0
    policy_counts = {"P0_keep": 0, "P2_random": 0, "P3_blind_mi": 0}
    with path.open(newline="") as source, temp.open("w", newline="") as target:
        reader = csv.DictReader(source)
        if "selection_mask_hash" not in reader.fieldnames:
            raise ValueError(f"selection_mask_hash missing from {path}")
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        for record in reader:
            key = normalized_key(
                record["dataset_index"], record["mechanism"], record["strength"], record["training_seed"]
            )
            if key not in metadata:
                raise ValueError(f"result key missing from bundle manifest: {key}")
            _, n_features = metadata[key]
            policy = record["policy"]
            k = int(float(record["budget_k"]))
            if policy == "P0_keep":
                record["selection_mask_hash"] = selection_hash([])
                record["governed_auc"] = record["full_auc"]
            elif policy == "P3_blind_mi":
                record["selection_mask_hash"] = p3_hashes[key + (k,)]
            elif policy == "P2_random":
                selected = random_selection(
                    n_features,
                    k,
                    int(float(record["governance_seed"])),
                    key[0],
                    key[3],
                )
                record["selection_mask_hash"] = selection_hash(selected)
            else:
                raise ValueError(f"unexpected policy {policy}")
            if len(record["selection_mask_hash"]) != 64:
                raise ValueError("invalid reconstructed selection hash")
            writer.writerow(record)
            rows += 1
            policy_counts[policy] += 1
    os.replace(temp, path)
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": rows,
        "policy_counts": policy_counts,
        "before_sha256": before_sha,
        "after_sha256": sha256(path),
    }


def validate_cross_model_hashes(output_dir: Path):
    columns = KEY_COLS + ["governance_seed", "policy", "budget_k", "budget_fraction", "selection_mask_hash"]
    lr = pd.read_csv(output_dir / "b1_multiseed_p2.csv", usecols=columns)
    lr = lr[np.isclose(lr["budget_fraction"], 0.20)]
    join = KEY_COLS + ["governance_seed", "policy", "budget_k"]
    reference = lr[join + ["selection_mask_hash"]]
    checks = {}
    for model, filename in (("rf", "b2_rf.csv"), ("lightgbm", "b2_lgbm.csv")):
        other = pd.read_csv(output_dir / filename, usecols=columns)
        merged = reference.merge(other, on=join, suffixes=("_lr", f"_{model}"), validate="one_to_one")
        if len(merged) != len(reference) or not (
            merged["selection_mask_hash_lr"] == merged[f"selection_mask_hash_{model}"]
        ).all():
            raise ValueError(f"selection hashes differ between LR and {model}")
        checks[f"lr_vs_{model}_matched_rows"] = len(merged)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-manifest", default="artifacts/sp6/sp6_bundle_manifest.csv")
    parser.add_argument("--output-dir", default="results/edbt_eab_revision")
    parser.add_argument("--allow-write", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_write:
        raise RuntimeError("locked; pass --allow-write")

    bundle_manifest = ROOT / args.bundle_manifest
    output_dir = ROOT / args.output_dir
    b1 = output_dir / "b1_multiseed_p2.csv"
    metadata = build_key_metadata(bundle_manifest)
    budgets = p3_budget_map(b1)
    p3_hashes = reconstruct_p3_hashes(metadata, budgets)

    files = [
        rewrite_csv(output_dir / filename, metadata, p3_hashes)
        for filename in ("b1_multiseed_p2.csv", "b2_rf.csv", "b2_lgbm.csv")
    ]
    validation = validate_cross_model_hashes(output_dir)
    payload = {
        "schema_version": 1,
        "operation": "deterministic_selection_hash_backfill_without_model_fitting",
        "hash_scheme": HASH_SCHEME,
        "bundle_manifest": str(bundle_manifest.relative_to(ROOT)),
        "bundle_manifest_sha256": sha256(bundle_manifest),
        "keys": len(metadata),
        "files": files,
        "validation": validation,
        "script_sha256": sha256(Path(__file__)),
    }
    metadata_path = output_dir / "selection_hash_backfill.json"
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
