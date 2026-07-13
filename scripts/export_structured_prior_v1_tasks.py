#!/usr/bin/env python3
"""Export immutable task bundles for a frozen structured-prior protocol.

This command materializes frozen inputs only.  It does not import a model
adapter or execute any model fit.  The protocol hash manifest and the planned
task grid are verified before any bundle is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.leakbench.datasets import build_panel_task  # noqa: E402
from src.leakbench.mechanisms.structured_prior_v1 import (  # noqa: E402
    AMENDMENT_VERSION,
    StructuredPriorV1Injector,
)
from src.leakbench.structured_prior_protocol import (  # noqa: E402
    PLAN_COLUMNS,
    base_task_sha256,
    build_mechanism_config,
    file_sha256,
    injected_task_sha256,
    load_protocol_config,
)


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def _verify_frozen_file(freeze, path):
    relative = _relative(path)
    entry = freeze["files"].get(relative)
    if entry is None:
        raise RuntimeError(f"File is not bound by the protocol freeze: {relative}")
    actual = file_sha256(path)
    if actual != entry["sha256"]:
        raise RuntimeError(f"Frozen file changed: {relative}")


def _load_frozen_contract(config_path, task_plan_path, freeze_path):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_ANY_MODEL_RUN":
        raise RuntimeError("structured-prior protocol is not in its pre-run frozen state")
    for relative in freeze.get("files", {}):
        _verify_frozen_file(freeze, ROOT / relative)
    if _relative(config_path) not in freeze["files"]:
        raise RuntimeError("requested config is not part of the joint protocol freeze")
    if _relative(task_plan_path) not in freeze["files"]:
        raise RuntimeError("requested task plan is not part of the joint protocol freeze")
    return freeze


def _validate_plan(plan, config, config_hash):
    protocol = config["protocol"]
    missing = set(PLAN_COLUMNS) - set(plan.columns)
    if missing:
        raise RuntimeError(f"frozen task plan is missing columns: {sorted(missing)}")
    if len(plan) != int(protocol["expected_task_variants"]):
        raise RuntimeError("frozen task-plan row count differs from the config")
    if set(plan["config_sha256"].astype(str)) != {config_hash}:
        raise RuntimeError("frozen task plan is bound to a different config")
    expected = {
        "protocol_version": {str(protocol["version"])},
        "study_namespace": {str(protocol["study_namespace"])},
        "dataset_namespace": {str(protocol["dataset_namespace"])},
        "dataset_index": set(int(value) for value in protocol["dataset_indices"]),
        "mechanism": set(protocol["mechanisms"]),
        "strength": set(protocol["strengths"]),
        "seed": set(int(value) for value in protocol["seeds"]),
    }
    for column, values in expected.items():
        actual = set(plan[column].astype(int if column in {"dataset_index", "seed"} else str))
        if actual != values:
            raise RuntimeError(f"frozen task plan has unexpected {column}: {sorted(actual)}")
    identity = ["dataset_index", "mechanism", "strength", "seed"]
    if plan.duplicated(identity).any() or plan["task_variant_id"].duplicated().any():
        raise RuntimeError("frozen task plan has duplicate task identities")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-plan", default=None)
    parser.add_argument("--freeze-manifest", default="protocols/structured_prior_v1/freeze_manifest_v1.json")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    config_path = (ROOT / args.config).resolve()
    config = load_protocol_config(config_path)
    protocol = config["protocol"]
    task_plan_path = (
        (ROOT / args.task_plan).resolve()
        if args.task_plan
        else (ROOT / protocol["frozen_task_plan"]).resolve()
    )
    freeze_path = (ROOT / args.freeze_manifest).resolve()
    freeze = _load_frozen_contract(config_path, task_plan_path, freeze_path)
    plan = pd.read_csv(task_plan_path)
    config_hash = file_sha256(config_path)
    _validate_plan(plan, config, config_hash)

    output = (
        (ROOT / args.output_dir).resolve()
        if args.output_dir
        else (ROOT / protocol["bundle_output"]).resolve()
    )
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    rows = []
    for dataset_index in protocol["dataset_indices"]:
        dataset_plan = plan.loc[plan["dataset_index"].astype(int) == int(dataset_index)]
        base = build_panel_task(
            int(dataset_index), namespace=str(protocol["dataset_namespace"])
        )
        expected_base_hashes = set(dataset_plan["base_task_sha256"].astype(str))
        actual_base_hash = base_task_sha256(base)
        if expected_base_hashes != {actual_base_hash}:
            raise RuntimeError(f"base task differs from freeze: {base.dataset_id}")

        arrays = {
            "base_X": base.X,
            "y": base.y,
            "train_idx": base.train_idx,
            "val_idx": base.val_idx,
            "test_idx": base.test_idx,
            "timestamps": base.timestamps,
            "base_entity_ids": base.entity_ids,
            "base_source_ids": base.source_ids,
        }
        dataset_rows = []
        for _, planned in dataset_plan.iterrows():
            mechanism = str(planned["mechanism"])
            strength = str(planned["strength"])
            seed = int(planned["seed"])
            mechanism_config = build_mechanism_config(
                mechanism, strength, config, seed
            )
            task = StructuredPriorV1Injector(seed=seed).inject(
                base.X,
                base.y,
                [mechanism_config],
                feature_names=list(base.feature_names),
                timestamps=base.timestamps,
                entity_ids=base.entity_ids,
                source_ids=base.source_ids,
                split_type="time",
            )
            if not (
                np.array_equal(task.train_idx, base.train_idx)
                and np.array_equal(task.val_idx, base.val_idx)
                and np.array_equal(task.test_idx, base.test_idx)
            ):
                raise RuntimeError("task export changed the frozen chronological split")
            key = str(planned["task_variant_id"])
            arrays[f"block__{key}"] = task.X[:, task.n_original :]
            arrays[f"leak_mask__{key}"] = task.leakage_mask
            arrays[f"entity_ids__{key}"] = task.entity_ids
            arrays[f"source_ids__{key}"] = task.source_ids
            dataset_rows.append({
                **{column: planned[column] for column in PLAN_COLUMNS},
                "bundle_key": key,
                "task_hash": injected_task_sha256(task),
                "split_hash": hashlib.sha256(task.test_idx.tobytes()).hexdigest(),
                "n_samples": len(task.y),
                "n_original": task.n_original,
                "n_injected": task.n_injected,
                "n_leak": int(task.leakage_mask.sum()),
                "diagnostic_ap": np.nan,
                "diagnostic_normalized_ap": np.nan,
                "top5_recall": np.nan,
                "diagnostic_status": "NOT_COMPUTED_PRE_RUN",
            })

        bundle = output / f"{base.dataset_id}.npz"
        np.savez_compressed(bundle, **arrays)
        bundle_hash = file_sha256(bundle)
        for row in dataset_rows:
            row["bundle_path"] = _relative(bundle)
            row["bundle_sha256"] = bundle_hash
        rows.extend(dataset_rows)
        print(f"exported {base.dataset_id}: {len(dataset_rows)} frozen task variants", flush=True)

    manifest = pd.DataFrame(rows)
    if len(manifest) != int(protocol["expected_task_variants"]):
        raise RuntimeError("exported task count differs from protocol")
    manifest_path = output / "task_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "schema_version": 2,
        "protocol_version": protocol["version"],
        "study_namespace": protocol["study_namespace"],
        "dataset_namespace": protocol["dataset_namespace"],
        "injector_amendment": AMENDMENT_VERSION,
        "config_path": _relative(config_path),
        "config_sha256": config_hash,
        "frozen_task_plan_path": _relative(task_plan_path),
        "frozen_task_plan_sha256": file_sha256(task_plan_path),
        "protocol_freeze_path": _relative(freeze_path),
        "protocol_freeze_sha256": file_sha256(freeze_path),
        "task_count": len(manifest),
        "expected_model_cells": int(protocol["expected_model_cells"]),
        "manifest_sha256": file_sha256(manifest_path),
        "bundle_sha256": sorted(set(manifest["bundle_sha256"].astype(str))),
        "frozen_files_verified": len(freeze["files"]),
        "models_executed": 0,
        "diagnostics_computed": 0,
    }
    (output / "bundle_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
