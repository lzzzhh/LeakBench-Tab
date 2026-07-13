#!/usr/bin/env python3
"""Run official TabM from immutable corrected_v2 task bundles.

This runner never regenerates datasets or leakage mechanisms.  It verifies the
exported manifest, bundle bytes, and reconstructed task bytes before model fits.
Confirmatory bundles remain locked behind ``--allow-confirmatory``.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.leakbench.models.official_tabm import fit_predict_official_tabm  # noqa: E402


RESULT_FIELDS = [
    "run_id", "dataset_id", "dataset_index", "dataset_namespace", "dataset_seed",
    "archetype", "n_samples", "n_original", "n_injected", "mechanism", "strength",
    "strength_value", "model", "seed", "status", "failure_reason", "diagnostic_ap",
    "diagnostic_normalized_ap", "top5_recall", "n_leak", "clean_auc", "full_auc",
    "paired_harm", "clean_runtime_sec", "full_runtime_sec", "implementation",
    "clean_best_epoch", "full_best_epoch", "split_hash", "task_hash",
    "task_manifest_sha256", "bundle_path", "bundle_sha256", "integrity_verified",
    "config_hash", "code_hash", "model_manifest_json",
]

REQUIRED_MANIFEST_COLUMNS = {
    "dataset_id", "dataset_index", "dataset_namespace", "dataset_seed", "archetype",
    "mechanism", "strength", "strength_value", "seed", "bundle_key", "task_hash",
    "split_hash", "n_samples", "n_original", "n_injected", "n_leak", "diagnostic_ap",
    "diagnostic_normalized_ap", "top5_recall", "bundle_path", "bundle_sha256",
}


@dataclass(frozen=True)
class FrozenTask:
    base_X: np.ndarray
    X: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    leakage_mask: np.ndarray
    entity_ids: np.ndarray
    source_ids: np.ndarray


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_sha256(task):
    digest = hashlib.sha256()
    for values in (
        task.X,
        task.y,
        task.train_idx,
        task.val_idx,
        task.test_idx,
        task.leakage_mask,
        task.entity_ids,
        task.source_ids,
    ):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def parse_selection(value, universe):
    if value in (None, "all"):
        return list(universe)
    selected = []
    for part in value.split(","):
        part = part.strip()
        if ":" in part and all(item.lstrip("-").isdigit() for item in part.split(":", 1)):
            start, stop = (int(item) for item in part.split(":", 1))
            selected.extend(range(start, stop))
        else:
            selected.append(type(universe[0])(part))
    unknown = set(selected) - set(universe)
    if unknown:
        raise ValueError(f"Unknown selection: {sorted(unknown)}")
    return selected


def empty_result_row():
    row = {field: "" for field in RESULT_FIELDS}
    for field in (
        "diagnostic_ap", "diagnostic_normalized_ap", "top5_recall", "clean_auc",
        "full_auc", "paired_harm", "clean_runtime_sec", "full_runtime_sec",
        "clean_best_epoch", "full_best_epoch",
    ):
        row[field] = np.nan
    row["integrity_verified"] = False
    return row


def append_result(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    unexpected = set(row) - set(RESULT_FIELDS)
    if unexpected:
        raise ValueError(f"Unexpected result fields: {sorted(unexpected)}")
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="raise")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _path_under_root(raw_path, root):
    path = Path(raw_path)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Bundle path escapes repository root: {raw_path}") from exc
    return resolved


def load_bundle_contract(manifest_path, config_hash):
    manifest_hash = file_sha256(manifest_path)
    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise RuntimeError(f"Task manifest is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise RuntimeError("Task manifest is empty")

    summary_path = manifest_path.parent / "bundle_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Missing bundle summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 1:
        raise RuntimeError("Unsupported bundle summary schema")
    if summary.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("Task manifest SHA256 does not match bundle summary")
    if summary.get("config_sha256") != config_hash:
        raise RuntimeError("Bundle config SHA256 does not match requested config")
    if int(summary.get("task_count", -1)) != len(manifest):
        raise RuntimeError("Bundle summary task_count does not match task manifest")

    namespaces = set(manifest["dataset_namespace"].astype(str))
    if namespaces != {str(summary.get("dataset_namespace"))}:
        raise RuntimeError("Bundle namespace is inconsistent across summary and manifest")
    identity_columns = ["dataset_index", "mechanism", "strength", "seed"]
    if manifest.duplicated(identity_columns).any():
        raise RuntimeError(f"Task manifest has duplicate identities: {identity_columns}")
    conflicts = manifest.groupby("bundle_path")["bundle_sha256"].nunique()
    if (conflicts != 1).any():
        raise RuntimeError("Task manifest assigns conflicting SHA256 values to a bundle")
    return manifest, manifest_hash, summary_path, summary


def _validate_task_shape(task, row):
    n_samples = int(row["n_samples"])
    n_original = int(row["n_original"])
    n_injected = int(row["n_injected"])
    n_leak = int(row["n_leak"])
    if task.base_X.shape != (n_samples, n_original):
        raise RuntimeError("Bundle base_X shape does not match manifest")
    if task.X.shape != (n_samples, n_original + n_injected):
        raise RuntimeError("Reconstructed X shape does not match manifest")
    if task.y.shape != (n_samples,):
        raise RuntimeError("Bundle y shape does not match manifest")
    if task.leakage_mask.shape != (n_original + n_injected,):
        raise RuntimeError("Leakage mask shape does not match reconstructed X")
    if int(task.leakage_mask.sum()) != n_leak:
        raise RuntimeError("Leakage mask count does not match manifest")
    if task.entity_ids.shape != (n_samples,) or task.source_ids.shape != (n_samples,):
        raise RuntimeError("Entity/source arrays do not match sample count")

    all_idx = np.concatenate((task.train_idx, task.val_idx, task.test_idx))
    if any(index.ndim != 1 for index in (task.train_idx, task.val_idx, task.test_idx)):
        raise RuntimeError("Split indices must be one-dimensional")
    if len(np.unique(all_idx)) != len(all_idx):
        raise RuntimeError("Split indices overlap")
    if len(all_idx) != n_samples or all_idx.min() < 0 or all_idx.max() >= n_samples:
        raise RuntimeError("Split indices do not form a full valid partition")


def load_verified_task(row, root=ROOT):
    bundle_path = _path_under_root(str(row["bundle_path"]), root)
    expected_bundle_hash = str(row["bundle_sha256"]).lower()
    actual_bundle_hash = file_sha256(bundle_path)
    if actual_bundle_hash != expected_bundle_hash:
        raise RuntimeError(
            f"Bundle SHA256 mismatch for {bundle_path}: "
            f"expected {expected_bundle_hash}, got {actual_bundle_hash}"
        )

    key = str(row["bundle_key"])
    required_keys = {
        "base_X", "y", "train_idx", "val_idx", "test_idx",
        f"block__{key}", f"leak_mask__{key}",
        f"entity_ids__{key}", f"source_ids__{key}",
    }
    with np.load(bundle_path, allow_pickle=False) as bundle:
        missing = required_keys - set(bundle.files)
        if missing:
            raise RuntimeError(f"Bundle is missing arrays: {sorted(missing)}")
        base_X = np.asarray(bundle["base_X"])
        block = np.asarray(bundle[f"block__{key}"])
        task = FrozenTask(
            base_X=base_X,
            X=np.concatenate((base_X, block), axis=1),
            y=np.asarray(bundle["y"]),
            train_idx=np.asarray(bundle["train_idx"]),
            val_idx=np.asarray(bundle["val_idx"]),
            test_idx=np.asarray(bundle["test_idx"]),
            leakage_mask=np.asarray(bundle[f"leak_mask__{key}"]),
            entity_ids=np.asarray(bundle[f"entity_ids__{key}"]),
            source_ids=np.asarray(bundle[f"source_ids__{key}"]),
        )

    _validate_task_shape(task, row)
    actual_split_hash = hashlib.sha256(task.test_idx.tobytes()).hexdigest()
    if actual_split_hash != str(row["split_hash"]):
        raise RuntimeError("Reconstructed split SHA256 does not match task manifest")
    actual_task_hash = task_sha256(task)
    if actual_task_hash != str(row["task_hash"]):
        raise RuntimeError(
            f"Reconstructed task SHA256 mismatch for {key}: "
            f"expected {row['task_hash']}, got {actual_task_hash}"
        )
    return task, bundle_path


def verify_before_fit(task, row, bundle_path):
    actual_bundle_hash = file_sha256(bundle_path)
    if actual_bundle_hash != str(row["bundle_sha256"]).lower():
        raise RuntimeError("Bundle SHA256 changed before model fit")
    actual_task_hash = task_sha256(task)
    if actual_task_hash != str(row["task_hash"]):
        raise RuntimeError("Reconstructed task SHA256 changed before model fit")


def save_predictions(directory, run_id, task, clean_probability, full_probability):
    directory.mkdir(parents=True, exist_ok=True)
    test = task.test_idx
    np.savez_compressed(
        directory / f"{run_id}.npz",
        row_id=test,
        y=task.y[test],
        clean_probability=clean_probability,
        full_probability=full_probability,
        entity_id=task.entity_ids[test],
        source_id=task.source_ids[test],
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument(
        "--task-manifest",
        default="results/corrected_v2/tabm_bundle_pilot_tasks/task_manifest.csv",
    )
    parser.add_argument(
        "--output",
        default="results/corrected_v2/tabm_bundle_pilot/tabm_official_cells.csv",
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mechanisms", default="all")
    parser.add_argument("--strengths", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--namespace", default="pilot")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--allow-confirmatory", action="store_true")
    args = parser.parse_args(argv)

    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    if args.namespace == protocol["dataset_namespace"] and not args.allow_confirmatory:
        raise RuntimeError(
            "Confirmatory TabM bundle execution is locked. Review the bundle pilot, "
            "then pass --allow-confirmatory explicitly."
        )

    manifest_path = (ROOT / args.task_manifest).resolve()
    config_hash = file_sha256(config_path)
    manifest, manifest_hash, summary_path, summary = load_bundle_contract(
        manifest_path, config_hash
    )
    manifest_namespace = str(summary["dataset_namespace"])
    if args.namespace != manifest_namespace:
        raise RuntimeError(
            f"Requested namespace {args.namespace!r} does not match bundle "
            f"namespace {manifest_namespace!r}"
        )

    protocol_mechanisms = list(protocol["mechanisms"])
    protocol_strengths = list(protocol["strengths"])
    protocol_seeds = [int(seed) for seed in protocol["seeds"]]
    if not set(manifest["mechanism"]).issubset(protocol_mechanisms):
        raise RuntimeError("Task manifest contains mechanisms outside the config protocol")
    if not set(manifest["strength"]).issubset(protocol_strengths):
        raise RuntimeError("Task manifest contains strengths outside the config protocol")
    if not set(manifest["seed"].astype(int)).issubset(protocol_seeds):
        raise RuntimeError("Task manifest contains seeds outside the config protocol")

    dataset_universe = sorted(manifest["dataset_index"].astype(int).unique().tolist())
    mechanism_universe = [item for item in protocol_mechanisms if item in set(manifest["mechanism"])]
    strength_universe = [item for item in protocol_strengths if item in set(manifest["strength"])]
    seed_universe = [item for item in protocol_seeds if item in set(manifest["seed"].astype(int))]
    datasets = parse_selection(args.datasets, dataset_universe)
    mechanisms = parse_selection(args.mechanisms, mechanism_universe)
    strengths = parse_selection(args.strengths, strength_universe)
    seeds = parse_selection(args.seeds, seed_universe)
    selected = manifest.loc[
        manifest["dataset_index"].astype(int).isin(datasets)
        & manifest["mechanism"].isin(mechanisms)
        & manifest["strength"].isin(strengths)
        & manifest["seed"].astype(int).isin(seeds)
    ].copy()
    if selected.empty:
        raise RuntimeError("Task selection is empty")

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_dir = output.parent / f"{output.stem}_predictions"
    code_paths = (ROOT / "src/leakbench/models/official_tabm.py", Path(__file__))
    code_hash = hashlib.sha256(
        "".join(file_sha256(path) for path in code_paths).encode()
    ).hexdigest()
    adapter_kwargs = {
        "device": args.device,
        "k": args.k,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    }
    adapter_hash = hashlib.sha256(
        json.dumps(adapter_kwargs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    completed = set()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"{output} exists; pass --resume or choose a new path")
        existing = pd.read_csv(output)
        if args.retry_failures:
            completed = set(existing.loc[existing["status"] == "SUCCESS", "run_id"].astype(str))
        else:
            completed = set(existing["run_id"].astype(str))

    clean_cache = {}
    total = len(selected)
    attempted = 0
    started = time.time()
    for _, manifest_row in selected.iterrows():
        seed = int(manifest_row["seed"])
        run_key = "|".join((
            "bundle", manifest_namespace, str(manifest_row["dataset_id"]),
            str(manifest_row["mechanism"]), str(manifest_row["strength"]), "tabm",
            str(seed), str(manifest_row["task_hash"]), manifest_hash, config_hash,
            code_hash, adapter_hash,
        ))
        run_id = hashlib.sha256(run_key.encode()).hexdigest()[:20]
        if run_id in completed:
            attempted += 1
            continue

        row = empty_result_row()
        row.update({
            "run_id": run_id,
            "dataset_id": manifest_row["dataset_id"],
            "dataset_index": int(manifest_row["dataset_index"]),
            "dataset_namespace": manifest_namespace,
            "dataset_seed": int(manifest_row["dataset_seed"]),
            "archetype": manifest_row["archetype"],
            "n_samples": int(manifest_row["n_samples"]),
            "n_original": int(manifest_row["n_original"]),
            "n_injected": int(manifest_row["n_injected"]),
            "mechanism": manifest_row["mechanism"],
            "strength": manifest_row["strength"],
            "strength_value": float(manifest_row["strength_value"]),
            "model": "tabm",
            "seed": seed,
            "status": "FAILURE",
            "failure_reason": "",
            "diagnostic_ap": float(manifest_row["diagnostic_ap"]),
            "diagnostic_normalized_ap": float(manifest_row["diagnostic_normalized_ap"]),
            "top5_recall": float(manifest_row["top5_recall"]),
            "n_leak": int(manifest_row["n_leak"]),
            "split_hash": manifest_row["split_hash"],
            "task_hash": manifest_row["task_hash"],
            "task_manifest_sha256": manifest_hash,
            "bundle_path": manifest_row["bundle_path"],
            "bundle_sha256": manifest_row["bundle_sha256"],
            "config_hash": config_hash,
            "code_hash": code_hash,
        })
        try:
            task, bundle_path = load_verified_task(manifest_row, ROOT)
            cache_key = (str(manifest_row["dataset_id"]), seed, str(manifest_row["bundle_sha256"]))
            if cache_key not in clean_cache:
                try:
                    verify_before_fit(task, manifest_row, bundle_path)
                    clean_cache[cache_key] = fit_predict_official_tabm(
                        task.base_X[task.train_idx],
                        task.y[task.train_idx],
                        task.base_X[task.val_idx],
                        task.y[task.val_idx],
                        task.base_X[task.test_idx],
                        seed=seed,
                        **adapter_kwargs,
                    )
                except Exception as exc:
                    clean_cache[cache_key] = exc
            clean_output = clean_cache[cache_key]
            if isinstance(clean_output, Exception):
                raise RuntimeError(f"clean TabM fit failed: {clean_output}") from clean_output

            verify_before_fit(task, manifest_row, bundle_path)
            row["integrity_verified"] = True
            full_output = fit_predict_official_tabm(
                task.X[task.train_idx],
                task.y[task.train_idx],
                task.X[task.val_idx],
                task.y[task.val_idx],
                task.X[task.test_idx],
                seed=seed,
                **adapter_kwargs,
            )
            y_test = task.y[task.test_idx]
            clean_auc = float(roc_auc_score(y_test, clean_output.probabilities))
            full_auc = float(roc_auc_score(y_test, full_output.probabilities))
            model_manifest = dict(full_output.manifest)
            model_manifest.update({
                "input_mode": "immutable_npz_bundle",
                "task_manifest_sha256": manifest_hash,
                "bundle_sha256": str(manifest_row["bundle_sha256"]),
                "task_sha256": str(manifest_row["task_hash"]),
            })
            row.update({
                "status": "SUCCESS",
                "clean_auc": clean_auc,
                "full_auc": full_auc,
                "paired_harm": full_auc - clean_auc,
                "clean_runtime_sec": clean_output.runtime_sec,
                "full_runtime_sec": full_output.runtime_sec,
                "implementation": full_output.implementation,
                "clean_best_epoch": clean_output.best_epoch,
                "full_best_epoch": full_output.best_epoch,
                "model_manifest_json": json.dumps(
                    model_manifest, sort_keys=True, separators=(",", ":")
                ),
            })
            if row["mechanism"] in protocol["predictions_retained_for"]:
                save_predictions(
                    prediction_dir,
                    run_id,
                    task,
                    clean_output.probabilities,
                    full_output.probabilities,
                )
        except Exception as exc:
            row["failure_reason"] = f"{type(exc).__name__}: {exc}"

        append_result(output, row)
        attempted += 1
        elapsed = time.time() - started
        print(
            f"{attempted}/{total} cells, status={row['status']}, {elapsed:.1f}s elapsed",
            flush=True,
        )

    result_frame = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=RESULT_FIELDS)
    latest_results = result_frame.drop_duplicates("run_id", keep="last")
    manifest_output = {
        "schema_version": 1,
        "protocol_version": protocol["version"],
        "input_mode": "immutable_npz_bundle",
        "model_identity": "tabm.TabM",
        "required_tabm_version": "0.0.3",
        "dataset_namespace": manifest_namespace,
        "confirmatory_authorized": bool(args.allow_confirmatory),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_hash": config_hash,
        "task_manifest_path": str(manifest_path.relative_to(ROOT)),
        "task_manifest_sha256": manifest_hash,
        "bundle_summary_path": str(summary_path.relative_to(ROOT)),
        "bundle_summary_sha256": file_sha256(summary_path),
        "bundle_sha256": sorted(set(selected["bundle_sha256"].astype(str))),
        "code_hash": code_hash,
        "code_files": [str(path.relative_to(ROOT)) for path in code_paths],
        "output": str(output.relative_to(ROOT)),
        "requested_cells": total,
        "success_cells": int((latest_results["status"] == "SUCCESS").sum()),
        "failure_cells": int((latest_results["status"] != "SUCCESS").sum()),
        "integrity_verified_cells": int(
            latest_results["integrity_verified"].astype(str).str.lower().eq("true").sum()
        ),
        "datasets": datasets,
        "mechanisms": mechanisms,
        "strengths": strengths,
        "seeds": seeds,
        "adapter_kwargs": adapter_kwargs,
        "adapter_hash": adapter_hash,
        "result_sha256": file_sha256(output) if output.exists() else "",
    }
    write_json_atomic(output.parent / f"{output.stem}_manifest.json", manifest_output)
    return 0 if manifest_output["failure_cells"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
