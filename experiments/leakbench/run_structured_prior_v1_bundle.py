#!/usr/bin/env python3
"""Run a frozen structured-prior bundle with a mask-derived strict comparator.

The runner never regenerates data or mechanisms.  It verifies the protocol
freeze, planned grid, bundle manifest, bundle bytes, reconstructed task bytes,
and literal strict/full views before each fit.  Model execution is locked until
``--allow-run`` is supplied explicitly.
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.leakbench.structured_prior_protocol import (  # noqa: E402
    PLAN_COLUMNS,
    file_sha256,
    injected_task_sha256,
    load_protocol_config,
)


STRICT_POLICY = "task.X[:, ~task.leakage_mask]"
FULL_POLICY = "task.X"
RESULT_FIELDS = [
    "run_id", "task_variant_id", "protocol_version", "study_namespace",
    "dataset_id", "dataset_index", "dataset_namespace", "dataset_seed",
    "archetype", "mechanism", "mechanism_family", "strength", "strength_value",
    "model", "seed", "status", "failure_reason", "n_samples", "n_original",
    "n_injected", "n_leak", "diagnostic_ap", "diagnostic_normalized_ap",
    "top5_recall", "strict_auc", "full_auc", "paired_harm",
    "strict_runtime_sec", "full_runtime_sec", "implementation",
    "strict_best_epoch", "full_best_epoch", "split_hash", "base_task_sha256",
    "task_hash", "strict_view_hash", "full_view_hash", "leakage_mask_hash",
    "task_manifest_sha256", "bundle_path", "bundle_sha256",
    "integrity_verified", "config_hash", "freeze_manifest_sha256", "code_hash",
    "adapter_hash", "model_manifest_json",
]

REQUIRED_BUNDLE_COLUMNS = set(PLAN_COLUMNS) | {
    "bundle_key", "task_hash", "split_hash", "n_samples", "n_original",
    "n_injected", "n_leak", "diagnostic_ap", "diagnostic_normalized_ap",
    "top5_recall", "bundle_path", "bundle_sha256",
}


@dataclass(frozen=True)
class FrozenTask:
    X: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    leakage_mask: np.ndarray
    entity_ids: np.ndarray
    source_ids: np.ndarray


def _relative(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def _array_sha256(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _path_under_root(raw_path):
    path = Path(raw_path)
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository root: {raw_path}") from exc
    return resolved


def _verify_frozen_file(freeze, path):
    relative = _relative(path)
    entry = freeze["files"].get(relative)
    if entry is None or file_sha256(path) != entry["sha256"]:
        raise RuntimeError(f"file differs from protocol freeze: {relative}")


def _load_contract(config_path, manifest_path, freeze_path):
    config = load_protocol_config(config_path)
    protocol = config["protocol"]
    if protocol["execution_gate"] != "explicit_allow_run":
        raise RuntimeError("protocol does not declare the explicit execution gate")
    if protocol["strict_policy"] != STRICT_POLICY or protocol["full_policy"] != FULL_POLICY:
        raise RuntimeError("configured strict/full policy differs from the frozen runner")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_ANY_MODEL_RUN":
        raise RuntimeError("protocol is not frozen before model execution")
    plan_path = (ROOT / protocol["frozen_task_plan"]).resolve()
    for relative in freeze.get("files", {}):
        _verify_frozen_file(freeze, ROOT / relative)
    if _relative(config_path) not in freeze["files"]:
        raise RuntimeError("requested config is not part of the joint protocol freeze")
    if _relative(plan_path) not in freeze["files"]:
        raise RuntimeError("requested task plan is not part of the joint protocol freeze")

    summary_path = manifest_path.parent / "bundle_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_hash = file_sha256(manifest_path)
    if summary.get("schema_version") != 2:
        raise RuntimeError("unsupported structured-prior bundle schema")
    if summary.get("protocol_version") != protocol["version"]:
        raise RuntimeError("bundle protocol version differs from requested config")
    if summary.get("study_namespace") != protocol["study_namespace"]:
        raise RuntimeError("bundle study namespace differs from requested config")
    if summary.get("dataset_namespace") != protocol["dataset_namespace"]:
        raise RuntimeError("bundle dataset namespace differs from requested config")
    if summary.get("manifest_sha256") != manifest_hash:
        raise RuntimeError("task manifest hash differs from bundle summary")
    if summary.get("config_sha256") != file_sha256(config_path):
        raise RuntimeError("bundle config hash differs from requested config")
    if summary.get("frozen_task_plan_sha256") != file_sha256(plan_path):
        raise RuntimeError("bundle task-plan hash differs from the frozen plan")
    if summary.get("protocol_freeze_sha256") != file_sha256(freeze_path):
        raise RuntimeError("bundle protocol-freeze hash differs from requested freeze")
    if int(summary.get("task_count", -1)) != int(protocol["expected_task_variants"]):
        raise RuntimeError("bundle task count differs from requested config")
    if int(summary.get("expected_model_cells", -1)) != int(protocol["expected_model_cells"]):
        raise RuntimeError("bundle model-cell count differs from requested config")
    if int(summary.get("models_executed", -1)) != 0:
        raise RuntimeError("input bundle summary is not a model-free export")

    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_BUNDLE_COLUMNS - set(manifest.columns)
    if missing:
        raise RuntimeError(f"task manifest is missing columns: {sorted(missing)}")
    plan = pd.read_csv(plan_path)
    if len(manifest) != int(protocol["expected_task_variants"]):
        raise RuntimeError("task manifest is not complete")
    plan_ids = set(plan["task_variant_id"].astype(str))
    manifest_ids = set(manifest["task_variant_id"].astype(str))
    if plan_ids != manifest_ids or len(manifest_ids) != len(manifest):
        raise RuntimeError("bundle task identities differ from the frozen task plan")
    compare = manifest.set_index("task_variant_id")
    frozen = plan.set_index("task_variant_id")
    for column in PLAN_COLUMNS[1:]:
        if compare[column].astype(str).to_dict() != frozen[column].astype(str).to_dict():
            raise RuntimeError(f"bundle task manifest differs from plan column {column}")
    return config, freeze, manifest, manifest_hash, summary_path


def _load_verified_task(row):
    bundle_path = _path_under_root(str(row["bundle_path"]))
    if file_sha256(bundle_path) != str(row["bundle_sha256"]).lower():
        raise RuntimeError("bundle SHA256 mismatch")
    key = str(row["bundle_key"])
    required = {
        "base_X", "y", "train_idx", "val_idx", "test_idx",
        f"block__{key}", f"leak_mask__{key}",
        f"entity_ids__{key}", f"source_ids__{key}",
    }
    with np.load(bundle_path, allow_pickle=False) as bundle:
        missing = required - set(bundle.files)
        if missing:
            raise RuntimeError(f"bundle is missing arrays: {sorted(missing)}")
        base_X = np.asarray(bundle["base_X"])
        block = np.asarray(bundle[f"block__{key}"])
        task = FrozenTask(
            X=np.concatenate((base_X, block), axis=1),
            y=np.asarray(bundle["y"]),
            train_idx=np.asarray(bundle["train_idx"]),
            val_idx=np.asarray(bundle["val_idx"]),
            test_idx=np.asarray(bundle["test_idx"]),
            leakage_mask=np.asarray(bundle[f"leak_mask__{key}"]),
            entity_ids=np.asarray(bundle[f"entity_ids__{key}"]),
            source_ids=np.asarray(bundle[f"source_ids__{key}"]),
        )
    n = int(row["n_samples"])
    if task.X.shape != (n, int(row["n_original"]) + int(row["n_injected"])):
        raise RuntimeError("reconstructed task shape differs from manifest")
    if task.y.shape != (n,) or task.leakage_mask.shape != (task.X.shape[1],):
        raise RuntimeError("reconstructed target or leakage-mask shape is invalid")
    if task.leakage_mask.dtype != np.bool_ or int(task.leakage_mask.sum()) != int(row["n_leak"]):
        raise RuntimeError("reconstructed leakage mask differs from manifest")
    all_idx = np.concatenate((task.train_idx, task.val_idx, task.test_idx))
    if len(all_idx) != n or len(np.unique(all_idx)) != n:
        raise RuntimeError("split indices do not form a complete non-overlapping partition")
    if hashlib.sha256(task.test_idx.tobytes()).hexdigest() != str(row["split_hash"]):
        raise RuntimeError("split hash differs from manifest")
    if injected_task_sha256(task) != str(row["task_hash"]):
        raise RuntimeError("reconstructed task hash differs from manifest")
    if task.leakage_mask[: int(row["n_original"])].any():
        raise RuntimeError("strict mask removes an original feature")
    return task, bundle_path


def _verify_views(task, row, bundle_path, strict_hash, full_hash, mask_hash, *, verify_bundle):
    if verify_bundle and file_sha256(bundle_path) != str(row["bundle_sha256"]).lower():
        raise RuntimeError("bundle changed before model fit")
    if injected_task_sha256(task) != str(row["task_hash"]):
        raise RuntimeError("reconstructed task changed during model execution")
    if _array_sha256(task.X[:, ~task.leakage_mask]) != strict_hash:
        raise RuntimeError("strict view changed during model execution")
    if _array_sha256(task.X) != full_hash:
        raise RuntimeError("full view changed during model execution")
    if _array_sha256(task.leakage_mask) != mask_hash:
        raise RuntimeError("leakage mask changed during model execution")


def _fit_model(model, task, X, seed, tabm_kwargs):
    train, val, test = task.train_idx, task.val_idx, task.test_idx
    if model == "tabm":
        from src.leakbench.models.official_tabm import fit_predict_official_tabm

        return fit_predict_official_tabm(
            X[train], task.y[train], X[val], task.y[val], X[test],
            seed=seed, **tabm_kwargs,
        )
    from src.leakbench.models.core_models import fit_predict_core_model

    return fit_predict_core_model(
        model, X[train], task.y[train], X[val], task.y[val], X[test], seed
    )


def _parse_selection(value, universe):
    if value in (None, "all"):
        return list(universe)
    selected = [type(universe[0])(item.strip()) for item in value.split(",")]
    unknown = set(selected) - set(universe)
    if unknown:
        raise ValueError(f"unknown selection: {sorted(unknown)}")
    return selected


def _empty_row():
    row = {field: "" for field in RESULT_FIELDS}
    for field in (
        "strict_auc", "full_auc", "paired_harm", "strict_runtime_sec",
        "full_runtime_sec", "strict_best_epoch", "full_best_epoch",
    ):
        row[field] = np.nan
    row["integrity_verified"] = False
    return row


def _append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="raise")
        if handle.tell() == 0:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--freeze-manifest", default="protocols/structured_prior_v1/freeze_manifest_v1.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mechanisms", default="all")
    parser.add_argument("--strengths", default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("model execution is locked; pass --allow-run after protocol review")

    config_path = (ROOT / args.config).resolve()
    manifest_path = (ROOT / args.task_manifest).resolve()
    freeze_path = (ROOT / args.freeze_manifest).resolve()
    config, freeze, manifest, manifest_hash, summary_path = _load_contract(
        config_path, manifest_path, freeze_path
    )
    protocol = config["protocol"]
    datasets = _parse_selection(args.datasets, [int(v) for v in protocol["dataset_indices"]])
    mechanisms = _parse_selection(args.mechanisms, list(protocol["mechanisms"]))
    strengths = _parse_selection(args.strengths, list(protocol["strengths"]))
    models = _parse_selection(args.models, list(protocol["core_models"]))
    seeds = _parse_selection(args.seeds, [int(v) for v in protocol["seeds"]])
    selected = manifest.loc[
        manifest["dataset_index"].astype(int).isin(datasets)
        & manifest["mechanism"].astype(str).isin(mechanisms)
        & manifest["strength"].astype(str).isin(strengths)
        & manifest["seed"].astype(int).isin(seeds)
    ].copy()
    if selected.empty:
        raise RuntimeError("task selection is empty")

    output = (
        (ROOT / args.output).resolve()
        if args.output
        else (ROOT / protocol["result_output"]).resolve()
    )
    completed = set()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"{output} exists; pass --resume or choose another output")
        existing = pd.read_csv(output)
        completed = set(existing["run_id"].astype(str))

    config_hash = file_sha256(config_path)
    freeze_hash = file_sha256(freeze_path)
    code_paths = (
        Path(__file__),
        ROOT / "src/leakbench/structured_prior_protocol.py",
        ROOT / "src/leakbench/models/core_models.py",
        ROOT / "src/leakbench/models/official_tabm.py",
    )
    code_hash = hashlib.sha256(
        "".join(file_sha256(path) for path in code_paths).encode()
    ).hexdigest()
    tabm_kwargs = dict(config["tabm"])
    tabm_kwargs.pop("required_version")
    if args.device:
        tabm_kwargs["device"] = args.device
    adapter_hash = hashlib.sha256(
        json.dumps(tabm_kwargs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    total = len(selected) * len(models)
    attempted = 0
    started = time.time()
    for _, source in selected.iterrows():
        for model in models:
            run_key = "|".join((
                str(source["task_variant_id"]), model, str(source["task_hash"]),
                manifest_hash, config_hash, freeze_hash, code_hash, adapter_hash,
            ))
            run_id = hashlib.sha256(run_key.encode()).hexdigest()[:20]
            if run_id in completed:
                attempted += 1
                continue
            row = _empty_row()
            row.update({
                "run_id": run_id,
                "task_variant_id": source["task_variant_id"],
                "protocol_version": protocol["version"],
                "study_namespace": protocol["study_namespace"],
                "dataset_id": source["dataset_id"],
                "dataset_index": int(source["dataset_index"]),
                "dataset_namespace": source["dataset_namespace"],
                "dataset_seed": int(source["dataset_seed"]),
                "archetype": source["archetype"],
                "mechanism": source["mechanism"],
                "mechanism_family": source["mechanism_family"],
                "strength": source["strength"],
                "strength_value": float(source["strength_value"]),
                "model": model,
                "seed": int(source["seed"]),
                "status": "FAILURE",
                "failure_reason": "",
                "n_samples": int(source["n_samples"]),
                "n_original": int(source["n_original"]),
                "n_injected": int(source["n_injected"]),
                "n_leak": int(source["n_leak"]),
                "diagnostic_ap": float(source["diagnostic_ap"]),
                "diagnostic_normalized_ap": float(source["diagnostic_normalized_ap"]),
                "top5_recall": float(source["top5_recall"]),
                "split_hash": source["split_hash"],
                "base_task_sha256": source["base_task_sha256"],
                "task_hash": source["task_hash"],
                "task_manifest_sha256": manifest_hash,
                "bundle_path": source["bundle_path"],
                "bundle_sha256": source["bundle_sha256"],
                "config_hash": config_hash,
                "freeze_manifest_sha256": freeze_hash,
                "code_hash": code_hash,
                "adapter_hash": adapter_hash,
            })
            try:
                task, bundle_path = _load_verified_task(source)
                strict_X = task.X[:, ~task.leakage_mask]
                strict_hash = _array_sha256(strict_X)
                full_hash = _array_sha256(task.X)
                mask_hash = _array_sha256(task.leakage_mask)
                _verify_views(
                    task, source, bundle_path, strict_hash, full_hash, mask_hash,
                    verify_bundle=True,
                )
                row.update({
                    "strict_view_hash": strict_hash,
                    "full_view_hash": full_hash,
                    "leakage_mask_hash": mask_hash,
                    "integrity_verified": True,
                })
                strict_output = _fit_model(
                    model, task, strict_X, int(source["seed"]), tabm_kwargs
                )
                _verify_views(
                    task, source, bundle_path, strict_hash, full_hash, mask_hash,
                    verify_bundle=False,
                )
                full_output = _fit_model(
                    model, task, task.X, int(source["seed"]), tabm_kwargs
                )
                _verify_views(
                    task, source, bundle_path, strict_hash, full_hash, mask_hash,
                    verify_bundle=False,
                )
                y_test = task.y[task.test_idx]
                strict_auc = float(roc_auc_score(y_test, strict_output.probabilities))
                full_auc = float(roc_auc_score(y_test, full_output.probabilities))
                model_manifest = getattr(full_output, "manifest", {})
                row.update({
                    "status": "SUCCESS",
                    "strict_auc": strict_auc,
                    "full_auc": full_auc,
                    "paired_harm": full_auc - strict_auc,
                    "strict_runtime_sec": strict_output.runtime_sec,
                    "full_runtime_sec": full_output.runtime_sec,
                    "implementation": full_output.implementation,
                    "strict_best_epoch": getattr(strict_output, "best_epoch", np.nan),
                    "full_best_epoch": getattr(full_output, "best_epoch", np.nan),
                    "model_manifest_json": json.dumps(
                        model_manifest, sort_keys=True, separators=(",", ":")
                    ),
                })
            except Exception as exc:
                row["failure_reason"] = f"{type(exc).__name__}: {exc}"
            _append_row(output, row)
            attempted += 1
            elapsed = time.time() - started
            print(f"{attempted}/{total} cells, status={row['status']}, {elapsed:.1f}s", flush=True)

    results = pd.read_csv(output)
    latest = results.drop_duplicates("run_id", keep="last")
    requested_ids = set()
    for _, source in selected.iterrows():
        for model in models:
            key = "|".join((
                str(source["task_variant_id"]), model, str(source["task_hash"]),
                manifest_hash, config_hash, freeze_hash, code_hash, adapter_hash,
            ))
            requested_ids.add(hashlib.sha256(key.encode()).hexdigest()[:20])
    latest = latest.loc[latest["run_id"].astype(str).isin(requested_ids)]
    run_manifest = {
        "schema_version": 1,
        "protocol_version": protocol["version"],
        "study_namespace": protocol["study_namespace"],
        "dataset_namespace": protocol["dataset_namespace"],
        "strict_policy": STRICT_POLICY,
        "full_policy": FULL_POLICY,
        "config_path": _relative(config_path),
        "config_sha256": config_hash,
        "freeze_manifest_sha256": freeze_hash,
        "task_manifest_path": _relative(manifest_path),
        "task_manifest_sha256": manifest_hash,
        "bundle_summary_sha256": file_sha256(summary_path),
        "code_hash": code_hash,
        "adapter_hash": adapter_hash,
        "requested_cells": total,
        "success_cells": int((latest["status"] == "SUCCESS").sum()),
        "failure_cells": int((latest["status"] != "SUCCESS").sum()),
        "integrity_verified_cells": int(
            latest["integrity_verified"].astype(str).str.lower().eq("true").sum()
        ),
        "models": models,
        "datasets": datasets,
        "mechanisms": mechanisms,
        "strengths": strengths,
        "seeds": seeds,
        "result_sha256": file_sha256(output),
        "explicit_run_authorized": True,
        "frozen_files_verified": len(freeze["files"]),
    }
    manifest_output = output.with_name(f"{output.stem}_manifest.json")
    temporary = manifest_output.with_suffix(manifest_output.suffix + ".tmp")
    temporary.write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_output)
    return 0 if run_manifest["failure_cells"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
