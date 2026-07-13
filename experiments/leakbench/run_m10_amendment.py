#!/usr/bin/env python3
"""Run the M10 strict-view protocol amendment from immutable task bundles.

The amendment changes only the strict comparator for M10.  The strict matrix is
defined literally as ``task.X[:, ~task.leakage_mask]`` so the injected,
legitimate duplicate remains available and only the contamination column is
removed.  The permissive matrix is the unmodified ``task.X``.

No dataset or mechanism generator is imported here.  Bundle, task, split,
configuration, view, runner, and model-adapter identities are recorded for
every attempted cell.  Confirmatory execution is fail-closed unless explicitly
authorized with ``--allow-confirmatory``.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
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

STRICT_POLICY = "task.X[:, ~task.leakage_mask]"
FULL_POLICY = "task.X"
AMENDMENT_VERSION = "m10_strict_mask_v1"

RESULT_FIELDS = [
    "run_id", "dataset_id", "dataset_index", "dataset_namespace", "dataset_seed",
    "archetype", "n_samples", "n_original", "n_injected", "mechanism", "strength",
    "strength_value", "model", "seed", "status", "failure_reason", "diagnostic_ap",
    "diagnostic_normalized_ap", "top5_recall", "n_leak", "clean_auc", "strict_auc",
    "full_auc", "paired_harm", "clean_runtime_sec", "strict_runtime_sec",
    "full_runtime_sec", "implementation", "clean_best_epoch", "strict_best_epoch",
    "full_best_epoch", "split_hash", "task_hash", "source_task_hash",
    "task_manifest_sha256", "bundle_summary_sha256", "bundle_path", "bundle_sha256",
    "integrity_verified", "amendment_version", "strict_policy", "full_policy",
    "strict_feature_count", "legitimate_injected_count", "contamination_removed_count",
    "strict_view_hash", "full_view_hash", "leakage_mask_hash", "config_hash",
    "amendment_config_hash", "runner_sha256", "model_adapter_sha256", "code_hash",
    "model_manifest_json",
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


@dataclass(frozen=True)
class StrictContract:
    X: np.ndarray
    strict_view_hash: str
    full_view_hash: str
    leakage_mask_hash: str
    legitimate_injected_count: int
    contamination_removed_count: int


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


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
        values = np.ascontiguousarray(values)
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
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


def _resolve_under_root(raw_path, root=None):
    root = ROOT if root is None else Path(root)
    path = Path(raw_path)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Path escapes repository root: {raw_path}") from exc
    return resolved


def load_amendment_config(config_path):
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    amendment = payload.get("amendment", {})
    if amendment.get("version") != AMENDMENT_VERSION:
        raise RuntimeError(f"Expected amendment version {AMENDMENT_VERSION}")
    if amendment.get("mechanism") != "M10":
        raise RuntimeError("The amendment configuration must be restricted to M10")
    if amendment.get("strict_policy") != STRICT_POLICY:
        raise RuntimeError("The configured strict policy is not the literal mask-derived view")
    if amendment.get("full_policy") != FULL_POLICY:
        raise RuntimeError("The configured permissive policy is not task.X")

    base_config_path = _resolve_under_root(amendment["base_config_path"])
    base_config_hash = file_sha256(base_config_path)
    if base_config_hash != str(amendment["base_config_sha256"]):
        raise RuntimeError("Base corrected_v2 config SHA256 differs from the amendment binding")
    base_protocol = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))["protocol"]
    if "M10" not in base_protocol["mechanisms"]:
        raise RuntimeError("The bound base protocol does not contain M10")
    return payload, amendment, base_protocol, base_config_path, base_config_hash


def load_bundle_contract(manifest_path, base_config_hash):
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
    if summary.get("config_sha256") != base_config_hash:
        raise RuntimeError("Bundle config SHA256 does not match the bound base config")
    if int(summary.get("task_count", -1)) != len(manifest):
        raise RuntimeError("Bundle summary task_count does not match task manifest")

    namespaces = set(manifest["dataset_namespace"].astype(str))
    if namespaces != {str(summary.get("dataset_namespace"))}:
        raise RuntimeError("Bundle namespace is inconsistent across summary and manifest")
    identity = ["dataset_index", "mechanism", "strength", "seed"]
    if manifest.duplicated(identity).any():
        raise RuntimeError(f"Task manifest has duplicate identities: {identity}")
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
    if task.leakage_mask.dtype != np.bool_:
        raise RuntimeError("Leakage mask must be boolean")
    if int(task.leakage_mask.sum()) != n_leak:
        raise RuntimeError("Leakage mask count does not match manifest")
    if task.entity_ids.shape != (n_samples,) or task.source_ids.shape != (n_samples,):
        raise RuntimeError("Entity/source arrays do not match sample count")

    split_arrays = (task.train_idx, task.val_idx, task.test_idx)
    if any(index.ndim != 1 for index in split_arrays):
        raise RuntimeError("Split indices must be one-dimensional")
    all_idx = np.concatenate(split_arrays)
    if len(np.unique(all_idx)) != len(all_idx):
        raise RuntimeError("Split indices overlap")
    if len(all_idx) != n_samples or all_idx.min() < 0 or all_idx.max() >= n_samples:
        raise RuntimeError("Split indices do not form a full valid partition")


def load_verified_task(row, root=ROOT):
    bundle_path = _resolve_under_root(str(row["bundle_path"]), root)
    expected_bundle_hash = str(row["bundle_sha256"]).lower()
    actual_bundle_hash = file_sha256(bundle_path)
    if actual_bundle_hash != expected_bundle_hash:
        raise RuntimeError(
            f"Bundle SHA256 mismatch for {bundle_path}: expected {expected_bundle_hash}, "
            f"got {actual_bundle_hash}"
        )

    key = str(row["bundle_key"])
    required_keys = {
        "base_X", "y", "train_idx", "val_idx", "test_idx", f"block__{key}",
        f"leak_mask__{key}", f"entity_ids__{key}", f"source_ids__{key}",
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
            f"Reconstructed task SHA256 mismatch for {key}: expected {row['task_hash']}, "
            f"got {actual_task_hash}"
        )
    return task, bundle_path


def derive_strict_contract(task, row, amendment):
    if str(row["mechanism"]) != "M10":
        raise RuntimeError("M10 amendment received a non-M10 task")
    n_original = int(row["n_original"])
    expected_injected = int(amendment["required_m10_injected_features"])
    expected_contamination = int(amendment["required_contamination_features"])
    expected_legitimate = int(amendment["required_legitimate_injected_features"])
    if int(row["n_injected"]) != expected_injected:
        raise RuntimeError("M10 injected-feature count differs from the amendment contract")
    if task.leakage_mask[:n_original].any():
        raise RuntimeError("M10 mask unexpectedly marks an original feature as contamination")

    injected_mask = task.leakage_mask[n_original:]
    contamination_count = int(injected_mask.sum())
    legitimate_count = int((~injected_mask).sum())
    if contamination_count != expected_contamination:
        raise RuntimeError("M10 must remove exactly one injected contamination feature")
    if legitimate_count != expected_legitimate:
        raise RuntimeError("M10 must retain exactly one injected legitimate feature")

    legitimate_indices = np.flatnonzero(~task.leakage_mask)
    injected_legitimate_indices = legitimate_indices[legitimate_indices >= n_original]
    if len(injected_legitimate_indices) != 1:
        raise RuntimeError("M10 injected legitimate feature is not uniquely identifiable")
    legitimate_column = int(injected_legitimate_indices[0])
    if not np.array_equal(task.X[:, legitimate_column], task.base_X[:, 0]):
        raise RuntimeError("M10 legitimate injected feature is not the frozen clean_0 duplicate")

    # This literal operation is the scientific amendment.  Do not replace it
    # with task.base_X: the legitimate injected duplicate must remain.
    strict_X = task.X[:, ~task.leakage_mask]
    if strict_X.shape[1] != task.X.shape[1] - contamination_count:
        raise RuntimeError("Strict M10 view removed an unexpected number of columns")
    if not np.array_equal(strict_X[:, -1], task.X[:, legitimate_column]):
        raise RuntimeError("Strict M10 view failed to retain the legitimate duplicate")
    return StrictContract(
        X=strict_X,
        strict_view_hash=array_sha256(strict_X),
        full_view_hash=array_sha256(task.X),
        leakage_mask_hash=array_sha256(task.leakage_mask),
        legitimate_injected_count=legitimate_count,
        contamination_removed_count=contamination_count,
    )


def verify_before_fit(task, row, bundle_path, contract):
    if file_sha256(bundle_path) != str(row["bundle_sha256"]).lower():
        raise RuntimeError("Bundle SHA256 changed before model fit")
    if task_sha256(task) != str(row["task_hash"]):
        raise RuntimeError("Reconstructed task SHA256 changed before model fit")
    if array_sha256(task.X[:, ~task.leakage_mask]) != contract.strict_view_hash:
        raise RuntimeError("Strict M10 view changed before model fit")
    if array_sha256(task.X) != contract.full_view_hash:
        raise RuntimeError("Full M10 view changed before model fit")


def _adapter_path(model):
    if model == "tabm":
        return ROOT / "src/leakbench/models/official_tabm.py"
    return ROOT / "src/leakbench/models/core_models.py"


def _fit_model(model, X_train, y_train, X_val, y_val, X_test, seed, tabm_kwargs):
    if model == "tabm":
        # Lazy by design: local CPU runs do not require torch or the tabm wheel.
        from src.leakbench.models.official_tabm import fit_predict_official_tabm

        return fit_predict_official_tabm(
            X_train, y_train, X_val, y_val, X_test, seed=seed, **tabm_kwargs
        )
    from src.leakbench.models.core_models import fit_predict_core_model

    return fit_predict_core_model(
        model, X_train, y_train, X_val, y_val, X_test, seed
    )


def _empty_result_row():
    row = {field: "" for field in RESULT_FIELDS}
    for field in (
        "diagnostic_ap", "diagnostic_normalized_ap", "top5_recall", "clean_auc",
        "strict_auc", "full_auc", "paired_harm", "clean_runtime_sec",
        "strict_runtime_sec", "full_runtime_sec", "clean_best_epoch",
        "strict_best_epoch", "full_best_epoch",
    ):
        row[field] = np.nan
    row["integrity_verified"] = False
    return row


def _append_result(path, row):
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


def _write_frame_atomic(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _prepare_resume(output, resume, retry_failures, expected_hashes):
    if not output.exists():
        return set()
    if not resume:
        raise FileExistsError(f"{output} exists; pass --resume or choose a new path")
    existing = pd.read_csv(output)
    missing = set(RESULT_FIELDS) - set(existing.columns)
    if missing:
        raise RuntimeError(f"Existing output is missing columns: {sorted(missing)}")
    if existing["run_id"].duplicated().any():
        raise RuntimeError("Existing M10 amendment output has duplicate run_id values")
    for field, expected in expected_hashes.items():
        actual = set(existing[field].dropna().astype(str))
        if actual and not actual.issubset(expected if isinstance(expected, set) else {expected}):
            raise RuntimeError(f"Existing output has incompatible {field}: {sorted(actual)}")

    if not retry_failures:
        return set(existing["run_id"].astype(str))

    failures = existing.loc[existing["status"] != "SUCCESS"].copy()
    if failures.empty:
        return set(existing["run_id"].astype(str))
    archive = output.with_name(f"{output.stem}_failure_attempts.jsonl")
    with archive.open("a", encoding="utf-8") as handle:
        for record in failures.to_dict(orient="records"):
            handle.write(json.dumps({
                "archived_at_utc": datetime.now(timezone.utc).isoformat(),
                "row": record,
            }, default=str, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    retained = existing.loc[existing["status"] == "SUCCESS", RESULT_FIELDS]
    _write_frame_atomic(output, retained)
    return set(retained["run_id"].astype(str))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/m10_amendment_v1.yaml")
    parser.add_argument(
        "--task-manifest",
        default="results/corrected_v2/m10_amendment_pilot_tasks/task_manifest.csv",
    )
    parser.add_argument(
        "--output", default="results/corrected_v2/m10_amendment_pilot/cpu_cells.csv"
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--strengths", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--models", default="lr,rf,catboost,lightgbm")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--allow-confirmatory", action="store_true")
    args = parser.parse_args(argv)

    config_path = _resolve_under_root(args.config)
    config_payload, amendment, base_protocol, _, base_config_hash = load_amendment_config(
        config_path
    )
    amendment_config_hash = file_sha256(config_path)
    manifest_path = _resolve_under_root(args.task_manifest)
    manifest, manifest_hash, summary_path, summary = load_bundle_contract(
        manifest_path, base_config_hash
    )
    manifest_namespace = str(summary["dataset_namespace"])
    namespace = args.namespace or manifest_namespace
    if namespace != manifest_namespace:
        raise RuntimeError(
            f"Requested namespace {namespace!r} does not match bundle namespace "
            f"{manifest_namespace!r}"
        )
    if namespace == amendment["confirmatory_namespace"]:
        if not args.allow_confirmatory:
            raise RuntimeError(
                "Confirmatory M10 amendment execution is locked; pass "
                "--allow-confirmatory only after the amendment freeze is reviewed"
            )
        expected_manifest = str(amendment["confirmatory_task_manifest_sha256"])
        if manifest_hash != expected_manifest:
            raise RuntimeError("Confirmatory task manifest differs from the amendment binding")
        if file_sha256(summary_path) != str(amendment["confirmatory_bundle_summary_sha256"]):
            raise RuntimeError("Confirmatory bundle summary differs from the amendment binding")

    m10 = manifest.loc[manifest["mechanism"].astype(str) == "M10"].copy()
    if m10.empty:
        raise RuntimeError("Task manifest contains no M10 tasks")
    dataset_universe = sorted(m10["dataset_index"].astype(int).unique().tolist())
    strength_universe = [
        value for value in amendment["strengths"] if value in set(m10["strength"].astype(str))
    ]
    seed_universe = [
        int(value) for value in base_protocol["seeds"]
        if int(value) in set(m10["seed"].astype(int))
    ]
    model_universe = list(amendment["cpu_models"]) + [str(amendment["official_model"])]
    datasets = parse_selection(args.datasets, dataset_universe)
    strengths = parse_selection(args.strengths, strength_universe)
    seeds = parse_selection(args.seeds, seed_universe)
    models = parse_selection(args.models, model_universe)
    selected = m10.loc[
        m10["dataset_index"].astype(int).isin(datasets)
        & m10["strength"].astype(str).isin(strengths)
        & m10["seed"].astype(int).isin(seeds)
    ].copy()
    if selected.empty:
        raise RuntimeError("M10 task selection is empty")

    runner_hash = file_sha256(Path(__file__))
    adapter_hashes = {model: file_sha256(_adapter_path(model)) for model in models}
    code_hashes = {
        model: hashlib.sha256((adapter_hashes[model] + runner_hash).encode()).hexdigest()
        for model in models
    }
    tabm_config = dict(config_payload["tabm"])
    tabm_config.pop("required_version")
    if args.device is not None:
        tabm_config["device"] = args.device
    tabm_kwargs = {
        "device": str(tabm_config["device"]),
        "k": int(tabm_config["k"]),
        "learning_rate": float(tabm_config["learning_rate"]),
        "weight_decay": float(tabm_config["weight_decay"]),
        "max_epochs": int(tabm_config["max_epochs"]),
        "batch_size": int(tabm_config["batch_size"]),
        "inference_batch_size": int(tabm_config["inference_batch_size"]),
        "patience": int(tabm_config["patience"]),
        "min_delta": float(tabm_config["min_delta"]),
    }

    output = _resolve_under_root(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _prepare_resume(output, args.resume, args.retry_failures, {
        "config_hash": base_config_hash,
        "amendment_config_hash": amendment_config_hash,
        "task_manifest_sha256": manifest_hash,
        "runner_sha256": runner_hash,
        "model_adapter_sha256": set(adapter_hashes.values()),
        "code_hash": set(code_hashes.values()),
        "amendment_version": AMENDMENT_VERSION,
        "strict_policy": STRICT_POLICY,
    })

    clean_cache = {}
    expected_run_ids = set()
    total = len(selected) * len(models)
    attempted = 0
    started = time.time()
    for _, manifest_row in selected.iterrows():
        seed = int(manifest_row["seed"])
        for model in models:
            run_key = "|".join((
                AMENDMENT_VERSION, namespace, str(manifest_row["dataset_id"]), "M10",
                str(manifest_row["strength"]), model, str(seed),
                str(manifest_row["task_hash"]), manifest_hash, base_config_hash,
                amendment_config_hash, code_hashes[model], STRICT_POLICY, FULL_POLICY,
            ))
            run_id = hashlib.sha256(run_key.encode()).hexdigest()[:20]
            expected_run_ids.add(run_id)
            if run_id in completed:
                attempted += 1
                continue

            row = _empty_result_row()
            row.update({
                "run_id": run_id,
                "dataset_id": manifest_row["dataset_id"],
                "dataset_index": int(manifest_row["dataset_index"]),
                "dataset_namespace": namespace,
                "dataset_seed": int(manifest_row["dataset_seed"]),
                "archetype": manifest_row["archetype"],
                "n_samples": int(manifest_row["n_samples"]),
                "n_original": int(manifest_row["n_original"]),
                "n_injected": int(manifest_row["n_injected"]),
                "mechanism": "M10",
                "strength": manifest_row["strength"],
                "strength_value": float(manifest_row["strength_value"]),
                "model": model,
                "seed": seed,
                "status": "FAILURE",
                "failure_reason": "",
                "diagnostic_ap": float(manifest_row["diagnostic_ap"]),
                "diagnostic_normalized_ap": float(
                    manifest_row["diagnostic_normalized_ap"]
                ),
                "top5_recall": float(manifest_row["top5_recall"]),
                "n_leak": int(manifest_row["n_leak"]),
                "split_hash": manifest_row["split_hash"],
                "task_hash": manifest_row["task_hash"],
                "source_task_hash": manifest_row["task_hash"],
                "task_manifest_sha256": manifest_hash,
                "bundle_summary_sha256": file_sha256(summary_path),
                "bundle_path": manifest_row["bundle_path"],
                "bundle_sha256": manifest_row["bundle_sha256"],
                "amendment_version": AMENDMENT_VERSION,
                "strict_policy": STRICT_POLICY,
                "full_policy": FULL_POLICY,
                "config_hash": base_config_hash,
                "amendment_config_hash": amendment_config_hash,
                "runner_sha256": runner_hash,
                "model_adapter_sha256": adapter_hashes[model],
                "code_hash": code_hashes[model],
            })
            try:
                task, bundle_path = load_verified_task(manifest_row, ROOT)
                contract = derive_strict_contract(task, manifest_row, amendment)
                row.update({
                    "strict_feature_count": int(contract.X.shape[1]),
                    "legitimate_injected_count": contract.legitimate_injected_count,
                    "contamination_removed_count": contract.contamination_removed_count,
                    "strict_view_hash": contract.strict_view_hash,
                    "full_view_hash": contract.full_view_hash,
                    "leakage_mask_hash": contract.leakage_mask_hash,
                })
                verify_before_fit(task, manifest_row, bundle_path, contract)
                row["integrity_verified"] = True

                cache_key = (
                    str(manifest_row["dataset_id"]), model, seed,
                    contract.strict_view_hash, str(manifest_row["split_hash"]),
                    code_hashes[model],
                )
                if cache_key not in clean_cache:
                    try:
                        clean_cache[cache_key] = _fit_model(
                            model,
                            contract.X[task.train_idx],
                            task.y[task.train_idx],
                            contract.X[task.val_idx],
                            task.y[task.val_idx],
                            contract.X[task.test_idx],
                            seed,
                            tabm_kwargs,
                        )
                    except Exception as exc:
                        clean_cache[cache_key] = exc
                strict_output = clean_cache[cache_key]
                if isinstance(strict_output, Exception):
                    raise RuntimeError(f"strict-view fit failed: {strict_output}") from strict_output

                verify_before_fit(task, manifest_row, bundle_path, contract)
                full_output = _fit_model(
                    model,
                    task.X[task.train_idx],
                    task.y[task.train_idx],
                    task.X[task.val_idx],
                    task.y[task.val_idx],
                    task.X[task.test_idx],
                    seed,
                    tabm_kwargs,
                )
                y_test = task.y[task.test_idx]
                strict_auc = float(roc_auc_score(y_test, strict_output.probabilities))
                full_auc = float(roc_auc_score(y_test, full_output.probabilities))
                model_manifest = dict(getattr(full_output, "manifest", {}))
                model_manifest.update({
                    "schema_version": 1,
                    "input_mode": "immutable_npz_bundle",
                    "amendment_version": AMENDMENT_VERSION,
                    "strict_policy": STRICT_POLICY,
                    "full_policy": FULL_POLICY,
                    "task_manifest_sha256": manifest_hash,
                    "bundle_sha256": str(manifest_row["bundle_sha256"]),
                    "task_sha256": str(manifest_row["task_hash"]),
                    "strict_view_sha256": contract.strict_view_hash,
                    "model_adapter_sha256": adapter_hashes[model],
                })
                strict_best_epoch = getattr(strict_output, "best_epoch", np.nan)
                full_best_epoch = getattr(full_output, "best_epoch", np.nan)
                row.update({
                    "status": "SUCCESS",
                    "clean_auc": strict_auc,
                    "strict_auc": strict_auc,
                    "full_auc": full_auc,
                    "paired_harm": full_auc - strict_auc,
                    "clean_runtime_sec": strict_output.runtime_sec,
                    "strict_runtime_sec": strict_output.runtime_sec,
                    "full_runtime_sec": full_output.runtime_sec,
                    "implementation": full_output.implementation,
                    "clean_best_epoch": strict_best_epoch,
                    "strict_best_epoch": strict_best_epoch,
                    "full_best_epoch": full_best_epoch,
                    "model_manifest_json": json.dumps(
                        model_manifest, sort_keys=True, separators=(",", ":")
                    ),
                })
            except Exception as exc:
                row["failure_reason"] = f"{type(exc).__name__}: {exc}"

            _append_result(output, row)
            attempted += 1
            elapsed = time.time() - started
            if attempted % 10 == 0 or attempted == total or row["status"] != "SUCCESS":
                print(
                    f"{attempted}/{total} M10 cells, status={row['status']}, "
                    f"{elapsed:.1f}s elapsed",
                    flush=True,
                )

    result_frame = pd.read_csv(output)
    if result_frame["run_id"].duplicated().any():
        raise RuntimeError("M10 amendment output contains duplicate run_id values")
    selected_results = result_frame.loc[
        result_frame["run_id"].astype(str).isin(expected_run_ids)
    ].copy()
    missing_ids = expected_run_ids - set(selected_results["run_id"].astype(str))
    if missing_ids:
        raise RuntimeError(f"M10 amendment output is missing {len(missing_ids)} requested cells")
    success_cells = int((selected_results["status"] == "SUCCESS").sum())
    failure_cells = int((selected_results["status"] != "SUCCESS").sum())
    manifest_output = {
        "schema_version": 1,
        "status": "PILOT" if namespace == amendment["pilot_namespace"] else "CONFIRMATORY",
        "amendment_version": AMENDMENT_VERSION,
        "strict_policy": STRICT_POLICY,
        "full_policy": FULL_POLICY,
        "input_mode": "immutable_npz_bundle",
        "dataset_namespace": namespace,
        "confirmatory_authorized": bool(args.allow_confirmatory),
        "base_config_sha256": base_config_hash,
        "amendment_config_path": str(config_path.relative_to(ROOT)),
        "amendment_config_sha256": amendment_config_hash,
        "task_manifest_path": str(manifest_path.relative_to(ROOT)),
        "task_manifest_sha256": manifest_hash,
        "bundle_summary_path": str(summary_path.relative_to(ROOT)),
        "bundle_summary_sha256": file_sha256(summary_path),
        "bundle_sha256": sorted(set(selected["bundle_sha256"].astype(str))),
        "runner_sha256": runner_hash,
        "model_adapter_sha256": adapter_hashes,
        "code_hash": code_hashes,
        "output": str(output.relative_to(ROOT)),
        "requested_cells": total,
        "success_cells": success_cells,
        "failure_cells": failure_cells,
        "integrity_verified_cells": int(
            selected_results["integrity_verified"].astype(str).str.lower().eq("true").sum()
        ),
        "datasets": datasets,
        "mechanism": "M10",
        "strengths": strengths,
        "models": models,
        "seeds": seeds,
        "tabm_kwargs": tabm_kwargs if "tabm" in models else {},
        "result_sha256": file_sha256(output),
    }
    manifest_output_path = output.with_name(f"{output.stem}_manifest.json")
    _write_json_atomic(manifest_output_path, manifest_output)
    return 0 if failure_cells == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
