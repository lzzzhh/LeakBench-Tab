#!/usr/bin/env python3
"""Run official TabM on corrected_v2 paired tasks.

The safe default is a disjoint pilot: one generated panel task, three mechanisms,
one strength, and one seed.  Confirmatory execution requires the explicit
``--allow-confirmatory`` gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.leakbench.datasets import build_panel_task  # noqa: E402
from src.leakbench.mechanisms import (  # noqa: E402
    LeakBenchInjector,
    MechanismConfig,
    MechanismID,
)
from src.leakbench.models.official_tabm import fit_predict_official_tabm  # noqa: E402


MECHANISM_IDS = {
    "M01": MechanismID.DIRECT_COPY,
    "M02": MechanismID.NOISY_PROXY,
    "M03": MechanismID.NONLINEAR,
    "M04": MechanismID.POST_OUTCOME,
    "M05": MechanismID.TEMPORAL_LEAK,
    "M06": MechanismID.REDUNDANT_CLUSTER,
    "M07": MechanismID.SPARSE_SUBGROUP,
    "M08": MechanismID.ENTITY_LEAK,
    "M09": MechanismID.SOURCE_LEAK,
    "M10": MechanismID.MIXED,
    "M11": MechanismID.GRAPH_MEDIATED,
}


RESULT_FIELDS = [
    "run_id", "dataset_id", "dataset_index", "dataset_namespace", "dataset_seed",
    "archetype", "n_samples", "n_original", "n_injected", "mechanism", "strength",
    "strength_value", "model", "seed", "status", "failure_reason", "diagnostic_ap",
    "diagnostic_normalized_ap", "top5_recall", "n_leak", "clean_auc", "full_auc",
    "paired_harm", "clean_runtime_sec", "full_runtime_sec", "implementation",
    "clean_best_epoch", "full_best_epoch", "split_hash", "task_hash", "config_hash",
    "code_hash", "model_manifest_json",
]


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


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_mechanism_config(mechanism, strength_name, config, seed):
    strength_names = list(config["protocol"]["strengths"])
    level = strength_names.index(strength_name)
    strength = float(config["protocol"]["strength_values"][strength_name])
    parameters = config["mechanism_parameters"].get(mechanism, {})
    kwargs = {
        "mechanism": MECHANISM_IDS[mechanism],
        "strength": strength,
        "noise_std": float(parameters.get("noise_std", 0.05)),
        "seed": seed,
    }
    if mechanism == "M05":
        kwargs["time_offset"] = float(parameters["time_offsets"][level])
    elif mechanism == "M06":
        kwargs["redundancy"] = int(parameters["redundancies"][level])
    elif mechanism == "M07":
        kwargs["subgroup_prevalence"] = float(parameters["prevalences"][level])
    elif mechanism == "M08":
        kwargs["prior_strength"] = float(parameters["prior_strength"])
    elif mechanism == "M09":
        kwargs["n_sources"] = int(parameters["n_sources"])
        kwargs["min_group_count"] = int(parameters["min_group_count"])
    elif mechanism == "M11":
        kwargs["n_leakage_features"] = int(parameters["components"][level])
    return MechanismConfig(**kwargs)


def diagnostic_metrics(task):
    scores = mutual_info_classif(
        task.X[task.train_idx], task.y[task.train_idx], random_state=42
    )
    scores = np.nan_to_num(scores, nan=0.0)
    truth = task.leakage_mask.astype(int)
    prevalence = float(truth.mean())
    ap = float(average_precision_score(truth, scores))
    normalized_ap = (ap - prevalence) / max(1e-12, 1.0 - prevalence)
    top = np.argsort(scores, kind="stable")[::-1][: min(5, len(scores))]
    recall = float(truth[top].sum() / max(1, truth.sum()))
    return ap, normalized_ap, recall


def save_predictions(directory, run_id, task, y_test, clean_probability, full_probability):
    directory.mkdir(parents=True, exist_ok=True)
    test = task.test_idx
    np.savez_compressed(
        directory / f"{run_id}.npz",
        row_id=test,
        y=y_test,
        clean_probability=clean_probability,
        full_probability=full_probability,
        entity_id=task.entity_ids[test],
        source_id=task.source_ids[test],
    )


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


def empty_result_row():
    row = {field: "" for field in RESULT_FIELDS}
    for field in (
        "diagnostic_ap", "diagnostic_normalized_ap", "top5_recall", "clean_auc",
        "full_auc", "paired_harm", "clean_runtime_sec", "full_runtime_sec",
        "clean_best_epoch", "full_best_epoch",
    ):
        row[field] = np.nan
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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--output", default="results/corrected_v2/pilot/tabm_official_cells.csv")
    parser.add_argument("--datasets", default="0:1")
    parser.add_argument("--mechanisms", default="M01,M08,M09")
    parser.add_argument("--strengths", default="S3")
    parser.add_argument("--seeds", default="13")
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

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    namespace = args.namespace
    if namespace == protocol["dataset_namespace"] and not args.allow_confirmatory:
        raise RuntimeError(
            "Confirmatory TabM execution is locked. Review the pilot, then pass "
            "--allow-confirmatory explicitly."
        )

    dataset_ids = parse_selection(args.datasets, list(range(int(protocol["dataset_count"]))))
    mechanisms = parse_selection(args.mechanisms, protocol["mechanisms"])
    strengths = parse_selection(args.strengths, protocol["strengths"])
    seeds = parse_selection(args.seeds, [int(seed) for seed in protocol["seeds"]])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_dir = output.parent / f"{output.stem}_predictions"

    config_hash = file_sha256(config_path)
    code_paths = (
        ROOT / "src/leakbench/datasets.py",
        ROOT / "src/leakbench/mechanisms/__init__.py",
        ROOT / "src/leakbench/models/official_tabm.py",
        Path(__file__),
    )
    code_hash = hashlib.sha256(
        "".join(file_sha256(path) for path in code_paths).encode()
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
    clean_cache = {}
    total = len(dataset_ids) * len(mechanisms) * len(strengths) * len(seeds)
    attempted = 0
    started = time.time()

    for dataset_index in dataset_ids:
        base = build_panel_task(dataset_index, namespace=namespace)
        for mechanism in mechanisms:
            for strength_name in strengths:
                for seed in seeds:
                    run_key = (
                        f"{namespace}|{base.dataset_id}|{mechanism}|{strength_name}|tabm|"
                        f"{seed}|{config_hash}|{code_hash}|{adapter_hash}"
                    )
                    run_id = hashlib.sha256(run_key.encode()).hexdigest()[:20]
                    if run_id in completed:
                        attempted += 1
                        continue

                    row = empty_result_row()
                    row.update({
                        "run_id": run_id,
                        "dataset_id": base.dataset_id,
                        "dataset_index": dataset_index,
                        "dataset_namespace": namespace,
                        "dataset_seed": base.generator_seed,
                        "archetype": base.archetype,
                        "n_samples": len(base.y),
                        "mechanism": mechanism,
                        "strength": strength_name,
                        "model": "tabm",
                        "seed": seed,
                        "status": "FAILURE",
                        "failure_reason": "",
                        "config_hash": config_hash,
                        "code_hash": code_hash,
                    })
                    try:
                        mechanism_config = build_mechanism_config(
                            mechanism, strength_name, config, seed
                        )
                        task = LeakBenchInjector(seed=seed).inject(
                            base.X,
                            base.y,
                            [mechanism_config],
                            feature_names=list(base.feature_names),
                            timestamps=base.timestamps,
                            entity_ids=base.entity_ids,
                            split_type="time",
                        )
                        if not (
                            np.array_equal(task.train_idx, base.train_idx)
                            and np.array_equal(task.val_idx, base.val_idx)
                            and np.array_equal(task.test_idx, base.test_idx)
                        ):
                            raise RuntimeError("injection changed the frozen chronological split")

                        diagnostic_ap, diagnostic_nap, top5_recall = diagnostic_metrics(task)
                        row.update({
                            "n_original": task.n_original,
                            "n_injected": task.n_injected,
                            "strength_value": mechanism_config.strength,
                            "diagnostic_ap": diagnostic_ap,
                            "diagnostic_normalized_ap": diagnostic_nap,
                            "top5_recall": top5_recall,
                            "n_leak": int(task.leakage_mask.sum()),
                            "split_hash": hashlib.sha256(task.test_idx.tobytes()).hexdigest(),
                            "task_hash": task_sha256(task),
                        })

                        cache_key = (dataset_index, seed)
                        if cache_key not in clean_cache:
                            try:
                                clean_cache[cache_key] = fit_predict_official_tabm(
                                    base.X[base.train_idx],
                                    base.y[base.train_idx],
                                    base.X[base.val_idx],
                                    base.y[base.val_idx],
                                    base.X[base.test_idx],
                                    seed=seed,
                                    **adapter_kwargs,
                                )
                            except Exception as exc:
                                clean_cache[cache_key] = exc
                        clean_output = clean_cache[cache_key]
                        if isinstance(clean_output, Exception):
                            raise RuntimeError(f"clean TabM fit failed: {clean_output}") from clean_output

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
                                full_output.manifest, sort_keys=True, separators=(",", ":")
                            ),
                        })
                        if mechanism in protocol["predictions_retained_for"]:
                            save_predictions(
                                prediction_dir,
                                run_id,
                                task,
                                y_test,
                                clean_output.probabilities,
                                full_output.probabilities,
                            )
                    except Exception as exc:
                        row["failure_reason"] = f"{type(exc).__name__}: {exc}"

                    append_result(output, row)
                    attempted += 1
                    elapsed = time.time() - started
                    print(
                        f"{attempted}/{total} cells, status={row['status']}, "
                        f"{elapsed:.1f}s elapsed",
                        flush=True,
                    )

    result_frame = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=RESULT_FIELDS)
    latest_results = result_frame.drop_duplicates("run_id", keep="last")
    manifest = {
        "schema_version": 1,
        "protocol_version": protocol["version"],
        "model_identity": "tabm.TabM",
        "required_tabm_version": "0.0.3",
        "dataset_namespace": namespace,
        "confirmatory_authorized": bool(args.allow_confirmatory),
        "config_path": str(config_path.relative_to(ROOT)),
        "config_hash": config_hash,
        "code_hash": code_hash,
        "code_files": [str(path.relative_to(ROOT)) for path in code_paths],
        "output": str(output.relative_to(ROOT)),
        "requested_cells": total,
        "success_cells": int((latest_results["status"] == "SUCCESS").sum()),
        "failure_cells": int((latest_results["status"] != "SUCCESS").sum()),
        "datasets": dataset_ids,
        "mechanisms": mechanisms,
        "strengths": strengths,
        "seeds": seeds,
        "adapter_kwargs": adapter_kwargs,
        "adapter_hash": adapter_hash,
        "result_sha256": file_sha256(output) if output.exists() else "",
    }
    write_json_atomic(output.parent / f"{output.stem}_manifest.json", manifest)
    return 0 if manifest["failure_cells"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
