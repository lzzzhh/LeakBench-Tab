#!/usr/bin/env python3
"""Run the corrected_v2 paired strict/permissive core matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.leakbench.datasets import build_panel_task
from src.leakbench.mechanisms import LeakBenchInjector, MechanismConfig, MechanismID
from src.leakbench.models.core_models import fit_predict_core_model


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


def parse_selection(value, universe):
    if value in (None, "all"):
        return list(universe)
    selected = []
    for part in value.split(","):
        part = part.strip()
        if ":" in part and all(str(item).lstrip("-").isdigit() for item in part.split(":", 1)):
            start, stop = (int(item) for item in part.split(":", 1))
            selected.extend(range(start, stop))
        else:
            cast = type(universe[0])
            selected.append(cast(part))
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
    train = task.train_idx
    scores = mutual_info_classif(task.X[train], task.y[train], random_state=42)
    scores = np.nan_to_num(scores, nan=0.0)
    truth = task.leakage_mask.astype(int)
    prevalence = float(truth.mean())
    ap = float(average_precision_score(truth, scores))
    normalized_ap = (ap - prevalence) / max(1e-12, 1.0 - prevalence)
    k = min(5, len(scores))
    top = np.argsort(scores, kind="stable")[::-1][:k]
    recall = float(truth[top].sum() / max(1, truth.sum()))
    return ap, normalized_ap, recall


def append_row(path, row):
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--output", default="results/corrected_v2/core_cells.csv")
    parser.add_argument("--datasets", default="all", help="all, comma list, or half-open range such as 0:3")
    parser.add_argument("--mechanisms", default="all")
    parser.add_argument("--strengths", default="all")
    parser.add_argument("--models", default="lr,rf,catboost,lightgbm")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    dataset_ids = parse_selection(args.datasets, list(range(int(protocol["dataset_count"]))))
    mechanisms = parse_selection(args.mechanisms, protocol["mechanisms"])
    strengths = parse_selection(args.strengths, protocol["strengths"])
    cpu_models = [model for model in protocol["core_models"] if model != "tabm"]
    models = parse_selection(args.models, cpu_models)
    seeds = parse_selection(args.seeds, [int(seed) for seed in protocol["seeds"]])
    namespace = args.namespace or protocol["dataset_namespace"]
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_dir = output.parent / "predictions"
    config_hash = file_sha256(config_path)
    code_hash = hashlib.sha256(
        "".join(
            file_sha256(path)
            for path in (
                ROOT / "src/leakbench/datasets.py",
                ROOT / "src/leakbench/mechanisms/__init__.py",
                ROOT / "src/leakbench/models/core_models.py",
                Path(__file__),
            )
        ).encode()
    ).hexdigest()

    completed = set()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"{output} exists; pass --resume or choose a new corrected_v2 path")
        existing = pd.read_csv(output)
        completed = set(existing.loc[existing["status"] == "SUCCESS", "run_id"].astype(str))

    clean_cache = {}
    total = len(dataset_ids) * len(mechanisms) * len(strengths) * len(models) * len(seeds)
    done = 0
    started = time.time()
    for dataset_index in dataset_ids:
        base = build_panel_task(dataset_index, namespace=namespace)
        for seed in seeds:
            for model in models:
                cache_key = (dataset_index, model, seed)
                clean_output = fit_predict_core_model(
                    model,
                    base.X[base.train_idx],
                    base.y[base.train_idx],
                    base.X[base.val_idx],
                    base.y[base.val_idx],
                    base.X[base.test_idx],
                    seed,
                )
                clean_auc = float(roc_auc_score(base.y[base.test_idx], clean_output.probabilities))
                clean_cache[cache_key] = (clean_output, clean_auc)

        for mechanism in mechanisms:
            for strength_name in strengths:
                for seed in seeds:
                    mechanism_config = build_mechanism_config(mechanism, strength_name, config, seed)
                    try:
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
                            raise RuntimeError("injection changed the frozen split")
                        diagnostic_ap, diagnostic_nap, top5_recall = diagnostic_metrics(task)
                    except Exception as exc:
                        for model in models:
                            run_key = f"{namespace}|{base.dataset_id}|{mechanism}|{strength_name}|{model}|{seed}|{config_hash}|{code_hash}"
                            run_id = hashlib.sha256(run_key.encode()).hexdigest()[:20]
                            if run_id not in completed:
                                append_row(output, {
                                    "run_id": run_id, "dataset_id": base.dataset_id, "dataset_index": dataset_index,
                                    "dataset_namespace": namespace, "archetype": base.archetype,
                                    "mechanism": mechanism, "strength": strength_name, "model": model, "seed": seed,
                                    "status": "INVALID", "failure_reason": f"{type(exc).__name__}: {exc}",
                                    "config_hash": config_hash, "code_hash": code_hash,
                                })
                            done += 1
                        continue

                    for model in models:
                        run_key = f"{namespace}|{base.dataset_id}|{mechanism}|{strength_name}|{model}|{seed}|{config_hash}|{code_hash}"
                        run_id = hashlib.sha256(run_key.encode()).hexdigest()[:20]
                        if run_id in completed:
                            done += 1
                            continue
                        row = {
                            "run_id": run_id, "dataset_id": base.dataset_id, "dataset_index": dataset_index,
                            "dataset_namespace": namespace, "dataset_seed": base.generator_seed,
                            "archetype": base.archetype, "n_samples": len(base.y), "n_original": task.n_original,
                            "n_injected": task.n_injected, "mechanism": mechanism, "strength": strength_name,
                            "strength_value": mechanism_config.strength, "model": model, "seed": seed,
                            "status": "FAILURE", "failure_reason": "", "diagnostic_ap": diagnostic_ap,
                            "diagnostic_normalized_ap": diagnostic_nap, "top5_recall": top5_recall,
                            "n_leak": int(task.leakage_mask.sum()), "config_hash": config_hash, "code_hash": code_hash,
                        }
                        try:
                            full_output = fit_predict_core_model(
                                model,
                                task.X[task.train_idx],
                                task.y[task.train_idx],
                                task.X[task.val_idx],
                                task.y[task.val_idx],
                                task.X[task.test_idx],
                                seed,
                            )
                            clean_output, clean_auc = clean_cache[(dataset_index, model, seed)]
                            full_auc = float(roc_auc_score(task.y[task.test_idx], full_output.probabilities))
                            row.update({
                                "status": "SUCCESS", "clean_auc": clean_auc, "full_auc": full_auc,
                                "paired_harm": full_auc - clean_auc,
                                "clean_runtime_sec": clean_output.runtime_sec,
                                "full_runtime_sec": full_output.runtime_sec,
                                "implementation": full_output.implementation,
                                "split_hash": hashlib.sha256(task.test_idx.tobytes()).hexdigest(),
                            })
                            if mechanism in protocol["predictions_retained_for"]:
                                save_predictions(
                                    prediction_dir, run_id, task, task.y[task.test_idx],
                                    clean_output.probabilities, full_output.probabilities,
                                )
                        except Exception as exc:
                            row["failure_reason"] = f"{type(exc).__name__}: {exc}"
                        append_row(output, row)
                        done += 1
                        if done % 25 == 0 or done == total:
                            elapsed = time.time() - started
                            print(f"{done}/{total} cells, {elapsed:.1f}s, {elapsed/max(1,done):.3f}s/cell", flush=True)

    manifest = {
        "schema_version": 1,
        "protocol_version": protocol["version"],
        "dataset_namespace": namespace,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_hash": config_hash,
        "code_hash": code_hash,
        "output": str(output.relative_to(ROOT)),
        "requested_cells": total,
        "datasets": dataset_ids,
        "mechanisms": mechanisms,
        "strengths": strengths,
        "models": models,
        "seeds": seeds,
    }
    (output.parent / f"{output.stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
