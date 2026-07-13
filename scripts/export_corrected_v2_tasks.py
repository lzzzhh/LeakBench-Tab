#!/usr/bin/env python3
"""Export immutable corrected_v2 task arrays for cross-environment model runs."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.leakbench.run_corrected_core import (
    build_mechanism_config,
    diagnostic_metrics,
    parse_selection,
)
from experiments.leakbench.run_corrected_tabm import task_sha256
from src.leakbench.datasets import build_panel_task
from src.leakbench.mechanisms import LeakBenchInjector


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--output-dir", default="results/corrected_v2/task_bundles")
    parser.add_argument("--namespace", default="confirmatory")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mechanisms", default="all")
    parser.add_argument("--strengths", default="all")
    parser.add_argument("--seeds", default="all")
    args = parser.parse_args(argv)
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    datasets = parse_selection(args.datasets, list(range(int(protocol["dataset_count"]))))
    mechanisms = parse_selection(args.mechanisms, protocol["mechanisms"])
    strengths = parse_selection(args.strengths, protocol["strengths"])
    seeds = parse_selection(args.seeds, [int(seed) for seed in protocol["seeds"]])
    output = ROOT / args.output_dir
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = []
    for dataset_index in datasets:
        base = build_panel_task(dataset_index, namespace=args.namespace)
        arrays = {
            "base_X": base.X,
            "y": base.y,
            "train_idx": base.train_idx,
            "val_idx": base.val_idx,
            "test_idx": base.test_idx,
            "timestamps": base.timestamps,
            "base_entity_ids": base.entity_ids,
        }
        dataset_rows = []
        for mechanism in mechanisms:
            for strength in strengths:
                for seed in seeds:
                    config_item = build_mechanism_config(mechanism, strength, config, seed)
                    task = LeakBenchInjector(seed=seed).inject(
                        base.X,
                        base.y,
                        [config_item],
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
                        raise RuntimeError("task export changed the frozen split")
                    key = f"{mechanism}_{strength}_{seed}"
                    arrays[f"block__{key}"] = task.X[:, task.n_original :]
                    arrays[f"leak_mask__{key}"] = task.leakage_mask
                    arrays[f"entity_ids__{key}"] = task.entity_ids
                    arrays[f"source_ids__{key}"] = task.source_ids
                    ap, normalized_ap, top5 = diagnostic_metrics(task)
                    dataset_rows.append({
                        "dataset_id": base.dataset_id,
                        "dataset_index": dataset_index,
                        "dataset_namespace": args.namespace,
                        "dataset_seed": base.generator_seed,
                        "archetype": base.archetype,
                        "mechanism": mechanism,
                        "strength": strength,
                        "strength_value": config_item.strength,
                        "seed": seed,
                        "bundle_key": key,
                        "task_hash": task_sha256(task),
                        "split_hash": hashlib.sha256(task.test_idx.tobytes()).hexdigest(),
                        "n_samples": len(task.y),
                        "n_original": task.n_original,
                        "n_injected": task.n_injected,
                        "n_leak": int(task.leakage_mask.sum()),
                        "diagnostic_ap": ap,
                        "diagnostic_normalized_ap": normalized_ap,
                        "top5_recall": top5,
                    })
        bundle = output / f"{base.dataset_id}.npz"
        np.savez_compressed(bundle, **arrays)
        bundle_hash = sha256(bundle)
        for row in dataset_rows:
            row["bundle_path"] = str(bundle.relative_to(ROOT))
            row["bundle_sha256"] = bundle_hash
        rows.extend(dataset_rows)
        print(f"exported {base.dataset_id}: {len(dataset_rows)} tasks, {bundle.stat().st_size / 1024**2:.1f} MiB", flush=True)

    manifest = pd.DataFrame(rows)
    manifest_path = output / "task_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "schema_version": 1,
        "dataset_namespace": args.namespace,
        "config_sha256": sha256(config_path),
        "task_count": len(manifest),
        "datasets": datasets,
        "mechanisms": mechanisms,
        "strengths": strengths,
        "seeds": seeds,
        "manifest_sha256": sha256(manifest_path),
    }
    (output / "bundle_summary.json").write_text(
        __import__("json").dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
