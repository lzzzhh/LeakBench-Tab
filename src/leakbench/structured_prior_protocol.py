"""Shared protocol helpers for the structured-prior amendment studies."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.leakbench.datasets import ARCHETYPES, build_panel_task
from src.leakbench.mechanisms import MechanismConfig, MechanismID


MECHANISM_IDS = {
    "M01": MechanismID.DIRECT_COPY,
    "M02": MechanismID.NOISY_PROXY,
    "M04": MechanismID.POST_OUTCOME,
    "M05": MechanismID.TEMPORAL_LEAK,
    "M06": MechanismID.REDUNDANT_CLUSTER,
    "M08": MechanismID.ENTITY_LEAK,
    "M09": MechanismID.SOURCE_LEAK,
    "M10": MechanismID.MIXED,
}

PLAN_COLUMNS = [
    "task_variant_id",
    "protocol_version",
    "study_namespace",
    "dataset_namespace",
    "dataset_index",
    "dataset_id",
    "dataset_seed",
    "archetype",
    "base_task_sha256",
    "mechanism",
    "mechanism_family",
    "strength",
    "strength_value",
    "seed",
    "model_ids",
    "expected_model_cells",
    "config_sha256",
]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _update_array_hash(digest, values):
    values = np.ascontiguousarray(values)
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())


def base_task_sha256(task):
    """Hash every frozen base-task array without exposing outcome summaries."""
    digest = hashlib.sha256()
    digest.update(task.dataset_id.encode())
    digest.update(str(task.generator_seed).encode())
    digest.update(task.archetype.encode())
    for values in (
        task.X,
        task.y,
        task.timestamps,
        task.entity_ids,
        task.source_ids,
        task.train_idx,
        task.val_idx,
        task.test_idx,
    ):
        _update_array_hash(digest, values)
    return digest.hexdigest()


def injected_task_sha256(task):
    """Hash the model-facing arrays used to reconstruct one injected task."""
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
        _update_array_hash(digest, values)
    return digest.hexdigest()


def load_protocol_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    indices = [int(value) for value in protocol["dataset_indices"]]
    mechanisms = list(protocol["mechanisms"])
    strengths = list(protocol["strengths"])
    seeds = [int(value) for value in protocol["seeds"]]
    models = list(protocol["core_models"])

    if len(indices) != int(protocol["dataset_count"]) or len(indices) != len(set(indices)):
        raise ValueError("dataset_indices must be unique and match dataset_count")
    if len(mechanisms) != len(set(mechanisms)) or not set(mechanisms) <= set(MECHANISM_IDS):
        raise ValueError("mechanisms must be unique supported mechanism IDs")
    if len(strengths) != len(set(strengths)):
        raise ValueError("strengths must be unique")
    if set(strengths) != set(protocol["strength_values"]):
        raise ValueError("strength_values must exactly cover strengths")
    if len(seeds) != len(set(seeds)) or len(models) != len(set(models)):
        raise ValueError("seeds and models must be unique")
    expected_variants = len(indices) * len(mechanisms) * len(strengths) * len(seeds)
    expected_cells = expected_variants * len(models)
    if expected_variants != int(protocol["expected_task_variants"]):
        raise ValueError("expected_task_variants arithmetic is inconsistent")
    if expected_cells != int(protocol["expected_model_cells"]):
        raise ValueError("expected_model_cells arithmetic is inconsistent")
    cpu_count = len([model for model in models if model != "tabm"])
    if expected_variants * cpu_count != int(protocol["expected_cpu_model_cells"]):
        raise ValueError("expected_cpu_model_cells arithmetic is inconsistent")
    if expected_variants * int("tabm" in models) != int(protocol["expected_tabm_model_cells"]):
        raise ValueError("expected_tabm_model_cells arithmetic is inconsistent")
    return config


def build_mechanism_config(mechanism, strength_name, config, seed):
    protocol = config["protocol"]
    level = list(protocol["strengths"]).index(strength_name)
    parameters = config["mechanism_parameters"].get(mechanism, {})
    kwargs = {
        "mechanism": MECHANISM_IDS[mechanism],
        "strength": float(protocol["strength_values"][strength_name]),
        "noise_std": float(parameters.get("noise_std", 0.05)),
        "seed": int(seed),
    }
    if mechanism == "M05":
        kwargs["time_offset"] = float(parameters["time_offsets"][level])
    elif mechanism == "M06":
        kwargs["redundancy"] = int(parameters["redundancies"][level])
    elif mechanism == "M08":
        kwargs["prior_strength"] = float(parameters["prior_strength"])
    elif mechanism == "M09":
        kwargs["n_sources"] = int(parameters["n_sources"])
        kwargs["min_group_count"] = int(parameters["min_group_count"])
    return MechanismConfig(**kwargs)


def build_frozen_task_plan(config_path):
    """Build the pre-outcome-analysis task grid and base-task identities."""
    config_path = Path(config_path)
    config = load_protocol_config(config_path)
    protocol = config["protocol"]
    config_hash = file_sha256(config_path)
    models = list(protocol["core_models"])
    simple = set(protocol.get("simple_mechanisms", []))
    structured = set(protocol.get("structured_mechanisms", []))
    rows = []
    tasks = {}
    for dataset_index in protocol["dataset_indices"]:
        task = build_panel_task(
            int(dataset_index), namespace=str(protocol["dataset_namespace"])
        )
        tasks[int(dataset_index)] = task
        task_hash = base_task_sha256(task)
        for mechanism in protocol["mechanisms"]:
            if mechanism in simple:
                family = "simple"
            elif mechanism in structured:
                family = "structured"
            else:
                family = "structured_replacement"
            for strength in protocol["strengths"]:
                for seed in protocol["seeds"]:
                    identity = "|".join((
                        str(protocol["study_namespace"]),
                        str(dataset_index),
                        task_hash,
                        str(mechanism),
                        str(strength),
                        str(seed),
                        config_hash,
                    ))
                    rows.append({
                        "task_variant_id": hashlib.sha256(identity.encode()).hexdigest()[:20],
                        "protocol_version": protocol["version"],
                        "study_namespace": protocol["study_namespace"],
                        "dataset_namespace": protocol["dataset_namespace"],
                        "dataset_index": int(dataset_index),
                        "dataset_id": task.dataset_id,
                        "dataset_seed": int(task.generator_seed),
                        "archetype": task.archetype,
                        "base_task_sha256": task_hash,
                        "mechanism": mechanism,
                        "mechanism_family": family,
                        "strength": strength,
                        "strength_value": float(protocol["strength_values"][strength]),
                        "seed": int(seed),
                        "model_ids": "|".join(models),
                        "expected_model_cells": len(models),
                        "config_sha256": config_hash,
                    })
    frame = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    if len(frame) != int(protocol["expected_task_variants"]):
        raise RuntimeError("generated task-plan count differs from the frozen config")
    identity = ["dataset_index", "mechanism", "strength", "seed"]
    if frame.duplicated(identity).any() or frame["task_variant_id"].duplicated().any():
        raise RuntimeError("generated task plan contains duplicate identities")
    if protocol["version"] == "independent_replication_v1":
        counts = frame.drop_duplicates("dataset_index")["archetype"].value_counts()
        expected = {name: 5 for name in ARCHETYPES}
        if counts.to_dict() != expected:
            raise RuntimeError(f"replication archetype counts differ: {counts.to_dict()}")
    return frame, tasks

