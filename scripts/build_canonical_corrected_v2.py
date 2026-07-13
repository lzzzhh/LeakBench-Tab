#!/usr/bin/env python3
"""Validate raw CPU/official-TabM outputs and build one canonical cell table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
M10_AMENDMENT_VERSION = "m10_strict_mask_v1"
M10_STRICT_POLICY = "task.X[:, ~task.leakage_mask]"
M10_FULL_POLICY = "task.X"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def builder_identity():
    path = Path(__file__).resolve()
    return {
        "path": str(path.relative_to(ROOT.resolve())),
        "sha256": sha256(path),
    }


def _assert_single_value(frame, field, expected, label):
    actual = set(frame[field].dropna().astype(str))
    if actual != {str(expected)}:
        raise ValueError(f"{label} {field} mismatch: {sorted(actual)}")


def _validate_result_manifest(output_path, expected_cells, expected_model):
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("amendment_version") != M10_AMENDMENT_VERSION:
        raise ValueError(f"{expected_model} amendment result manifest has wrong version")
    if manifest.get("strict_policy") != M10_STRICT_POLICY:
        raise ValueError(f"{expected_model} amendment result manifest has wrong strict policy")
    if (
        int(manifest.get("requested_cells", -1)) != expected_cells
        or int(manifest.get("success_cells", -1)) != expected_cells
        or int(manifest.get("failure_cells", -1)) != 0
        or int(manifest.get("integrity_verified_cells", -1)) != expected_cells
    ):
        raise ValueError(f"{expected_model} amendment result manifest is incomplete")
    if manifest.get("result_sha256") != sha256(output_path):
        raise ValueError(f"{expected_model} amendment result hash mismatch")
    if set(manifest.get("models", [])) != {expected_model} and expected_model == "tabm":
        raise ValueError("TabM amendment result manifest has wrong model identity")
    return manifest, manifest_path


def _validate_m10_amendment_rows(
    frame,
    tasks,
    *,
    label,
    expected_cells,
    expected_models,
    expected_config_hash,
    expected_amendment_config_hash,
    expected_manifest_hash,
    expected_summary_hash,
    expected_runner_hash,
    expected_adapter_hash,
    expected_code_hash,
):
    required = {
        "run_id", "dataset_id", "dataset_namespace", "n_original", "n_injected", "n_leak",
        "mechanism", "strength", "model", "seed", "status", "clean_auc",
        "strict_auc", "full_auc", "paired_harm", "split_hash", "task_hash",
        "source_task_hash", "task_manifest_sha256", "bundle_summary_sha256",
        "bundle_path", "bundle_sha256", "integrity_verified", "amendment_version",
        "strict_policy", "full_policy", "strict_feature_count",
        "legitimate_injected_count", "contamination_removed_count", "strict_view_hash",
        "full_view_hash", "leakage_mask_hash", "config_hash", "amendment_config_hash",
        "runner_sha256", "model_adapter_sha256", "code_hash",
        "diagnostic_ap", "diagnostic_normalized_ap", "top5_recall",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} amendment is missing columns: {sorted(missing)}")
    if len(frame) != expected_cells:
        raise ValueError(f"{label} M10 replacement coverage mismatch: {len(frame)}/{expected_cells}")
    if set(frame["model"].astype(str)) != set(expected_models):
        raise ValueError(f"{label} M10 replacement models are incorrect")
    if set(frame["mechanism"].astype(str)) != {"M10"}:
        raise ValueError(f"{label} amendment contains a non-M10 cell")
    if set(frame["dataset_namespace"].astype(str)) != {"confirmatory"}:
        raise ValueError(f"{label} amendment is not exclusively confirmatory")
    if not (frame["status"].astype(str) == "SUCCESS").all():
        raise ValueError(f"{label} amendment contains a failed cell")
    if not frame["integrity_verified"].astype(str).str.lower().eq("true").all():
        raise ValueError(f"{label} amendment contains an unverified cell")
    scientific_key = ["dataset_id", "mechanism", "strength", "model", "seed"]
    if frame["run_id"].duplicated().any() or frame.duplicated(scientific_key).any():
        raise ValueError(f"{label} amendment contains duplicate cells")

    for field, expected in (
        ("amendment_version", M10_AMENDMENT_VERSION),
        ("strict_policy", M10_STRICT_POLICY),
        ("full_policy", M10_FULL_POLICY),
        ("config_hash", expected_config_hash),
        ("amendment_config_hash", expected_amendment_config_hash),
        ("task_manifest_sha256", expected_manifest_hash),
        ("bundle_summary_sha256", expected_summary_hash),
        ("runner_sha256", expected_runner_hash),
        ("model_adapter_sha256", expected_adapter_hash),
        ("code_hash", expected_code_hash),
    ):
        _assert_single_value(frame, field, expected, label)
    if not np.allclose(frame["clean_auc"], frame["strict_auc"], atol=0, rtol=0):
        raise ValueError(f"{label} clean_auc is not the amended strict_auc")
    if not np.allclose(
        frame["paired_harm"], frame["full_auc"] - frame["strict_auc"], atol=1e-12, rtol=0
    ):
        raise ValueError(f"{label} paired_harm is not full_auc - strict_auc")
    if not (frame["n_injected"].astype(int) == 2).all():
        raise ValueError(f"{label} M10 does not contain exactly two injected features")
    if not (frame["n_leak"].astype(int) == 1).all():
        raise ValueError(f"{label} M10 does not mark exactly one contamination feature")
    if not (frame["legitimate_injected_count"].astype(int) == 1).all():
        raise ValueError(f"{label} M10 did not retain exactly one legitimate injected feature")
    if not (frame["contamination_removed_count"].astype(int) == 1).all():
        raise ValueError(f"{label} M10 did not remove exactly one contamination feature")
    if not (
        frame["strict_feature_count"].astype(int) == frame["n_original"].astype(int) + 1
    ).all():
        raise ValueError(f"{label} strict feature count does not retain the legitimate duplicate")
    for field in ("strict_view_hash", "full_view_hash", "leakage_mask_hash"):
        if not frame[field].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"{label} contains an invalid {field}")

    key = ["dataset_id", "mechanism", "strength", "seed"]
    task_fields = key + [
        "task_hash", "split_hash", "diagnostic_ap", "diagnostic_normalized_ap",
        "top5_recall", "n_leak", "bundle_path", "bundle_sha256",
    ]
    m10_tasks = tasks.loc[tasks["mechanism"].astype(str) == "M10", task_fields]
    unique_cells = frame.drop_duplicates(key)
    checked = unique_cells.merge(
        m10_tasks, on=key, suffixes=("", "_task"), validate="one_to_one"
    )
    if len(checked) != len(m10_tasks):
        raise ValueError(f"{label} amendment does not cover every frozen M10 task")
    for field in ("task_hash", "split_hash", "bundle_path", "bundle_sha256"):
        if not (checked[field].astype(str) == checked[f"{field}_task"].astype(str)).all():
            raise ValueError(f"{label} amendment {field} differs from frozen tasks")
    if not (checked["source_task_hash"].astype(str) == checked["task_hash_task"].astype(str)).all():
        raise ValueError(f"{label} amendment source_task_hash differs from frozen tasks")
    for field in ("diagnostic_ap", "diagnostic_normalized_ap", "top5_recall", "n_leak"):
        if not np.allclose(checked[field], checked[f"{field}_task"], atol=1e-12, rtol=0):
            raise ValueError(f"{label} amendment {field} differs from frozen tasks")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", default="results/corrected_v2/core_cpu_cells.csv")
    parser.add_argument("--tabm", default="results/corrected_v2/tabm_bundle_confirmatory/tabm_cells.csv")
    parser.add_argument(
        "--cpu-m10-amendment",
        default="results/corrected_v2/m10_amendment_confirmatory/cpu_cells.csv",
    )
    parser.add_argument(
        "--tabm-m10-amendment",
        default="results/corrected_v2/m10_amendment_confirmatory/tabm_cells.csv",
    )
    parser.add_argument("--tasks", default="results/corrected_v2/task_bundles/task_manifest.csv")
    parser.add_argument("--config", default="configs/paper/corrected_v2.yaml")
    parser.add_argument("--m10-config", default="configs/paper/m10_amendment_v1.yaml")
    parser.add_argument(
        "--m10-freeze",
        default="results/corrected_v2/m10_amendment_protocol_freeze.json",
    )
    parser.add_argument("--output", default="results/corrected_v2/canonical_cells.csv")
    args = parser.parse_args(argv)
    paths = {name: ROOT / value for name, value in vars(args).items()}
    if paths["output"].exists():
        raise FileExistsError(paths["output"])
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))["protocol"]
    cpu = pd.read_csv(paths["cpu"])
    tabm = pd.read_csv(paths["tabm"])
    m10_cpu = pd.read_csv(paths["cpu_m10_amendment"])
    m10_tabm = pd.read_csv(paths["tabm_m10_amendment"])
    tasks = pd.read_csv(paths["tasks"])
    m10_config_payload = yaml.safe_load(paths["m10_config"].read_text(encoding="utf-8"))
    m10_config = m10_config_payload["amendment"]
    m10_freeze = json.loads(paths["m10_freeze"].read_text(encoding="utf-8"))
    expected_config_hash = sha256(paths["config"])
    expected_manifest_hash = sha256(paths["tasks"])
    expected_cpu_code_hash = hashlib.sha256(
        "".join(
            sha256(ROOT / path)
            for path in (
                "src/leakbench/datasets.py",
                "src/leakbench/mechanisms/__init__.py",
                "src/leakbench/models/core_models.py",
                "experiments/leakbench/run_corrected_core.py",
            )
        ).encode()
    ).hexdigest()
    expected_tabm_code_hash = hashlib.sha256(
        "".join(
            sha256(ROOT / path)
            for path in (
                "src/leakbench/models/official_tabm.py",
                "experiments/leakbench/run_corrected_tabm_bundle.py",
            )
        ).encode()
    ).hexdigest()
    expected_m10_config_hash = sha256(paths["m10_config"])
    expected_m10_runner_hash = sha256(
        ROOT / "experiments/leakbench/run_m10_amendment.py"
    )
    expected_m10_cpu_adapter_hash = sha256(
        ROOT / "src/leakbench/models/core_models.py"
    )
    expected_m10_tabm_adapter_hash = sha256(
        ROOT / "src/leakbench/models/official_tabm.py"
    )
    expected_m10_cpu_code_hash = hashlib.sha256(
        (expected_m10_cpu_adapter_hash + expected_m10_runner_hash).encode()
    ).hexdigest()
    expected_m10_tabm_code_hash = hashlib.sha256(
        (expected_m10_tabm_adapter_hash + expected_m10_runner_hash).encode()
    ).hexdigest()
    if m10_config.get("version") != M10_AMENDMENT_VERSION:
        raise ValueError("M10 amendment config version mismatch")
    if m10_config.get("strict_policy") != M10_STRICT_POLICY:
        raise ValueError("M10 amendment config strict policy mismatch")
    if m10_config.get("full_policy") != M10_FULL_POLICY:
        raise ValueError("M10 amendment config full policy mismatch")
    if m10_config.get("base_config_sha256") != expected_config_hash:
        raise ValueError("M10 amendment is bound to a different corrected_v2 config")
    if m10_config.get("confirmatory_task_manifest_sha256") != expected_manifest_hash:
        raise ValueError("M10 amendment is bound to a different task manifest")
    if m10_freeze.get("status") != "FROZEN_BEFORE_M10_AMENDMENT_CONFIRMATORY_RUN":
        raise ValueError("M10 amendment protocol was not frozen before confirmation")
    if m10_freeze.get("amendment_version") != M10_AMENDMENT_VERSION:
        raise ValueError("M10 amendment freeze version mismatch")
    if m10_freeze.get("strict_policy") != M10_STRICT_POLICY:
        raise ValueError("M10 amendment freeze strict policy mismatch")
    if m10_freeze.get("source_task_manifest", {}).get("sha256") != expected_manifest_hash:
        raise ValueError("M10 amendment freeze references a different task manifest")
    expected_outputs = {
        "cpu": str(paths["cpu_m10_amendment"].relative_to(ROOT)),
        "tabm": str(paths["tabm_m10_amendment"].relative_to(ROOT)),
    }
    if m10_freeze.get("outputs") != expected_outputs:
        raise ValueError("M10 amendment inputs do not match the frozen output paths")
    for relative, entry in m10_freeze.get("frozen_files", {}).items():
        frozen_path = ROOT / relative
        if (
            not frozen_path.is_file()
            or frozen_path.stat().st_size != int(entry["size_bytes"])
            or sha256(frozen_path) != entry["sha256"]
        ):
            raise ValueError(f"M10 amendment frozen file changed: {relative}")
    expected_cpu = config["dataset_count"] * len(config["mechanisms"]) * len(config["strengths"]) * 4 * len(config["seeds"])
    expected_tabm = config["dataset_count"] * len(config["mechanisms"]) * len(config["strengths"]) * len(config["seeds"])
    if len(cpu) != expected_cpu or set(cpu["model"]) != {"lr", "rf", "catboost", "lightgbm"}:
        raise ValueError(f"CPU coverage mismatch: {len(cpu)}/{expected_cpu}")
    if len(tabm) != expected_tabm or set(tabm["model"]) != {"tabm"}:
        raise ValueError(f"TabM coverage mismatch: {len(tabm)}/{expected_tabm}")
    if not (cpu["status"] == "SUCCESS").all() or not (tabm["status"] == "SUCCESS").all():
        raise ValueError("Canonical inputs contain failed cells")
    if set(cpu["dataset_namespace"].astype(str)) != {"confirmatory"}:
        raise ValueError("CPU input is not exclusively confirmatory")
    if set(tabm["dataset_namespace"].astype(str)) != {"confirmatory"}:
        raise ValueError("TabM input is not exclusively confirmatory")
    if set(tasks["dataset_namespace"].astype(str)) != {"confirmatory"} or len(tasks) != expected_tabm:
        raise ValueError("Frozen task manifest is not the full confirmatory matrix")
    if set(cpu["config_hash"].astype(str)) != {expected_config_hash}:
        raise ValueError("CPU config hash differs from the frozen config")
    if set(tabm["config_hash"].astype(str)) != {expected_config_hash}:
        raise ValueError("TabM config hash differs from the frozen config")
    if set(cpu["code_hash"].astype(str)) != {expected_cpu_code_hash}:
        raise ValueError("CPU code hash differs from the frozen implementation")
    if set(tabm["code_hash"].astype(str)) != {expected_tabm_code_hash}:
        raise ValueError("TabM code hash differs from the frozen implementation")
    if set(tabm["task_manifest_sha256"].astype(str)) != {expected_manifest_hash}:
        raise ValueError("TabM task-manifest hash differs from the canonical manifest")
    if not tabm["integrity_verified"].astype(str).str.lower().eq("true").all():
        raise ValueError("TabM includes a cell without pre-fit integrity verification")
    key = ["dataset_id", "mechanism", "strength", "seed"]
    if cpu.duplicated(key + ["model"]).any() or tabm.duplicated(key + ["model"]).any():
        raise ValueError("Duplicate raw scientific cell")

    task_fields = key + [
        "task_hash", "split_hash", "diagnostic_ap", "diagnostic_normalized_ap",
        "top5_recall", "n_leak", "bundle_path", "bundle_sha256",
    ]
    tabm_checked = tabm.merge(tasks[task_fields], on=key, suffixes=("", "_task"), validate="one_to_one")
    for field in ("task_hash", "split_hash", "bundle_path", "bundle_sha256"):
        if not (tabm_checked[field].astype(str) == tabm_checked[f"{field}_task"].astype(str)).all():
            raise ValueError(f"TabM {field} differs from frozen task manifest")
    for field in ("diagnostic_ap", "diagnostic_normalized_ap", "top5_recall", "n_leak"):
        if not np.allclose(tabm_checked[field], tabm_checked[f"{field}_task"], atol=1e-12, rtol=0):
            raise ValueError(f"TabM {field} differs from frozen task manifest")

    cpu_task = cpu.groupby(key, as_index=False).agg(
        diagnostic_ap=("diagnostic_ap", "mean"),
        diagnostic_ap_range=("diagnostic_ap", lambda values: float(values.max() - values.min())),
        diagnostic_normalized_ap=("diagnostic_normalized_ap", "mean"),
        diagnostic_normalized_ap_range=("diagnostic_normalized_ap", lambda values: float(values.max() - values.min())),
        top5_recall=("top5_recall", "mean"),
        top5_recall_range=("top5_recall", lambda values: float(values.max() - values.min())),
        n_leak=("n_leak", "mean"),
        split_hash=("split_hash", "first"),
        split_hash_count=("split_hash", "nunique"),
    ).merge(tasks[task_fields], on=key, suffixes=("", "_task"), validate="one_to_one")
    for field in ("diagnostic_ap", "diagnostic_normalized_ap", "top5_recall", "n_leak"):
        if cpu_task.get(f"{field}_range", pd.Series([0.0])).max() > 1e-12:
            raise ValueError(f"CPU {field} differs across models")
        if not np.allclose(cpu_task[field], cpu_task[f"{field}_task"], atol=1e-12, rtol=0):
            raise ValueError(f"CPU {field} differs from frozen task manifest")
    if cpu_task["split_hash_count"].max() != 1:
        raise ValueError("CPU split hash differs across models")
    if not (cpu_task["split_hash"].astype(str) == cpu_task["split_hash_task"].astype(str)).all():
        raise ValueError("CPU split hash differs from frozen task manifest")

    expected_m10_cpu = int(m10_freeze.get("expected_cpu_cells", -1))
    expected_m10_tabm = int(m10_freeze.get("expected_tabm_cells", -1))
    expected_m10_total = int(m10_freeze.get("expected_replacement_cells", -1))
    if (expected_m10_cpu, expected_m10_tabm, expected_m10_total) != (2000, 500, 2500):
        raise ValueError("Frozen M10 replacement-cell counts are not 2,000 + 500 = 2,500")
    if (
        int(m10_config.get("expected_confirmatory_cpu_cells", -1)) != expected_m10_cpu
        or int(m10_config.get("expected_confirmatory_tabm_cells", -1)) != expected_m10_tabm
        or int(m10_config.get("expected_replacement_cells", -1)) != expected_m10_total
    ):
        raise ValueError("M10 amendment config/freeze cell counts disagree")

    m10_cpu_manifest, m10_cpu_manifest_path = _validate_result_manifest(
        paths["cpu_m10_amendment"], expected_m10_cpu, "cpu"
    )
    m10_tabm_manifest, m10_tabm_manifest_path = _validate_result_manifest(
        paths["tabm_m10_amendment"], expected_m10_tabm, "tabm"
    )
    source_manifest_paths = {
        "cpu": paths["cpu"].with_name(f"{paths['cpu'].stem}_manifest.json"),
        "tabm": paths["tabm"].with_name(f"{paths['tabm'].stem}_manifest.json"),
        "m10_cpu": m10_cpu_manifest_path,
        "m10_tabm": m10_tabm_manifest_path,
    }
    if any(not path.is_file() for path in source_manifest_paths.values()):
        missing = [str(path) for path in source_manifest_paths.values() if not path.is_file()]
        raise FileNotFoundError(f"Canonical source manifests are incomplete: {missing}")
    if set(m10_cpu_manifest.get("models", [])) != {"lr", "rf", "catboost", "lightgbm"}:
        raise ValueError("CPU M10 amendment manifest has the wrong model set")
    if m10_cpu_manifest.get("runner_sha256") != expected_m10_runner_hash:
        raise ValueError("CPU M10 amendment manifest has the wrong runner hash")
    if m10_tabm_manifest.get("runner_sha256") != expected_m10_runner_hash:
        raise ValueError("TabM M10 amendment manifest has the wrong runner hash")
    if m10_cpu_manifest.get("amendment_config_sha256") != expected_m10_config_hash:
        raise ValueError("CPU M10 amendment manifest has the wrong config hash")
    if m10_tabm_manifest.get("amendment_config_sha256") != expected_m10_config_hash:
        raise ValueError("TabM M10 amendment manifest has the wrong config hash")

    expected_summary_hash = str(m10_config["confirmatory_bundle_summary_sha256"])
    _validate_m10_amendment_rows(
        m10_cpu,
        tasks,
        label="CPU",
        expected_cells=expected_m10_cpu,
        expected_models={"lr", "rf", "catboost", "lightgbm"},
        expected_config_hash=expected_config_hash,
        expected_amendment_config_hash=expected_m10_config_hash,
        expected_manifest_hash=expected_manifest_hash,
        expected_summary_hash=expected_summary_hash,
        expected_runner_hash=expected_m10_runner_hash,
        expected_adapter_hash=expected_m10_cpu_adapter_hash,
        expected_code_hash=expected_m10_cpu_code_hash,
    )
    _validate_m10_amendment_rows(
        m10_tabm,
        tasks,
        label="TabM",
        expected_cells=expected_m10_tabm,
        expected_models={"tabm"},
        expected_config_hash=expected_config_hash,
        expected_amendment_config_hash=expected_m10_config_hash,
        expected_manifest_hash=expected_manifest_hash,
        expected_summary_hash=expected_summary_hash,
        expected_runner_hash=expected_m10_runner_hash,
        expected_adapter_hash=expected_m10_tabm_adapter_hash,
        expected_code_hash=expected_m10_tabm_code_hash,
    )
    for manifest, label in ((m10_cpu_manifest, "CPU"), (m10_tabm_manifest, "TabM")):
        if manifest.get("task_manifest_sha256") != expected_manifest_hash:
            raise ValueError(f"{label} M10 amendment manifest has the wrong task hash")
        if manifest.get("bundle_summary_sha256") != expected_summary_hash:
            raise ValueError(f"{label} M10 amendment manifest has the wrong summary hash")

    scientific_key = ["dataset_id", "mechanism", "strength", "model", "seed"]
    original_cpu_m10 = cpu.loc[cpu["mechanism"].astype(str) == "M10"]
    original_tabm_m10 = tabm.loc[tabm["mechanism"].astype(str) == "M10"]
    if set(map(tuple, original_cpu_m10[scientific_key].to_numpy())) != set(
        map(tuple, m10_cpu[scientific_key].to_numpy())
    ):
        raise ValueError("CPU M10 amendment identities do not exactly replace original M10")
    if set(map(tuple, original_tabm_m10[scientific_key].to_numpy())) != set(
        map(tuple, m10_tabm[scientific_key].to_numpy())
    ):
        raise ValueError("TabM M10 amendment identities do not exactly replace original M10")

    # Keep the common CPU schema plus official TabM provenance columns. Missing
    # fields remain NaN by design; scientific keys and metrics are shared.
    base_cpu = cpu.loc[cpu["mechanism"].astype(str) != "M10"].copy()
    base_tabm = tabm.loc[tabm["mechanism"].astype(str) != "M10"].copy()
    canonical = pd.concat(
        [base_cpu, base_tabm, m10_cpu, m10_tabm], ignore_index=True, sort=False
    )
    canonical = canonical.sort_values(
        ["dataset_id", "mechanism", "strength", "model", "seed"], kind="stable"
    ).reset_index(drop=True)
    if canonical.duplicated(scientific_key).any() or len(canonical) != config["expected_model_training_cells"]:
        raise ValueError("Canonical matrix is not exactly the frozen 27,500-cell design")
    canonical_m10 = canonical.loc[canonical["mechanism"].astype(str) == "M10"]
    if len(canonical_m10) != expected_m10_total:
        raise ValueError("Canonical M10 is not exactly the 2,500-cell amendment")
    if set(canonical_m10["amendment_version"].astype(str)) != {M10_AMENDMENT_VERSION}:
        raise ValueError("Canonical retained an original, unamended M10 cell")
    canonical["evidence_tier"] = "confirmatory"
    canonical["task_source"] = "frozen_local_bundle_base_protocol"
    canonical.loc[
        canonical["amendment_version"].astype(str) == M10_AMENDMENT_VERSION,
        "task_source",
    ] = "frozen_local_bundle_m10_amendment"
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(paths["output"], index=False)
    manifest = {
        "schema_version": 1,
        "status": "CANONICAL",
        "builder": builder_identity(),
        "cells": len(canonical),
        "successful_cells": int((canonical["status"] == "SUCCESS").sum()),
        "datasets": int(canonical["dataset_id"].nunique()),
        "mechanisms": int(canonical["mechanism"].nunique()),
        "strengths": int(canonical["strength"].nunique()),
        "models": sorted(canonical["model"].unique()),
        "seeds": sorted(int(seed) for seed in canonical["seed"].unique()),
        "source_sha256": {
            "cpu": sha256(paths["cpu"]),
            "tabm": sha256(paths["tabm"]),
            "m10_cpu": sha256(paths["cpu_m10_amendment"]),
            "m10_tabm": sha256(paths["tabm_m10_amendment"]),
            "tasks": sha256(paths["tasks"]),
        },
        "source_manifest_sha256": {
            name: sha256(path) for name, path in source_manifest_paths.items()
        },
        "validated_code_sha256": {
            "cpu": expected_cpu_code_hash,
            "tabm": expected_tabm_code_hash,
            "m10_cpu": expected_m10_cpu_code_hash,
            "m10_tabm": expected_m10_tabm_code_hash,
        },
        "m10_amendment": {
            "version": M10_AMENDMENT_VERSION,
            "replacement_cells": expected_m10_total,
            "cpu_cells": expected_m10_cpu,
            "tabm_cells": expected_m10_tabm,
            "protocol_freeze_sha256": sha256(paths["m10_freeze"]),
        },
        "config_sha256": expected_config_hash,
        "canonical_sha256": sha256(paths["output"]),
    }
    (paths["output"].parent / "canonical_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
