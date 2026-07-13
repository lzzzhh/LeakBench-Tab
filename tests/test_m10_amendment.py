import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.leakbench import run_m10_amendment as runner
from scripts import build_canonical_corrected_v2 as canonical_builder


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_contract(tmp_path, namespace="pilot", bad_bundle_hash=False):
    base_config = tmp_path / "base.yaml"
    base_config.write_text(yaml.safe_dump({
        "protocol": {
            "mechanisms": ["M10"],
            "seeds": [13],
        }
    }), encoding="utf-8")
    base_hash = _sha256(base_config)

    n = 12
    base_X = np.column_stack((np.arange(n), np.linspace(-1, 1, n))).astype(np.float32)
    y = (np.arange(n) % 2).astype(np.float32)
    block = np.column_stack((base_X[:, 0], y + 0.1)).astype(np.float32)
    mask = np.array([False, False, False, True])
    train_idx = np.arange(0, 6)
    val_idx = np.arange(6, 9)
    test_idx = np.arange(9, 12)
    entity_ids = np.arange(n, dtype=np.int64)
    source_ids = np.zeros(n, dtype=np.int64)
    task = runner.FrozenTask(
        base_X=base_X,
        X=np.concatenate((base_X, block), axis=1),
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        leakage_mask=mask,
        entity_ids=entity_ids,
        source_ids=source_ids,
    )

    bundle_dir = tmp_path / "tasks"
    bundle_dir.mkdir()
    bundle_path = bundle_dir / "panel_00.npz"
    np.savez_compressed(
        bundle_path,
        base_X=base_X,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        block__M10_S1_13=block,
        leak_mask__M10_S1_13=mask,
        entity_ids__M10_S1_13=entity_ids,
        source_ids__M10_S1_13=source_ids,
    )
    bundle_hash = _sha256(bundle_path)
    row = {
        "dataset_id": "panel_00",
        "dataset_index": 0,
        "dataset_namespace": namespace,
        "dataset_seed": 1,
        "archetype": "linear",
        "mechanism": "M10",
        "strength": "S1",
        "strength_value": 0.2,
        "seed": 13,
        "bundle_key": "M10_S1_13",
        "task_hash": runner.task_sha256(task),
        "split_hash": hashlib.sha256(test_idx.tobytes()).hexdigest(),
        "n_samples": n,
        "n_original": 2,
        "n_injected": 2,
        "n_leak": 1,
        "diagnostic_ap": 1.0,
        "diagnostic_normalized_ap": 1.0,
        "top5_recall": 1.0,
        "bundle_path": str(bundle_path.relative_to(tmp_path)),
        "bundle_sha256": "0" * 64 if bad_bundle_hash else bundle_hash,
    }
    manifest_path = bundle_dir / "task_manifest.csv"
    pd.DataFrame([row]).to_csv(manifest_path, index=False)
    summary_path = bundle_dir / "bundle_summary.json"
    summary_path.write_text(json.dumps({
        "schema_version": 1,
        "dataset_namespace": namespace,
        "config_sha256": base_hash,
        "task_count": 1,
        "datasets": [0],
        "mechanisms": ["M10"],
        "strengths": ["S1"],
        "seeds": [13],
        "manifest_sha256": _sha256(manifest_path),
    }), encoding="utf-8")

    amendment_config = tmp_path / "amendment.yaml"
    amendment_config.write_text(yaml.safe_dump({
        "amendment": {
            "version": runner.AMENDMENT_VERSION,
            "mechanism": "M10",
            "strict_policy": runner.STRICT_POLICY,
            "full_policy": runner.FULL_POLICY,
            "confirmatory_namespace": "confirmatory",
            "pilot_namespace": "pilot",
            "base_config_path": str(base_config.relative_to(tmp_path)),
            "base_config_sha256": base_hash,
            "confirmatory_task_manifest_sha256": _sha256(manifest_path),
            "confirmatory_bundle_summary_sha256": _sha256(summary_path),
            "cpu_models": ["lr", "rf", "catboost", "lightgbm"],
            "official_model": "tabm",
            "strengths": ["S1"],
            "required_m10_injected_features": 2,
            "required_contamination_features": 1,
            "required_legitimate_injected_features": 1,
        },
        "tabm": {
            "required_version": "0.0.3",
            "device": "cuda",
            "k": 32,
            "learning_rate": 0.003,
            "weight_decay": 0.0001,
            "max_epochs": 200,
            "batch_size": 256,
            "inference_batch_size": 1024,
            "patience": 20,
            "min_delta": 0.00001,
        },
    }), encoding="utf-8")
    for relative in (
        "src/leakbench/models/core_models.py",
        "src/leakbench/models/official_tabm.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test adapter identity\n", encoding="utf-8")
    return amendment_config, manifest_path, row, task


def _fake_output(probabilities, model="lr"):
    return SimpleNamespace(
        probabilities=np.asarray(probabilities, dtype=float),
        runtime_sec=0.01,
        implementation=f"test.{model}",
    )


def test_strict_contract_on_real_pilot_retains_duplicate_and_only_removes_masked_column():
    config_path = Path("configs/paper/m10_amendment_v1.yaml").resolve()
    _, amendment, _, _, base_hash = runner.load_amendment_config(config_path)
    manifest_path = Path(
        "results/corrected_v2/m10_amendment_pilot_tasks/task_manifest.csv"
    ).resolve()
    manifest, _, _, _ = runner.load_bundle_contract(manifest_path, base_hash)
    assert len(manifest) == 45
    for _, row in manifest.iterrows():
        task, _ = runner.load_verified_task(row)
        contract = runner.derive_strict_contract(task, row, amendment)
        assert np.array_equal(contract.X, task.X[:, ~task.leakage_mask])
        assert contract.X.shape[1] == int(row["n_original"]) + 1
        assert contract.legitimate_injected_count == 1
        assert contract.contamination_removed_count == 1
        assert np.array_equal(contract.X[:, -1], task.base_X[:, 0])


def test_runner_fits_literal_strict_view_then_full_view(monkeypatch, tmp_path):
    config_path, manifest_path, _, task = _write_contract(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    calls = []

    def fake_fit(model, X_train, y_train, X_val, y_val, X_test, seed, tabm_kwargs):
        calls.append((X_train.copy(), X_val.copy(), X_test.copy()))
        return _fake_output([0.8, 0.2, 0.9], model)

    monkeypatch.setattr(runner, "_fit_model", fake_fit)
    output = tmp_path / "results/cells.csv"
    exit_code = runner.main([
        "--config", str(config_path),
        "--task-manifest", str(manifest_path),
        "--output", str(output),
        "--models", "lr",
    ])
    assert exit_code == 0
    assert len(calls) == 2
    strict = task.X[:, ~task.leakage_mask]
    assert np.array_equal(calls[0][0], strict[task.train_idx])
    assert np.array_equal(calls[1][0], task.X[task.train_idx])
    result = pd.read_csv(output).iloc[0]
    assert result["status"] == "SUCCESS"
    assert result["strict_policy"] == runner.STRICT_POLICY
    assert result["strict_feature_count"] == 3
    assert result["legitimate_injected_count"] == 1
    assert result["contamination_removed_count"] == 1
    assert result["clean_auc"] == result["strict_auc"]
    assert bool(result["integrity_verified"])


def test_bundle_hash_mismatch_fails_before_fit(tmp_path):
    _, manifest_path, _, _ = _write_contract(tmp_path, bad_bundle_hash=True)
    row = pd.read_csv(manifest_path).iloc[0]
    with pytest.raises(RuntimeError, match="Bundle SHA256 mismatch"):
        runner.load_verified_task(row, tmp_path)


def test_confirmatory_execution_is_locked_before_adapter_resolution(monkeypatch, tmp_path):
    config_path, manifest_path, _, _ = _write_contract(
        tmp_path, namespace="confirmatory"
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="Confirmatory M10 amendment execution is locked"):
        runner.main([
            "--config", str(config_path),
            "--task-manifest", str(manifest_path),
            "--output", str(tmp_path / "results/cells.csv"),
            "--models", "lr",
        ])


def test_official_tabm_adapter_import_is_lazy():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
        elif isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
    assert "src.leakbench.models.official_tabm" not in top_level_imports


def test_canonical_amendment_validator_rejects_old_clean_semantics():
    values = {
        "run_id": "amended-cell",
        "dataset_id": "panel_00",
        "dataset_namespace": "confirmatory",
        "n_original": 2,
        "n_injected": 2,
        "n_leak": 1,
        "mechanism": "M10",
        "strength": "S1",
        "model": "lr",
        "seed": 13,
        "status": "SUCCESS",
        "clean_auc": 0.7,
        "strict_auc": 0.7,
        "full_auc": 0.8,
        "paired_harm": 0.1,
        "split_hash": "1" * 64,
        "task_hash": "2" * 64,
        "source_task_hash": "2" * 64,
        "task_manifest_sha256": "3" * 64,
        "bundle_summary_sha256": "4" * 64,
        "bundle_path": "results/corrected_v2/task_bundles/panel_00.npz",
        "bundle_sha256": "5" * 64,
        "integrity_verified": True,
        "amendment_version": runner.AMENDMENT_VERSION,
        "strict_policy": runner.STRICT_POLICY,
        "full_policy": runner.FULL_POLICY,
        "strict_feature_count": 3,
        "legitimate_injected_count": 1,
        "contamination_removed_count": 1,
        "strict_view_hash": "6" * 64,
        "full_view_hash": "7" * 64,
        "leakage_mask_hash": "8" * 64,
        "config_hash": "9" * 64,
        "amendment_config_hash": "a" * 64,
        "runner_sha256": "b" * 64,
        "model_adapter_sha256": "c" * 64,
        "code_hash": "d" * 64,
        "diagnostic_ap": 1.0,
        "diagnostic_normalized_ap": 1.0,
        "top5_recall": 1.0,
    }
    frame = pd.DataFrame([values])
    tasks = pd.DataFrame([{
        "dataset_id": "panel_00",
        "mechanism": "M10",
        "strength": "S1",
        "seed": 13,
        "task_hash": "2" * 64,
        "split_hash": "1" * 64,
        "diagnostic_ap": 1.0,
        "diagnostic_normalized_ap": 1.0,
        "top5_recall": 1.0,
        "n_leak": 1,
        "bundle_path": "results/corrected_v2/task_bundles/panel_00.npz",
        "bundle_sha256": "5" * 64,
    }])
    kwargs = {
        "label": "CPU",
        "expected_cells": 1,
        "expected_models": {"lr"},
        "expected_config_hash": "9" * 64,
        "expected_amendment_config_hash": "a" * 64,
        "expected_manifest_hash": "3" * 64,
        "expected_summary_hash": "4" * 64,
        "expected_runner_hash": "b" * 64,
        "expected_adapter_hash": "c" * 64,
        "expected_code_hash": "d" * 64,
    }
    canonical_builder._validate_m10_amendment_rows(frame, tasks, **kwargs)
    old_semantics = frame.copy()
    old_semantics.loc[0, "clean_auc"] = 0.6
    with pytest.raises(ValueError, match="clean_auc is not the amended strict_auc"):
        canonical_builder._validate_m10_amendment_rows(old_semantics, tasks, **kwargs)
