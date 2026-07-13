"""Contract tests for immutable-bundle official TabM execution."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.leakbench import run_corrected_tabm_bundle as runner


def _write_contract(root, *, task_hash=None, bundle_hash=None, namespace="pilot"):
    config = {
        "protocol": {
            "version": "corrected_v2",
            "dataset_namespace": "confirmatory",
            "mechanisms": ["M01"],
            "strengths": ["S1"],
            "seeds": [13],
            "predictions_retained_for": ["M08", "M09"],
        }
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (root / "src/leakbench/models").mkdir(parents=True)
    (root / "src/leakbench/models/official_tabm.py").write_text(
        "# test code-hash fixture\n", encoding="utf-8"
    )
    runner_fixture = root / "experiments/leakbench/run_corrected_tabm_bundle.py"
    runner_fixture.parent.mkdir(parents=True)
    runner_fixture.write_text("# test code-hash fixture\n", encoding="utf-8")

    base_X = np.arange(20, dtype=np.float32).reshape(10, 2)
    block = np.linspace(0.0, 1.0, 10, dtype=np.float32).reshape(10, 1)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32)
    train_idx = np.arange(6, dtype=np.int64)
    val_idx = np.arange(6, 8, dtype=np.int64)
    test_idx = np.arange(8, 10, dtype=np.int64)
    leakage_mask = np.array([False, False, True])
    entity_ids = np.arange(10, dtype=np.int64)
    source_ids = np.arange(10, dtype=np.int64) % 2
    frozen = runner.FrozenTask(
        base_X=base_X,
        X=np.concatenate((base_X, block), axis=1),
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        leakage_mask=leakage_mask,
        entity_ids=entity_ids,
        source_ids=source_ids,
    )

    bundle_dir = root / "bundles"
    bundle_dir.mkdir()
    bundle_path = bundle_dir / "panel_00.npz"
    key = "M01_S1_13"
    np.savez_compressed(
        bundle_path,
        base_X=base_X,
        y=y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        block__M01_S1_13=block,
        leak_mask__M01_S1_13=leakage_mask,
        entity_ids__M01_S1_13=entity_ids,
        source_ids__M01_S1_13=source_ids,
    )
    actual_bundle_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    row = {
        "dataset_id": "panel_00",
        "dataset_index": 0,
        "dataset_namespace": namespace,
        "dataset_seed": 30260732,
        "archetype": "linear",
        "mechanism": "M01",
        "strength": "S1",
        "strength_value": 0.2,
        "seed": 13,
        "bundle_key": key,
        "task_hash": task_hash or runner.task_sha256(frozen),
        "split_hash": hashlib.sha256(test_idx.tobytes()).hexdigest(),
        "n_samples": 10,
        "n_original": 2,
        "n_injected": 1,
        "n_leak": 1,
        "diagnostic_ap": 0.123,
        "diagnostic_normalized_ap": 0.456,
        "top5_recall": 0.789,
        "bundle_path": "bundles/panel_00.npz",
        "bundle_sha256": bundle_hash or actual_bundle_hash,
    }
    manifest_path = bundle_dir / "task_manifest.csv"
    pd.DataFrame([row]).to_csv(manifest_path, index=False)
    summary = {
        "schema_version": 1,
        "dataset_namespace": namespace,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "task_count": 1,
        "datasets": [0],
        "mechanisms": ["M01"],
        "strengths": ["S1"],
        "seeds": [13],
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (bundle_dir / "bundle_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return config_path, manifest_path, row


def _fake_output(probabilities):
    return SimpleNamespace(
        probabilities=np.asarray(probabilities, dtype=float),
        runtime_sec=0.1,
        implementation="tabm.TabM@0.0.3",
        best_epoch=1,
        manifest={"model_class": "tabm.TabM"},
    )


def test_bundle_runner_has_no_generator_dependency():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "build_panel_task" not in referenced_names
    assert "LeakBenchInjector" not in referenced_names
    assert "mutual_info_classif" not in referenced_names


def test_bundle_sha256_mismatch_fails_closed(tmp_path):
    _, manifest_path, _ = _write_contract(tmp_path, bundle_hash="0" * 64)
    row = pd.read_csv(manifest_path).iloc[0]
    with pytest.raises(RuntimeError, match="Bundle SHA256 mismatch"):
        runner.load_verified_task(row, tmp_path)


def test_task_hash_mismatch_blocks_fit_and_preserves_nan_failure(monkeypatch, tmp_path):
    config_path, manifest_path, _ = _write_contract(tmp_path, task_hash="f" * 64)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "__file__", str(tmp_path / "experiments/leakbench/run_corrected_tabm_bundle.py")
    )
    fit_calls = []
    monkeypatch.setattr(
        runner,
        "fit_predict_official_tabm",
        lambda *args, **kwargs: fit_calls.append((args, kwargs)),
    )
    output = tmp_path / "results/failure.csv"
    exit_code = runner.main([
        "--config", str(config_path),
        "--task-manifest", str(manifest_path),
        "--output", str(output),
        "--namespace", "pilot",
        "--device", "cpu",
    ])
    result = pd.read_csv(output).iloc[0]
    assert exit_code == 1
    assert fit_calls == []
    assert result["status"] == "FAILURE"
    assert "Reconstructed task SHA256 mismatch" in result["failure_reason"]
    assert pd.isna(result["clean_auc"])
    assert pd.isna(result["full_auc"])
    assert pd.isna(result["paired_harm"])


def test_success_uses_manifest_diagnostics_and_records_verified_hashes(monkeypatch, tmp_path):
    config_path, manifest_path, expected = _write_contract(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "__file__", str(tmp_path / "experiments/leakbench/run_corrected_tabm_bundle.py")
    )
    outputs = iter((_fake_output([0.1, 0.9]), _fake_output([0.2, 0.8])))
    monkeypatch.setattr(
        runner,
        "fit_predict_official_tabm",
        lambda *args, **kwargs: next(outputs),
    )
    output = tmp_path / "results/success.csv"
    exit_code = runner.main([
        "--config", str(config_path),
        "--task-manifest", str(manifest_path),
        "--output", str(output),
        "--namespace", "pilot",
        "--device", "cpu",
        "--max-epochs", "1",
        "--patience", "1",
    ])
    result = pd.read_csv(output).iloc[0]
    assert exit_code == 0
    assert result["status"] == "SUCCESS"
    assert result["diagnostic_ap"] == pytest.approx(0.123)
    assert result["diagnostic_normalized_ap"] == pytest.approx(0.456)
    assert result["top5_recall"] == pytest.approx(0.789)
    assert result["task_hash"] == expected["task_hash"]
    assert result["bundle_sha256"] == expected["bundle_sha256"]
    assert bool(result["integrity_verified"])


def test_bundle_confirmatory_namespace_requires_explicit_gate(monkeypatch, tmp_path):
    config_path, _, _ = _write_contract(tmp_path, namespace="confirmatory")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "results/must_not_exist.csv"
    with pytest.raises(RuntimeError, match="Confirmatory TabM bundle execution is locked"):
        runner.main([
            "--config", str(config_path),
            "--namespace", "confirmatory",
            "--output", str(output),
        ])
    assert not output.exists()
