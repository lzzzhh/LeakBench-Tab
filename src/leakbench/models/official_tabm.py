"""Auditable adapter for the published ``tabm`` package.

This module deliberately has no substitute implementation.  If the pinned
package or requested accelerator is unavailable, callers receive an exception
and must record a failed cell rather than a neutral-looking score.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import os
import random
import time
from typing import Any

import numpy as np


PINNED_TABM_VERSION = "0.0.3"


@dataclass(frozen=True)
class OfficialTabMOutput:
    probabilities: np.ndarray
    runtime_sec: float
    best_epoch: int
    best_validation_loss: float
    implementation: str
    manifest: dict[str, Any]


def _validate_arrays(X_train, y_train, X_val, y_val, X_test):
    arrays = {
        "X_train": np.asarray(X_train),
        "X_val": np.asarray(X_val),
        "X_test": np.asarray(X_test),
    }
    labels = {
        "y_train": np.asarray(y_train).reshape(-1),
        "y_val": np.asarray(y_val).reshape(-1),
    }
    feature_count = arrays["X_train"].shape[1] if arrays["X_train"].ndim == 2 else None
    for name, values in arrays.items():
        if values.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional array")
        if values.shape[1] != feature_count:
            raise ValueError("train, validation, and test feature counts must match")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    for split, target in (("train", labels["y_train"]), ("validation", labels["y_val"])):
        if len(target) != len(arrays[f"X_{'val' if split == 'validation' else split}"]):
            raise ValueError(f"{split} X and y row counts must match")
        if not set(np.unique(target)).issubset({0, 1}):
            raise ValueError(f"{split} labels must be binary")
    if len(np.unique(labels["y_train"])) != 2:
        raise ValueError("training labels must contain both classes")
    return arrays, labels


def fit_predict_official_tabm(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    *,
    seed: int,
    device: str = "cuda",
    k: int = 32,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    batch_size: int = 256,
    inference_batch_size: int = 1024,
    patience: int = 20,
    min_delta: float = 1e-5,
) -> OfficialTabMOutput:
    """Fit official TabM and return mean-probability ensemble predictions.

    Numerical preprocessing is fitted only on the training rows.  During
    training, BCE is evaluated independently for all ``k`` predictions; only
    inference averages the per-member probabilities.
    """
    if k <= 0 or max_epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("k, max_epochs, batch_size, and patience must be positive")
    arrays, labels = _validate_arrays(X_train, y_train, X_val, y_val, X_test)

    try:
        package_version = importlib.metadata.version("tabm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Official tabm=={PINNED_TABM_VERSION} is required; no fallback is permitted"
        ) from exc
    if package_version != PINNED_TABM_VERSION:
        raise RuntimeError(
            f"Expected tabm=={PINNED_TABM_VERSION}, found {package_version}; "
            "update the experiment lock deliberately before running"
        )

    # This variable must be set before the first CUDA BLAS operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as functional
    from sklearn.preprocessing import StandardScaler
    from tabm import TabM

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is permitted")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(arrays["X_train"]).astype(np.float32, copy=False)
    val_scaled = scaler.transform(arrays["X_val"]).astype(np.float32, copy=False)
    test_scaled = scaler.transform(arrays["X_test"]).astype(np.float32, copy=False)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    started = time.time()
    try:
        train_tensor = torch.as_tensor(train_scaled, device=requested_device)
        train_target = torch.as_tensor(labels["y_train"], dtype=torch.float32, device=requested_device)
        val_tensor = torch.as_tensor(val_scaled, device=requested_device)
        val_target = torch.as_tensor(labels["y_val"], dtype=torch.float32, device=requested_device)

        model = TabM.make(
            n_num_features=train_tensor.shape[1],
            cat_cardinalities=[],
            d_out=1,
            k=k,
        ).to(requested_device)
        model_identity = f"{model.__class__.__module__}.{model.__class__.__name__}"
        if model_identity != "tabm.TabM" or model.k != k:
            raise RuntimeError(f"Unexpected model identity: {model_identity}, k={model.k}")

        optimizer = torch.optim.AdamW(
            model.parameters(), learning_rate, weight_decay=weight_decay
        )
        best_state = None
        best_loss = float("inf")
        best_epoch = -1
        stale_epochs = 0

        for epoch in range(max_epochs):
            model.train()
            permutation = np.random.RandomState(seed + epoch).permutation(len(train_tensor))
            for start in range(0, len(permutation), batch_size):
                indices = torch.as_tensor(
                    permutation[start : start + batch_size], device=requested_device
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_tensor[indices]).squeeze(-1)
                target = train_target[indices, None].expand_as(logits)
                # Mean of member losses.  Never compute loss on mean logits.
                loss = functional.binary_cross_entropy_with_logits(logits, target)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(val_tensor).squeeze(-1)
                val_labels = val_target[:, None].expand_as(val_logits)
                validation_loss = float(
                    functional.binary_cross_entropy_with_logits(val_logits, val_labels).cpu()
                )
            if validation_loss < best_loss - min_delta:
                best_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        if best_state is None:
            raise RuntimeError("Early stopping never produced a finite validation checkpoint")
        model.load_state_dict(best_state)
        model.eval()

        probability_chunks = []
        with torch.no_grad():
            for start in range(0, len(test_scaled), inference_batch_size):
                batch = torch.as_tensor(
                    test_scaled[start : start + inference_batch_size], device=requested_device
                )
                logits = model(batch).squeeze(-1)
                probability_chunks.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
        probabilities = np.concatenate(probability_chunks).astype(float, copy=False)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise RuntimeError("Official TabM produced invalid probabilities")

        if requested_device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(requested_device)
            max_memory_mb = float(torch.cuda.max_memory_allocated(requested_device) / 1024**2)
        else:
            gpu_name = ""
            max_memory_mb = 0.0
        manifest = {
            "schema_version": 1,
            "model_id": "tabm",
            "model_class": model_identity,
            "tabm_version": package_version,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": str(requested_device),
            "gpu_name": gpu_name,
            "max_gpu_memory_mb": max_memory_mb,
            "seed": int(seed),
            "k": int(k),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "max_epochs": int(max_epochs),
            "best_epoch": int(best_epoch),
            "patience": int(patience),
            "batch_size": int(batch_size),
            "preprocessing": "sklearn.StandardScaler fit on train only",
            "training_loss": "mean of per-k BCEWithLogits losses",
            "inference_aggregation": "mean of per-k sigmoid probabilities",
            "deterministic_algorithms": True,
        }
        return OfficialTabMOutput(
            probabilities=probabilities,
            runtime_sec=time.time() - started,
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            implementation=f"tabm.TabM@{package_version}",
            manifest=manifest,
        )
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
