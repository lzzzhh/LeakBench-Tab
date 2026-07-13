"""Compatibility entry point for the official published TabM implementation.

The former random-input-mask MLP was not TabM and has been removed.  This
module never substitutes another architecture under the TabM name.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from benchmark_v2.models.downstream import ModelResult


class TabMEnsemble:
    """Identity guard for callers that still instantiate the removed proxy."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "TabMEnsemble was an identity-invalid random-mask MLP and has been removed. "
            "Use the pinned official tabm.TabM adapter instead."
        )


def train_evaluate_tabm(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    *,
    n_ensemble=32,
    lr=3e-3,
    epochs=200,
    batch_size=256,
    patience=20,
    seed=42,
    device="cuda",
    hidden_dim=None,
    n_layers=None,
    dropout=None,
):
    """Preserve the legacy call shape while routing only to official TabM."""
    if any(value is not None for value in (hidden_dim, n_layers, dropout)):
        raise ValueError(
            "hidden_dim, n_layers, and dropout belonged to the removed proxy; "
            "configure official TabM through its audited adapter"
        )
    from src.leakbench.models.official_tabm import fit_predict_official_tabm

    output = fit_predict_official_tabm(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        seed=seed,
        device=device,
        k=n_ensemble,
        learning_rate=lr,
        max_epochs=epochs,
        batch_size=batch_size,
        patience=patience,
    )
    target = np.asarray(y_test).reshape(-1)
    probabilities = output.probabilities
    return ModelResult(
        model_name="TabM",
        auc=float(roc_auc_score(target, probabilities)),
        pr_auc=float(average_precision_score(target, probabilities)),
        log_loss=float(log_loss(target, probabilities)),
        brier=float(np.mean((probabilities - target) ** 2)),
        runtime_sec=output.runtime_sec,
        seed=seed,
        n_features=np.asarray(X_train).shape[1],
    )
