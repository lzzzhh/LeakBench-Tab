#!/usr/bin/env python3
"""T0-B V3 Selectors — pure functions for P2-P6, using only synthetic fixtures.

NEVER receives leak_mask, evaluation labels, or test labels.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance


def score_mi(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """P3: mutual information scores per column."""
    scores = mutual_info_classif(X_train, y_train, random_state=42)
    scores = np.nan_to_num(scores, nan=0.0)
    return scores


def score_point_biserial(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """P4: absolute point-biserial correlation per column."""
    n_cols = X_train.shape[1]
    scores = np.zeros(n_cols)
    for j in range(n_cols):
        col = X_train[:, j]
        if np.std(col) < 1e-15:
            scores[j] = 0.0
            continue
        corr = np.corrcoef(col, y_train)[0, 1]
        if np.isnan(corr) or np.isinf(corr):
            scores[j] = 0.0
        else:
            scores[j] = abs(corr)
    return scores


def score_lr_coef(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """P5: abs(standardized LR coefficient). Scaler fit on training only."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    lr.fit(X_scaled, y_train)
    return np.abs(lr.coef_[0])


def score_rf_permutation(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """P6: 3-fold cross-fitted RF permutation importance.

    Ranking RF: n_estimators=100, random_state=42, n_jobs=1.
    StratifiedKFold(3, shuffle=True, random_state=42).
    n_repeats=5, fold permutation seed = 42 + fold_id.
    """
    n_cols = X_train.shape[1]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    importances = np.zeros((3, n_cols))

    for fold_id, (fit_idx, eval_idx) in enumerate(cv.split(X_train, y_train)):
        X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
        X_eval, y_eval = X_train[eval_idx], y_train[eval_idx]

        rf = RandomForestClassifier(
            n_estimators=100, criterion="gini", max_depth=None,
            min_samples_split=2, min_samples_leaf=1, max_features="sqrt",
            bootstrap=True, class_weight=None, random_state=42, n_jobs=1,
        )
        rf.fit(X_fit, y_fit)

        result = permutation_importance(
            rf, X_eval, y_eval, scoring="roc_auc", n_repeats=5,
            random_state=42 + fold_id, n_jobs=1,
        )
        importances[fold_id] = result.importances_mean

    # Negative importance retained, NaN → 0
    scores = importances.mean(axis=0)
    scores = np.nan_to_num(scores, nan=0.0)
    return scores


def group_max_score(column_scores: np.ndarray, groups: list[dict]) -> list[tuple[str, float]]:
    """Compute group score = max(member column scores)."""
    result = []
    for g in groups:
        member_scores = [column_scores[idx] for idx in g["member_encoded_indices"]]
        result.append((g["opaque_group_id"], float(max(member_scores))))
    return result


def top_k_groups(group_scores: list[tuple[str, float]], k: int) -> list[str]:
    """Select top-k groups: score descending, then neutral group_id ascending."""
    sorted_scores = sorted(group_scores, key=lambda x: (-x[1], x[0]))
    return [gid for gid, _ in sorted_scores[:k]]


def top_k_columns(column_scores: np.ndarray, k: int) -> np.ndarray:
    """Select top-k encoded columns: score descending, then index ascending."""
    order = np.lexsort((np.arange(len(column_scores)), -column_scores))
    return order[:k]
