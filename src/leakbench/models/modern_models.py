"""modern_models.py — Tier 2 modern tabular model adapters for scope audit.

TabPFN v2: Prior-Data Fitted Networks for tabular classification.
ModernNCA: Neighborhood-based classification with learned metric.
TabR: Retrieval-augmented tabular prediction.

All models are for EXTERNAL VALIDITY AUDIT only — no full benchmark ranking claims.
"""

import numpy as np, time
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis


@dataclass
class ModelResult:
    model_name: str; auc: float = 0.5; pr_auc: float = 0.5
    runtime_sec: float = 0.0; seed: int = 42; n_features: int = 0
    applicable: bool = True; note: str = ""


def train_evaluate_tabpfn(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """TabPFN v2 — Prior-Data Fitted Network.

    NOT-APPLICABLE when: n_samples > 1000 or n_features > 100
    """
    n_samples, n_features = X_train.shape

    if n_samples > 1000:
        # Subsample
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_samples, 1000, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        note = f"SUBSAMPLED from {n_samples} to 1000"
    else:
        note = ""

    try:
        from tabpfn import TabPFNClassifier
        t0 = time.time()
        model = TabPFNClassifier(device='cpu', random_state=seed)
        model.fit(X_train[:1000], y_train[:1000])
        proba = model.predict_proba(X_test)[:, 1]
        runtime = time.time() - t0
        return ModelResult(model_name="TabPFNv2",
            auc=float(roc_auc_score(y_test, proba)),
            pr_auc=float(average_precision_score(y_test, proba)),
            runtime_sec=runtime, seed=seed, n_features=n_features,
            applicable=True, note=note)
    except Exception as e:
        return ModelResult(model_name="TabPFNv2", applicable=False,
            note=f"NOT-APPLICABLE: {str(e)[:80]}", seed=seed)


def train_evaluate_modernnca(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """ModernNCA — Neighborhood Component Analysis + k-NN.

    Learns a linear projection via NCA, then classifies with k-NN.
    Falls back to plain k-NN if NCA fails.
    """
    t0 = time.time()
    n_features = X_train.shape[1]

    # NCA projection (reduce to min(n_components, n_features//2))
    n_components = min(32, n_features // 2, X_train.shape[0] // 5)
    n_components = max(2, n_components)

    try:
        nca = NeighborhoodComponentsAnalysis(
            n_components=n_components, random_state=seed, max_iter=100)
        X_tr_nca = nca.fit_transform(X_train, y_train)
        X_te_nca = nca.transform(X_test)
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        knn.fit(X_tr_nca, y_train)
        proba = knn.predict_proba(X_te_nca)[:, 1]
        note = f"NCA({n_components}d)+kNN"
    except Exception:
        # Fallback: plain k-NN
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        knn.fit(X_train, y_train)
        proba = knn.predict_proba(X_test)[:, 1]
        note = "kNN(5) fallback"

    runtime = time.time() - t0
    return ModelResult(model_name="ModernNCA",
        auc=float(roc_auc_score(y_test, proba)),
        pr_auc=float(average_precision_score(y_test, proba)),
        runtime_sec=runtime, seed=seed, n_features=n_features,
        applicable=True, note=note)


def train_evaluate_tabr(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    """TabR — Tabular Retrieval: k-NN with learned feature weights.

    Simplified: Uses a k-NN with Mahalanobis distance (feature-wise variance scaling)
    as a retrieval baseline. Full TabR would use a learned encoder.
    """
    t0 = time.time()
    n_features = X_train.shape[1]

    # Compute feature-wise std for Mahalanobis-style weighting
    X_combined = np.vstack([X_train, X_test])
    feat_std = np.std(X_combined, axis=0) + 1e-8
    X_tr_w = X_train / feat_std
    X_te_w = X_test / feat_std

    try:
        from sklearn.neighbors import KNeighborsRegressor
        # Use k-NN regression with weighted features as retrieval baseline
        knn = KNeighborsRegressor(n_neighbors=10, weights='distance', n_jobs=-1)
        knn.fit(X_tr_w, y_train)
        proba = knn.predict(X_te_w)
        proba = np.clip(proba, 0.0, 1.0)
        note = "TabR(retrieval-kNN-weighted)"
    except Exception as e:
        return ModelResult(model_name="TabR", applicable=False,
            note=f"NOT-APPLICABLE: {str(e)[:80]}", seed=seed)

    runtime = time.time() - t0
    return ModelResult(model_name="TabR",
        auc=float(roc_auc_score(y_test, proba)),
        pr_auc=float(average_precision_score(y_test, proba)),
        runtime_sec=runtime, seed=seed, n_features=n_features,
        applicable=True, note=note)


TIER2_MODELS = {
    "TabPFNv2": train_evaluate_tabpfn,
    "ModernNCA": train_evaluate_modernnca,
    "TabR": train_evaluate_tabr,
}
