"""Verified core-model adapters used by the corrected_v2 matrix."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class CoreModelOutput:
    model_id: str
    probabilities: np.ndarray
    runtime_sec: float
    implementation: str


def fit_predict_core_model(model_id, X_train, y_train, X_val, y_val, X_test, seed):
    """Fit one named implementation and return test probabilities.

    All preprocessing is fit on training rows only.  Unknown or unavailable
    models fail explicitly; there is no silent fallback under another identity.
    """
    started = time.time()
    if model_id == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, random_state=seed, C=1.0),
        )
        implementation = "sklearn.StandardScaler+LogisticRegression"
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
    elif model_id == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=1,
        )
        implementation = "sklearn.RandomForestClassifier"
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
    elif model_id == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=250,
            learning_rate=0.05,
            depth=6,
            random_seed=seed,
            eval_metric="AUC",
            early_stopping_rounds=30,
            verbose=False,
            task_type="CPU",
            thread_count=1,
            allow_writing_files=False,
        )
        implementation = "catboost.CatBoostClassifier"
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        probabilities = model.predict_proba(X_test)[:, 1]
    elif model_id == "lightgbm":
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        model = LGBMClassifier(
            objective="binary",
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
        implementation = "lightgbm.LGBMClassifier"
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(30, verbose=False), log_evaluation(0)],
        )
        probabilities = model.booster_.predict(X_test)
    else:
        raise ValueError(f"Unknown verified core model: {model_id}")

    return CoreModelOutput(
        model_id=model_id,
        probabilities=np.asarray(probabilities, dtype=float),
        runtime_sec=time.time() - started,
        implementation=implementation,
    )
