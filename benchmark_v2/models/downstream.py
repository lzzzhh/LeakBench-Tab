"""benchmark_v2/models/downstream.py — CatBoost + Logistic Regression trainers."""
from __future__ import annotations
import numpy as np, time
from dataclasses import dataclass

@dataclass
class ModelResult:
    model_name: str; auc: float = 0.5; pr_auc: float = 0.5
    log_loss: float = 0.0; brier: float = 0.0
    runtime_sec: float = 0.0; seed: int = 42; n_features: int = 0

def train_evaluate_catboost(X_train, y_train, X_val, y_val, X_test, y_test,
    iterations=300, seed=42):
    from catboost import CatBoostClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
    t0 = time.time()
    m = CatBoostClassifier(iterations=iterations, learning_rate=0.05, depth=6,
        random_seed=seed, eval_metric='AUC', early_stopping_rounds=30,
        verbose=False, task_type='CPU', thread_count=4)
    m.fit(X_train, y_train, eval_set=(X_val, y_val))
    proba = m.predict_proba(X_test)[:,1]
    return ModelResult(model_name="CatBoost", auc=float(roc_auc_score(y_test,proba)),
        pr_auc=float(average_precision_score(y_test,proba)),
        log_loss=float(log_loss(y_test,proba)), brier=float(brier_score_loss(y_test,proba)),
        runtime_sec=time.time()-t0, seed=seed, n_features=X_train.shape[1])

def train_evaluate_lr(X_train, y_train, X_val, y_val, X_test, y_test, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
    t0 = time.time()
    m = LogisticRegression(max_iter=1000, random_state=seed)
    m.fit(X_train, y_train)
    proba = m.predict_proba(X_test)[:,1]
    return ModelResult(model_name="LogisticRegression", auc=float(roc_auc_score(y_test,proba)),
        pr_auc=float(average_precision_score(y_test,proba)),
        log_loss=float(log_loss(y_test,proba)), brier=float(brier_score_loss(y_test,proba)),
        runtime_sec=time.time()-t0, seed=seed, n_features=X_train.shape[1])

def compute_utility_metrics(results):
    clean = results.get("clean"); method = results.get("method")
    if clean is None or method is None: return {}
    return {"residual_inflation": max(0, method.auc - clean.auc),
            "utility_damage": max(0, clean.auc - method.auc),
            "auc_delta": method.auc - clean.auc}


def train_evaluate_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test,
    seed=42, n_estimators=300, learning_rate=0.05, max_depth=6,
    num_leaves=31, min_child_samples=20):
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
    t0 = time.time()
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    params = {'objective':'binary','metric':'auc','boosting_type':'gbdt',
              'num_leaves':num_leaves,'learning_rate':learning_rate,
              'max_depth':max_depth,'min_child_samples':min_child_samples,
              'seed':seed,'verbose':-1,'n_jobs':-1}
    model = lgb.train(params, dtrain, num_boost_round=n_estimators,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    proba = model.predict(X_test)
    return ModelResult(model_name="LightGBM", auc=float(roc_auc_score(y_test,proba)),
        pr_auc=float(average_precision_score(y_test,proba)),
        log_loss=float(log_loss(y_test,proba)), brier=float(brier_score_loss(y_test,proba)),
        runtime_sec=time.time()-t0, seed=seed, n_features=X_train.shape[1])
