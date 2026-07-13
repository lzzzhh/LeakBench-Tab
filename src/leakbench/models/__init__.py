"""models.py — Model adapters for LeakBench-Tab."""
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class ModelResult:
    model_name: str; auc: float = 0.5; pr_auc: float = 0.5
    log_loss: float = 0.0; brier: float = 0.0
    runtime_sec: float = 0.0; seed: int = 42; n_features: int = 0


def train_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test, seed=42,
                       n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    t0 = time.time()
    m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=min_samples_leaf, max_features="sqrt", class_weight="balanced",
        n_jobs=n_jobs, random_state=seed)
    m.fit(X_train, y_train)
    proba = m.predict_proba(X_test)[:, 1]
    return ModelResult(model_name="RandomForest",
        auc=float(roc_auc_score(y_test, proba)),
        pr_auc=float(average_precision_score(y_test, proba)),
        runtime_sec=time.time()-t0, seed=seed, n_features=X_train.shape[1])


def train_evaluate_mlp(X_train, y_train, X_val, y_val, X_test, y_test, seed=42,
                        hidden_dims=None, lr=1e-3, batch_size=1024, max_epochs=100,
                        patience=10, device="cpu"):
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, average_precision_score
    if hidden_dims is None: hidden_dims = [256, 128]
    torch.manual_seed(seed); dev = torch.device(device)
    layers = []; in_dim = X_train.shape[1]
    for h in hidden_dims:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1)])
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers).to(dev)
    Xt = torch.FloatTensor(X_train.astype(np.float32)).to(dev)
    yt = torch.FloatTensor(y_train.reshape(-1,1).astype(np.float32)).to(dev)
    Xv = torch.FloatTensor(X_val.astype(np.float32)).to(dev)
    yv = torch.FloatTensor(y_val.reshape(-1,1).astype(np.float32)).to(dev)
    Xs = torch.FloatTensor(X_test.astype(np.float32)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    t0 = time.time(); best_vl = float("inf"); best_st = None; ni = 0
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad(); crit(model(Xt[idx]), yt[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = crit(model(Xv), yv).item()
        if vl < best_vl - 1e-6: best_vl = vl; best_st = {k:v.clone() for k,v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= patience: break
    if best_st: model.load_state_dict(best_st)
    model.eval()
    with torch.no_grad(): proba = torch.sigmoid(model(Xs)).cpu().numpy().flatten()
    return ModelResult(model_name="MLP", auc=float(roc_auc_score(y_test, proba)),
        pr_auc=float(average_precision_score(y_test, proba)),
        runtime_sec=time.time()-t0, seed=seed, n_features=X_train.shape[1])
