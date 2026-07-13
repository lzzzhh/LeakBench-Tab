"""capacity.py — Model capacity vs leakage exploitation diagnostics."""
import numpy as np
import time
from dataclasses import dataclass, field


@dataclass
class CapacityResult:
    model_name: str; mechanism_id: str; strength: float; n_leakage: int
    n_total_features: int; clean_auc: float = 0.5; leak_auc: float = 0.5
    inflation: float = 0.0; exploitation_ratio: float = 0.0
    runtime_sec: float = 0.0; seed: int = 42; n_samples: int = 0


@dataclass
class CapacityProfile:
    model_name: str; results: list[CapacityResult] = field(default_factory=list)


def evaluate_model_capacity(task, train_fn, model_name, mechanism_id, strength, n_leakage, seed=42):
    X, y = task.X, task.y
    tr, va, te = task.train_idx, task.val_idx, task.test_idx
    legit_idx = np.where(task.legitimate_mask)[0]
    use_leak = np.where(task.leakage_mask)[0][:n_leakage]

    t0 = time.time()
    try:
        if len(legit_idx) > 0:
            cr = train_fn(X[tr][:, legit_idx], y[tr], X[va][:, legit_idx], y[va], X[te][:, legit_idx], y[te], seed=seed)
            clean_auc = cr.auc
        else: clean_auc = 0.5
        full_idx = np.concatenate([legit_idx, use_leak])
        fr = train_fn(X[tr][:, full_idx], y[tr], X[va][:, full_idx], y[va], X[te][:, full_idx], y[te], seed=seed)
        leak_auc = fr.auc
    except Exception:
        clean_auc = 0.5; leak_auc = 0.5

    inflation = leak_auc - clean_auc
    return CapacityResult(model_name=model_name, mechanism_id=mechanism_id, strength=strength,
        n_leakage=n_leakage, n_total_features=task.X.shape[1], clean_auc=clean_auc,
        leak_auc=leak_auc, inflation=inflation,
        exploitation_ratio=min(1.0, inflation / max(0.001, 1.0 - clean_auc)),
        runtime_sec=time.time()-t0, seed=seed, n_samples=len(y))
