"""ModernNCA adapter for SP6 (official vendored model, thin wrapper).

The adapter owns preprocessing, training loop, early stopping, and
predict_proba. It NEVER modifies the official ModernNCA forward, never lets
test rows enter candidate memory, and keeps strict/full pipelines separate.
"""
from __future__ import annotations
import hashlib
import sys
from pathlib import Path
import numpy as np

_TP = Path(__file__).resolve().parents[3] / "third_party/modernnca"
if str(_TP) not in sys.path:
    sys.path.insert(0, str(_TP))


def _import_official():
    from modernNCA import ModernNCA  # noqa: official vendored, byte-identical
    return ModernNCA


class ModernNCAAdapter:
    """Train-only-candidate ModernNCA. All numeric features (LeakBench cells
    have no categorical columns in the injected panels)."""

    name = "modernnca"

    def __init__(self, config: dict):
        self.cfg = config
        self.model = None
        self._mu = None
        self._sd = None
        self._cand_x = None
        self._cand_y = None
        self._meta = {}

    # ---- preprocessing fit on TRAIN ONLY ----
    def _fit_scaler(self, X):
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0) + 1e-8

    def _scale(self, X):
        return (X - self._mu) / self._sd

    def fit(self, X_train, y_train, X_valid, y_valid, *, categorical_indices, seed, device):
        import torch
        ModernNCA = _import_official()
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no CPU fallback")

        self._fit_scaler(X_train)  # TRAIN ONLY
        Xtr = self._scale(X_train).astype(np.float32)
        Xva = self._scale(X_valid).astype(np.float32)
        d_num = Xtr.shape[1]  # all numeric
        mc = self.cfg["model"]
        self.model = ModernNCA(
            d_in=d_num, d_num=d_num, d_out=2, dim=mc["dim"], dropout=mc["dropout"],
            d_block=mc["d_block"], n_blocks=mc["n_blocks"],
            num_embeddings=mc.get("num_embeddings"),
            temperature=mc["temperature"], sample_rate=mc["sample_rate"],
        ).to(dev)

        xtr = torch.as_tensor(Xtr, device=dev)
        ytr = torch.as_tensor(y_train.astype(np.int64), device=dev)
        xva = torch.as_tensor(Xva, device=dev)
        # candidate memory = TRAIN ONLY (never valid/test)
        self._cand_x = xtr
        self._cand_y = ytr
        self._cand_row_ids = np.arange(len(X_train))

        opt = torch.optim.AdamW(self.model.parameters(),
                                lr=self.cfg["training"]["lr"],
                                weight_decay=self.cfg["training"]["weight_decay"])
        loss_fn = torch.nn.NLLLoss()
        best_auc, best_state, patience = -1.0, None, 0
        max_epochs = self.cfg["training"]["max_epochs"]
        pat_limit = self.cfg["training"]["patience"]
        bs = min(self.cfg["training"]["batch_size"], len(xtr))
        from sklearn.metrics import roc_auc_score

        for epoch in range(max_epochs):
            self.model.train()
            perm = torch.randperm(len(xtr), device=dev)
            for i in range(0, len(xtr), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                logits = self.model(xtr[idx], ytr[idx], self._cand_x, self._cand_y, is_train=True)
                loss = loss_fn(logits, ytr[idx])
                loss.backward()
                opt.step()
            # validation retrieves from TRAIN candidates only
            self.model.eval()
            with torch.no_grad():
                vlogits = self.model(xva, None, self._cand_x, self._cand_y, is_train=False)
                vproba = torch.exp(vlogits)[:, 1].cpu().numpy()
            try:
                auc = roc_auc_score(y_valid, vproba)
            except ValueError:
                auc = 0.5
            if auc > best_auc + 1e-6:
                best_auc, patience = auc, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                self._best_epoch = epoch
            else:
                patience += 1
                if patience >= pat_limit:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._meta = {"best_val_auc": float(best_auc), "best_epoch": int(getattr(self, "_best_epoch", 0)),
                      "device": str(dev), "candidate_rows": int(len(self._cand_row_ids))}
        return self

    def predict_proba(self, X):
        import torch
        dev = next(self.model.parameters()).device
        Xs = self._scale(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.as_tensor(Xs, device=dev), None,
                                self._cand_x, self._cand_y, is_train=False)
            proba = torch.exp(logits).cpu().numpy()
        proba = np.clip(proba, 0.0, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba

    def candidate_row_ids_hash(self):
        return hashlib.sha256(np.ascontiguousarray(self._cand_row_ids).tobytes()).hexdigest()

    def candidate_tensor_hash(self):
        import torch
        return hashlib.sha256(self._cand_x.detach().cpu().numpy().tobytes()).hexdigest()

    def runtime_metadata(self):
        return dict(self._meta)
