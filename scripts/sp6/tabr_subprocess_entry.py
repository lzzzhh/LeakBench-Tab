#!/usr/bin/env python3
"""tabr_subprocess_entry.py — runs inside the official tabr-official-sp6 env.

Protocol glue ONLY. Loads an immutable exchange package (X_train/y_train/
X_valid/y_valid/X_test, NO y_test), instantiates the UNMODIFIED official
bin.tabr.Model, trains with the official apply_model retrieval pattern
(train-only candidates), and writes test probabilities. Never receives y_test.

Requires: PROJECT_DIR + PYTHONPATH pointing at the official tabr repo checkout.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from bin.tabr import Model  # official, unmodified


def _sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    pkg = np.load(args.exchange, allow_pickle=False)
    assert "y_test" not in pkg.files, "y_test must NOT be in exchange package"
    Xtr = pkg["X_train"].astype(np.float32); ytr = pkg["y_train"].astype(np.int64)
    Xva = pkg["X_valid"].astype(np.float32); yva = pkg["y_valid"].astype(np.int64)
    Xte = pkg["X_test"].astype(np.float32)
    seed = int(pkg["seed"]); cfg = json.loads(pkg["model_config"].item())

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no CPU fallback")
    import delu
    delu.random.seed(seed)

    d_num = Xtr.shape[1]
    # train-only preprocessing: quantile-standard on train
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    def scale(X): return (X - mu) / sd
    xtr = torch.as_tensor(scale(Xtr), device=dev)
    xva = torch.as_tensor(scale(Xva), device=dev)
    xte = torch.as_tensor(scale(Xte), device=dev)
    Ytr = torch.as_tensor(ytr, device=dev)

    model = Model(
        n_num_features=d_num, n_bin_features=0, cat_cardinalities=[], n_classes=2,
        num_embeddings=None, d_main=cfg["d_main"], d_multiplier=cfg["d_multiplier"],
        encoder_n_blocks=cfg["encoder_n_blocks"], predictor_n_blocks=cfg["predictor_n_blocks"],
        mixer_normalization=cfg["mixer_normalization"], context_dropout=cfg["context_dropout"],
        dropout0=cfg["dropout0"], dropout1=cfg["dropout1"], normalization=cfg["normalization"],
        activation=cfg["activation"], memory_efficient=False, candidate_encoding_batch_size=None,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    context_size = int(cfg["context_size"])
    n_train = len(xtr)
    train_idx_all = torch.arange(n_train, device=dev)

    def apply_train(idx):
        # official pattern: remove batch from candidates; forward re-adds it
        mask = ~torch.isin(train_idx_all, idx)
        cand_x = {"num": xtr[mask]}
        cand_y = Ytr[mask]
        return model(x_={"num": xtr[idx]}, y=Ytr[idx], candidate_x_=cand_x,
                     candidate_y=cand_y, context_size=context_size, is_train=True).squeeze(-1)

    @torch.inference_mode()
    def predict(xq):
        model.eval()
        cand_x = {"num": xtr}; cand_y = Ytr
        out = model(x_={"num": xq}, y=None, candidate_x_=cand_x, candidate_y=cand_y,
                    context_size=context_size, is_train=False).squeeze(-1)
        return out

    from sklearn.metrics import roc_auc_score
    # official get_loss_fn(BINCLASS) = binary_cross_entropy_with_logits; output is [batch] single logit
    loss_fn = F.binary_cross_entropy_with_logits
    bs = min(cfg["batch_size"], n_train)
    best_auc, best_state, patience, best_epoch = -1.0, None, 0, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        for i in range(0, n_train, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = apply_train(idx)
            loss = loss_fn(logits, Ytr[idx].float())
            loss.backward()
            opt.step()
        with torch.inference_mode():
            vp = predict(xva)
            vproba = torch.sigmoid(vp).cpu().numpy()
        try:
            auc = roc_auc_score(yva, vproba)
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-6:
            best_auc, patience, best_epoch = auc, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.inference_mode():
        tp = predict(xte)
        test_proba = torch.sigmoid(tp).cpu().numpy()
    test_proba = np.clip(test_proba, 0.0, 1.0).astype(np.float64)

    np.savez(args.output, test_proba=test_proba,
             candidate_row_ids=np.arange(n_train),
             preprocessor=np.concatenate([mu, sd]),
             best_epoch=np.array([best_epoch]), best_val_auc=np.array([best_auc]),
             device=np.array([str(dev)]))
    meta = {"best_epoch": best_epoch, "best_val_auc": float(best_auc),
            "device": str(dev), "candidate_row_ids_hash": _sha(np.arange(n_train)),
            "preprocessor_hash": _sha(np.concatenate([mu, sd])),
            "test_proba_hash": _sha(test_proba)}
    Path(args.output + ".meta.json").write_text(json.dumps(meta))
    print("TABR_CELL_DONE", json.dumps(meta))


if __name__ == "__main__":
    raise SystemExit(main())
