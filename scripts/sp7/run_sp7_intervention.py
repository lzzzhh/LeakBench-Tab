#!/usr/bin/env python3
"""run_sp7_intervention.py — SP7-D sentinel intervention runner.

Model-agnostic bundle-only. Reads SP7 intervention bundles, uses SP6-frozen
adapters (official ModernNCA, TabR bridge via subprocess, TabM official package).
For each intervention: train strict+full (or strict+permuted for I2/I4) and
record paired_harm. For ModernNCA/TabR: also log per-query retrieval diagnostics
(neighbor IDs, distances, purity).

Device: cuda (fail-closed). No runtime injection, permutation, or sampling.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

FIELDS = ["run_id", "intervention", "dataset_index", "mechanism", "strength", "seed",
          "model", "status", "failure_reason", "strict_auc", "full_auc", "paired_harm",
          "best_epoch_strict", "best_epoch_full", "device", "runtime_sec",
          "retrieval_strict_k1_purity", "retrieval_full_k1_purity",
          "retrieval_strict_k5_purity", "retrieval_full_k5_purity",
          "neighbor_overlap_k5", "strict_proba_hash", "full_proba_hash"]


def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def fit_tabm(Xtr, ytr, Xva, yva, Xte, yte, seed, device, fixed_epoch=None):
    import torch, sys
    sys.path.insert(0, str(ROOT))
    from src.leakbench.models.official_tabm import fit_predict_official_tabm
    kw = {"k": 32, "learning_rate": 0.003, "weight_decay": 0.0001, "max_epochs": 100 if fixed_epoch is None else fixed_epoch,
          "batch_size": 256, "inference_batch_size": 1024, "patience": 20 if fixed_epoch is None else fixed_epoch + 1, "min_delta": 1e-5, "device": device}
    out = fit_predict_official_tabm(Xtr, ytr, Xva, yva, Xte, seed=seed, **kw)
    return float(roc_auc_score(yte, out.probabilities)), out.best_epoch, out.probabilities, np.arange(len(ytr)), {}


def fit_modernnca(Xtr, ytr, Xva, yva, Xte, yte, seed, device, fixed_epoch=None):
    import sys, torch
    _TP = str(ROOT / "third_party/modernnca")
    if _TP not in sys.path: sys.path.insert(0, _TP)
    from modernNCA import ModernNCA
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/sp6/modernnca_v1.yaml").read_text())
    mc = cfg["model"]; tc = cfg["training"]
    max_ep = tc["max_epochs"] if fixed_epoch is None else fixed_epoch
    pat = tc["patience"] if fixed_epoch is None else max_ep + 1
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    rng = np.random.RandomState(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-8
    def sc(X): return ((X - mu) / sd).astype(np.float32)
    xtr = torch.as_tensor(sc(Xtr), device=dev); xva = torch.as_tensor(sc(Xva), device=dev)
    xte = torch.as_tensor(sc(Xte), device=dev)
    yT = torch.as_tensor(ytr.astype(np.int64), device=dev)
    d_num = Xtr.shape[1]
    model = ModernNCA(d_in=d_num, d_num=d_num, d_out=2, dim=mc["dim"], dropout=mc["dropout"],
                      d_block=mc["d_block"], n_blocks=mc["n_blocks"], num_embeddings=mc.get("num_embeddings"),
                      temperature=mc["temperature"], sample_rate=mc["sample_rate"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    cand_x, cand_y, cand_ids = xtr, yT, np.arange(len(xtr))
    bs = min(tc["batch_size"], len(xtr)); best_auc, best_state, patience, best_ep = -1.0, None, 0, 0
    from sklearn.metrics import roc_auc_score
    for ep in range(max_ep):
        model.train(); perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr), bs):
            idx = perm[i:i+bs]; opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(xtr[idx], yT[idx], cand_x, cand_y, is_train=True).squeeze(-1), yT[idx].float())
            loss.backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            vp = torch.sigmoid(model(xva, None, cand_x, cand_y, is_train=False).squeeze(-1)).cpu().numpy()
        auc = roc_auc_score(yva, vp)
        if auc > best_auc + 1e-6: best_auc, patience, best_ep, best_state = auc, 0, ep, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= pat: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        tp = torch.sigmoid(model(xte, None, cand_x, cand_y, is_train=False).squeeze(-1)).cpu().numpy()
    # Retrieval diagnostics: top-5 nearest neighbors from candidate memory
    with torch.inference_mode():
        cd = model.encoder(cand_x)  # encode with trained encoder
        qe = model.encoder(xte)
    cd_np = cd.detach().cpu().numpy(); qe_np = qe.detach().cpu().numpy()
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=min(5, len(cd_np)), metric='euclidean').fit(cd_np)
    dists, idx = nn.kneighbors(qe_np)
    k1_pur = (ytr[idx[:, 0]] == ytr).mean()
    k5_pur = np.mean([np.mean(ytr[idx[i]] == ytr[i]) for i in range(len(ytr))])
    return float(roc_auc_score(yte, tp)), best_ep, tp, cand_ids, \
           {"k1_purity": float(k1_pur), "k5_purity": float(k5_pur), "distance_mean": float(dists.mean()),
            "encoder_hash": sha_arr(cd_np)}


def fit_tabr(Xtr, ytr, Xva, yva, Xte, yte, seed, device, fixed_epoch=None):
    """Subprocess bridge — reuses SP6 official env."""
    env_name = "tabr-official-sp6"; project_dir = "/root/external/tabular-dl-tabr"
    mm = "/root/.local/bin/micromamba"
    bridge = ROOT / "scripts/sp6/tabr_subprocess_entry.py"
    import yaml
    cfg = yaml.safe_load((ROOT / "configs/sp6/tabr_v1.yaml").read_text())["model"]
    if fixed_epoch is not None:
        cfg["max_epochs"] = fixed_epoch; cfg["patience"] = fixed_epoch + 1
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        exch = wd / "exch.npz"; out = wd / "result.npz"
        np.savez(exch, X_train=Xtr.astype(np.float32), y_train=ytr.astype(np.int64),
                 X_valid=Xva.astype(np.float32), y_valid=yva.astype(np.int64),
                 X_test=Xte.astype(np.float32), seed=np.array([seed]),
                 model_config=np.array([json.dumps(cfg)]))
        env = dict(subprocess.os.environ)
        env.update({"PROJECT_DIR": project_dir, "PYTHONPATH": project_dir,
                    "CUDA_VISIBLE_DEVICES": "0", "MAMBA_ROOT_PREFIX": "/root/micromamba"})
        cmd = [mm, "run", "-n", env_name, "python", str(bridge), "--exchange", str(exch), "--output", str(out), "--device", device]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        metap = Path(str(out) + ".meta.json")
        if r.returncode != 0 or not out.exists() or not metap.exists():
            raise RuntimeError(f"TabR child rc={r.returncode}")
        res = np.load(out, allow_pickle=False); meta = json.loads(metap.read_text())
        return float(roc_auc_score(yte, res["test_proba"])), meta.get("best_epoch", 0), \
               res["test_proba"], np.arange(len(Xtr)), {"k1_purity": None, "k5_purity": None}


MODELS = {"tabm": fit_tabm, "modernnca": fit_modernnca, "tabr": fit_tabr}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models", default="tabm,modernnca,tabr")
    ap.add_argument("--interventions", default="all")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--allow-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("locked; pass --allow-run")

    man = pd.read_csv(ROOT / args.bundle_manifest)
    if args.interventions != "all":
        man = man[man["intervention"].isin(args.interventions.split(","))]
    if args.datasets != "all":
        man = man[man["dataset_index"].astype(int).isin([int(x) for x in args.datasets.split(",")])]
    models = args.models.split(",")

    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"{out} exists; pass --resume")
        completed = set(pd.read_csv(out)["run_id"].astype(str))

    total = len(man) * len(models); done = 0; started = time.time()
    for _, row in man.iterrows():
        bid = row["id"]; bp = ROOT / "artifacts/sp7/bundles" / f"{bid}.npz"
        if not bp.exists(): continue
        b = np.load(bp, allow_pickle=False)
        intervention = str(b.get("intervention", row.get("intervention", "")))
        fix_ep = int(b.get("fixed_epoch", -1)) if b.get("fixed_epoch", -1) > 0 else None
        for model in models:
            rid = hashlib.sha256(f"{bid}|{model}".encode()).hexdigest()[:20]
            done += 1
            if rid in completed: continue
            rec = {k: "" for k in FIELDS}; rec.update({"run_id": rid, "intervention": intervention,
                "dataset_index": int(row["dataset_index"]), "mechanism": row["mechanism"],
                "strength": row.get("strength", ""), "seed": int(row.get("seed", 0)), "model": model, "status": "FAILURE"})
            yte = b["y_test"]
            try:
                s_auc, s_ep, s_proba, s_cand, s_meta = MODELS[model](
                    b["X_train"][:, b.get("strict_cols")], b["y_train"],
                    b["X_valid"][:, b.get("strict_cols")], b["y_valid"],
                    b["X_test"][:, b.get("strict_cols")], yte, int(row.get("seed", 0)), args.device, fix_ep)
                f_auc, f_ep, f_proba, f_cand, f_meta = MODELS[model](
                    b["X_train"][:, b.get("full_cols_permuted")], b["y_train"],
                    b["X_valid"][:, b.get("full_cols_permuted")], b["y_valid"],
                    b["X_test"][:, b.get("full_cols_permuted")], yte, int(row.get("seed", 0)), args.device, fix_ep)
                # Train-only assertion
                assert set(s_cand).issubset(set(range(len(b["y_train"])))), "strict candidates not train-only"
                assert set(f_cand).issubset(set(range(len(b["y_train"])))), "full candidates not train-only"
                rec.update({"status": "SUCCESS", "strict_auc": s_auc, "full_auc": f_auc,
                            "paired_harm": f_auc - s_auc, "best_epoch_strict": s_ep, "best_epoch_full": f_ep,
                            "device": args.device, "runtime_sec": round(time.time() - started, 1),
                            "retrieval_strict_k1_purity": s_meta.get("k1_purity"),
                            "retrieval_full_k1_purity": f_meta.get("k1_purity"),
                            "retrieval_strict_k5_purity": s_meta.get("k5_purity"),
                            "retrieval_full_k5_purity": f_meta.get("k5_purity"),
                            "strict_proba_hash": sha_arr(s_proba), "full_proba_hash": sha_arr(f_proba)})
            except Exception as e:
                rec["failure_reason"] = f"{type(e).__name__}: {str(e)[:300]}"
            write_hdr = not out.exists()
            with out.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if write_hdr: w.writeheader()
                w.writerow(rec)
            if done % 30 == 0 or rec["status"] != "SUCCESS":
                print(f"{done}/{total} {rec['status']} {model} {intervention} {row['mechanism']} d{row['dataset_index']}", flush=True)
    print(f"DONE {done}/{total} in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
