#!/usr/bin/env python3
"""run_sp6_tabr_bridge.py — main-env orchestrator for official TabR (SP6).

Reads frozen SP5 bundles, builds per-cell immutable exchange packages WITHOUT
y_test, invokes tabr_subprocess_entry.py inside the official micromamba env,
receives test probabilities, and computes paired_harm using y_test kept ONLY in
the main env. Strict and full views run as separate subprocess calls.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]

RESULT_FIELDS = [
    "run_id", "dataset_index", "mechanism", "strength", "seed", "model", "status",
    "failure_reason", "strict_auc", "full_auc", "paired_harm", "n_original",
    "n_injected", "n_leak", "task_hash", "split_hash", "strict_view_hash",
    "full_view_hash", "strict_preprocessor_hash", "full_preprocessor_hash",
    "strict_candidate_row_ids_hash", "full_candidate_row_ids_hash",
    "child_received_y_test", "bundle_source", "bundle_sha256", "runner_hash",
    "bridge_hash", "config_hash", "device", "best_epoch_strict", "best_epoch_full",
    "runtime_sec",
]


def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_cell(row):
    bundle = ROOT / row["bundle_path"]
    if sha_file(bundle) != str(row["bundle_sha256"]).lower():
        raise RuntimeError("bundle SHA256 mismatch")
    key = str(row["bundle_key"])
    with np.load(bundle, allow_pickle=False) as b:
        base_X = np.asarray(b["base_X"]); y = np.asarray(b["y"])
        tr, va, te = np.asarray(b["train_idx"]), np.asarray(b["val_idx"]), np.asarray(b["test_idx"])
        block = np.asarray(b[f"block__{key}"]); mask = np.asarray(b[f"leak_mask__{key}"])
    X = np.concatenate((base_X, block), axis=1)
    if hashlib.sha256(te.tobytes()).hexdigest() != str(row["split_hash"]):
        raise RuntimeError("split hash mismatch")
    return X, y, tr, va, te, mask


def run_view(X, y, tr, va, te, cols, seed, cfg, mm, env_name, project_dir, bridge, device, workdir):
    Xtr, Xva, Xte = X[tr][:, cols], X[va][:, cols], X[te][:, cols]
    ytr, yva, yte = y[tr], y[va], y[te]
    exch = workdir / "exchange.npz"
    np.savez(exch, X_train=Xtr, y_train=ytr, X_valid=Xva, y_valid=yva, X_test=Xte,
             seed=np.array([seed]), model_config=np.array([json.dumps(cfg)]))
    out = workdir / "result.npz"
    cmd = [mm, "run", "-n", env_name, "python", str(bridge),
           "--exchange", str(exch), "--output", str(out), "--device", device]
    import os
    full_env = dict(os.environ)
    full_env.update({"PROJECT_DIR": project_dir, "PYTHONPATH": project_dir,
                     "CUDA_VISIBLE_DEVICES": "0", "MAMBA_ROOT_PREFIX": "/root/micromamba"})
    r = subprocess.run(cmd, capture_output=True, text=True, env=full_env, timeout=1800)
    metap = Path(str(out) + ".meta.json")
    if r.returncode != 0 or not out.exists() or not metap.exists():
        raise RuntimeError(f"tabr child rc={r.returncode} out={out.exists()} meta={metap.exists()} STDERR[{r.stderr[-1000:]}]")
    res = np.load(out, allow_pickle=False)
    proba = res["test_proba"]
    auc = float(roc_auc_score(yte, proba))  # y_test used ONLY here in main env
    meta = json.loads(metap.read_text())
    exch.unlink(); out.unlink(); metap.unlink()
    return auc, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--micromamba", default="/root/.local/bin/micromamba")
    ap.add_argument("--env-name", default="tabr-official-sp6")
    ap.add_argument("--project-dir", default="/root/external/tabular-dl-tabr")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--mechanisms", default="all")
    ap.add_argument("--strengths", default="all")
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("locked; pass --allow-run")

    import yaml
    cfg = yaml.safe_load((ROOT / args.config).read_text())["model"]
    config_hash = sha_file(ROOT / args.config)
    bridge = ROOT / "scripts/sp6/tabr_subprocess_entry.py"
    bridge_hash = sha_file(bridge)
    runner_hash = sha_file(Path(__file__))

    man = pd.read_csv(ROOT / args.bundle_manifest)
    if args.datasets != "all":
        man = man[man["dataset_index"].astype(int).isin([int(x) for x in args.datasets.split(",")])]
    if args.mechanisms != "all":
        man = man[man["mechanism"].isin(args.mechanisms.split(","))]
    if args.strengths != "all":
        man = man[man["strength"].isin(args.strengths.split(","))]
    if args.seeds != "all":
        man = man[man["seed"].astype(int).isin([int(x) for x in args.seeds.split(",")])]

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        if not args.resume:
            raise FileExistsError(f"{out} exists; pass --resume")
        completed = set(pd.read_csv(out)["run_id"].astype(str))

    total = len(man); done = 0; started = time.time()
    for _, row in man.iterrows():
        rid_key = f"tabr|{row['dataset_index']}|{row['mechanism']}|{row['strength']}|{row['seed']}|{row['task_hash']}|{config_hash}|{bridge_hash}|{runner_hash}"
        run_id = hashlib.sha256(rid_key.encode()).hexdigest()[:20]
        done += 1
        if run_id in completed:
            continue
        rec = {k: "" for k in RESULT_FIELDS}
        rec.update({"run_id": run_id, "dataset_index": int(row["dataset_index"]),
                    "mechanism": row["mechanism"], "strength": row["strength"],
                    "seed": int(row["seed"]), "model": "tabr", "status": "FAILURE",
                    "n_original": int(row["n_original"]), "n_injected": int(row["n_injected"]),
                    "n_leak": int(row["n_leak"]), "task_hash": row["task_hash"],
                    "split_hash": row["split_hash"], "bundle_source": row["bundle_source"],
                    "bundle_sha256": row["bundle_sha256"], "runner_hash": runner_hash,
                    "bridge_hash": bridge_hash, "config_hash": config_hash,
                    "device": args.device, "child_received_y_test": "NO"})
        try:
            X, y, tr, va, te, mask = load_cell(row)
            strict_cols = np.where(~mask)[0]; full_cols = np.arange(X.shape[1])
            rec["strict_view_hash"] = sha_arr(X[:, strict_cols])
            rec["full_view_hash"] = sha_arr(X)
            with tempfile.TemporaryDirectory() as td:
                sd = Path(td) / "strict"; sd.mkdir()
                fd = Path(td) / "full"; fd.mkdir()
                s_auc, s_meta = run_view(X, y, tr, va, te, strict_cols, int(row["seed"]), cfg,
                                         args.micromamba, args.env_name, args.project_dir, bridge, args.device, sd)
                f_auc, f_meta = run_view(X, y, tr, va, te, full_cols, int(row["seed"]), cfg,
                                         args.micromamba, args.env_name, args.project_dir, bridge, args.device, fd)
            rec.update({"status": "SUCCESS", "strict_auc": s_auc, "full_auc": f_auc,
                        "paired_harm": f_auc - s_auc,
                        "strict_preprocessor_hash": s_meta["preprocessor_hash"],
                        "full_preprocessor_hash": f_meta["preprocessor_hash"],
                        "strict_candidate_row_ids_hash": s_meta["candidate_row_ids_hash"],
                        "full_candidate_row_ids_hash": f_meta["candidate_row_ids_hash"],
                        "best_epoch_strict": s_meta["best_epoch"], "best_epoch_full": f_meta["best_epoch"],
                        "runtime_sec": round(time.time() - started, 1)})
        except Exception as e:
            rec["failure_reason"] = f"{type(e).__name__}: {str(e)[:300]}"
        write_header = not out.exists()
        with out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
            if write_header: w.writeheader()
            w.writerow(rec)
        if done % 10 == 0 or rec["status"] != "SUCCESS":
            print(f"{done}/{total} {rec['status']} {row['mechanism']} d{row['dataset_index']} {row['strength']} s{row['seed']}", flush=True)
    print(f"DONE {done}/{total} in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
