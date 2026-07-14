#!/usr/bin/env python3
"""run_sp6_bundle_model.py — model-agnostic, bundle-only SP6 runner.

Reads frozen SP5 bundles (via sp6_bundle_manifest.csv), constructs strict/full
views, calls a model adapter with TRAIN-ONLY candidate memory, and records
paired_harm = full_auc - strict_auc. NEVER injects, splits, or regenerates.
Strict and full pipelines are fully isolated per cell.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

RESULT_FIELDS = [
    "run_id", "dataset_index", "mechanism", "strength", "seed", "model", "status",
    "failure_reason", "strict_auc", "full_auc", "paired_harm", "n_original",
    "n_injected", "n_leak", "task_hash", "split_hash", "strict_view_hash",
    "full_view_hash", "strict_preprocessor_hash", "full_preprocessor_hash",
    "strict_candidate_row_ids_hash", "full_candidate_row_ids_hash",
    "strict_candidate_tensor_hash", "full_candidate_tensor_hash",
    "bundle_source", "bundle_sha256", "runner_hash", "adapter_hash",
    "upstream_model_hash", "config_hash", "device", "best_epoch_strict",
    "best_epoch_full", "runtime_sec",
]


def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_adapter(model, config):
    if model == "modernnca":
        from src.sp6.adapters.modernnca_adapter import ModernNCAAdapter
        return ModernNCAAdapter(config)
    raise ValueError(f"unknown model {model}")


def load_cell(row):
    bundle = ROOT / row["bundle_path"]
    if sha_file(bundle) != str(row["bundle_sha256"]).lower():
        raise RuntimeError("bundle SHA256 mismatch")
    key = str(row["bundle_key"])
    with np.load(bundle, allow_pickle=False) as b:
        base_X = np.asarray(b["base_X"]); y = np.asarray(b["y"])
        tr, va, te = np.asarray(b["train_idx"]), np.asarray(b["val_idx"]), np.asarray(b["test_idx"])
        block = np.asarray(b[f"block__{key}"])
        mask = np.asarray(b[f"leak_mask__{key}"])
    X = np.concatenate((base_X, block), axis=1)
    if hashlib.sha256(te.tobytes()).hexdigest() != str(row["split_hash"]):
        raise RuntimeError("split hash mismatch")
    return X, y, tr, va, te, mask


def run_view(adapter_factory, X, y, tr, va, te, cols, seed, device):
    """Train on a view (column subset). Returns (auc, meta, hashes)."""
    Xtr, Xva, Xte = X[tr][:, cols], X[va][:, cols], X[te][:, cols]
    ytr, yva, yte = y[tr], y[va], y[te]
    ad = adapter_factory()
    ad.fit(Xtr, ytr, Xva, yva, categorical_indices=[], seed=seed, device=device)
    # train-only candidate assertion: candidate rows are within train range only
    proba = ad.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))
    pre_hash = sha_arr(np.concatenate([ad._mu, ad._sd]))
    return auc, ad.runtime_metadata(), pre_hash, ad.candidate_row_ids_hash(), ad.candidate_tensor_hash(), len(tr)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bundle-manifest", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--mechanisms", default="all")
    ap.add_argument("--strengths", default="all")
    ap.add_argument("--seeds", default="all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.allow_run:
        raise RuntimeError("model execution locked; pass --allow-run")

    import yaml
    config = yaml.safe_load((ROOT / args.config).read_text())
    config_hash = sha_file(ROOT / args.config)
    runner_hash = sha_file(Path(__file__))
    adapter_file = ROOT / f"src/sp6/adapters/{args.model}_adapter.py"
    adapter_hash = sha_file(adapter_file)
    upstream_hash = sha_file(ROOT / "third_party/modernnca/modernNCA.py") if args.model == "modernnca" else ""

    man = pd.read_csv(ROOT / args.bundle_manifest)
    def sel(v, col, cast=str):
        if v == "all": return man
        vals = [cast(x) for x in v.split(",")]
        return man[man[col].astype(type(vals[0]) if vals else str).isin(vals)]
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

    total = len(man)
    started = time.time()
    done = 0
    for _, row in man.iterrows():
        rid_key = f"{args.model}|{row['dataset_index']}|{row['mechanism']}|{row['strength']}|{row['seed']}|{row['task_hash']}|{config_hash}|{adapter_hash}|{runner_hash}"
        run_id = hashlib.sha256(rid_key.encode()).hexdigest()[:20]
        done += 1
        if run_id in completed:
            continue
        rec = {k: "" for k in RESULT_FIELDS}
        rec.update({"run_id": run_id, "dataset_index": int(row["dataset_index"]),
                    "mechanism": row["mechanism"], "strength": row["strength"],
                    "seed": int(row["seed"]), "model": args.model, "status": "FAILURE",
                    "n_original": int(row["n_original"]), "n_injected": int(row["n_injected"]),
                    "n_leak": int(row["n_leak"]), "task_hash": row["task_hash"],
                    "split_hash": row["split_hash"], "bundle_source": row["bundle_source"],
                    "bundle_sha256": row["bundle_sha256"], "runner_hash": runner_hash,
                    "adapter_hash": adapter_hash, "upstream_model_hash": upstream_hash,
                    "config_hash": config_hash, "device": args.device})
        try:
            X, y, tr, va, te, mask = load_cell(row)
            strict_cols = np.where(~mask)[0]
            full_cols = np.arange(X.shape[1])
            rec["strict_view_hash"] = sha_arr(X[:, strict_cols])
            rec["full_view_hash"] = sha_arr(X)
            fac = lambda: load_adapter(args.model, config)
            s_auc, s_meta, s_pre, s_rid, s_ten, ntr = run_view(fac, X, y, tr, va, te, strict_cols, int(row["seed"]), args.device)
            f_auc, f_meta, f_pre, f_rid, f_ten, _ = run_view(fac, X, y, tr, va, te, full_cols, int(row["seed"]), args.device)
            # train-only + isolation assertions
            assert s_pre != f_pre or np.array_equal(strict_cols, full_cols), "strict/full preprocessor not isolated"
            rec.update({"status": "SUCCESS", "strict_auc": s_auc, "full_auc": f_auc,
                        "paired_harm": f_auc - s_auc,
                        "strict_preprocessor_hash": s_pre, "full_preprocessor_hash": f_pre,
                        "strict_candidate_row_ids_hash": s_rid, "full_candidate_row_ids_hash": f_rid,
                        "strict_candidate_tensor_hash": s_ten, "full_candidate_tensor_hash": f_ten,
                        "best_epoch_strict": s_meta.get("best_epoch"), "best_epoch_full": f_meta.get("best_epoch"),
                        "runtime_sec": round(time.time() - started, 2)})
        except Exception as e:
            rec["failure_reason"] = f"{type(e).__name__}: {e}"
        # append
        write_header = not out.exists()
        with out.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
            if write_header: w.writeheader()
            w.writerow(rec)
        if done % 20 == 0 or rec["status"] != "SUCCESS":
            print(f"{done}/{total} {rec['status']} {row['mechanism']} d{row['dataset_index']} {row['strength']} s{row['seed']}", flush=True)
    print(f"DONE {done}/{total} in {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
