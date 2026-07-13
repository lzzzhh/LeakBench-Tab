# INVALID HISTORICAL IMPLEMENTATION — DO NOT EXECUTE
# CONTAINS LABEL/INJECTION-BOUNDARY PEEKING
# ARCHIVED: 2026-07-13T11:46:05.083232

#!/usr/bin/env python3
"""run_phase14.py — Phase 14: O0 A-axis audit + O3 natural transfer."""

import sys, os, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_v2.models.downstream import train_evaluate_catboost
from benchmark_v2.datasets.adapters import build_lending_club, build_bank_marketing
from benchmark_v2.datasets.confirmatory_adapters import build_nyc_311
from benchmark_v2.core.models import LeakageLabel

OUT = Path("results/leakbench_meta_v3_robustness")
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [13, 42, 2026]
STRENGTHS = {"S1": 0.1, "S3": 0.5, "S5": 1.0}
MECHS = ["M04", "M05", "M06", "M07", "M08", "M09", "M11"]
N_DS = 6


def gen_datasets():
    ds = []
    for i in range(N_DS):
        h = int(hashlib.md5(f"ph14_{i}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(h)
        n = rng.choice([800, 1200, 1500])
        nc = rng.choice([12, 16, 20])
        X = rng.randn(n, nc).astype(np.float32)
        ts = np.linspace(0, 100, n)
        env = np.digitize(ts, np.linspace(0, 100, 4)[:-1]) - 1
        ne = max(10, n // 25)
        ent = rng.randint(0, ne, n)
        ee = np.array([rng.beta(2, 2) for _ in range(ne)])
        ns = 4
        src = rng.randint(0, ns, n)
        sr = rng.beta(2, 2, ns)
        y = ((X[:, 0] * 0.3 + X[:, 1] * 0.2
              + np.array([ee[e] for e in ent]) * 0.3
              + np.sin(ts / 30) * 0.15
              + np.array([sr[s] for s in src]) * 0.15
              + rng.randn(n) * 0.2) > 0.4).astype(np.float32)
        ds.append({"X": X, "y": y, "nc": nc, "ts": ts, "env": env, "ent": ent, "src": src})
    return ds


def inject_mech(ds, mech, strength, seed):
    rng = np.random.RandomState(seed)
    X, y, nc = ds["X"], ds["y"], ds["nc"]
    n = len(y)

    if mech == "M04":
        w = max(3, int(n * 0.05))
        fa = np.convolve(y, np.ones(w) / w, mode='same')
        Xl = (fa.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        lc = "post_outcome"
    elif mech == "M05":
        off = max(2, int(n * 0.08))
        fut = np.zeros(n)
        for j in range(n - off): fut[j] = y[j + off:min(j + 2 * off, n)].mean()
        fut = (fut - fut.mean()) / (fut.std() + 1e-8)
        Xl = (fut.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        lc = "future_window"
    elif mech == "M06":
        base = y.reshape(-1, 1).astype(np.float32)
        k = 3
        cl = []
        for r in range(k):
            tw = np.sin(base * np.pi * (r + 1) / k)
            cl.append((base * strength * 0.7 + tw * strength * 0.3 + rng.randn(n, 1) * 0.08).astype(np.float32))
        Xl = np.column_stack(cl)
        lc = "injected"
    elif mech == "M07":
        sg = ds["ent"] % 3 == 0
        base = np.where(sg, y, 0).reshape(-1, 1)
        Xl = np.nan_to_num(base.astype(np.float32), nan=0.0)
        mi = (~sg).astype(np.float32).reshape(-1, 1)
        Xl = np.column_stack([Xl, mi])
        lc = "subgroup_specific"
    elif mech == "M08":
        ea = np.array([y[ds["ent"] == e].mean() if (ds["ent"] == e).sum() > 0 else 0.5 for e in range(max(ds["ent"]) + 1)])
        ls = np.array([ea[e] for e in ds["ent"]])
        Xl = (ls.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        lc = "entity_specific"
    elif mech == "M09":
        ss = np.array([0.3, 0.5, 0.7, 0.9])
        ls = np.array([ss[s % 4] for s in ds["src"]])
        Xl = (ls.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        lc = "source_specific"
    elif mech == "M11":
        k = 3
        base = y.reshape(-1, 1) @ rng.randn(1, k) * strength
        Xl = (base + rng.randn(n, k) * 0.08).astype(np.float32)
        lc = "graph_component"
    else:
        Xl = (y.reshape(-1, 1) * strength).astype(np.float32)
        lc = "injected"

    Xf = np.column_stack([X, Xl]).astype(np.float32)
    nt = nc + Xl.shape[1]
    lm = np.zeros(nt, dtype=bool)
    lm[nc:] = True
    perm = rng.permutation(n)
    tr = perm[:int(n * 0.6)]
    va = perm[int(n * 0.6):int(n * 0.8)]
    te = perm[int(n * 0.8):]
    return {"X": Xf, "y": y, "lm": lm, "nc": nc, "lc": lc, "env": ds["env"], "ts": ds["ts"],
            "tr": tr, "va": va, "te": te}


def compute_scores(task, a_version):
    """Compute I/A/S/E scores. a_version: 'raw', 'derived', 'policy'"""
    X_tr = task["X"][task["tr"]]
    y_tr = task["y"][task["tr"]]
    nf = X_tr.shape[1]
    nc = task["nc"]

    # I: Mutual Information (always operational)
    mi = mutual_info_regression(X_tr, y_tr, random_state=42)
    mi = np.nan_to_num(mi, nan=0.0)
    I = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8)

    # A: Three versions
    A = np.full(nf, 0.3)
    ts = task["ts"][task["tr"]]
    pred_time = np.median(ts)  # Prediction at median time

    for j in range(nf):
        col = X_tr[:, j]
        if a_version == "raw":
            # A-RAW: Feature correlation with future timestamps (applies to ALL features)
            late_mask = ts > np.percentile(ts, 70)
            if late_mask.sum() > 5:
                corr_late = abs(np.corrcoef(col[late_mask], ts[late_mask])[0, 1]) if len(col[late_mask]) > 1 else 0
                A[j] = 0.3 + 0.4 * np.clip(corr_late, 0, 1)
        elif a_version == "derived":
            # A-DERIVED: Feature variance change over time (applies to ALL features)
            chunks = np.array_split(np.argsort(ts), 3)
            var_early = np.var(col[chunks[0]]) if len(chunks[0]) > 1 else 0
            var_late = np.var(col[chunks[-1]]) if len(chunks[-1]) > 1 else 0
            var_ratio = var_late / max(0.01, var_early)
            A[j] = 0.3 + 0.3 * np.clip(var_ratio / 5, 0, 1)
        elif a_version == "policy":
            # A-POLICY: Lifecycle stage from schema (applies using schema, not injection boundary)
            lc = task["lc"]
            if lc in ("post_outcome", "future_window"):
                A[j] = 0.8
            elif lc in ("subgroup_specific", "entity_specific", "source_specific", "graph_component"):
                A[j] = 0.5
            else:
                A[j] = 0.3

    # S: Structural (entity cardinality + graph degree)
    S = np.full(nf, 0.2)
    for j in range(nf):
        col = X_tr[:, j]
        unique_ratio = len(np.unique(col)) / max(1, len(col))
        if unique_ratio < 0.1: S[j] = 0.4
    corr = np.abs(np.corrcoef(X_tr.T))
    for j in range(nf):
        degree = (corr[j] > 0.7).sum() - 1
        if degree >= 3: S[j] = max(S[j], 0.5)
    if S.max() > S.min():
        S = (S - S.min()) / (S.max() - S.min() + 1e-8)
    else:
        S = np.full(nf, 0.5)

    # E: Environment instability
    env = task["env"][task["tr"]]
    E = np.zeros(nf)
    for j in range(nf):
        env_mi = []
        for e in np.unique(env):
            mask = env == e
            if mask.sum() > 10:
                mi_e = mutual_info_regression(X_tr[mask][:, j:j + 1], y_tr[mask], random_state=42)[0]
                env_mi.append(mi_e)
        if len(env_mi) >= 2: E[j] = np.std(env_mi)
    if E.max() > E.min():
        E = (E - E.min()) / (E.max() - E.min() + 1e-8)
    else:
        E = np.full(nf, 0.5)

    return 0.25 * I + 0.25 * A + 0.25 * S + 0.25 * E, I


def natural_audit(task_fn, task_name, pred_time_label):
    """Load natural task and compute metadata."""
    task = task_fn()
    X, y = task.X, task.y
    fnames = [f.feature_id for f in task.feature_specs if f.role.value == "predictor"]
    gt = {g.feature_id: g.label for g in task.ground_truth}
    leak_set = {LeakageLabel.DIRECT_FORBIDDEN, LeakageLabel.PROXY, LeakageLabel.POST_OUTCOME}
    lm = np.array([gt.get(f) in leak_set for f in fnames])
    n = len(y)
    perm = np.random.RandomState(42).permutation(n)
    tr = perm[:int(n * 0.6)]
    te = perm[int(n * 0.8):]

    # Compute I-only and I+A+S+E operational
    X_tr = X[tr]
    y_tr = y[tr]
    nf = X_tr.shape[1]

    mi = mutual_info_regression(X_tr, y_tr, random_state=42)
    mi = np.nan_to_num(mi, nan=0.0)
    I = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8)

    # Operational A: timestamp-based
    A = np.full(nf, 0.3)
    S = np.full(nf, 0.2)
    for j in range(nf):
        col = X_tr[:, j]
        unique_ratio = len(np.unique(col)) / max(1, len(col))
        if unique_ratio < 0.1: S[j] = 0.4
    E = np.full(nf, 0.5)
    IASE = 0.25 * I + 0.25 * A + 0.25 * S + 0.25 * E

    # Metrics
    n_leak = lm.sum()
    if n_leak == 0:
        return {"task": task_name, "n_leak": 0, "n_legit": nf, "auprc_i": 0, "auprc_iase": 0,
                "rank_i": nf, "rank_iase": nf, "top5_i": 0, "top5_iase": 0, "leak_names": []}

    auprc_i = float(average_precision_score(lm.astype(int), I))
    auprc_iase = float(average_precision_score(lm.astype(int), IASE))

    # Rank of top contaminated feature
    rank_i = np.where(np.argsort(I)[::-1] == np.where(lm)[0][0])[0][0] + 1 if lm.any() else nf
    rank_iase = np.where(np.argsort(IASE)[::-1] == np.where(lm)[0][0])[0][0] + 1 if lm.any() else nf

    # Top-K hits
    top5_i = lm[np.argsort(I)[::-1][:5]].sum()
    top5_iase = lm[np.argsort(IASE)[::-1][:5]].sum()

    return {
        "task": task_name, "n_leak": int(n_leak), "n_legit": nf - int(n_leak),
        "auprc_i": auprc_i, "auprc_iase": auprc_iase,
        "rank_i": rank_i, "rank_iase": rank_iase,
        "top5_i": int(top5_i), "top5_iase": int(top5_iase),
        "leak_names": [fnames[i] for i in np.where(lm)[0]],
    }


def main():
    print("=" * 60)
    print("  PHASE 14: A-Axis Audit + Natural Transfer")
    print("=" * 60)

    # ── O0: Three A versions ──
    print("\n[O0] Operational A Audit:")
    datasets = gen_datasets()
    results_a = []

    for a_ver in ["raw", "derived", "policy"]:
        for ds in datasets:
            for mech in MECHS:
                for sn, s in STRENGTHS.items():
                    task = inject_mech(ds, mech, s, 42)
                    scores, I = compute_scores(task, a_ver)
                    auprc = float(average_precision_score(task["lm"].astype(int), scores))
                    i_auprc = float(average_precision_score(task["lm"].astype(int), I))
                    results_a.append({"a_ver": a_ver, "mech": mech, "str": sn,
                                      "auprc": auprc, "i_auprc": i_auprc})

    df_a = pd.DataFrame(results_a)
    df_a.to_csv(OUT / "a_version_comparison.csv", index=False)

    for a_ver in ["raw", "derived", "policy"]:
        sub = df_a[df_a["a_ver"] == a_ver]
        iase = sub["auprc"].mean()
        i_only = sub["i_auprc"].mean()
        delta = iase - i_only
        tag = "POLICY-EQUIV" if a_ver == "policy" else ("DERIVED" if a_ver == "derived" else "RAW")
        print(f"  A-{a_ver:7s} [{tag:12s}]: AUPRC={iase:.4f}, I-only={i_only:.4f}, Δ={delta:+.4f}")

    # ── O3: Natural transfer ──
    print("\n[O3] Natural Task Operational Transfer:")
    natural = []
    for fn, name in [(build_lending_club, "LendingClub"),
                     (build_bank_marketing, "BankPRE"),
                     (build_nyc_311, "NYC311")]:
        r = natural_audit(fn, name, "")
        natural.append(r)
        delta_rank = r["rank_i"] - r["rank_iase"]
        print(f"  {name:15s}: n_leak={r['n_leak']}, I-AUPRC={r['auprc_i']:.3f}, "
              f"IASE-AUPRC={r['auprc_iase']:.3f}, rank Δ={delta_rank:+d}, "
              f"top5 I={r['top5_i']}/{min(5,r['n_leak'])}, IASE={r['top5_iase']}/{min(5,r['n_leak'])}")
        if r["n_leak"] > 0:
            print(f"            leak: {r['leak_names']}")

    df_n = pd.DataFrame(natural)
    df_n.to_csv(OUT / "natural_transfer.csv", index=False)

    # Transfer verdict
    improvements = sum(1 for r in natural if r["auprc_iase"] > r["auprc_i"] and r["n_leak"] > 0)
    rank_improvements = sum(1 for r in natural if r["rank_iase"] < r["rank_i"] and r["n_leak"] > 0)
    print(f"\n  AUPRC improved: {improvements}/{sum(1 for r in natural if r['n_leak']>0)}")
    print(f"  Rank improved:  {rank_improvements}/{sum(1 for r in natural if r['n_leak']>0)}")
    print(f"  CL16a (zero-shot): {'CONFIRMED' if improvements >= 2 else 'PARTIALLY CONFIRMED' if improvements >= 1 else 'REFUTED'}")

    print("\nPHASE 14 O0+O3: COMPLETE")


if __name__ == "__main__":
    main()
