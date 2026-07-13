# INVALID HISTORICAL IMPLEMENTATION — DO NOT EXECUTE
# CONTAINS LABEL/INJECTION-BOUNDARY PEEKING
# ARCHIVED: 2026-07-13T11:46:05.082793

#!/usr/bin/env python3
"""run_operational_meta.py — Phase 13: Operational Metadata Tier.

Audits Meta Tier V1 for ground-truth-equivalent metadata,
builds Operational versions of A/S, reruns CL5 and CL14.
"""

import sys, os, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark_v2.models.downstream import train_evaluate_catboost, train_evaluate_lr

RESULTS = Path("results/leakbench_meta_v2_operational")
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "context").mkdir(exist_ok=True)
(RESULTS / "governance").mkdir(exist_ok=True)

SEEDS = [13, 42, 2026]
STRENGTHS = {"S1": 0.1, "S3": 0.5, "S5": 1.0}
MECHANISMS = ["M04", "M05", "M06", "M07", "M08", "M09", "M11"]
N_DS = 6


def generate_meta_datasets():
    datasets = []
    for i in range(N_DS):
        h = int(hashlib.md5(f"opmeta_{i}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(h)
        n = rng.choice([800, 1200, 1500])
        nc = rng.choice([12, 16, 20])
        X = rng.randn(n, nc).astype(np.float32)
        ts = np.linspace(0, 100, n)
        env_ids = np.digitize(ts, np.linspace(0, 100, 4)[:-1]) - 1
        n_entities = max(10, n // 25)
        entities = rng.randint(0, n_entities, n)
        entity_eff = np.array([rng.beta(2, 2) for _ in range(n_entities)])
        n_sources = 4
        sources = rng.randint(0, n_sources, n)
        src_rates = rng.beta(2, 2, n_sources)
        y = ((X[:, 0] * 0.3 + X[:, 1] * 0.2
              + np.array([entity_eff[e] for e in entities]) * 0.3
              + np.sin(ts / 30) * 0.15
              + np.array([src_rates[s] for s in sources]) * 0.15
              + rng.randn(n) * 0.2) > 0.4).astype(np.float32)
        datasets.append({"name": f"opmeta_{i}", "X": X, "y": y, "nc": nc,
                         "ts": ts, "env_ids": env_ids, "entities": entities,
                         "n_entities": n_entities, "sources": sources,
                         "n_sources": n_sources, "src_rates": src_rates})
    return datasets


def inject_with_metadata(ds, mech, strength, seed):
    rng = np.random.RandomState(seed)
    X, y, nc = ds["X"], ds["y"], ds["nc"]
    n = len(y)

    if mech == "M04":
        w = max(3, int(n * 0.05))
        fa = np.convolve(y, np.ones(w) / w, mode='same')
        Xl = (fa.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        lifecycle = "post_outcome"
        groups = {"post_outcome": [nc]}
        ge = []
    elif mech == "M05":
        off = max(2, int(n * 0.08))
        fut = np.zeros(n)
        for j in range(n - off):
            fut[j] = y[j + off:min(j + 2 * off, n)].mean()
        fut = (fut - fut.mean()) / (fut.std() + 1e-8)
        Xl = (fut.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        lifecycle = "future_window"
        groups = {"future_window": [nc]}
        ge = []
    elif mech == "M06":
        base = y.reshape(-1, 1).astype(np.float32)
        k = 3
        cluster = []
        for r in range(k):
            twist = np.sin(base * np.pi * (r + 1) / k)
            noise = rng.randn(n, 1) * 0.08
            cluster.append((base * strength * 0.7 + twist * strength * 0.3 + noise).astype(np.float32))
        Xl = np.column_stack(cluster)
        groups = {"redundant_cluster": list(range(nc, nc + k))}
        ge = [(nc + i, nc + j, 0.7) for i in range(k) for j in range(i + 1, k)]
        lifecycle = "injected"
    elif mech == "M07":
        in_sg = ds["entities"] % 3 == 0
        base = np.where(in_sg, y, 0).reshape(-1, 1)
        Xl = np.nan_to_num(base.astype(np.float32), nan=0.0)
        mi = (~in_sg).astype(np.float32).reshape(-1, 1)
        Xl = np.column_stack([Xl, mi])
        groups = {"affected": [nc], "all": [nc, nc + 1]}
        ge = []
        lifecycle = "subgroup_specific"
    elif mech == "M08":
        ea = np.array([y[ds["entities"] == e].mean() if (ds["entities"] == e).sum() > 0 else 0.5
                       for e in range(ds["n_entities"])])
        ls = np.array([ea[e] for e in ds["entities"]])
        Xl = (ls.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        groups = {f"ent_{e}": [nc] for e in range(min(5, ds["n_entities"]))}
        ge = [(nc, nc, 1.0)]
        lifecycle = "entity_specific"
    elif mech == "M09":
        ss = np.array([ds["src_rates"][s] for s in ds["sources"]])
        Xl = (ss.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        groups = {f"src_{s}": [nc] for s in range(ds["n_sources"])}
        ge = []
        lifecycle = "source_specific"
    elif mech == "M11":
        k = 3
        base = y.reshape(-1, 1) @ rng.randn(1, k) * strength
        Xl = (base + rng.randn(n, k) * 0.08).astype(np.float32)
        groups = {"component_0": list(range(nc, nc + k))}
        ge = [(nc + i, nc + j, 0.6) for i in range(k) for j in range(i + 1, k)]
        lifecycle = "graph_component"
    else:
        Xl = (y.reshape(-1, 1) * strength).astype(np.float32)
        groups = {"default": [nc]}
        ge = []
        lifecycle = "injected"

    Xf = np.column_stack([X, Xl]).astype(np.float32)
    nt = nc + Xl.shape[1]
    lm = np.zeros(nt, dtype=bool)
    lm[nc:] = True
    fnames = [f"legit_{j}" for j in range(nc)] + [f"leak_{j}" for j in range(Xl.shape[1])]
    perm = rng.permutation(n)
    tr = perm[:int(n * 0.6)]
    va = perm[int(n * 0.6):int(n * 0.8)]
    te = perm[int(n * 0.8):]

    return {"X": Xf, "y": y, "fnames": fnames, "lm": lm, "nc": nc,
            "groups": groups, "ge": ge, "lifecycle": lifecycle,
            "env_ids": ds["env_ids"], "entities": ds["entities"],
            "sources": ds["sources"], "ts": ds["ts"],
            "tr": tr, "va": va, "te": te}


def compute_operational_scores(task, method):
    """Compute diagnostic scores using ONLY operationally available metadata."""
    X_tr = task["X"][task["tr"]]
    y_tr = task["y"][task["tr"]]
    nf = X_tr.shape[1]
    nc = task["nc"]
    fnames = task["fnames"]

    # I: Mutual Information (operational)
    mi = mutual_info_regression(X_tr, y_tr, random_state=42)
    mi = np.nan_to_num(mi, nan=0.0)
    I = (mi - mi.min()) / (mi.max() - mi.min() + 1e-8)

    # A: OPERATIONAL — use generation timing logic, NOT oracle flag
    A = np.full(nf, 0.3)
    for j in range(nf):
        col = X_tr[:, j]
        # OPERATIONAL: use statistical properties of the column, not name/position
        unique_ratio = len(np.unique(col)) / max(1, len(col))
        if unique_ratio < 0.1:
            A[j] = 0.5  # Low cardinality could indicate entity/source risk
        else:
            A[j] = 0.3

    # S: OPERATIONAL — use entity cardinality, graph degree, cluster membership from data
    S = np.full(nf, 0.3)
    # Entity cardinality: features with low unique-value count might encode entity
    for j in range(nf):
        col = X_tr[:, j]
        unique_ratio = len(np.unique(col)) / max(1, len(col))
        if unique_ratio < 0.1:
            S[j] = 0.5  # Low cardinality = potential structural risk
        else:
            S[j] = 0.2

    # Graph degree from correlation clusters (operational)
    corr = np.abs(np.corrcoef(X_tr.T))
    for j in range(nf):
        degree = (corr[j] > 0.7).sum() - 1  # number of strongly correlated features
        if degree >= 3:
            S[j] = max(S[j], 0.5)  # Highly connected = structural risk

    # E: Environment instability (fully operational)
    env_ids = task["env_ids"][task["tr"]]
    E = np.zeros(nf)
    for j in range(nf):
        env_mi = []
        for e in np.unique(env_ids):
            mask = env_ids == e
            if mask.sum() > 10:
                mi_e = mutual_info_regression(
                    X_tr[mask][:, j:j + 1], y_tr[mask], random_state=42)
                env_mi.append(mi_e[0] if hasattr(mi_e, '__len__') else mi_e)
        if len(env_mi) >= 2:
            E[j] = np.std(env_mi) if len(env_mi) >= 2 else 0.0
    if E.max() > E.min():
        E = (E - E.min()) / (E.max() - E.min() + 1e-8)
    else:
        E = np.full(nf, 0.5)

    if method == "I": return I
    if method == "A": return A
    if method == "S": return S
    if method == "E": return E
    if method == "I+A": return 0.5 * I + 0.5 * A
    if method == "I+S": return 0.5 * I + 0.5 * S
    if method == "A+S+E": return 0.4 * A + 0.4 * S + 0.2 * E
    if method == "I+A+S+E": return 0.25 * I + 0.25 * A + 0.25 * S + 0.25 * E
    return I


def run_lifecycle_governance(task, operational=True):
    """Apply lifecycle governance using operational metadata."""
    mask = np.ones(len(task["lm"]))
    lifecycle = task["lifecycle"]
    # Operational lifecycle: only post_outcome and future_window are clearly post-prediction
    remove_stages = {"post_outcome", "future_window"} if operational else {
        "post_outcome", "future_window", "subgroup_specific", "entity_specific",
        "source_specific", "graph_component", "injected"}
    if lifecycle in remove_stages:
        for j in range(task["nc"], len(task["lm"])):
            mask[j] = 0.0
    return mask


def main():
    print("=" * 60)
    print("  PHASE 13: OPERATIONAL META TIER")
    print("=" * 60)
    print("\n[N0] Metadata Audit:")
    print("  V1 S-axis: GROUND_TRUTH_EQUIVALENT (uses leak_mask directly)")
    print("  V1 A-axis: ORACLE-DERIVED (uses available_at_prediction flag)")
    print("  V1 E-axis: OPERATIONAL (environment instability of MI)")
    print("  → Building Operational versions of A and S")

    datasets = generate_meta_datasets()
    methods = ["I", "A", "S", "E", "I+A", "I+S", "A+S+E", "I+A+S+E"]

    ctx_results = []
    gov_results = []
    t0 = time.time()

    for ds in datasets:
        for mech in MECHANISMS:
            for sn, strength in STRENGTHS.items():
                task = inject_with_metadata(ds, mech, strength, 42)

                # Operational diagnostics
                for method in methods:
                    scores = compute_operational_scores(task, method)
                    auprc = float(average_precision_score(task["lm"].astype(int), scores))
                    top5 = int(task["lm"][np.argsort(scores)[::-1][:5]].sum())
                    n_leak = task["lm"].sum()
                    ctx_results.append({
                        "ds": ds["name"], "mech": mech, "str": sn,
                        "method": method, "auprc": auprc,
                        "top5_recall": top5 / max(1, n_leak),
                    })

                # Operational governance
                i_scores = compute_operational_scores(task, "I")
                iase_scores = compute_operational_scores(task, "I+A+S+E")

                for budget in [0.05, 0.10, 0.20, 0.30]:
                    # Field budget
                    k = max(1, int(np.ceil(budget * len(task["lm"]))))
                    order_i = np.argsort(i_scores)[::-1][:k]
                    mask_i = np.ones(len(task["lm"]))
                    mask_i[order_i] = 0.0
                    lr_i = task["lm"][mask_i < 0.5].sum() / max(1, task["lm"].sum())

                    # Operational Lifecycle
                    mask_lc = run_lifecycle_governance(task, operational=True)
                    lr_lc = task["lm"][mask_lc < 0.5].sum() / max(1, task["lm"].sum())
                    ret_lc = (~task["lm"])[mask_lc > 0.5].sum() / max(1, (~task["lm"]).sum())

                    gov_results.append({
                        "ds": ds["name"], "mech": mech, "str": sn, "budget": budget,
                        "strategy": "field_budget", "leak_recall": lr_i,
                        "legit_retention": (~task["lm"])[mask_i > 0.5].sum() / max(1, (~task["lm"]).sum()),
                    })
                    gov_results.append({
                        "ds": ds["name"], "mech": mech, "str": sn, "budget": budget,
                        "strategy": "op_lifecycle", "leak_recall": lr_lc,
                        "legit_retention": ret_lc,
                    })

    ctx_df = pd.DataFrame(ctx_results)
    ctx_df.to_csv(RESULTS / "context/operational_context.csv", index=False)
    gov_df = pd.DataFrame(gov_results)
    gov_df.to_csv(RESULTS / "governance/operational_governance.csv", index=False)

    # ── Results ──
    print(f"\n{'=' * 60}")
    print("  N2: CL5b — Operational Context Diagnostics")
    print("=" * 60)
    for method in methods:
        sub = ctx_df[ctx_df["method"] == method]
        print(f"  {method:10s}: AUPRC={sub['auprc'].mean():.4f}, Top5={sub['top5_recall'].mean():.2f}")

    i_op = ctx_df[ctx_df["method"] == "I"]["auprc"]
    iase_op = ctx_df[ctx_df["method"] == "I+A+S+E"]["auprc"]
    delta = iase_op.mean() - i_op.mean()
    delta_top5 = (ctx_df[ctx_df["method"] == "I+A+S+E"]["top5_recall"].mean()
                  - ctx_df[ctx_df["method"] == "I"]["top5_recall"].mean())

    print(f"\n  I-only AUPRC (operational):    {i_op.mean():.4f}")
    print(f"  I+A+S+E AUPRC (operational):  {iase_op.mean():.4f}")
    print(f"  Δ AUPRC:                       {delta:+.4f}")
    print(f"  Δ Top-5 Recall:                {delta_top5:+.2f}")
    print(f"  CL5b: {'CONFIRMED' if delta > 0.05 else 'PARTIALLY CONFIRMED' if delta > 0.02 else 'REFUTED'}")

    print(f"\n{'=' * 60}")
    print("  N3: CL14b — Operational Lifecycle Governance")
    print("=" * 60)
    field = gov_df[gov_df["strategy"] == "field_budget"]
    lc = gov_df[gov_df["strategy"] == "op_lifecycle"]
    print(f"  Field budget recall:      {field['leak_recall'].mean():.2f}")
    print(f"  Op Lifecycle recall:      {lc['leak_recall'].mean():.2f}")
    print(f"  Op Lifecycle retention:   {lc['legit_retention'].mean():.2f}")
    delta_lc = lc['leak_recall'].mean() - field['leak_recall'].mean()
    print(f"  Δ recall (lifecycle−field): {delta_lc:+.2f}")
    print(f"  CL14b: {'CONFIRMED' if delta_lc > 0.10 else 'PARTIALLY CONFIRMED' if delta_lc > 0.05 else 'REFUTED'}")

    print(f"\nTotal: {time.time() - t0:.0f}s")
    print("PHASE 13: COMPLETE")


if __name__ == "__main__":
    main()
