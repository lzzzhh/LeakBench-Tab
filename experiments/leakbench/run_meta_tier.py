#!/usr/bin/env python3
"""run_meta_tier.py — Phase 12: Metadata-Complete Evaluation Tier.

Generates 6 datasets with full I/A/S/E metadata, runs context diagnostics
and governance on 7 structured mechanisms, validating CL5 and CL14.
"""

import sys, os, json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_v2.models.downstream import train_evaluate_catboost, train_evaluate_lr
from benchmark_v2.models.tabm import train_evaluate_tabm
from src.leakbench.diagnostics import (
    OperationalFeatureMetadata,
    OperationalMetadata,
    OracleMetadata,
    compute_operational_diagnostics,
)
from src.leakbench.governance import (
    GovernanceStatus,
    GovernanceStrategy,
    apply_strategy,
)

RESULTS = Path("results/leakbench_meta")
SEEDS = [13, 42, 2026]
STRENGTHS = {"S1": 0.1, "S3": 0.5, "S5": 1.0}
N_DATASETS = 6


def _stable_feature_id(dataset_name, logical_name):
    digest = hashlib.sha256(f"{dataset_name}:{logical_name}".encode()).hexdigest()[:16]
    return f"fid_{digest}"

# ── Metadata-complete dataset generator ──
def generate_meta_datasets():
    """Generate 6 datasets with built-in time/entity/source/graph/metadata."""
    datasets = []
    for i in range(N_DATASETS):
        h = int(hashlib.md5(f"meta_{i}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(h)
        n = rng.choice([800, 1200, 1500, 2000])
        n_clean = rng.choice([12, 16, 20])
        n_envs = 3

        # Legitimate features
        X_legit = rng.randn(n, n_clean).astype(np.float32)

        # Timestamps for temporal environments
        timestamps = np.linspace(0, 100, n)
        env_ids = np.digitize(timestamps, np.linspace(0, 100, n_envs + 1)[:-1]) - 1

        # Entity IDs for structural metadata
        n_entities = max(10, n // 30)
        entities = rng.randint(0, n_entities, n)

        # Source IDs
        n_sources = 4
        sources = rng.randint(0, n_sources, n)
        source_rates = rng.beta(2, 2, n_sources)

        # Target with entity + time + source effects
        entity_effect = np.array([rng.beta(2, 2) for _ in range(n_entities)])
        time_effect = np.sin(timestamps / 30) * 0.2
        source_effect = np.array([source_rates[s] for s in sources]) * 0.3
        y = ((X_legit[:, 0] * 0.3 + X_legit[:, 1] * 0.2
              + np.array([entity_effect[e] for e in entities]) * 0.3
              + time_effect * 0.15 + source_effect * 0.15
              + rng.randn(n) * 0.2) > 0.4).astype(np.float32)

        datasets.append({
            "name": f"meta_{i}", "X": X_legit, "y": y,
            "n_clean": n_clean, "n_samples": n,
            "timestamps": timestamps, "env_ids": env_ids,
            "entities": entities, "n_entities": n_entities,
            "sources": sources, "n_sources": n_sources,
            "source_rates": source_rates,
        })
    return datasets


def inject_with_metadata(ds, mechanism, strength, seed):
    """Inject contamination with full A/S/E metadata."""
    rng = np.random.RandomState(seed)
    X, y = ds["X"], ds["y"]
    n, n_clean = X.shape

    # Common: add contaminated features based on mechanism
    if mechanism == "M04":  # Post-outcome aggregation
        window = max(3, int(n * 0.05))
        future_agg = np.convolve(y, np.ones(window) / window, mode='same')
        X_leak = (future_agg.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        lifecycle = "post_outcome"
        groups = {"post_outcome": [n_clean]}
        graph_edges = [(n_clean - 1, n_clean, 0.8)] if n_clean > 1 else []
        a_avail = False  # NOT available at prediction time

    elif mechanism == "M05":  # Temporal look-ahead
        offset = max(2, int(n * 0.08))
        future = np.zeros(n)
        for j in range(n - offset):
            future[j] = y[j+offset:min(j+2*offset, n)].mean()
        future = (future - future.mean()) / (future.std() + 1e-8)
        X_leak = (future.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        lifecycle = "future_window"
        groups = {"future_window": [n_clean]}
        graph_edges = []
        a_avail = False

    elif mechanism == "M06":  # Redundant cluster
        base = y.reshape(-1, 1).astype(np.float32)
        cluster = []
        k = 3  # 3 redundant copies
        for r in range(k):
            twist = np.sin(base * np.pi * (r + 1) / k)
            noise = rng.randn(n, 1) * 0.08
            cluster.append((base * strength * 0.7 + twist * strength * 0.3 + noise).astype(np.float32))
        X_leak = np.column_stack(cluster)
        groups = {"redundant_cluster": list(range(n_clean, n_clean + k))}
        graph_edges = [(n_clean + i, n_clean + j, 0.7) for i in range(k) for j in range(i+1, k)]
        lifecycle = "injected"
        a_avail = True

    elif mechanism == "M07":  # Sparse subgroup
        in_subgroup = ds["entities"] % 3 == 0
        base = np.where(in_subgroup, y, 0).reshape(-1, 1)
        X_leak = np.nan_to_num(base.astype(np.float32), nan=0.0)
        missing_indicator = (~in_subgroup).astype(np.float32).reshape(-1, 1)
        X_leak = np.column_stack([X_leak, missing_indicator])
        groups = {"affected_subgroup": [n_clean], "all": [n_clean, n_clean + 1]}
        graph_edges = []
        lifecycle = "subgroup_specific"
        a_avail = False  # Only available for in-subgroup

    elif mechanism == "M08":  # Entity leakage
        entity_avg = np.array([ds["y"][ds["entities"] == e].mean() if (ds["entities"] == e).sum() > 0 else 0.5
                               for e in range(ds["n_entities"])])
        leak_signal = np.array([entity_avg[e] for e in ds["entities"]])
        X_leak = (leak_signal.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        groups = {"entity_bundle": [n_clean]}
        graph_edges = [(n_clean, n_clean, 1.0)]
        lifecycle = "entity_specific"
        a_avail = True  # Entity ID IS available, but leaks label distribution

    elif mechanism == "M09":  # Source leakage
        src_signal = np.array([ds["source_rates"][s] for s in ds["sources"]])
        X_leak = (src_signal.reshape(-1, 1) * strength + rng.randn(n, 1) * 0.05).astype(np.float32)
        groups = {"source_bundle": [n_clean]}
        graph_edges = []
        lifecycle = "source_specific"
        a_avail = True

    elif mechanism == "M11":  # Graph-mediated
        k = 3
        base = y.reshape(-1, 1) @ rng.randn(1, k) * strength
        X_leak = (base + rng.randn(n, k) * 0.08).astype(np.float32)
        groups = {"component_0": list(range(n_clean, n_clean + k))}
        graph_edges = [(n_clean + i, n_clean + j, 0.6) for i in range(k) for j in range(i+1, k)]
        lifecycle = "graph_component"
        a_avail = True

    else:
        X_leak = (y.reshape(-1, 1) * strength).astype(np.float32)
        groups = {"default": [n_clean]}
        graph_edges = []
        lifecycle = "injected"
        a_avail = True

    X_full = np.column_stack([X, X_leak]).astype(np.float32)
    n_total = X_full.shape[1]
    feature_ids = [
        _stable_feature_id(ds["name"], f"input_{j}") for j in range(n_clean)
    ] + [
        _stable_feature_id(ds["name"], f"{mechanism}_{j}") for j in range(X_leak.shape[1])
    ]

    # Translate construction-time indices exactly once. Downstream operational
    # code sees only stable IDs, never the injection boundary or display names.
    stable_groups = {
        group_id: [feature_ids[index] for index in indices]
        for group_id, indices in groups.items()
    }
    stable_graph = tuple(
        (feature_ids[src], feature_ids[dst], float(weight))
        for src, dst, weight in graph_edges
    )
    group_by_feature = {}
    for group_id, members in sorted(
        stable_groups.items(), key=lambda item: len(item[1]), reverse=True
    ):
        for feature_id in members:
            group_by_feature.setdefault(feature_id, group_id)

    operational_features = {
        feature_id: OperationalFeatureMetadata(stable_id=feature_id)
        for feature_id in feature_ids[:n_clean]
    }
    for feature_id in feature_ids[n_clean:]:
        operational_features[feature_id] = OperationalFeatureMetadata(
            stable_id=feature_id,
            available_at_prediction=a_avail,
            lifecycle=lifecycle,
            group_id=group_by_feature.get(feature_id),
            outcome_descendant=mechanism in {"M04", "M06"},
            post_event_table=mechanism == "M04",
        )
    operational_metadata = OperationalMetadata(
        features=operational_features,
        graph_edges=stable_graph,
    )
    oracle_metadata = OracleMetadata(
        leakage_by_feature_id={
            **{feature_id: False for feature_id in feature_ids[:n_clean]},
            **{feature_id: True for feature_id in feature_ids[n_clean:]},
        }
    )

    # Splits
    perm = rng.permutation(n)
    tr = perm[:int(n * 0.6)]
    va = perm[int(n * 0.6):int(n * 0.8)]
    te = perm[int(n * 0.8):]

    return {
        "X": X_full, "y": y, "feature_ids": feature_ids,
        "operational_metadata": operational_metadata,
        "oracle_metadata": oracle_metadata,
        "env_ids": ds["env_ids"], "entities": ds["entities"],
        "sources": ds["sources"], "timestamps": ds["timestamps"],
        "train_idx": tr, "val_idx": va, "test_idx": te,
    }


def compute_diagnostic_scores(
        X, y, feature_ids, operational_metadata, train_idx, env_ids, method="I"):
    """Compute blind per-feature scores from operationally visible inputs."""
    diagnostics = compute_operational_diagnostics(
        X[train_idx],
        y[train_idx],
        feature_ids,
        operational_metadata,
        environment_ids=env_ids[train_idx],
    )
    I = diagnostics.predictive_impact
    A = diagnostics.availability_risk
    S = diagnostics.structural_risk
    E = diagnostics.environment_instability
    combinations = {
        "I": I,
        "A": A,
        "S": S,
        "E": E,
        "I+A": 0.5 * I + 0.5 * A,
        "I+S": 0.5 * I + 0.5 * S,
        "A+S+E": 0.4 * A + 0.4 * S + 0.2 * E,
        "I+A+S+E": 0.25 * I + 0.25 * A + 0.25 * S + 0.25 * E,
    }
    return combinations.get(method, I)


def compute_auprc(scores, feature_ids, oracle_metadata):
    from sklearn.metrics import average_precision_score
    truth = oracle_metadata.leakage_mask(feature_ids)
    if truth.sum() == 0:
        return 0.0
    return float(average_precision_score(truth.astype(int), scores))


def run_governance_strategy(task, scores, strategy, budget):
    """Apply a stable-ID policy; oracle metadata is evaluation-only."""
    strategy_map = {
        "no_removal": GovernanceStrategy.NO_REMOVAL,
        "oracle": GovernanceStrategy.ORACLE_REMOVE_ALL,
        "field_budget": GovernanceStrategy.FIXED_FIELD_BUDGET,
        "group_budget": GovernanceStrategy.FIXED_GROUP_BUDGET,
        "lifecycle": GovernanceStrategy.LIFECYCLE_REMOVAL,
        "graph_cut": GovernanceStrategy.GRAPH_CUT,
    }
    result = apply_strategy(
        strategy_map[strategy],
        task["feature_ids"],
        scores,
        task["operational_metadata"],
        oracle_metadata=task["oracle_metadata"],
        budget=budget,
    )
    mask = result.feature_mask
    kept = np.where(mask > 0.5)[0]

    return {
        "status": result.status.value,
        "status_reason": result.reason,
        "n_quarantined": result.n_quarantined,
        "review_units": result.review_units,
        "leak_recall": result.leakage_recall,
        "legit_retention": result.legitimate_retention,
        "residual_contam": 1.0 - result.leakage_recall
        if result.status == GovernanceStatus.APPLIED else np.nan,
        "kept_indices": kept.tolist(),
        "mask_hash": hashlib.md5(mask.tobytes()).hexdigest()[:8],
    }


def main():
    raise RuntimeError(
        "INTEGRITY HOLD: run_meta_tier.py retains the legacy metadata-task "
        "generator and is not valid for corrected_v2 evidence. Its M04/M08/M09 "
        "constructions do not satisfy the frozen corrected_v2 invariants. Do not "
        "use this entry point for paper claims; metadata/governance remain PENDING "
        "until a runner consumes immutable corrected_v2 task bundles."
    )
    print("=" * 60)
    print("  PHASE 12: META TIER")
    print("=" * 60)

    datasets = generate_meta_datasets()
    mechanisms = ["M04", "M05", "M06", "M07", "M08", "M09", "M11"]
    diag_methods = ["I", "A", "S", "E", "I+A", "I+S", "A+S+E", "I+A+S+E"]
    gov_strategies = ["no_removal", "oracle", "field_budget", "group_budget", "lifecycle", "graph_cut"]

    ctx_results = []
    gov_results = []
    t0 = time.time()
    n_done = 0
    n_ctx = N_DATASETS * len(mechanisms) * len(STRENGTHS) * len(SEEDS) * len(diag_methods)
    n_gov = (N_DATASETS * len(mechanisms) * len(STRENGTHS) * len(SEEDS)
             * len(gov_strategies) * 4)  # 4 budgets

    for ds in datasets:
        for mech in mechanisms:
            for sn, strength in STRENGTHS.items():
                for seed in SEEDS:
                    task = inject_with_metadata(ds, mech, strength, seed)
                    truth = task["oracle_metadata"].leakage_mask(task["feature_ids"])

                    # ── M2: Context diagnostics ──
                    for method in diag_methods:
                        scores = compute_diagnostic_scores(
                            task["X"], task["y"], task["feature_ids"],
                            task["operational_metadata"], task["train_idx"],
                            task["env_ids"], method,
                        )
                        auprc = compute_auprc(
                            scores, task["feature_ids"], task["oracle_metadata"]
                        )
                        top5 = int(truth[np.argsort(scores, kind="stable")[::-1][:5]].sum())
                        n_leak = int(truth.sum())
                        ctx_results.append({
                            "ds": ds["name"], "mech": mech, "strength": sn,
                            "seed": seed, "method": method, "auprc": auprc,
                            "top5_recall": top5 / max(1, n_leak),
                            "n_leak": n_leak,
                        })
                        n_done += 1

                    # ── M3: Governance ──
                    i_scores = compute_diagnostic_scores(
                        task["X"], task["y"], task["feature_ids"],
                        task["operational_metadata"], task["train_idx"],
                        task["env_ids"], "I",
                    )
                    iase_scores = compute_diagnostic_scores(
                        task["X"], task["y"], task["feature_ids"],
                        task["operational_metadata"], task["train_idx"],
                        task["env_ids"], "I+A+S+E",
                    )

                    for budget in [0.05, 0.10, 0.20, 0.30]:
                        gm_field = run_governance_strategy(
                            task, i_scores, "field_budget", budget
                        )
                        for strategy in gov_strategies:
                            scores = iase_scores if strategy != "field_budget" else i_scores
                            gm = run_governance_strategy(task, scores, strategy, budget)
                            degenerate = (
                                gm["mask_hash"] == gm_field["mask_hash"]
                                if gm["status"] == GovernanceStatus.APPLIED.value else np.nan
                            )

                            gov_results.append({
                                "ds": ds["name"], "mech": mech, "strength": sn,
                                "seed": seed, "strategy": strategy, "budget": budget,
                                "status": gm["status"],
                                "status_reason": gm["status_reason"],
                                "leak_recall": gm["leak_recall"],
                                "legit_retention": gm["legit_retention"],
                                "residual_contam": gm["residual_contam"],
                                "n_quarantined": gm["n_quarantined"],
                                "review_units": gm["review_units"],
                                "degenerate": degenerate,
                            })
                            n_done += 1

                if n_done % 50 == 0:
                    print(f"  {n_done}/{n_ctx + n_gov} ({time.time()-t0:.0f}s)")

    # ── SAVE ──
    ctx_df = pd.DataFrame(ctx_results)
    ctx_df.to_csv(RESULTS / "context/meta_context_results.csv", index=False)
    gov_df = pd.DataFrame(gov_results)
    gov_df.to_csv(RESULTS / "governance/meta_governance_results.csv", index=False)

    # ── M2: CL5 Analysis ──
    print(f"\n{'='*60}")
    print("  M2: CL5 — Context Diagnostic Results")
    print("=" * 60)
    for method in diag_methods:
        sub = ctx_df[ctx_df["method"] == method]
        print(f"  {method:10s}: AUPRC={sub['auprc'].mean():.4f}, Top5={sub['top5_recall'].mean():.2f}")

    # I-only vs I+A+S+E
    i_only = ctx_df[ctx_df["method"] == "I"]["auprc"]
    iase = ctx_df[ctx_df["method"] == "I+A+S+E"]["auprc"]
    i_top5 = ctx_df[ctx_df["method"] == "I"]["top5_recall"]
    iase_top5 = ctx_df[ctx_df["method"] == "I+A+S+E"]["top5_recall"]
    delta_auprc = iase.mean() - i_only.mean()
    delta_top5 = iase_top5.mean() - i_top5.mean()
    print(f"\n  I-only AUPRC:      {i_only.mean():.4f}")
    print(f"  I+A+S+E AUPRC:      {iase.mean():.4f}")
    print(f"  Δ AUPRC:            {delta_auprc:+.4f}")
    print(f"  Δ Top-5 Recall:     {delta_top5:+.2f}")
    print(f"  CL5: {'CONFIRMED' if delta_auprc > 0.05 else 'PARTIALLY CONFIRMED' if delta_auprc > 0.02 else 'REFUTED'}")

    # ── M3: CL14 Analysis ──
    print(f"\n{'='*60}")
    print("  M3: CL14 — Governance Results")
    print("=" * 60)
    for strategy in gov_strategies:
        sub = gov_df[gov_df["strategy"] == strategy]
        non_degen = sub[~sub["degenerate"]]
        print(f"  {strategy:15s}: leak_recall={sub['leak_recall'].mean():.2f}, "
              f"legit_ret={sub['legit_retention'].mean():.2f}, "
              f"degen={sub['degenerate'].mean():.0%}")

    field = gov_df[gov_df["strategy"] == "field_budget"]
    group = gov_df[gov_df["strategy"] == "group_budget"]
    lc = gov_df[gov_df["strategy"] == "lifecycle"]
    gc = gov_df[gov_df["strategy"] == "graph_cut"]

    print(f"\n  Field budget recall:     {field['leak_recall'].mean():.2f}")
    print(f"  Group budget recall:     {group['leak_recall'].mean():.2f}")
    print(f"  Lifecycle recall:        {lc['leak_recall'].mean():.2f}")
    print(f"  Graph cut recall:        {gc['leak_recall'].mean():.2f}")

    delta_gf = group['leak_recall'].mean() - field['leak_recall'].mean()
    print(f"\n  Group vs Field Δ recall: {delta_gf:+.2f}")
    print(f"  CL14: {'CONFIRMED' if delta_gf > 0.10 else 'PARTIALLY CONFIRMED' if delta_gf > 0.05 else 'REFUTED' if delta_gf < 0 else 'UNCONFIRMED'}")

    print(f"\nTotal: {time.time()-t0:.0f}s")
    print("PHASE 12 META TIER: COMPLETE")


if __name__ == "__main__":
    main()
