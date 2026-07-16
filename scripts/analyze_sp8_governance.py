#!/usr/bin/env python3
"""analyze_sp8_governance.py — SP8 governance bootstrap analysis.

Reads artifacts/sp8/governance_clean.csv, computes per-budget and per-category
P3 vs P2 paired cluster-bootstrap over datasets, emits bootstrap_analysis.json
with observed_diff (mean of per-dataset effects) and bootstrap statistics.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "artifacts/sp8/governance_clean.csv"
OUT = ROOT / "artifacts/sp8/bootstrap_analysis.json"
SEED = 20260716
NBOOT = 10000

CAT = {"M01": "simple", "M02": "simple", "M06": "simple", "M10": "simple",
       "M04": "structured", "M05": "structured", "M08": "structured", "M09": "structured",
       "M03": "boundary", "M07": "boundary", "M11": "boundary"}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def paired_cluster_boot(df_p3, df_p2, nboot=NBOOT, seed=SEED):
    """Cluster bootstrap over datasets. Returns observed_diff, bootstrap_mean, CI, P3_better_prob."""
    rng = np.random.RandomState(seed)
    ds = sorted(df_p3["dataset_index"].unique())
    # Per-dataset effects
    p3g = {u: df_p3[df_p3.dataset_index == u]["strict_distance_reduction"].mean() for u in ds}
    p2g = {u: df_p2[df_p2.dataset_index == u]["strict_distance_reduction"].mean() for u in ds}
    observed_diff = float(np.mean([p3g[u] - p2g[u] for u in ds]))
    # Bootstrap
    boot_diffs = [np.mean([p3g[u] - p2g[u] for u in rng.choice(ds, len(ds), True)]) for _ in range(nboot)]
    boot_mean = float(np.mean(boot_diffs))
    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    p3_better = float(np.mean(np.asarray(boot_diffs) > 0))
    return observed_diff, boot_mean, ci_lo, ci_hi, p3_better, {f"ds_{int(u)}": {"p3_sdr": float(p3g[u]), "p2_sdr": float(p2g[u]),
                        "diff": float(p3g[u] - p2g[u])} for u in ds}


def main():
    d = pd.read_csv(CSV).drop_duplicates("run_id", keep="last")
    d["cat"] = d["mechanism"].map(CAT)
    results = {}
    for label, (p3, p2) in [
        ("budget_0.01", (d[(d.policy == "P3_blind_mi") & (d.budget_fraction == 0.01)],
                         d[(d.policy == "P2_random") & (d.budget_fraction == 0.01)])),
        ("budget_0.05", (d[(d.policy == "P3_blind_mi") & (d.budget_fraction == 0.05)],
                         d[(d.policy == "P2_random") & (d.budget_fraction == 0.05)])),
        ("budget_0.10", (d[(d.policy == "P3_blind_mi") & (d.budget_fraction == 0.10)],
                         d[(d.policy == "P2_random") & (d.budget_fraction == 0.10)])),
        ("budget_0.20", (d[(d.policy == "P3_blind_mi") & (d.budget_fraction == 0.20)],
                         d[(d.policy == "P2_random") & (d.budget_fraction == 0.20)])),
    ]:
        obs_diff, boot_mean, ci_lo, ci_hi, p3b, ds_eff = paired_cluster_boot(p3, p2)
        results[label] = {"observed_diff": round(obs_diff, 6), "bootstrap_mean": round(boot_mean, 6),
                          "ci_lo": round(ci_lo, 6), "ci_hi": round(ci_hi, 6),
                          "p3_better_prob": round(p3b, 6),
                          "p3_sdr": round(float(p3["strict_distance_reduction"].mean()), 6),
                          "p2_sdr": round(float(p2["strict_distance_reduction"].mean()), 6),
                          "p3_recall": round(float(p3["leak_recall"].mean()), 4),
                          "p3_retention": round(float(p3["legit_retention"].mean()), 4),
                          "dataset_effects": ds_eff}

    # Per category at 20%
    p3_20 = d[(d.policy == "P3_blind_mi") & (d.budget_fraction == 0.20)]
    p2_20 = d[(d.policy == "P2_random") & (d.budget_fraction == 0.20)]
    for cat in ["simple", "structured", "boundary"]:
        p3c = p3_20[p3_20.cat == cat]; p2c = p2_20[p2_20.cat == cat]
        if len(p3c) < 10: continue
        obs_diff, boot_mean, ci_lo, ci_hi, p3b, _ = paired_cluster_boot(p3c, p2c)
        results[f"category_{cat}"] = {"observed_diff": round(obs_diff, 6), "bootstrap_mean": round(boot_mean, 6),
                                       "ci_lo": round(ci_lo, 6), "ci_hi": round(ci_hi, 6),
                                       "p3_better_prob": round(p3b, 6),
                                       "p3_sdr": round(float(p3c["strict_distance_reduction"].mean()), 6),
                                       "p2_sdr": round(float(p2c["strict_distance_reduction"].mean()), 6),
                                       "p3_recall": round(float(p3c["leak_recall"].mean()), 4),
                                       "p3_retention": round(float(p3c["legit_retention"].mean()), 4)}

    manifest = {
        "analysis_seed": SEED, "bootstrap_reps": NBOOT, "bootstrap_unit": "dataset_index",
        "runner_sha": sha("scripts/run_sp8_clean.py"),
        "csv_sha": sha(CSV),
        "analysis_script_sha": sha(__file__),
        "policy_registry_sha": sha("artifacts/sp8/protocol/policy_registry.yaml"),
        "bundle_manifest_sha": sha("artifacts/sp6/sp6_bundle_manifest.csv"),
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2))
    # Print summary
    for k in ["budget_0.20", "category_simple", "category_structured"]:
        v = results.get(k, {})
        if "observed_diff" in v:
            print(f"{k}: obs_diff {v['observed_diff']:+.4f} boot_mean {v['bootstrap_mean']:+.4f} "
                  f"CI[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}] P3_better={v.get('p3_better_prob',0):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
