#!/usr/bin/env python3
"""analyze_governance_revision.py — Formal multi-seed EDBT governance analysis.

Reads B1 (LR multi-seed), B2 (RF, LightGBM) CSVs.
Generates ALL statistics: A1 mechanism-level, A2 gap stratification, A3 archetypes,
three-model comparison, learner interaction bootstrap.
Outputs: analysis_summary.json, updated CSVs, per-mechanism tables.
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260717
NBOOT = 10000
BUDGET = 0.20

CATS = {"M01": "simple", "M02": "simple", "M06": "simple", "M10": "simple",
        "M04": "structured", "M05": "structured", "M08": "structured", "M09": "structured",
        "M03": "boundary", "M07": "boundary", "M11": "boundary"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def task_boot(p3g, p2g, nb=NBOOT):
    rng = np.random.RandomState(SEED)
    ds = sorted(set(p3g.keys()) & set(p2g.keys()))
    observed = float(np.mean([p3g[u] - p2g[u] for u in ds]))
    diffs = [np.mean([p3g[u] - p2g[u] for u in rng.choice(ds, len(ds), True)]) for _ in range(nb)]
    bmean = float(np.mean(diffs))
    lo = float(np.percentile(diffs, 2.5)); hi = float(np.percentile(diffs, 97.5))
    p3b = float(np.mean([np.mean([p3g[u] for u in rng.choice(ds, len(ds), True)]) >
                         np.mean([p2g[u] for u in rng.choice(ds, len(ds), True)]) for _ in range(nb)]))
    return observed, bmean, lo, hi, p3b


def learner_interaction_boot(mg_a, mg_b, col_a='paired', col_b='paired', nb=NBOOT):
    """Bootstrap the difference between two learners' paired effects on matched keys."""
    key_cols = ['dataset_index', 'mechanism', 'strength', 'training_seed']
    merged = mg_a[key_cols + [col_a]].merge(mg_b[key_cols + [col_b]], on=key_cols, suffixes=('_a', '_b'))
    merged['diff'] = merged[f'{col_a}_a'] - merged[f'{col_b}_b']
    rng = np.random.RandomState(SEED)
    ds = sorted(merged['dataset_index'].unique())
    g = {u: merged[merged.dataset_index == u]['diff'].values for u in ds}
    diffs = [np.concatenate([g[u] for u in rng.choice(ds, len(ds), True)]).mean() for _ in range(nb)]
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def get_paired(df):
    ok = df[df.status == 'SUCCESS']
    p2 = ok[ok.policy == 'P2_random'].groupby(['dataset_index', 'mechanism', 'strength', 'training_seed'])['strict_distance_reduction'].mean()
    p3 = ok[ok.policy == 'P3_blind_mi'].set_index(['dataset_index', 'mechanism', 'strength', 'training_seed'])['strict_distance_reduction']
    mg = p3.reset_index().merge(p2.reset_index().rename(columns={'strict_distance_reduction': 'P2_bar_SDR'}),
                                 on=['dataset_index', 'mechanism', 'strength', 'training_seed'])
    mg['paired'] = mg['strict_distance_reduction'] - mg['P2_bar_SDR']
    mg['family'] = mg['mechanism'].map(CATS)
    # initial gap from P0
    p0 = ok[ok.policy == 'P0_keep'][['dataset_index', 'mechanism', 'strength', 'training_seed', 'initial_gap']].drop_duplicates()
    mg = mg.merge(p0, on=['dataset_index', 'mechanism', 'strength', 'training_seed'], how='left')
    return mg


def compute_summary(models_dict):
    results = {}
    # Overall per model
    for name, mg in models_dict.items():
        p3g = mg.groupby('dataset_index')['strict_distance_reduction'].mean()
        p2g = mg.groupby('dataset_index')['P2_bar_SDR'].mean()
        obs, bmean, lo, hi, p3b = task_boot(dict(p3g), dict(p2g))
        results[f'{name}_overall'] = {"paired": round(obs, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                                       "P3_better": round(p3b, 4), "n_keys": len(mg)}
    # Per family per model
    for fam in ['simple', 'structured', 'boundary']:
        for name, mg in models_dict.items():
            s = mg[mg.family == fam]
            p3g = s.groupby('dataset_index')['strict_distance_reduction'].mean()
            p2g = s.groupby('dataset_index')['P2_bar_SDR'].mean()
            obs, _, lo, hi, p3b = task_boot(dict(p3g), dict(p2g))
            results[f'{name}_{fam}'] = {"paired": round(obs, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)}
    # Per mechanism per model
    for mech in sorted(CATS):
        for name, mg in models_dict.items():
            s = mg[mg.mechanism == mech]
            p3g = s.groupby('dataset_index')['strict_distance_reduction'].mean()
            p2g = s.groupby('dataset_index')['P2_bar_SDR'].mean()
            obs, _, lo, hi, _ = task_boot(dict(p3g), dict(p2g))
            gap = s['initial_gap'].mean() if 'initial_gap' in s.columns else 0
            results[f'{name}_{mech}'] = {"paired": round(obs, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                                          "initial_gap": round(float(gap), 6)}
    # Learner interaction
    names = list(models_dict.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            diff, lo, hi = learner_interaction_boot(models_dict[a], models_dict[b])
            results[f'interaction_{a}_vs_{b}'] = {"diff": round(diff, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)}
    # Gap stratification (LR only — gap is model-independent)
    lr_mg = models_dict['LR']
    lr_mg['gap_quartile'] = pd.qcut(lr_mg['initial_gap'].fillna(0), 4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
    for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
        s = lr_mg[lr_mg.gap_quartile == q]
        p3g = s.groupby('dataset_index')['strict_distance_reduction'].mean()
        p2g = s.groupby('dataset_index')['P2_bar_SDR'].mean()
        obs, _, lo, hi, _ = task_boot(dict(p3g), dict(p2g))
        results[f'gap_quartile_{q}'] = {"paired": round(obs, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                                         "gap_range": [round(float(s.initial_gap.min()), 4), round(float(s.initial_gap.max()), 4)],
                                         "n_keys": len(s)}
    # Archetype (LR, gap is model-independent)
    core = pd.read_csv(ROOT / 'results/corrected_v2/core_cpu_cells.csv')
    arch_map = core[['dataset_index', 'archetype']].drop_duplicates()
    lr_arch = lr_mg.merge(arch_map, on='dataset_index')
    for arch in sorted(lr_arch.archetype.unique()):
        s = lr_arch[lr_arch.archetype == arch]
        p3g = s.groupby('dataset_index')['strict_distance_reduction'].mean()
        p2g = s.groupby('dataset_index')['P2_bar_SDR'].mean()
        obs, _, lo, hi, _ = task_boot(dict(p3g), dict(p2g))
        results[f'archetype_{arch}'] = {"paired": round(obs, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                                         "n_tasks": len(p3g)}
    # LOAO
    all_ds = sorted(core.dataset_index.unique())
    for arch in sorted(lr_arch.archetype.unique()):
        arch_ds = arch_map[arch_map.archetype == arch]['dataset_index'].unique()
        loao_ds = [d for d in all_ds if d not in arch_ds]
        s = lr_arch[lr_arch.dataset_index.isin(loao_ds)]
        p3g = s.groupby('dataset_index')['strict_distance_reduction'].mean()
        p2g = s.groupby('dataset_index')['P2_bar_SDR'].mean()
        obs, _, lo, hi, _ = task_boot(dict(p3g), dict(p2g))
        results[f'archetype_LOAO_{arch}'] = {"paired": round(obs, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)}
    return results


def write_a1(table_dir, models_dict):
    """A1: mechanism-level decomposition for LR (gap is model-independent)."""
    lr_mg = models_dict['LR']
    rows = []
    for mech in sorted(lr_mg.mechanism.unique()):
        s = lr_mg[lr_mg.mechanism == mech]
        p3g = s.groupby('dataset_index')['strict_distance_reduction'].mean()
        p2g = s.groupby('dataset_index')['P2_bar_SDR'].mean()
        obs, bmean, lo, hi, p3b = task_boot(dict(p3g), dict(p2g))
        gap = s['initial_gap'].mean() if 'initial_gap' in s.columns else 0
        rows.append({"mechanism": mech, "family": CATS[mech], "initial_gap": round(float(gap), 6),
                     "P3_SDR": round(s.strict_distance_reduction.mean(), 6),
                     "P2_bar_SDR": round(s.P2_bar_SDR.mean(), 6),
                     "paired_effect": round(obs, 6), "bootstrap_mean": round(bmean, 6),
                     "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                     "P3_better_prob": round(p3b, 6), "n_cells": len(s)})
    pd.DataFrame(rows).to_csv(table_dir / "a1_mechanism_level.csv", index=False)
    print(f"  A1: {len(rows)} mechanisms")


def write_a2(table_dir, models_dict):
    """A2: initial-gap stratification."""
    lr_mg = models_dict['LR']
    lr_mg['gap_quartile'] = pd.qcut(lr_mg['initial_gap'].fillna(0), 4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'])
    rows = []
    for q in ['Q1_low', 'Q2', 'Q3', 'Q4_high']:
        s = lr_mg[lr_mg.gap_quartile == q]
        p3g = s.groupby('dataset_index')['strict_distance_reduction'].mean()
        p2g = s.groupby('dataset_index')['P2_bar_SDR'].mean()
        obs, _, lo, hi, _ = task_boot(dict(p3g), dict(p2g))
        rows.append({"quartile": q, "gap_range": f"[{s.initial_gap.min():.4f},{s.initial_gap.max():.4f}]",
                     "n_keys": len(s), "paired_effect": round(obs, 6),
                     "ci_lo": round(lo, 6), "ci_hi": round(hi, 6)})
    pd.DataFrame(rows).to_csv(table_dir / "a2_gap_stratification.csv", index=False)
    print(f"  A2: {len(rows)} quartiles")


def write_a3(table_dir, models_dict):
    """A3: archetype robustness."""
    core = pd.read_csv(ROOT / 'results/corrected_v2/core_cpu_cells.csv')
    arch_map = core[['dataset_index', 'archetype']].drop_duplicates()
    lr_arch = models_dict['LR'].merge(arch_map, on='dataset_index')
    rows = []
    for arch in sorted(lr_arch.archetype.unique()):
        s = lr_arch[lr_arch.archetype == arch]
        p3g = s.groupby('dataset_index')['strict_distance_reduction'].mean()
        p2g = s.groupby('dataset_index')['P2_bar_SDR'].mean()
        obs, _, lo, hi, _ = task_boot(dict(p3g), dict(p2g))
        pos = (s.groupby('dataset_index')['paired'].mean() > 0).mean()
        rows.append({"archetype": arch, "n_tasks": len(p3g), "paired_effect": round(obs, 6),
                     "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                     "positive_task_proportion": round(pos, 4)})
    all_ds = sorted(core.dataset_index.unique())
    for arch in sorted(lr_arch.archetype.unique()):
        arch_ds = arch_map[arch_map.archetype == arch]['dataset_index'].unique()
        loao_ds = [d for d in all_ds if d not in arch_ds]
        s = lr_arch[lr_arch.dataset_index.isin(loao_ds)]
        p3g = s.groupby('dataset_index')['strict_distance_reduction'].mean()
        p2g = s.groupby('dataset_index')['P2_bar_SDR'].mean()
        obs, _, lo, hi, _ = task_boot(dict(p3g), dict(p2g))
        rows.append({"archetype": f"LOAO-{arch}", "n_tasks": len(p3g), "paired_effect": round(obs, 6),
                     "ci_lo": round(lo, 6), "ci_hi": round(hi, 6), "positive_task_proportion": 0})
    pd.DataFrame(rows).to_csv(table_dir / "a3_archetype.csv", index=False)
    print(f"  A3: {len(rows)} archetypes (incl LOAO)")


def main():
    out_dir = ROOT / "results/edbt_eab_revision"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    lr = pd.read_csv(out_dir / "b1_multiseed_p2.csv")
    rf = pd.read_csv(out_dir / "b2_rf.csv")
    lgbm = pd.read_csv(out_dir / "b2_lgbm.csv")
    models = {"LR": get_paired(lr), "RF": get_paired(rf), "LightGBM": get_paired(lgbm)}

    print("=== EDBT Governance Revision — Formal Analysis ===")
    write_a1(out_dir, models)
    write_a2(out_dir, models)
    write_a3(out_dir, models)

    summary = compute_summary(models)
    # Add metadata
    summary["analysis_seed"] = SEED
    summary["bootstrap_reps"] = NBOOT
    summary["bootstrap_unit"] = "dataset_index"
    summary["budget"] = BUDGET
    summary["input_hashes"] = {
        "b1_lr": sha(out_dir / "b1_multiseed_p2.csv"),
        "b2_rf": sha(out_dir / "b2_rf.csv"),
        "b2_lgbm": sha(out_dir / "b2_lgbm.csv"),
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print key results
    for k in ["LR_overall", "RF_overall", "LightGBM_overall",
              "LR_structured", "interaction_RF_vs_LR", "interaction_LightGBM_vs_LR",
              "gap_quartile_Q4_high", "archetype_sparse", "archetype_LOAO_sparse"]:
        if k in summary:
            v = summary[k]
            print(f"  {k}: paired={v.get('paired',v.get('diff','?')):+.4f} CI[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
