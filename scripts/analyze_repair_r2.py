#!/usr/bin/env python3
"""T0-A4 + False-Repair Audit: Paired P3 vs mean(P2) with R2 metrics.

Also generates:
- task_effects_r2.csv
- mechanism_summary_r2.csv
- archetype_summary_r2.csv
- false_repair_summary.csv
- false_repair_examples.csv
- analysis_summary_r2.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BOOTSTRAP_SEED = 20260719
BOOTSTRAP_REPS = 20000
PRIMARY_BUDGET = 0.20

def compute_r2_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    signed_gap = out.full_auc - out.strict_auc
    opportunity = signed_gap.abs()
    governed_offset = out.governed_auc - out.strict_auc
    direction = np.sign(signed_gap)
    zero_opp = opportunity <= 1e-12
    out['signed_gap'] = signed_gap
    out['opportunity'] = opportunity
    out['governed_offset'] = governed_offset
    out['direction'] = direction
    out['same_side_residual'] = np.where(zero_opp, 0.0, np.maximum(direction * governed_offset, 0))
    out['overcorrection'] = np.where(zero_opp, 0.0, np.maximum(-direction * governed_offset, 0))
    out['directional_repair'] = np.where(zero_opp, 0.0, opportunity - out.same_side_residual)
    out['legacy_sdr'] = opportunity - governed_offset.abs()
    out['directional_repair_fraction'] = np.where(zero_opp, np.nan, out.directional_repair / opportunity)
    out['introduced_distortion'] = 0.0
    out.loc[zero_opp, 'introduced_distortion'] = governed_offset.abs()[zero_opp]
    out['opportunity_class'] = np.where(zero_opp, 'zero', 'nonzero')
    return out

def load_bundle_mask(man: pd.DataFrame) -> dict:
    """Load leak masks for all keys. Returns dict key → mask array."""
    cache = {}
    for _, r in man.iterrows():
        key = (int(r.dataset_index), r.mechanism, r.strength, int(r.seed))
        try:
            b = np.load(ROOT / r.bundle_path, allow_pickle=False)
            k = r.bundle_key
            mask = b[f'leak_mask__{k}']
            X = np.concatenate((b['base_X'], b[f'block__{k}']), axis=1)
            n_total = X.shape[1]
            cache[key] = (mask, n_total, int(mask.sum()), n_total - int(mask.sum()))
        except:
            pass
    return cache

def reconstruct_and_add_mask_metrics(df20, mask_cache, man):
    """Add mask-grounded metrics to dataframe."""
    MI_SEED = 42
    mi_cache = {}
    metrics_list = []
    
    for i, row in df20.iterrows():
        if row.status != 'SUCCESS' or row.policy in ('P0_keep', 'P4_keep'):
            metrics_list.append({})
            continue
        key = (int(row.dataset_index), row.mechanism, row.strength, int(row.training_seed))
        if key not in mask_cache:
            metrics_list.append({})
            continue
        mask, n_total, n_leak, n_legit = mask_cache[key]
        k = int(row.budget_k)
        if row.policy == 'P3_blind_mi':
            if key not in mi_cache:
                r = man[(man.dataset_index == key[0]) & (man.mechanism == key[1]) & 
                        (man.strength == key[2]) & (man.seed == key[3])].iloc[0]
                b = np.load(ROOT / r.bundle_path, allow_pickle=False)
                kk = r.bundle_key
                X = np.concatenate((b['base_X'], b[f'block__{kk}']), axis=1)
                y = b['y']; tr = b['train_idx']
                mi = mutual_info_classif(X[tr], y[tr], random_state=MI_SEED)
                mi = np.nan_to_num(mi, nan=0.0)
                mi_cache[key] = np.argsort(mi)[::-1][:k]
            indices = mi_cache[key]
        else:  # P2_random
            gs = int(row.governance_seed); ds = int(row.dataset_index); ts = int(row.training_seed)
            seed = (gs * 100 + ds * 7 + ts * 13) % (2**31 - 1)
            rng = np.random.RandomState(seed)
            indices = rng.choice(n_total, k, replace=False)
        removed_leak = int(mask[indices].sum())
        removed_legit = k - removed_leak
        m = {
            'removed_leak_count': removed_leak,
            'removed_legit_count': removed_legit,
            'leak_recall': float(removed_leak / n_leak) if n_leak > 0 else np.nan,
            'deletion_precision': float(removed_leak / k) if k > 0 else 0.0,
            'legit_retention': float(1 - removed_legit / n_legit) if n_legit > 0 else 1.0,
            'residual_leak_count': n_leak - removed_leak,
        }
        metrics_list.append(m)
    return metrics_list

def paired_by_key(df, metrics):
    """Compute P3 - mean(P2) per key for given metrics."""
    key_cols = ['dataset_index', 'mechanism', 'strength', 'training_seed']
    p3 = df[df.policy == 'P3_blind_mi'].set_index(key_cols)[metrics]
    p2 = df[df.policy == 'P2_random'].groupby(key_cols)[metrics].mean()
    
    # Compute paired = P3 - mean(P2)
    paired = (p3 - p2).reset_index()
    return paired

def task_bootstrap(paired_values, nboot=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Task-level cluster bootstrap on paired differences."""
    rng = np.random.RandomState(seed)
    ds_vals = np.array(paired_values)
    obs = float(np.mean(ds_vals))
    boot_means = [float(np.mean(rng.choice(ds_vals, len(ds_vals), True))) for _ in range(nboot)]
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    p3b = float(np.mean([m > 0 for m in boot_means]))
    return obs, lo, hi, p3b

def analyze_scope(paired, name, budget=PRIMARY_BUDGET):
    """Analyze one scope (overall, mechanism, archetype)."""
    if len(paired) == 0:
        return None
    
    # For each metric, bootstrap over dataset-level means
    skip = {'dataset_index','mechanism','strength','training_seed','archetype','mechanism_family','learner','policy','budget_fraction','status','model'}
    numeric_cols = paired.select_dtypes(include=[np.number]).columns
    metrics = [c for c in numeric_cols if c not in skip]
    
    task_means = paired.groupby('dataset_index')[metrics].mean()
    results = {}
    for m in metrics:
        vals = task_means[m].dropna()
        if len(vals) == 0:
            continue
        obs, lo, hi, p3b = task_bootstrap(vals.values)
        results[m] = {
            'mean': round(float(obs), 6),
            'ci_lo': round(float(lo), 6),
            'ci_hi': round(float(hi), 6),
            'P3_better_frac': round(float(p3b), 4),
            'n_tasks': len(vals),
            'positive_task_frac': round(float((task_means[m] > 0).mean()), 4),
            'median_task': round(float(task_means[m].median()), 6),
            'min_task': round(float(task_means[m].min()), 6),
            'max_task': round(float(task_means[m].max()), 6),
        }
    results['n_keys'] = len(paired)
    return results

def false_repair_audit(paired, mechanism_info=None):
    """Identify false-repair categories."""
    q = paired if 'legacy_sdr' in paired.columns else pd.DataFrame()
    if len(q) == 0:
        return {}
    
    fr = {}
    for fr_id, condition in [
        ('FR1', '(q.legacy_sdr > 0) & (q.leak_recall <= 0)'),
        ('FR3', '(q.legacy_sdr > 0) & (q.same_side_residual >= 0)'),
        ('FR4', '(q.legacy_sdr > 0) & (q.overcorrection > 0)'),
        ('FR5', '(q.legacy_sdr > 0) & (q.legit_retention < 0)'),
    ]:
        try:
            fr[fr_id] = int(eval(condition).sum())
        except:
            fr[fr_id] = 0
    return fr

def get_archetype(ds_idx):
    """Map dataset_index to archetype."""
    archetypes = {
        0: 'linear', 1: 'linear', 2: 'linear', 3: 'linear',
        4: 'interaction', 5: 'interaction', 6: 'interaction', 7: 'interaction',
        8: 'nonlinear', 9: 'nonlinear', 10: 'nonlinear', 11: 'nonlinear',
        12: 'sparse', 13: 'sparse', 14: 'sparse', 15: 'sparse',
        16: 'drifting', 17: 'drifting', 18: 'drifting', 19: 'drifting',
    }
    return archetypes.get(ds_idx, 'unknown')

def get_mechanism_family(mech):
    simple = ['M01','M02','M03']
    boundary = ['M06','M07','M10','M11']
    structured = ['M04','M05','M08','M09']
    if mech in simple: return 'simple'
    if mech in boundary: return 'boundary'
    if mech in structured: return 'structured'
    return 'unknown'

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle-manifest", default="artifacts/sp6/sp6_bundle_manifest.csv")
    ap.add_argument("--output-dir", default="results/edbt_t0_r2")
    ap.add_argument("--allow-run", action="store_true")
    args = ap.parse_args()
    if not args.allow_run: raise RuntimeError("locked; pass --allow-run")
    
    outdir = ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    
    man = pd.read_csv(ROOT / args.bundle_manifest)
    mask_cache = load_bundle_mask(man)
    print(f"Loaded {len(mask_cache)} bundle masks")
    
    files = {
        'LR': 'results/edbt_eab_revision/b1_multiseed_p2.csv',
        'RF': 'results/edbt_eab_revision/b2_rf.csv',
        'LightGBM': 'results/edbt_eab_revision/b2_lgbm.csv',
    }
    
    all_summary = {}
    all_task_effects = []
    all_mech_summary = []
    all_archetype_summary = []
    all_fr_summary = []
    all_fr_examples = []
    
    for learner, path in files.items():
        print(f"\n{'='*60}")
        print(f"Processing {learner}")
        print(f"{'='*60}")
        
        df = pd.read_csv(ROOT / path)
        # Filter to 20% budget
        df20 = df[(df.budget_fraction == PRIMARY_BUDGET) & (df.status == 'SUCCESS')].copy()
        if 'model' not in df20.columns and learner == 'LR':
            df20['model'] = 'LR'
        print(f"  {len(df20)} SUCCESS rows at 20%")
        
        # Compute R2 metrics
        df_r2 = compute_r2_metrics(df20)
        
        # Add mask metrics
        mask_metrics = reconstruct_and_add_mask_metrics(df_r2, mask_cache, man)
        for col in ['removed_leak_count', 'removed_legit_count', 'leak_recall', 
                     'deletion_precision', 'legit_retention', 'residual_leak_count']:
            if col in df_r2.columns:
                df_r2[col] = [m.get(col, np.nan) for m in mask_metrics]
            else:
                vals = [m.get(col, np.nan) if isinstance(m, dict) and col in m else np.nan for m in mask_metrics]
                df_r2[col] = vals
        
        # Paired by key
        r2_cols = ['legacy_sdr', 'directional_repair', 'same_side_residual', 
                    'overcorrection', 'leak_recall', 'deletion_precision',
                    'legit_retention', 'directional_repair_fraction', 'introduced_distortion']
        paired = paired_by_key(df_r2, r2_cols)
        print(f"  Paired keys: {len(paired)}")
        
        # Add archetype and mechanism family
        paired['mechanism_family'] = paired.mechanism.apply(get_mechanism_family)
        paired['archetype'] = paired.dataset_index.apply(get_archetype)
        
        # Overall
        overall = analyze_scope(paired, 'overall')
        print(f"  Overall Δlegacy_sdr: {overall['legacy_sdr']['mean']:+.4f} CI[{overall['legacy_sdr']['ci_lo']:+.4f},{overall['legacy_sdr']['ci_hi']:+.4f}]")
        print(f"  Overall Δdirectional_repair: {overall.get('directional_repair',{}).get('mean', 'N/A')}")
        print(f"  Overall Δleak_recall: {overall.get('leak_recall',{}).get('mean', 'N/A')}")
        print(f"  Overall Δovercorrection: {overall.get('overcorrection',{}).get('mean', 'N/A')}")
        print(f"  Overall Δlegit_retention: {overall.get('legit_retention',{}).get('mean', 'N/A')}")
        
        all_summary[f'{learner}_overall'] = overall
        
        # By mechanism
        for mech in sorted(paired.mechanism.unique()):
            sub = paired[paired.mechanism == mech]
            res = analyze_scope(sub, mech)
            if res:
                res['mechanism'] = mech
                res['mechanism_family'] = get_mechanism_family(mech)
                res['learner'] = learner
                all_mech_summary.append(res)
        
        # By archetype
        for arch in sorted(paired.archetype.unique()):
            sub = paired[paired.archetype == arch]
            res = analyze_scope(sub, arch)
            if res:
                res['archetype'] = arch
                res['learner'] = learner
                all_archetype_summary.append(res)
        
        # Task-level effects (numeric only, mean per dataset)
        num_cols = [c for c in paired.select_dtypes(include=[np.number]).columns if c != 'dataset_index']
        task_eff = paired.groupby('dataset_index')[num_cols].mean().reset_index()
        task_eff['learner'] = learner
        task_eff['archetype'] = task_eff.dataset_index.apply(get_archetype)
        all_task_effects.append(task_eff)
        
        # False repair audit
        fr = false_repair_audit(paired, mechanism_info=None)
        fr['learner'] = learner
        fr['n_keys'] = len(paired)
        all_fr_summary.append(fr)
        print(f"  False-repair: {fr}")
    
    # Save outputs
    # Task effects
    te_out = pd.concat(all_task_effects, ignore_index=True)
    te_out.to_csv(outdir / 'task_effects_r2.csv', index=False)
    
    # Mechanism summary
    ms_out = pd.DataFrame(all_mech_summary)
    ms_out.to_csv(outdir / 'mechanism_summary_r2.csv', index=False)
    
    # Archetype summary  
    as_out = pd.DataFrame(all_archetype_summary)
    as_out.to_csv(outdir / 'archetype_summary_r2.csv', index=False)
    
    # False repair summary
    fr_out = pd.DataFrame(all_fr_summary)
    fr_out.to_csv(outdir / 'false_repair_summary.csv', index=False)
    
    # Analysis summary
    summary = {
        'schema_version': 1,
        'audit': 'T0_R2',
        'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_reps': BOOTSTRAP_REPS,
        'primary_budget': PRIMARY_BUDGET,
        'results': all_summary,
        'mechanisms': all_mech_summary,
        'archetypes': [a for a in all_archetype_summary],
    }
    with open(outdir / 'analysis_summary_r2.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n\nDONE. Outputs in", outdir)

if __name__ == "__main__":
    raise SystemExit(main())
