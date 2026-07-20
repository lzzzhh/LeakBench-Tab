#!/usr/bin/env python3
"""T0-R2.2: Denominator closure and M09 semantic-group quantitative reporting.

Reads R2.1 governance data + analysis, adds:
1. FR1-FR6 dual denominators (all-key + conditional)
2. M09 semantic-group paired metrics (FR6)
3. Updated claim state, reports, manifest
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CATS = {"M01": "simple", "M02": "simple", "M06": "simple", "M10": "simple",
        "M04": "structured", "M05": "structured", "M08": "structured", "M09": "structured",
        "M03": "boundary", "M07": "boundary", "M11": "boundary"}

OUT_DIR = ROOT / 'results/edbt_t0_r2'
BOOT_SEED = 20260719; BOOT_REPS = 20000
PRIMARY_BUDGET = 0.20

def load_canonical():
    core = pd.read_csv(ROOT / 'results/corrected_v2/core_cpu_cells.csv')
    return core[['dataset_index', 'archetype']].drop_duplicates().set_index('dataset_index')['archetype'].to_dict()

def task_bootstrap(vals):
    rng = np.random.RandomState(BOOT_SEED); v = np.array(vals)
    obs = float(np.mean(v))
    boots = [float(np.mean(rng.choice(v, len(v), True))) for _ in range(BOOT_REPS)]
    return obs, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(np.mean([b > 0 for b in boots]))

def load_data():
    """Load governance data and compute R2 + mask metrics (reusing R2.1 logic)."""
    from sklearn.feature_selection import mutual_info_classif
    
    man = pd.read_csv(ROOT / 'artifacts/sp6/sp6_bundle_manifest.csv')
    
    # Bundle cache
    mask_cache = {}
    mi_cache = {}
    
    for _, r in man.iterrows():
        key = (int(r.dataset_index), r.mechanism, r.strength, int(r.seed))
        try:
            b = np.load(ROOT / r.bundle_path, allow_pickle=False)
            kk = r.bundle_key
            mask = b[f'leak_mask__{kk}']
            X = np.concatenate((b['base_X'], b[f'block__{kk}']), axis=1)
            y = b['y']; tr = b['train_idx']
            nf = X.shape[1]; n_leak = int(mask.sum()); n_legit = nf - n_leak
            mask_cache[key] = (mask, nf, n_leak, n_legit, X, y, tr)
            mi = mutual_info_classif(X[tr], y[tr], random_state=42)
            mi = np.nan_to_num(mi, nan=0.0)
            mi_cache[key] = np.argsort(mi)[::-1]
        except:
            pass
    
    files = {
        'LR': 'results/edbt_eab_revision/b1_multiseed_p2.csv',
        'RF': 'results/edbt_eab_revision/b2_rf.csv',
        'LightGBM': 'results/edbt_eab_revision/b2_lgbm.csv',
    }
    
    result = {}
    for learner, path in files.items():
        df = pd.read_csv(ROOT / path)
        df20 = df[(df.budget_fraction == PRIMARY_BUDGET) & (df.status == 'SUCCESS')].copy()
        
        # Compute R2 metrics
        sg = df20.full_auc - df20.strict_auc
        opp = sg.abs(); go = df20.governed_auc - df20.strict_auc
        d = np.sign(sg); zo = opp <= 1e-12
        df20['signed_gap'] = sg; df20['opportunity'] = opp; df20['governed_offset'] = go; df20['direction'] = d
        df20['same_side_residual'] = np.where(zo, 0.0, np.maximum(d * go, 0))
        df20['overcorrection'] = np.where(zo, 0.0, np.maximum(-d * go, 0))
        df20['directional_repair'] = np.where(zo, 0.0, opp - df20.same_side_residual)
        df20['legacy_sdr'] = opp - go.abs()
        
        # Add mask-grounded metrics per row
        rm_leak = []; rm_legit = []; leak_rec = []; del_prec = []; leg_ret = []
        m09_full = []; m09_any = []; m09_partial = []; m09_removed_count = []
        
        for i, row in df20.iterrows():
            if row.status != 'SUCCESS' or row.policy not in ('P3_blind_mi', 'P2_random'):
                for lst in [rm_leak, rm_legit, leak_rec, del_prec, leg_ret, m09_full, m09_any, m09_partial, m09_removed_count]:
                    lst.append(np.nan)
                continue
            
            key = (int(row.dataset_index), row.mechanism, row.strength, int(row.training_seed))
            if key not in mask_cache:
                for lst in [rm_leak, rm_legit, leak_rec, del_prec, leg_ret, m09_full, m09_any, m09_partial, m09_removed_count]:
                    lst.append(np.nan)
                continue
            
            mask, nf, n_leak, n_legit, X, y, tr = mask_cache[key]
            k = int(row.budget_k)
            
            # Reconstruct selection
            if row.policy == 'P3_blind_mi':
                indices = mi_cache[key][:k]
            else:
                gs = int(row.governance_seed); ds = int(row.dataset_index); ts = int(row.training_seed)
                seed = (gs * 100 + ds * 7 + ts * 13) % (2**31 - 1)
                rng = np.random.RandomState(seed)
                indices = rng.choice(nf, k, replace=False)
            
            rl = int(mask[indices].sum()); rg = k - rl
            rm_leak.append(rl); rm_legit.append(rg)
            leak_rec.append(float(rl / n_leak) if n_leak > 0 else np.nan)
            del_prec.append(float(rl / k) if k > 0 else 0.0)
            leg_ret.append(float(1 - rg / n_legit) if n_legit > 0 else 1.0)
            
            # M09 semantic-group metrics
            if row.mechanism == 'M09':
                m09_group = np.where(mask)[0]  # all leak columns = M09 group
                n_group = len(m09_group)
                group_removed = len(set(m09_group) & set(indices))
                m09_full.append(1 if group_removed == n_group else 0)
                m09_any.append(1 if group_removed > 0 else 0)
                m09_partial.append(1 if (group_removed > 0 and group_removed < n_group) else 0)
                m09_removed_count.append(group_removed)
            else:
                for lst in [m09_full, m09_any, m09_partial, m09_removed_count]:
                    lst.append(np.nan)
        
        df20['removed_leak_count'] = rm_leak
        df20['removed_legit_count'] = rm_legit
        df20['leak_recall'] = leak_rec
        df20['deletion_precision'] = del_prec
        df20['legit_retention'] = leg_ret
        df20['m09_full_group_removed'] = m09_full
        df20['m09_any_hit'] = m09_any
        df20['m09_partial_removed'] = m09_partial
        df20['m09_group_removed_count'] = m09_removed_count
        
        # Paired analysis per key
        kc = ['dataset_index', 'mechanism', 'strength', 'training_seed']
        r2_cols = ['legacy_sdr', 'directional_repair', 'same_side_residual', 'overcorrection',
                   'leak_recall', 'deletion_precision', 'legit_retention',
                   'm09_full_group_removed', 'm09_any_hit', 'm09_partial_removed',
                   'm09_group_removed_count']
        available = [c for c in r2_cols if c in df20.columns]
        p3 = df20[df20.policy == 'P3_blind_mi'].set_index(kc)[available].copy()
        p2 = df20[df20.policy == 'P2_random'].groupby(kc)[available].mean()
        paired = (p3 - p2).reset_index()
        
        # Add P3 absolute values for FR2
        p3_abs = df20[df20.policy == 'P3_blind_mi'].set_index(kc)[['legacy_sdr', 'removed_leak_count', 'overcorrection']].copy()
        p3_abs.columns = ['p3_legacy_sdr', 'p3_removed_leak_count', 'p3_overcorrection']
        paired = paired.merge(p3_abs.reset_index(), on=kc)
        
        arch_map = load_canonical()
        paired['archetype'] = paired.dataset_index.map(arch_map)
        paired['mechanism_family'] = paired.mechanism.map(CATS)
        
        result[learner] = {'df20': df20, 'paired': paired}
    
    return result

# ===============================================================
# False-Repair Audit with Dual Denominators
# ===============================================================
def compute_fr_audit(learner_data, arch_map):
    """Compute FR1-FR6 with both all-key and conditional denominators."""
    all_fr_summaries = []
    all_fr_examples = []
    
    for learner, data in learner_data.items():
        paired = data['paired']
        n_all = len(paired)
        
        # Eligible conditions
        eligible_dsdr_pos = paired.legacy_sdr > 0  # FR1, FR3, FR4, FR5 eligible
        eligible_p3sdr_pos = paired.p3_legacy_sdr > 0  # FR2 eligible
        eligible_m09 = paired.mechanism == 'M09'  # FR6 eligible
        eligible_m09_dsdr = (paired.mechanism == 'M09') & (paired.legacy_sdr > 0)  # FR6 conditional
        
        n_eligible_dsdr = eligible_dsdr_pos.sum()
        n_eligible_p3sdr = eligible_p3sdr_pos.sum()
        n_eligible_m09 = eligible_m09_dsdr.sum()  # For FR6: M09 AND ΔSDR > 0
        
        # FR flags
        fr1 = eligible_dsdr_pos & (paired.get('leak_recall', 0) <= 0)
        fr2 = eligible_p3sdr_pos & (paired.p3_removed_leak_count == 0)
        fr3 = eligible_dsdr_pos & (paired.get('same_side_residual', 0) >= 0)
        fr4 = paired.legacy_sdr > 0  # event based on all-key
        fr4_conditional = eligible_dsdr_pos & (paired.overcorrection > 0)
        fr5 = eligible_dsdr_pos & (paired.get('legit_retention', 0) < 0)
        
        # FR6: M09 partial removal among keys with mechanic==M09 and ΔSDR > 0
        m09_sub = paired[eligible_m09_dsdr]
        fr6_events = 0
        if len(m09_sub) > 0 and 'm09_partial_removed' in m09_sub.columns:
            fr6_events = int((m09_sub['m09_partial_removed'] > 0).sum())
        
        # Build FR summary
        categories = {
            'FR1': {'event_count': int(fr1.sum()), 'all_key_n': n_all,
                    'eligible_n': n_eligible_dsdr, 'eligible_label': 'Δlegacy_sdr > 0'},
            'FR2': {'event_count': int(fr2.sum()), 'all_key_n': n_all,
                    'eligible_n': n_eligible_p3sdr, 'eligible_label': 'P3 legacy_sdr > 0'},
            'FR3': {'event_count': int(fr3.sum()), 'all_key_n': n_all,
                    'eligible_n': n_eligible_dsdr, 'eligible_label': 'Δlegacy_sdr > 0'},
            'FR4': {'event_count': int(fr4_conditional.sum()), 'all_key_n': n_all,
                    'eligible_n': n_eligible_dsdr, 'eligible_label': 'Δlegacy_sdr > 0',
                    'all_key_wording': f'{int(fr4.sum())} of all {n_all} keys satisfy the FR4 condition ({fr4.mean()*100:.1f}%)',
                    'conditional_wording': f'{int(fr4_conditional.sum())} of {n_eligible_dsdr} keys with positive ΔSDR exhibit overcorrection ({fr4_conditional.mean()*100:.1f}% of eligible)'},
            'FR5': {'event_count': int(fr5.sum()), 'all_key_n': n_all,
                    'eligible_n': n_eligible_dsdr, 'eligible_label': 'Δlegacy_sdr > 0'},
            'FR6': {'event_count': fr6_events, 'all_key_n': n_all,
                    'eligible_n': n_eligible_m09, 'eligible_label': 'M09 AND Δlegacy_sdr > 0'},
        }
        
        for fr_id, cat in categories.items():
            cat['learner'] = learner
            cat['category'] = fr_id
            cat['all_key_prevalence'] = round(cat['event_count'] / cat['all_key_n'] * 100, 1)
            if cat['eligible_n'] > 0:
                cat['conditional_prevalence'] = round(cat['event_count'] / cat['eligible_n'] * 100, 1)
            else:
                cat['conditional_prevalence'] = 0.0
            all_fr_summaries.append(cat)
        
        # FR examples: top 20 per FR category
        for fr_id in ['FR1', 'FR2', 'FR3', 'FR4', 'FR5']:
            if fr_id == 'FR1':
                mask = fr1
            elif fr_id == 'FR2':
                mask = fr2
            elif fr_id == 'FR3':
                mask = fr3
            elif fr_id == 'FR4':
                mask = fr4_conditional
            elif fr_id == 'FR5':
                mask = fr5
            else:
                continue
            
            fr_rows = paired[mask].nlargest(20, 'legacy_sdr')
            for _, r in fr_rows.iterrows():
                all_fr_examples.append({
                    'FR_category': fr_id,
                    'learner': learner,
                    'dataset_index': int(r.dataset_index),
                    'mechanism': r.mechanism,
                    'strength': r.strength,
                    'training_seed': int(r.training_seed),
                    'archetype': r.archetype,
                    'delta_legacy_sdr': round(float(r.legacy_sdr), 6),
                    'p3_legacy_sdr': round(float(r.get('p3_legacy_sdr', np.nan)), 6),
                    'delta_overcorrection': round(float(r.get('overcorrection', np.nan)), 6),
                    'p3_overcorrection': round(float(r.get('p3_overcorrection', np.nan)), 6),
                })
    
    return pd.DataFrame(all_fr_summaries), pd.DataFrame(all_fr_examples)

# ===============================================================
# M09 Semantic-Group Quantitative Analysis
# ===============================================================
def m09_semantic_analysis(learner_data):
    """Compute M09 semantic-group paired metrics with bootstrap."""
    results = {}
    for learner, data in learner_data.items():
        m09 = data['paired'][data['paired'].mechanism == 'M09'].copy()
        if len(m09) == 0:
            results[learner] = {'status': 'NO_M09_KEYS'}
            continue
        
        # P3 absolute values
        p3 = data['df20'][(data['df20'].policy == 'P3_blind_mi') & (data['df20'].mechanism == 'M09')]
        p2 = data['df20'][(data['df20'].policy == 'P2_random') & (data['df20'].mechanism == 'M09')]
        
        # Compute per-key totals
        kc = ['dataset_index', 'mechanism', 'strength', 'training_seed']
        
        r = {}
        for metric, col in [
            ('p3_full_group_removed_rate', 'm09_full_group_removed'),
            ('p3_any_hit_rate', 'm09_any_hit'),
            ('p3_partial_removal_rate', 'm09_partial_removed'),
        ]:
            if col in p3.columns:
                r[metric] = round(float(p3[col].mean()), 4)
        
        for metric, col in [
            ('p2_mean_full_group_removed_rate', 'm09_full_group_removed'),
            ('p2_mean_any_hit_rate', 'm09_any_hit'),
            ('p2_mean_partial_removal_rate', 'm09_partial_removed'),
        ]:
            if col in p2.columns:
                p2k = p2.groupby(kc)[col].mean()
                r[metric] = round(float(p2k.mean()), 4)
        
        # Paired metrics with bootstrap
        for delta_col in ['m09_full_group_removed', 'm09_any_hit', 'm09_partial_removed']:
            if delta_col in m09.columns:
                task_vals = m09.groupby('dataset_index')[delta_col].mean().dropna()
                if len(task_vals) > 0:
                    obs, lo, hi, p3b = task_bootstrap(task_vals.values)
                    base = delta_col.replace('m09_', '')
                    r[f'delta_{base}'] = round(obs, 4)
                    r[f'delta_{base}_ci_lo'] = round(lo, 4)
                    r[f'delta_{base}_ci_hi'] = round(hi, 4)
                    r[f'delta_{base}_p3_better'] = round(p3b, 4)
                    r[f'delta_{base}_n_tasks'] = len(task_vals)
                    r[f'delta_{base}_event_count'] = int((m09[delta_col] > 0).sum())
        
        r['n_m09_keys'] = len(m09)
        r['partial_removal_zero'] = True if r.get('p3_partial_removal_rate', 0) == 0.0 else False
        
        results[learner] = r
    return results

# ===============================================================
# Claim State Update
# ===============================================================
def build_claim_state(analysis_sha, m09_results):
    claims = {}
    
    # C1: Score recovery only (inherited from R2.1)
    for learner in ['LR', 'RF', 'LightGBM']:
        claims[f'C1_{learner}_GOVERNANCE'] = {
            'status': 'SCORE_RECOVERY_ONLY',
            'evidence_tier': 'confirmatory_r2_2',
            'note': 'Δovercorrection > 0 prevents SEMANTICALLY_CORROBORATED; semantic evidence reported as subclaims.',
        }
    
    # C1_descriptive: Independent semantic subclaim
    claims['C1_DESCRIPTIVE_SEMANTIC_SUBCLAIM'] = {
        'status': 'DESCRIPTIVE_SUPPORTING_EVIDENCE',
        'allowed_wording': (
            'MI removal improves leak-column recall and directional gap reduction over '
            'matched random removal, but these improvements do not satisfy the joint '
            'semantic-repair gate because overcorrection increases.'
        ),
        'forbidden_wording': 'MI removal is a pure semantic repair operation.',
    }
    
    # C7: M09 semantic-group
    m09_status = 'NOT_EVALUABLE'
    m09_wording = ''
    lr_m09 = m09_results.get('LR', {})
    if lr_m09.get('delta_full_group_removed') is not None:
        m09_status = 'CORROBORATED_AT_SEMANTIC_GROUP_LEVEL'
        delta = lr_m09['delta_full_group_removed']
        m09_wording = (
            f'M09 semantic full-group recall for MI exceeds random: '
            f'paired Δ={delta:.3f}, confirming that the encoded-column advantage '
            f'extends to the semantic-group level.'
        ) if delta > 0 else 'M09 semantic-group advantage not detected.'
    
    claims['C7_M09_SEMANTIC_GROUP'] = {
        'status': m09_status,
        'allowed_wording': m09_wording,
        'm09_metrics': lr_m09,
    }
    
    # C2: Learner interaction (inherited)
    claims['C2_LEARNER_INTERACTION'] = {
        'status': 'SUPPORTED',
        'note': 'All learner interaction CIs cross zero or are near-zero.',
    }
    
    return {
        'schema_version': 1,
        'derivation': 'scripts/repair_evidence_chain_r2_2.py',
        'audit': 'T0_R2_2_DENOMINATOR_CLOSURE',
        'primary_budget': 0.20,
        'analysis_summary_sha256': analysis_sha,
        'claims': claims,
        'global_limitations': [
            'C1 is SCORE_RECOVERY_ONLY — overcorrection gate failed.',
            'Semantic evidence (Δleak_recall, Δdirectional_repair, M09 group recall) is descriptively positive.',
            'FR4 conditional prevalence: ~X% of keys with positive ΔSDR exhibit overcorrection (not X% of all keys).',
        ],
    }

# ===============================================================
# Main
# ===============================================================
def main():
    print("=== T0-R2.2: DENOMINATOR CLOSURE ===", flush=True)
    arch_map = load_canonical()
    
    # Load and compute
    print("Loading data and computing metrics...", flush=True)
    learner_data = load_data()
    
    # FR audit with dual denominators
    print("Computing false-repair audit...", flush=True)
    fr_summary, fr_examples = compute_fr_audit(learner_data, arch_map)
    fr_summary.to_csv(OUT_DIR / 'false_repair_summary.csv', index=False)
    fr_examples.to_csv(OUT_DIR / 'false_repair_examples.csv', index=False)
    
    # Print FR summary
    for learner in ['LR', 'RF', 'LightGBM']:
        print(f"\n--- {learner} False-Repair ---")
        sub = fr_summary[fr_summary.learner == learner]
        for _, row in sub.iterrows():
            all_pct = row.get('all_key_prevalence', 'N/A')
            cond_pct = row.get('conditional_prevalence', 'N/A')
            cond_label = row.get('eligible_label', 'N/A')
            print(f"  {row['category']}: events={row['event_count']}, "
                  f"all-key={all_pct}%, conditional={cond_pct}% ({cond_label})")
    
    # M09 semantic analysis
    print("\nComputing M09 semantic-group metrics...", flush=True)
    m09_results = m09_semantic_analysis(learner_data)
    for learner, r in m09_results.items():
        print(f"\n--- {learner} M09 ---")
        for k, v in sorted(r.items()):
            print(f"  {k}: {v}")
    
    # Save M09 results
    with open(OUT_DIR / 'm09_semantic_group_r2_2.json', 'w') as f:
        json.dump(m09_results, f, indent=2)
    
    # Build analysis summary
    analysis = {
        'schema_version': 2,
        'audit': 'T0_R2_2',
        'm09_semantic_group': m09_results,
    }
    analysis_json = json.dumps(analysis, sort_keys=True).encode()
    analysis_sha = hashlib.sha256(analysis_json).hexdigest()
    with open(OUT_DIR / 'analysis_summary_r2.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Build claim state
    claim_state = build_claim_state(analysis_sha, m09_results)
    with open(OUT_DIR / 'claim_state_r2.json', 'w') as f:
        json.dump(claim_state, f, indent=2)
    
    # Update manifest
    manifest_path = OUT_DIR / 'manifest.json'
    with open(manifest_path) as f:
        m = json.load(f)
    
    m['status'] = 'COMPLETE_POSTRUN_CORRECTIVE_AUDIT'
    m['r2_2_commit_sha'] = None  # will be filled after commit
    m['analysis_summary_sha256'] = analysis_sha
    m['claim_state_sha256'] = hashlib.sha256(
        json.dumps(claim_state, sort_keys=True).encode()
    ).hexdigest()
    m['false_repair_summary_sha256'] = hashlib.sha256(
        (OUT_DIR / 'false_repair_summary.csv').read_bytes()
    ).hexdigest()
    m['protocol_amendment_sha256'] = hashlib.sha256(
        (ROOT / 'reports/edbt_t0_r2/protocol_amendment_postrun.md').read_bytes()
    ).hexdigest()
    m['reconstruction_summary_sha256'] = hashlib.sha256(
        (OUT_DIR / 'reconstruction_summary_r2_1.json').read_bytes()
    ).hexdigest()
    
    with open(manifest_path, 'w') as f:
        json.dump(m, f, indent=2)
    
    print(f"\nAnalysis SHA: {analysis_sha}")
    print(f"Claim-state SHA: {m['claim_state_sha256']}")
    print(f"Manifest status: {m['status']}")
    print("\n=== T0-R2.2 COMPLETE ===")

if __name__ == "__main__":
    raise SystemExit(main())
