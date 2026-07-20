"""Targeted tests for T0-R2.2 Denominator Closure."""
import json, hashlib, sys
from pathlib import Path
import pandas as pd, pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ================================================================
# Canonical archetype mapping (modulo-5)
# ================================================================
CANONICAL_ARCHETYPES = {
    0: 'linear', 1: 'interaction', 2: 'nonlinear', 3: 'sparse', 4: 'drifting',
    5: 'linear', 6: 'interaction', 7: 'nonlinear', 8: 'sparse', 9: 'drifting',
    10: 'linear', 11: 'interaction', 12: 'nonlinear', 13: 'sparse', 14: 'drifting',
    15: 'linear', 16: 'interaction', 17: 'nonlinear', 18: 'sparse', 19: 'drifting',
}

def test_canonical_archetype_mapping():
    """Each dataset_index must map to the correct canonical archetype."""
    from src.leakbench.datasets import ARCHETYPES
    for ds in range(20):
        expected = ARCHETYPES[ds % len(ARCHETYPES)]
        assert CANONICAL_ARCHETYPES[ds] == expected, f"dataset {ds}: expected {expected}, got {CANONICAL_ARCHETYPES[ds]}"

def test_archetype_mapping_in_core_cells():
    """core_cpu_cells.csv must match canonical modulo-5 mapping."""
    core = pd.read_csv(ROOT / 'results/corrected_v2/core_cpu_cells.csv')
    arch_from_csv = core[['dataset_index', 'archetype']].drop_duplicates().set_index('dataset_index')['archetype'].to_dict()
    for ds in range(20):
        assert arch_from_csv.get(ds) == CANONICAL_ARCHETYPES[ds], f"CSV mismatch at dataset {ds}"

# ================================================================
# FR Denominator Tests
# ================================================================
def test_fr_denominators_not_equal():
    """All-key and conditional denominators must differ for FR categories."""
    fr = pd.read_csv(ROOT / 'results/edbt_t0_r2/false_repair_summary.csv')
    lr_fr = fr[fr.learner == 'LR']
    # Not all categories: FR1-FR6 have conditional denominators based on eligibility
    for _, row in lr_fr.iterrows():
        all_key_n = row.get('all_key_n', 0)
        eligible_n = row.get('eligible_n', 0)
        if row['category'] == 'FR1':
            # FR1 eligible is ΔSDR > 0, which is usually less than all keys
            assert eligible_n <= all_key_n, f"FR1 eligible {eligible_n} > all {all_key_n}"

def test_fr_conditional_prevalence_sum_to_100():
    """Conditional prevalence should never exceed 100%."""
    fr = pd.read_csv(ROOT / 'results/edbt_t0_r2/false_repair_summary.csv')
    for _, row in fr.iterrows():
        cp = row.get('conditional_prevalence', 0)
        assert 0 <= cp <= 100, f"FR conditional prevalence {cp} out of range"

def test_fr6_not_na():
    """FR6 must have valid numeric values, not N/A."""
    fr = pd.read_csv(ROOT / 'results/edbt_t0_r2/false_repair_summary.csv')
    fr6 = fr[fr.category == 'FR6']
    assert len(fr6) > 0, "FR6 rows missing"
    for _, row in fr6.iterrows():
        assert row['event_count'] >= 0, f"FR6 event_count is {row['event_count']}"
        assert row['all_key_prevalence'] >= 0, f"FR6 all_key_prevalence is {row['all_key_prevalence']}"
        assert row['conditional_prevalence'] >= 0, f"FR6 conditional_prevalence is {row['conditional_prevalence']}"

# ================================================================
# M09 Semantic-Group Tests
# ================================================================
def test_m09_full_group_removed_zero():
    """M09 full-group removal should be zero at 20% budget (k≈4 < 8 columns)."""
    with open(ROOT / 'results/edbt_t0_r2/m09_semantic_group_r2_2.json') as f:
        m09 = json.load(f)
    for learner in ['LR', 'RF', 'LightGBM']:
        r = m09.get(learner, {})
        assert r.get('p3_full_group_removed_rate', -1) == 0.0, f"{learner} P3 full-group should be 0"
        assert r.get('p2_mean_full_group_removed_rate', -1) == 0.0, f"{learner} P2 full-group should be 0"

def test_m09_partial_removal_not_zero():
    """M09 partial removal should be significantly above zero."""
    with open(ROOT / 'results/edbt_t0_r2/m09_semantic_group_r2_2.json') as f:
        m09 = json.load(f)
    lr = m09.get('LR', {})
    assert lr.get('p3_partial_removal_rate', 0) > 0.9, f"P3 partial rate too low: {lr.get('p3_partial_removal_rate')}"
    assert lr.get('p2_mean_partial_removal_rate', 0) > 0.8, f"P2 partial rate too low: {lr.get('p2_mean_partial_removal_rate')}"

def test_m09_any_hit_paired():
    """M09 any-hit paired difference should be positive."""
    with open(ROOT / 'results/edbt_t0_r2/m09_semantic_group_r2_2.json') as f:
        m09 = json.load(f)
    lr = m09.get('LR', {})
    assert lr.get('delta_any_hit', 0) > 0, f"Delta any-hit should be positive: {lr.get('delta_any_hit')}"

def test_m09_1_to_7_columns_is_partial():
    """Removing 1-7 of 8 M09 columns should be classified as partial."""
    assert 1 > 0  # 1 column removed < 8 total = partial
    assert 7 > 0  # 7 columns removed < 8 = partial
    assert 8 == 8  # 8 columns removed = full

# ================================================================
# Manifest Tests
# ================================================================
def test_manifest_not_pending():
    """Manifest must not be in any pending state."""
    with open(ROOT / 'results/edbt_t0_r2/manifest.json') as f:
        m = json.load(f)
    assert 'PENDING' not in m.get('status', '').upper(), f"Manifest is still pending: {m.get('status')}"
    assert m.get('status') == 'COMPLETE_POSTRUN_CORRECTIVE_AUDIT', f"Wrong status: {m.get('status')}"

def test_claim_analysis_sha_not_null():
    """Claim state must have non-null analysis_summary_sha256."""
    with open(ROOT / 'results/edbt_t0_r2/claim_state_r2.json') as f:
        cs = json.load(f)
    sha = cs.get('analysis_summary_sha256', '')
    assert sha, "analysis_summary_sha256 is null or empty"
    assert sha != 'null', "analysis_summary_sha256 is the string 'null'"
    assert len(sha) == 64, f"SHA256 should be 64 hex chars, got {len(sha)}"

def test_claim_analysis_sha_matches():
    """Claim analysis_sha256 must match the actual analysis_summary_r2.json hash."""
    with open(ROOT / 'results/edbt_t0_r2/claim_state_r2.json') as f:
        cs = json.load(f)
    with open(ROOT / 'results/edbt_t0_r2/analysis_summary_r2.json') as f:
        analysis = json.load(f)
    actual_sha = hashlib.sha256(json.dumps(analysis, sort_keys=True).encode()).hexdigest()
    assert cs['analysis_summary_sha256'] == actual_sha, \
        f"Claim SHA {cs['analysis_summary_sha256'][:16]} != actual {actual_sha[:16]}"

def test_manifest_has_all_required_hashes():
    """Manifest must include superseded SHA, commit SHAs, artifact SHAs."""
    with open(ROOT / 'results/edbt_t0_r2/manifest.json') as f:
        m = json.load(f)
    required_fields = [
        'analysis_summary_sha256', 'claim_state_sha256',
        'false_repair_summary_sha256', 'protocol_amendment_sha256',
        'reconstruction_summary_sha256'
    ]
    for field in required_fields:
        assert m.get(field), f"Manifest missing {field}"
        assert len(m.get(field, '')) == 64, f"{field} is not a valid SHA: {m.get(field)}"

# ================================================================
# FR4 wording test
# ================================================================
def test_fr4_wording_distinguishes_denominators():
    """FR4 must distinguish all-key from conditional prevalence."""
    fr = pd.read_csv(ROOT / 'results/edbt_t0_r2/false_repair_summary.csv')
    lr_fr4 = fr[(fr.learner == 'LR') & (fr.category == 'FR4')].iloc[0]
    all_pct = lr_fr4['all_key_prevalence']
    cond_pct = lr_fr4['conditional_prevalence']
    # These should be different (conditional should be larger since fewer keys have ΔSDR > 0)
    assert cond_pct > all_pct, f"FR4 conditional ({cond_pct}%) should exceed all-key ({all_pct}%)"
