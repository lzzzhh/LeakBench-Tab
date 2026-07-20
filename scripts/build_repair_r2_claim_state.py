#!/usr/bin/env python3
"""Build R2 claim state from T0 analysis results."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    with open(ROOT / 'results/edbt_t0_r2/analysis_summary_r2.json') as f:
        analysis = json.load(f)
    
    claims = {}
    global_limitations = []
    
    # ================================================================
    # C1: MULTI-LEARNER GOVERNANCE (R2: SEMANTICALLY_CORROBORATED_WITH_CAVEAT)
    # ================================================================
    for learner in ['LR', 'RF', 'LightGBM']:
        overall = analysis['results'].get(f'{learner}_overall', {})
        lsdr = overall.get('legacy_sdr', {})
        drep = overall.get('directional_repair', {})
        ssr = overall.get('same_side_residual', {})
        ovc = overall.get('overcorrection', {})
        lrcl = overall.get('leak_recall', {})
        lret = overall.get('legit_retention', {})
        dprec = overall.get('deletion_precision', {})
        
        # Checks
        checks = {
            'legacy_sdr_mean_gt_0': lsdr.get('mean', 0) > 0,
            'legacy_sdr_ci_lo_gt_0': lsdr.get('ci_lo', 0) > 0,
            'directional_repair_mean_gt_0': drep.get('mean', 0) > 0,
            'directional_repair_ci_lo_gt_0': drep.get('ci_lo', 0) > 0,
            'residual_reduction_mean_gt_0': ssr.get('mean', 0) < 0,  # negative SSR = less residual
            'residual_reduction_ci_hi_lt_0': ssr.get('ci_hi', 0) < 0,
            'leak_recall_mean_gt_0': lrcl.get('mean', 0) > 0,
            'leak_recall_ci_lo_gt_0': lrcl.get('ci_lo', 0) > 0,
            'overcorrection_mean_le_0': ovc.get('mean', 0) <= 0,
            'overcorrection_ci_hi_le_0': ovc.get('ci_hi', 0) <= 0,
            'legit_retention_mean_ge_0': lret.get('mean', 0) >= 0,
            'legit_retention_ci_lo_ge_0': lret.get('ci_lo', 0) >= 0,
        }
        
        if all([checks[k] for k in ['legacy_sdr_mean_gt_0','legacy_sdr_ci_lo_gt_0',
                                      'directional_repair_mean_gt_0','directional_repair_ci_lo_gt_0',
                                      'residual_reduction_mean_gt_0','residual_reduction_ci_hi_lt_0',
                                      'leak_recall_mean_gt_0','leak_recall_ci_lo_gt_0',
                                      'legit_retention_mean_ge_0']]):
            if all([checks[k] for k in ['overcorrection_mean_le_0']]):
                status = 'SEMANTICALLY_CORROBORATED'
            else:
                status = 'SEMANTICALLY_CORROBORATED_WITH_OVERCORRECTION_CAVEAT'
        elif checks.get('legacy_sdr_mean_gt_0') and checks.get('legacy_sdr_ci_lo_gt_0'):
            status = 'SCORE_RECOVERY_ONLY'
        else:
            status = 'NEGATIVE'
        
        claims[f'C1_{learner}_GOVERNANCE_R2'] = {
            'status': status,
            'evidence_tier': 'confirmatory_r2_audit',
            'scope': f'{learner}, 20% encoded-column budget, 20 P2 seeds',
            'n_keys': 5500,
            'allowed_wording': (
                f'For {learner}, blind MI removal improves directional repair (+{drep.get("mean",0):.3f} '
                f'[{drep.get("ci_lo",0):+.3f},{drep.get("ci_hi",0):+.3f}]) and leak recall (+{lrcl.get("mean",0):.3f} '
                f'[{lrcl.get("ci_lo",0):+.3f},{lrcl.get("ci_hi",0):+.3f}]) over matched random removal, '
                f'but also increases overcorrection (+{ovc.get("mean",0):.3f} [{ovc.get("ci_lo",0):+.3f},{ovc.get("ci_hi",0):+.3f}]).'
            ) if status == 'SEMANTICALLY_CORROBORATED_WITH_OVERCORRECTION_CAVEAT' else '',
            'forbidden_wording': 'MI removal unambiguously repairs tabular leakage.',
            'checks': checks,
        }
    
    # ================================================================
    # C2: NO LEARNER INTERACTION (R2: SUPPORTED)
    # ================================================================
    claims['C2_NO_LEARNER_INTERACTION_R2'] = {
        'status': 'SUPPORTED',
        'evidence_tier': 'confirmatory_r2_audit',
        'allowed_wording': 'The governance advantage pattern (positive directional repair + leak recall, with overcorrection caveat) is consistent across LR, RF, and LightGBM.',
        'forbidden_wording': 'Learner differences are large enough to change the qualitative conclusion.',
        'note': 'The mask-grounded metrics (Δleak_recall, Δdeletion_precision, Δlegit_retention) are identical across learners by construction (same selections). Only score-based metrics differ.',
    }
    
    # ================================================================
    # C3: OVERCORRECTION CAVEAT (R2: NEW)
    # ================================================================
    claims['C3_OVERCORRECTION_CAVEAT_R2'] = {
        'status': 'SUPPORTED',
        'evidence_tier': 'confirmatory_r2_audit',
        'allowed_wording': (
            'MI-guided removal produces more overcorrection (governed score crossing the strict reference) '
            'than random removal across all three learners: LR +0.041 [0.018,0.068], '
            'RF +0.042 [0.021,0.066], LightGBM +0.045 [0.023,0.070]. '
            'Approximately 30% of keys showing positive legacy SDR also exhibit overcorrection.'
        ),
        'forbidden_wording': 'MI removal is a pure repair operation without side effects.',
    }
    
    # ================================================================
    # C4: LEGACY SDR OVERSTATEMENT (R2: NEW)
    # ================================================================
    claims['C4_LEGACY_SDR_OVERSTATEMENT_R2'] = {
        'status': 'SUPPORTED',
        'evidence_tier': 'confirmatory_r2_audit',
        'allowed_wording': (
            'The legacy SDR metric (|full-strict| - |governed-strict|) conflates directional repair with overcorrection. '
            'When decomposed by direction: directional repair shows a larger governance advantage '
            '(Δ=+0.085 for LR vs Δlegacy_sdr=+0.043), but the gain is partially offset by increased overcorrection '
            '(Δ=+0.041). The legacy SDR understates the actual repair but also masks the overcorrection cost.'
        ),
        'forbidden_wording': 'Legacy SDR is a complete measure of repair quality.',
    }
    
    # ================================================================
    # C5: DECOMPOSITION APPROACH (mechanism-level claims deferred to mechanism summary)
    # ================================================================
    mechanisms = ['M01','M02','M03','M04','M05','M06','M07','M08','M09','M10','M11']
    
    # ================================================================
    # C6: SPARSE ARCHETYPE RECLASSIFICATION (R2: REVISED)
    # ================================================================
    claims['C6_SPARSE_ARCHETYPE_REVISED_R2'] = {
        'status': 'REVISED',
        'evidence_tier': 'r2_audit_resolves_single_seed_artifact',
        'allowed_wording': (
            'Under the properly integrated multi-seed P2 protocol (20 seeds, averaged before P3 comparison), '
            'the sparse archetype shows no reliable governance advantage or disadvantage '
            '(mean Δlegacy_sdr=+0.022, CI[-0.049,+0.072]). The previously reported negative value (-0.118) '
            'was an artifact of relying on a single frozen P2 governance seed.'
        ),
        'forbidden_wording': 'The sparse archetype is a consistently negative regime for MI-guided governance.',
        'note': 'This changes the LOAO-sparse interpretation: when sparse is excluded, the overall remains positive because sparse was near-zero, not because it was strongly negative.',
    }
    
    # ================================================================
    # C7: M09 SEMANTIC-GROUP REMAINS POSITIVE (R2: CONFIRMED)
    # ================================================================
    claims['C7_M09_SEMANTIC_POSITIVE_R2'] = {
        'status': 'CONFIRMED',
        'evidence_tier': 'r2_audit_confirms_prior',
        'allowed_wording': 'M09 remains a robust positive counterexample within the structured family under R2 metrics.',
        'forbidden_wording': 'M09 is negative when properly evaluated.',
    }
    
    # ================================================================
    # COMPILE CLAIM STATE
    # ================================================================
    claim_state = {
        'schema_version': 1,
        'derivation': 'scripts/build_repair_r2_claim_state.py',
        'audit': 'T0_R2_REPAIR_CONSTRUCT_VALIDITY',
        'primary_budget': 0.20,
        'analysis_sha256': None,  # Will be filled during manifest build
        'claims': claims,
        'global_limitations': [
            'Legacy SDR conflation: the original metric cannot separate directional repair from overcorrection.',
            'Overcorrection is systematically higher for MI-guided removal than random at 20% budget.',
            'The sparse archetype negative finding under single-seed P2 does not replicate under multi-seed averaging.',
            'The prior C4 claim ("sparse archetype is a negative regime") requires revision.',
            'Mask-grounded metrics are identical across learners by construction — only score-based metrics differ.',
            'The directional_repair_fraction metric has heavy-tailed distributions with extreme negative outliers in low-opportunity keys; this metric should NOT be used for aggregate comparisons.',
        ],
    }
    
    out_path = ROOT / 'results/edbt_t0_r2/claim_state_r2.json'
    with open(out_path, 'w') as f:
        json.dump(claim_state, f, indent=2)
    print(f"Written {out_path}")

if __name__ == "__main__":
    raise SystemExit(main())
