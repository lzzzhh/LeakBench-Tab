#!/usr/bin/env python3
"""T0-R2.3 Release Validator — checks final_report, claim_state, manifest, M09, sparse."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / 'results/edbt_t0_r2'
REPORT_DIR = ROOT / 'reports/edbt_t0_r2'

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    errors = []
    warnings = []

    # ============================================================
    # 1. Final report stale-string scan
    # ============================================================
    final_report = (REPORT_DIR / 'final_report.md').read_text()
    
    stale = [
        'SEMANTICALLY_CORROBORATED_WITH_OVERCORRECTION_CAVEAT',
        'sparse archetype is NOT reliably negative',
        '346,500 selection hashes',
        '29–31% of keys with positive SDR',
        'single-seed artifact',
        '346,500 rows',
    ]
    for s in stale:
        if s in final_report:
            errors.append(f"final_report.md contains stale string: '{s}'")
    
    # Check required positive strings exist
    # Check required phrases (normalize whitespace for multi-line matches)
    fr_normalized = ' '.join(final_report.split())
    required = [
        '709,500 ROWS',
        'SCORE_RECOVERY_ONLY',
        '47.6%',
        '48.3%',
        '51.0%',
        'full-group repair not corroborated',
        'any-hit localization',
        'score proximity alone',
        '0.118',
        '0.068',
        '0.054',
    ]
    for r in required:
        fr_check = fr_normalized.replace('\u2212', '-')
        if r not in fr_check:
            errors.append(f"final_report.md missing required string: '{r}'")
    
    # ============================================================
    # 2. Claim state validation
    # ============================================================
    with open(OUT_DIR / 'claim_state_r2.json') as f:
        cs = json.load(f)
    
    # No CORROBORATED_AT_SEMANTIC_GROUP_LEVEL
    claim_text = json.dumps(cs)
    if 'CORROBORATED_AT_SEMANTIC_GROUP_LEVEL' in claim_text:
        errors.append("claim_state contains CORROBORATED_AT_SEMANTIC_GROUP_LEVEL")
    if '~X%' in claim_text:
        errors.append("claim_state contains placeholder '~X%'")
    if 'TBD' in claim_text:
        errors.append("claim_state contains 'TBD'")
    
    # C7 must be NOT_EVALUABLE_FOR_FULL_GROUP_REPAIR
    c7 = cs['claims'].get('C7_M09_SEMANTIC_GROUP', {})
    if c7.get('status') != 'NOT_EVALUABLE_FOR_FULL_GROUP_REPAIR':
        errors.append(f"C7 status is {c7.get('status')}, expected NOT_EVALUABLE_FOR_FULL_GROUP_REPAIR")
    
    # C7 allowed wording must mention any-hit, not full repair
    c7_wording = c7.get('allowed_wording', '')
    if 'full-group repair' in c7_wording and 'not corroborated' not in c7_wording:
        warnings.append("C7 wording mentions full-group repair without 'not corroborated'")
    
    # C2 must be scoped to legacy SDR
    assert 'C2_LEGACY_SDR_LEARNER_INTERACTION' in cs['claims'], "C2 not scoped to legacy SDR"
    
    # C1 must be SCORE_RECOVERY_ONLY
    for learner in ['LR', 'RF', 'LightGBM']:
        c1 = cs['claims'].get(f'C1_{learner}_GOVERNANCE', {})
        if c1.get('status') != 'SCORE_RECOVERY_ONLY':
            errors.append(f"C1_{learner} status is {c1.get('status')}, expected SCORE_RECOVERY_ONLY")
    
    # Analysis hash must be non-null and match
    stored_sha = cs.get('analysis_summary_sha256', '')
    if not stored_sha or stored_sha == 'null':
        errors.append("claim_state analysis_summary_sha256 is null")
    else:
        actual_sha = sha256(OUT_DIR / 'analysis_summary_r2.json')
        if stored_sha != actual_sha:
            errors.append(f"claim_state analysis SHA mismatch: stored={stored_sha[:16]} actual={actual_sha[:16]}")
    
    # C1_DESCRIPTIVE evidence_tier must be 'descriptive'
    c1d = cs['claims'].get('C1_DESCRIPTIVE_SEMANTIC_SUBCLAIM', {})
    if c1d.get('evidence_tier') != 'descriptive':
        errors.append(f"C1_DESCRIPTIVE evidence_tier is {c1d.get('evidence_tier')}, expected 'descriptive'")
    
    # ============================================================
    # 3. Manifest validation
    # ============================================================
    with open(OUT_DIR / 'manifest.json') as f:
        m = json.load(f)
    
    if m.get('claim_verdict') != 'SCORE_RECOVERY_ONLY':
        errors.append(f"manifest claim_verdict is {m.get('claim_verdict')}")
    if m.get('status') != 'COMPLETE_POSTRUN_CORRECTIVE_AUDIT':
        errors.append(f"manifest status is {m.get('status')}")
    if m['tests']['repository_suite'].get('failed', -1) != 0:
        errors.append("manifest tests.repository_suite.failed != 0")
    if m['tests']['repository_suite'].get('passed', 0) <= 0:
        errors.append("manifest tests.repository_suite.passed <= 0")
    if m['tests']['r2_targeted_suite'].get('failed', -1) != 0:
        errors.append("manifest tests.r2_targeted_suite.failed != 0")
    if m['tests']['r2_targeted_suite'].get('passed', 0) <= 0:
        errors.append("manifest tests.r2_targeted_suite.passed <= 0")
    
    # Validation receipt
    receipt_path = m['tests'].get('receipt', '')
    if not receipt_path or not Path(receipt_path).exists():
        errors.append(f"validation receipt missing: {receipt_path}")
    else:
        with open(receipt_path) as f:
            receipt = json.load(f)
        if receipt.get('integration_merge_sha') != '2ae0fa50449a84f1f4a8b27ac086b695fd9b0a73':
            errors.append("receipt integration_merge_sha mismatch")
        if receipt.get('validation_scope') != 'LOCAL_VALIDATION_ONLY':
            errors.append(f"receipt validation_scope: {receipt.get('validation_scope')}")
        if receipt['repository_suite'].get('passed') != m['tests']['repository_suite']['passed']:
            errors.append("receipt repository_suite.passed != manifest")
        if receipt['repository_suite'].get('failed') != 0:
            errors.append("receipt repository_suite.failed != 0")
        if receipt['r2_targeted_suite'].get('passed') != m['tests']['r2_targeted_suite']['passed']:
            errors.append("receipt r2_targeted_suite.passed != manifest")
    
    # Validator self-reference: this script must be in artifact list
    validator_path = 'scripts/validate_release_r2_3.py'
    validator_art = [a for a in m.get('artifacts', []) if a['path'] == validator_path]
    if not validator_art:
        errors.append(f"validator script not in artifact list: {validator_path}")
    else:
        actual_validator_sha = sha256(validator_path)
        if validator_art[0]['sha256'] != actual_validator_sha:
            errors.append(f"validator SHA mismatch: recorded={validator_art[0]['sha256'][:16]} actual={actual_validator_sha[:16]}")
    
    # Receipt must be in artifact list
    if receipt_path:
        receipt_art = [a for a in m.get('artifacts', []) if a['path'] == receipt_path]
        if not receipt_art:
            errors.append(f"receipt not in artifact list: {receipt_path}")
        else:
            actual_receipt_sha = sha256(receipt_path)
            if receipt_art[0]['sha256'] != actual_receipt_sha:
                errors.append(f"receipt SHA mismatch: recorded={receipt_art[0]['sha256'][:16]} actual={actual_receipt_sha[:16]}")
    
    # Artifact SHA consistency: every artifact in list must match disk
    for art in m.get('artifacts', []):
        path = art['path']
        recorded = art['sha256']
        if not Path(path).exists():
            errors.append(f"manifest artifact missing on disk: {path}")
        else:
            actual = sha256(path)
            if actual != recorded:
                errors.append(f"manifest artifact SHA mismatch: {path} recorded={recorded[:16]} actual={actual[:16]}")
    
    # Top-level SHAs must match disk
    for key, path in [
        ('analysis_summary_sha256', 'results/edbt_t0_r2/analysis_summary_r2.json'),
        ('claim_state_sha256', 'results/edbt_t0_r2/claim_state_r2.json'),
        ('false_repair_summary_sha256', 'results/edbt_t0_r2/false_repair_summary.csv'),
        ('reconstruction_summary_sha256', 'results/edbt_t0_r2/reconstruction_summary_r2_1.json'),
        ('protocol_amendment_sha256', 'reports/edbt_t0_r2/protocol_amendment_postrun.md'),
        ('final_report_sha256', 'reports/edbt_t0_r2/final_report.md'),
    ]:
        if m.get(key):
            actual = sha256(path)
            if m[key] != actual:
                errors.append(f"manifest {key} mismatch: recorded={m[key][:16]} actual={actual[:16]}")
        else:
            errors.append(f"manifest missing {key}")
    
    # No stale artifact hashes
    recorded_shas = {a['sha256'] for a in m.get('artifacts', [])}
    disk_shas = {sha256(a['path']) for a in m.get('artifacts', []) if Path(a['path']).exists()}
    if recorded_shas != disk_shas:
        errors.append("manifest artifact SHA set does not match disk")
    
    # ============================================================
    # 4. M09 semantic-group metric validation
    # ============================================================
    if (OUT_DIR / 'm09_semantic_group_r2_2.json').exists():
        with open(OUT_DIR / 'm09_semantic_group_r2_2.json') as f:
            m09 = json.load(f)
        for learner in ['LR', 'RF', 'LightGBM']:
            r = m09.get(learner, {})
            if r.get('p3_full_group_removed_rate', -1) != 0.0:
                errors.append(f"M09 {learner} P3 full_group should be 0.0, got {r.get('p3_full_group_removed_rate')}")
            if r.get('p2_mean_full_group_removed_rate', -1) != 0.0:
                errors.append(f"M09 {learner} P2 mean full_group should be 0.0")
            if r.get('delta_any_hit', -1) <= 0:
                errors.append(f"M09 {learner} delta_any_hit should be > 0")
    
    # ============================================================
    # 5. Sparse archetype validation
    # ============================================================
    # From R2.1 analysis: sparse is negative for all learners
    expected_sparse = {'LR': (-0.118, -0.160, -0.093),
                       'RF': (-0.068, -0.083, -0.044),
                       'LightGBM': (-0.054, -0.083, -0.009)}
    
    # Check final_report mentions correct sparse numbers
    for learner, (mean, lo, _) in expected_sparse.items():
        mean_str = f'{mean:.3f}'
        lo_str = f'{lo:.3f}'
        if mean_str not in final_report:
            warnings.append(f"final_report may not mention {learner} sparse mean {mean_str}")
    
    # Check no incorrect current sparse claims
    # (mentioning old values in revision context is acceptable)
    
    # ============================================================
    # 6. Planned outputs check
    # ============================================================
    planned = [
        'reports/edbt_t0_r2/protocol.md',
        'reports/edbt_t0_r2/protocol_amendment_postrun.md',
        'reports/edbt_t0_r2/final_report.md',
        'reports/edbt_t0_r2/baseline_continuity_report.md',
        'reports/edbt_t0_r2/false_repair_report.md',
        'results/edbt_t0_r2/protocol_freeze.json',
        'results/edbt_t0_r2/b1_sp8_baseline_continuity.csv',
        'results/edbt_t0_r2/analysis_summary_r2.json',
        'results/edbt_t0_r2/claim_state_r2.json',
        'results/edbt_t0_r2/false_repair_summary.csv',
        'results/edbt_t0_r2/false_repair_examples.csv',
        'results/edbt_t0_r2/manifest.json',
        'results/edbt_t0_r2/task_effects_r2.csv',
        'results/edbt_t0_r2/mechanism_summary_r2.csv',
        'results/edbt_t0_r2/archetype_summary_r2.csv',
        'results/edbt_t0_r2/reconstruction_summary_r2_1.json',
        'results/edbt_t0_r2/m09_semantic_group_r2_2.json',
    ]
    for p in planned:
        if not (ROOT / p).exists():
            errors.append(f"Planned output missing: {p}")
    
    # ============================================================
    # Summary
    # ============================================================
    print(f"\n=== T0-R2.3 RELEASE VALIDATOR ===")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  ERROR: {e}")
    print(f"Warnings: {len(warnings)}")
    for w in warnings:
        print(f"  WARN: {w}")
    
    if errors:
        print("\nVALIDATOR: FAIL")
        sys.exit(1)
    else:
        print("\nVALIDATOR: PASS (LOCAL_VALIDATION_ONLY)")
        sys.exit(0)

if __name__ == "__main__":
    main()
