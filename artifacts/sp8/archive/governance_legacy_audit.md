# Governance Legacy Audit (SP8-A)

5 sources inspected. 0 claim-eligible. All SUPERSEDED.

`run_meta_tier.py` remains INTEGRITY_HOLD (locked, never to be executed).

Replaced by read-only bundle-consuming governance runner.
- **run_meta_tier.py**: INTEGRITY_HOLD; does own inject_with_metadata; int-as-string; seed=42; S-axis reads leak_mask
- **meta_governance_results.csv**: runtime injection; non-frozen; non-bundle
- **operational_governance.csv**: operational metadata generated at runtime from non-frozen split
- **governance_matrix.csv**: pre-corrected_v2
- **governance_v2.csv**: pre-corrected_v2
