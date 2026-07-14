# SP5-G Source Registry Report

10 sources inspected. 6 claim-eligible (formal), 4 excluded.

## Formal (claim-eligible)
| source | status | role | mechanisms |
|--------|--------|------|-----------|
| core_cpu | FORMAL_ACTIVE | base | 7 non-amended × 4 CPU models |
| base7_tabm | FORMAL_ACTIVE | base | 7 non-amended × TabM |
| sp4 | FORMAL_REPLACEMENT | exact_replacement | M04/M05/M08 × 5 models |
| sp4_detectability | FORMAL_REPLACEMENT | detectability | M04/M05/M08 (derived from frozen bundles) |
| m10_cpu | FORMAL_REPLACEMENT | exact_replacement | M10 × 4 CPU |
| m10_tabm | FORMAL_REPLACEMENT | exact_replacement | M10 × TabM |

## Excluded
| source | status | reason |
|--------|--------|--------|
| old_tabm_checkpoint | CODE_DRIFT_EXCLUDED | code_hash 99b17868 ≠ current; unreproducible |
| interim_m08 | INTERIM_EXCLUDED | protocol substitution (entity-mean, synthetic) |
| profiles_v2 | SUPERSEDED_EXCLUDED | pre-correction aggregates |
| ce2r_neural | SUPERSEDED_EXCLUDED | legacy pre-corrected_v2 |

Machine-readable: source_registry.csv / source_registry.json.
All excluded sources archived (not deleted), disconnected from active pipeline.
