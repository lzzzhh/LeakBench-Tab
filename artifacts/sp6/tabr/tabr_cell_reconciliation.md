# TabR Cell Reconciliation (SP6-G1)

**Raw rows: 5524 → Clean ledger: 5500. 12 retry keys, 4 original timeouts, all resolved.**

## Raw composition (5524 rows)
- 5500 unique frozen-grid `(dataset_index,mechanism,strength,seed)` keys.
- 24 extra rows = 12 keys with 2 rows each (original + retry).
- 4 keys had an original subprocess timeout (1200s limit).
- All 4 timeout keys succeeded on retry (last_status=SUCCESS).

## Extra 24 rows (explained)
The retry command `--resume` over datasets `1,5`, mechanisms `M01,M05`,
strengths `S3,S4`, seeds `13,2026,7777` matched 24 cells (2 ds × 2 mech ×
2 str × 3 seed). 12 of these already had SUCCESS rows; the resume appended a
second (identical-key) SUCCESS row → 12 duplicate-key pairs = 24 extra rows.

The 4 timeout keys were among those 24. Their retries succeeded.

## Deduplication
Clean ledger = 5500 rows, one per unique key, preferring SUCCESS → latest.
0 duplicates, 0 missing, 0 non-finite, 0 timeout. All SUCCESS.

## Timeout audit
| key (ds,mech,str,seed) | timeout? | resolved? |
|---|---|---|
| 1 M01 S3 2026 | YES | YES (retry SUCCESS) |
| 1 M01 S4 13 | YES | YES (retry SUCCESS) |
| 1 M01 S4 2026 | YES | YES (retry SUCCESS) |
| 5 M05 S3 7777 | YES | YES (retry SUCCESS) |

No pattern (mixed mechanisms, datasets, seeds). Likely transient GPU kernel
contention (the parent process had a 1200s wall-clock timeout). The 4 cells
retried with identical config in ~100s each.

## Cell classification
- FORMAL_SUCCESS: 5500
- FORMAL_TIMEOUT_RESOLVED: 4
- RETRY_DUPLICATE_EXCLUDED: 24
- All other categories: 0

Clean claim-eligible: `tabr_claim_eligible_cells.csv` (5500 rows, sha in manifest).
