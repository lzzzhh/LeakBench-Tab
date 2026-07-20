# T0-B1 Dry-Run Report

**Status:** COMPLETE — INTEGRITY PASS
**Runner seal:** ed17b5a

## Counts
| Ledger | Expected | Actual |
|--------|----------|--------|
| Baseline | 8 | 8 |
| Governed | 576 | 576 |
| Selection | 576 | 576 |
| Failure | 0 | 0 |
| Ranking fits | 16 | 16 |

## Integrity
- Bundle SHA: 4/4 verified
- Split hashes: 12/12 verified
- Semantic partial violations: 0
- M09 atomic groups: verified
- Run ID uniqueness: 576/576 unique
- Baseline parity: SP8 diff ~2e-4 (benign max_iter=2000 vs 1000 convergence)

## Runtime
- Wall clock: 6s (single worker, 4 keys)
- Per-key: ~1.5s (including P6 RF permutation)

## Calibrated Full-Run Estimate
- B1 (803k fits): ~6s × 5500/4 ≈ 2.3 hours (single worker)
- B2 (286k fits): similar per-fit time → ~0.8 hours
- Total: ~3 hours single-worker, ~10 min with 20 workers
- P6 cost is dominant (permutation importance per key)
