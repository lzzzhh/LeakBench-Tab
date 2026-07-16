# LeakBench-Tab — Current Status

**Date:** 2026-07-16
**Tests:** 279 passed, 1 skipped, 0 failed

## Frozen (DO NOT MODIFY)
- SP5: 27,500 cells, claim_ledger_v2 (sha ccb2549f)
- SP6: 38,500 cells, 7 models
- SP7: mechanism study closed, CL13a PARTIALLY_CONFIRMED

## UNDER AUDIT
### SP8 — Governance
- Clean runner: oracle-isolated, P0/P1/P2/P3 matched-cost, 55,000 cells
- 20% budget: P3 blind MI strictly outperforms P2 random (diff +0.051 CI[+0.008,+0.087], 99% bootstrap)
- Simple mechs highly governable (recall 0.93), structured not (recall 0.30, CI crosses 0)
- Bootstrap analysis + per-dataset effects committed
- 14 SP8 tests (oracle isolation, matched cost, metrics)

## Deferred
- Natural-task governance (SP8-D)
- TabPFNv2 / TabICL
