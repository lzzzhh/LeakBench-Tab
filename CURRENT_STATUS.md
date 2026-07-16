# LeakBench-Tab — Current Status

**Date:** 2026-07-16
**Tests:** 265 passed, 1 skipped, 0 failed
**GitHub:** main / origin synchronized, working tree clean

## Frozen (DO NOT MODIFY)

### SP5 — Core Benchmark
- 27,500 cells, 5 models, 11 mechanisms
- `claim_ledger_v2.csv` + `claim_evidence_matrix_v2.*`
- CL2 DOWNGRADED_PARTIAL, CL3 PARTIALLY_CONFIRMED, CL4 CONFIRMED_WITH_REVISED_MAGNITUDES, CL10 BROADLY_CONSISTENT_WITH_EXCEPTIONS

### SP6 — Modern Model Expansion
- 38,500 cells, 7 models
- ModernNCA (official vendor) + TabR (official env)
- SP5 Claim Matrix V2 unchanged

### SP7 — Negative-Harm Study (CLOSED)
- CL13a PARTIALLY_CONFIRMED, CL13 UNCONFIRMED
- Mechanism study closed; no SP7-E

## UNDER AUDIT

### SP8 — Governance
- **Status:** UNDER_AUDIT — claims not yet restored
- Clean runner (`run_sp8_clean.py`) implemented: oracle-isolated, P0/P1/P2/P3 with matched budget.
- Old 77,000 rows: NON_CLAIM_ELIGIBLE (retained for provenance).
- Old governance runner (`run_governance_bundle.py`): non-oracle paths read leakage_mask; NOT_APPLICABLE saved as SUCCESS.
- Pending: full manifest, claim decision, per-dataset P3 vs P2 cluster-bootstrap.

## Deferred
- TabPFNv2 / TabICL
- Natural-task governance (SP8-D) — blocked until SP8 clean runner is finalized
