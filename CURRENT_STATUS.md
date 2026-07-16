# LeakBench-Tab — Current Status

**Date:** 2026-07-17
**Tests:** run the current suite with `python -m pytest -q`

**Canonical project-material map:** `reports/edbt_eab/PROJECT_MATERIALS.md`

## EDBT EA&B Paper-Facing State
- Corrected-v2 canonical ledger: 27,500 successful cells, SHA-256 `25c21440...`
- Paper claims and claim state: byte-identical, confirmatory evidence tier
- Paper numeric inputs: three governed CSVs (12 + 7 + 5 rows)
- Current official-template draft: 10 A4 pages including references
- EDBT paper source and evidence packages are built independently under `release/`

## Frozen (DO NOT MODIFY)
- SP5: 27,500 cells, claim_ledger_v2 (sha ccb2549f)
- SP6: 38,500 cells, 7 models
- SP7: mechanism study closed, CL13a PARTIALLY_CONFIRMED

## COMPLETE_FROZEN
### SP8 — Governance
- Clean runner: oracle-isolated, P0/P1/P2/P3 matched-cost, 55,000 rows and 5,500 keys
- 20% budget: P3 blind MI outperforms P2 random (diff +0.051 CI[+0.008,+0.087], paired P(diff>0)=98.94%)
- Simple mechs highly governable (recall 0.93), structured not (recall 0.30, CI crosses 0)
- Bootstrap analysis + per-dataset effects committed
- 15 SP8 tests (oracle isolation, matched cost, metrics, paired bootstrap, hash binding)

## Deferred
- Natural-task governance (SP8-D)
- TabPFNv2 / TabICL
- Author front-matter finalization
