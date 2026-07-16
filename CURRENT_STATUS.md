# LeakBench-Tab — Current Status

**Date:** 2026-07-16
**Tests:** 265 passed, 1 skipped, 0 failed
**GitHub:** main / origin synchronized, working tree clean

## Frozen (DO NOT MODIFY)

### SP5 — Core Benchmark
- 27,500 cells, 5 models, 11 mechanisms, SHA-256 verified
- `claim_ledger_v2.csv` + `claim_evidence_matrix_v2.*`
- CL2 DOWNGRADED_PARTIAL, CL3 PARTIALLY_CONFIRMED, CL4 CONFIRMED_WITH_REVISED_MAGNITUDES, CL10 BROADLY_CONSISTENT_WITH_EXCEPTIONS
- Paper macros (`result_macros.tex`) generated from this ledger

### SP6 — Modern Model Expansion
- 38,500 cells, 7 models
- ModernNCA (official vendor) + TabR (official env, subprocess bridge)
- M04/M05 negative harm first observed across three modern models
- SP5 Claim Matrix V2 unchanged

### SP7 — Negative-Harm Study (CLOSED)
- CL13a PARTIALLY_CONFIRMED (cross-model directional replication)
- CL13 UNCONFIRMED (causal mechanism not identified)
- H2/H3/H4 NOT SUPPORTED by pre-registered sentinel interventions
- Mechanism study formally closed; no SP7-E

## UNDER AUDIT

### SP8 — Governance
- **Status:** UNDER_AUDIT (all claims G1-G4)
- **77,000 existing rows:** NON_CLAIM_ELIGIBLE — retained for provenance only
- Issues found:
  1. Non-oracle policies read leakage_mask to construct group/lifecycle metadata
  2. Protocol-required P2 random matched-cost baseline not implemented
  3. strict_distance_reduction metric not computed (CSV uses wrong formula)
  4. NOT_APPLICABLE strategies saved as SUCCESS (16,500 rows graph-cut zero-removal)
  5. Lifecycle achieves recall=1/retention=1 via oracle information in metadata
  6. Matched-cost runner (`run_sp8_matched_cost_governance.py`) broken
  7. No final manifest, claim decision, or result hash binding
- Required: fix runner, re-run with proper protocol, then re-audit
- Next phase: SP8 governance protocol correction (NOT SP8-D natural)

## Deferred

- TabPFNv2 / TabICL (SP6 external-validity expansion)
- Track B formal governance experiments (pending SP8 runner fix)
</EOF>

# Update README governance section
python3 -c "
r=open('README.md').read()
# Replace SP8 claims section
r=r.replace('**G1/G3/G4 SUPPORTED; G2 INCONCLUSIVE**','**G1-G4 UNDER_AUDIT** (see CURRENT_STATUS.md)')
r=r.replace('field-score budget (P3): 20%','field-score budget (P3) provisional: 20%')
open('README.md','w').write(r)
print('README updated')
"

# Update HANDOFF SP8 section
python3 -c "
h=open('HANDOFF.md').read()
h=h.replace('### SP8 — Governance (77,000 cells, LR)','### SP8 — Governance (77,000 cells, LR) ⚠️ UNDER_AUDIT')
h=h.replace('- **Claims:** G1/G3/G4 SUPPORTED; G2 INCONCLUSIVE','- **Claims:** G1-G4 UNDER_AUDIT (protocol violations found 2026-07-16 — non-oracle paths read leakage_mask, no matched-cost P2, wrong metric)')
open('HANDOFF.md','w').write(h)
print('HANDOFF updated')
"