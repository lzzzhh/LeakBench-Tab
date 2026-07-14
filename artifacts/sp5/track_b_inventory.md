# Track B — Static Inventory (SP5-G, NO formal experiments)

**Date:** 2026-07-14
**Constraint:** static audit only. **No formal governance cells were run.**
**Formal Track B experiments started: NO.**

## Scope
Governance-relevant mechanisms M06, M09, M10, M11 (redundant / source / mixed /
graph). Governance logic lives in `src/leakbench/governance/__init__.py`
(module fixed earlier) and the experiment entrypoint
`experiments/leakbench/run_meta_tier.py` (currently under INTEGRITY_HOLD).

## Findings (all four mechanisms)

| Aspect | Status |
|--------|--------|
| Runner | `run_meta_tier.py` — `main()` raises INTEGRITY_HOLD, does not execute |
| Freeze status | NOT frozen; not hash-bound to a protocol |
| Runtime injection | YES — `inject_with_metadata()` re-implements mechanisms (invalid for corrected_v2) |
| Bundle compatible | Partial — SP2/SP4/M10 immutable-bundle infra exists and works, but run_meta_tier does not consume it |
| Oracle dependency | governance module has oracle strategies (evaluation-only, acceptable) |
| Resume / deterministic run_id | no |
| Hash validation | governance module fixed; entrypoint not bundle-verified |
| Cross-platform | unknown; WSL backslash-path risk (as hit in TabM runner) is plausible |
| Known bugs | INTEGRITY_HOLD; legacy int-as-string, seed=42, S-axis reads leak_mask |
| Protocol amendment required | **YES** |

## Recommendation
Track B needs a **read-only governance runner that consumes frozen task bundles**
(same pattern as `run_structured_prior_v1_bundle.py` / `run_m10_amendment.py`):
never inject at runtime, verify bundle/task/mask hashes, deterministic run_id,
fail-closed. M10 already has an amendment bundle; M06/M09/M11 would need bundle
export first.

## Explicitly NOT done (per directive)
- No formal governance cells executed.
- No governance performance results generated.
- No governance claim (CL7/CL14-*) upgraded.
- Runner compatibility is NOT interpreted as governance effectiveness.

## Governance claim status (unchanged, pending)
- CL7: PENDING CORRECTED RERUN
- CL14-R / CL14-G / CL14-T: PENDING CORRECTED RERUN / UNCONFIRMED

Track B remains a future phase; SP5-G does not start it.
