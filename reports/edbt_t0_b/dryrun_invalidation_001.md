# T0-B1 Dry Run Invalidation 001

**Original runner seal:** ed17b5a5cbddb5f0dff56155e7f3ba3b334356d7
**Status:** INVALIDATED_DIAGNOSTIC_ONLY
**Existing outputs:** results/edbt_t0_b_dryrun/ (preserved, not deleted)

Reasons:
1. RUNNER_MODIFIED_AFTER_SEAL: parity tolerance relaxed from 1e-6 to 5e-4 to warning after observing ~2e-4 difference
2. RESUME_NOT_TESTED
3. DRYRUN_VALIDATOR_INCOMPLETE: only 4 of 25 required gate checks
4. PARITY_CONTRACT_MISHANDLED: SP8 factory (max_iter=1000) ≠ V4 factory (max_iter=2000), requiring factory-conditional parity not numeric comparison

Outcomes preserved for diagnostic reference only. Not valid for policy/method/runtime claims.
