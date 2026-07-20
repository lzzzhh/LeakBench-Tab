# T0-B Amendment 004 — V4.1 Validation Receipt Closure

**Status:** PRE_OUTCOME_VALIDATION_CLOSURE
**Date:** 2026-07-20
**Scientific freeze:** V4 (ff347b0)

V4 old receipt recorded 353/1 at time of generation (test_old_p2_formula_not_in_v3 transient failure, later fixed). V4 validator did not check receipt.failed field, producing a false pass.

V4.1 fixes: new receipt with actual test results (354/0, 13/0), enhanced validator with recursive hash closure, regression tests for receipt validation, exact 4-key dry-run bundle/split/mapping binding.

No scientific protocol changes. No outcomes observed.
