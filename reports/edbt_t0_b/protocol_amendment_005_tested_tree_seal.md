# T0-B Amendment 005 — Tested-Tree Provenance Seal

**Status:** PRE_OUTCOME_VALIDATION_CLOSURE
**Date:** 2026-07-20
**Scientific freeze:** V4 (ff347b0)

Old V4.1 receipt (4cb54f0) had `tested_git_sha = ff347b`, which is the scientific freeze commit — not the commit where the V4.1 validator and tests were actually implemented. This meant the receipt did not prove that the tested code matched the receipt.

This amendment fixes the provenance chain:
1. Implementation seal commit (4cfec1f) hardens validator+test code.
2. New receipt binds `tested_git_sha = 4cfec1f` (the implementation seal).
3. `validate_tested_tree()` verifies that between tested commit and HEAD, only receipt/manifest/amendment files changed — no validator, test, or scientific config changes.

Scientific design unchanged. No outcomes observed.
