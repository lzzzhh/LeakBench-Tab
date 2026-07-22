# Notes: Reviewer T0 Resolution

## 2026-07-22 Current-State Audit

- Branch: `t0/multipolicy-semantic-cost`
- Audited HEAD: `56088357e7a61abc11316846607c82aa13822f5b`
- Targeted Full-B1 tests: 240 passed.
- Repository tests: 653 passed.
- Worktree was clean before remediation began.
- Scientific state: `FULL_B1_EXECUTION_INFRASTRUCTURE_PRE_OUTCOME`;
  `full_b1_outcomes_observed=false`.

## RC3 Blocking Evidence

The formal 5,500-key manifest is rejected by the current R10c production schema:

1. missing `mode`;
2. missing `execution_contract_version`;
3. missing `selection_rows`;
4. missing `failure_rows`;
5. mode cannot be proven production.

The production plan builder reproduces this legacy manifest. Its receipt checks
only key count, run-ID uniqueness, and shard balance. Conversely,
`run_full_b1_shard.py --validate-only` returns `VALIDATION_PASS` for the
inadmissible plan because it does not reuse the R10c plan and global-scope gates.

## Additional Contract Gaps

- Merge source-aggregate validation uses whole-file `read_bytes`,
  `gzip.decompress`, and complete row lists.
- `header\n\n` is accepted as a header-only ledger.
- Merge/admission use `expected_mode=None` when `--synthetic` is absent, so
  production mode is not enforced.
- Tool-seal validation does not require an exact 40-character lowercase Git SHA.
- No current R10c test receipt is bound to the audited HEAD.

## Remediation Evidence in Progress

- Strict merge suite expanded from 5 to 23 tests.
- New fail-closed coverage includes mode mismatch, exact 40-character tool
  seal, blank row, missing newline, key-plan SHA mismatch, missing run plan,
  source lock, unsorted/duplicate/failure/digest source mutations,
  output-parent symlink, fsync failure, and partial-output absence.
- The canonical empty failure ledger producer now emits exactly `run_id\n`.
- Candidate/source exactness and artifact hashing now stream file content.
- Focused strict merge suite: 23 passed.
- Pre-regeneration targeted suite: 255 passed, 3 legacy-mode tests failed;
  those tests omitted the now-required `--synthetic` flag and were updated.

## Outcome Boundary

No Full-B1 scientific outcomes have been observed. Infrastructure corrections
therefore do not select on experimental results and may be recorded as a
pre-outcome contract amendment.
