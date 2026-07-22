# T0-B Protocol Amendment 002 — Full-B1 Production Contract Closure

**Status:** PRE_OUTCOME_INFRASTRUCTURE_CORRECTIVE  
**Date:** 2026-07-22  
**Scientific design modified:** No  
**Full-B1 outcomes observed:** No

## Reason

The R10c merge contract had become stricter than the formal Full-B1 plan
builder. The old production manifest omitted fields required by shard admission
and global merge, while runner `--validate-only` did not reuse those gates. This
could allow an expensive execution to begin with a plan that could not later be
admitted or merged.

## Corrections

1. The production plan now declares and binds `mode`, execution-contract
   version, selection/failure counts, policy/semantic mappings, and an exact
   40-character Git tool seal.
2. Runner validation and execution use the same plan-schema and global-scope
   gates as shard admission and merge.
3. Absence of `--synthetic` now means production; synthetic plans fail closed.
4. Source and candidate ledger comparison, including SHA-256 hashing, is
   streaming.
5. Empty ledgers have one canonical representation (`header\n`); blank physical
   rows and missing terminal newlines are rejected.
6. Candidate validation rebuilds the source-shard snapshot under lock rather
   than trusting only a caller-provided snapshot.
7. Negative tests cover corrupt plans, locked/mutated shards, digest and
   selection closure, physical CSV violations, symlink parents, fsync failures,
   and absence of partial publication.

## Scientific Invariants Preserved

- Policies P0-P6 are unchanged.
- Governance seeds, budgets, cost contracts, mechanisms, datasets, and learners
  are unchanged.
- The 5,500 canonical keys and 803,000 B1 downstream runs are unchanged.
- Run-ID namespace formulas are unchanged.
- No outcome value informed any correction in this amendment.

## Frozen Execution Inputs

- Scientific freeze: `ff347b0657e8faf5d0ec1a4ca283185ffe2f5845`
- Tool seal: `156b6e887bc97d0099ec0a700e748ae5cf561f6d`
- Formal plan manifest SHA-256:
  `b36551fe7cce614204a8417f11ae55079339b4638c42dadd6ba7b81ac938f2f1`
- Formal run plan SHA-256:
  `4fdfe21c3861ef45fcab693d8af6e64de27bd92e972af077bee00d5b84da3aef`

This amendment authorizes validation and canary execution only after the
regenerated production plan passes every shared preflight gate.
