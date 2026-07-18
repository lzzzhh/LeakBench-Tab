# Working Notes

This file records evidence discovered during implementation. It intentionally begins before protocol freezing or experiment execution.

## Pre-run inventory

- Five local natural datasets are present and bound by the existing v2 train-fitted-category freeze: Bank Marketing, Lending Club, BTS Flights, Chicago Food, and NYC311.
- Natural retained shapes range from 16 to 140 features; every case contains both legitimate and boundary-invalid fields.
- The frozen 5,500-key synthetic bundle manifest is complete.
- M09 is the only mechanism that expands one semantic source field into multiple encoded columns: eight complete one-hot columns. Other mechanisms have identity semantic-to-encoded group mappings under the current frozen representation; M10's two injected columns have distinct roles and remain distinct groups.
- The primary natural and semantic sensitivity budget is 20%, with 20 governance seeds and LR as the directly comparable learner.

## Execution audit

- Natural governance completed 315/315 rows with no failures or duplicate run IDs.
- Semantic v1 and v2 exited before writing rows because of frozen-baseline lookup bugs.
- Semantic v3 completed all fits but was excluded wholesale because its run ID omitted dataset index, producing cross-dataset collisions. The excluded 10,500-row snapshot is retained under `artifacts/archive/semantic_group_v3_duplicate_ids/`.
- Semantic v4 corrected only the run identity, reran all fits, and completed 10,500/10,500 rows with no failures or duplicate run IDs.
- No frozen SP5--SP8 result was modified.

## Result boundary

- Natural evidence is mixed: four cases favor P3 and NYC311 favors P2.
- Semantic grouping reduces the M09 effect but it remains positive with an interval above zero.
- Replacing encoded-cost M09 with semantic-cost M09 causes the full-panel interval to cross zero, so cost invariance is not supported.
