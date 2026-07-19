# Working notes

## Current direction

The candidate method story is a contract-grounded architecture that separates:

1. semantic validity under an availability contract;
2. blind statistical detectability/localization;
3. learner-conditional exploitability;
4. opportunity-conditioned, matched-cost repair response.

LeakBench-Tab becomes the controlled instantiation and stress-test environment,
not the paper's sole contribution.

## Evidence boundaries to retain

- Encoded-column aggregate repair is positive across LR, RF, and LightGBM.
- M04/M05/M08 have little initial repair opportunity; M09 is a structured counterexample.
- Semantic grouping preserves M09 but weakens the aggregate interval across zero.
- Natural governance is positive in four of five cases but formally mixed.
- Sparse is the only negative archetype and must remain visible.

## Open decisions

- Final architecture name and title.
- Whether the repair-opportunity object is a separate `O` stage or an explicit
  conditioning variable inside the repair stage.
- Exact composition of the three main tables.

