# Argument map

## Scientific tension

Tabular leakage is usually treated as one detection problem, but the action an
auditor must take depends on four questions that can disagree: whether a field
is valid, whether a blind score can find it, whether a learner exploits it, and
whether deleting it improves a boundary-valid evaluation under an explicit
cost. A detector-only benchmark cannot adjudicate the last question.

## Central research question

How should tabular leakage repair be evaluated when semantic invalidity,
statistical localization, learner exploitation, and intervention cost do not
coincide?

## Central thesis

CDXR turns leakage governance into a contract-grounded, matched-cost evaluation
problem. Its controlled instantiation shows that blind MI ranking adds repair
value beyond equal-count deletion across three learners under encoded cost,
while opportunity, task archetype, representation cost, and natural context
define visible boundaries.

## Supporting arguments

1. The semantic contract must precede statistics because association cannot
   determine prediction-time availability.
2. D and X require different conditionings; crossed mechanism profiles show
   that neither substitutes for the other.
3. R requires a matched random negative control because arbitrary deletion can
   move AUROC toward the strict reference.
4. Repair opportunity explains why a family average can hide opposite regimes;
   M09 prevents the structured category from becoming a false failure claim.
5. Cost representation belongs inside the intervention contract; semantic
   grouping changes the aggregate support boundary.
6. Natural and archetype sensitivities show both portability and failure cases,
   without converting the designed registry into a population sample.
7. Role-separated interfaces and a machine-readable claim certificate turn the
   architecture into a falsification procedure rather than a renamed taxonomy.

## Counterarguments and answers

- **This is still only a benchmark.** The contribution is the CDXR evaluation
  architecture, its access-controlled interfaces, and its claim gates;
  LeakBench-Tab is the executable controlled instantiation used to test them.
- **MI is not novel.** MI is deliberately a representative blind selector. The
  method novelty is the contract-grounded matched-cost evaluation of repair,
  not a new feature score.
- **Structured leakage is weak, so there is nothing to repair.** Exactly: CDXR
  exposes repair opportunity as a separate object and reports M09 as the
  counterexample.
- **Twenty synthetic tasks are not independent applications.** Intervals are
  registry-reweighting summaries; archetype and natural-case results expose the
  transport boundary.
- **Encoded-column cost biases wide representations.** Semantic-group
  sensitivity narrows the aggregate claim and is reported as such.
- **Sparse is an unexplained systematic failure.** Hash-verified post-hoc
  anatomy shows the negative response across all four tasks and most mechanisms,
  while the MI policy frequently removes the sparse generator's concentrated
  legitimate signal. The paper reports this as diagnosis, not causal proof.
- **Cross-learner evidence was added after seeing LR.** Correct: the extension
  answers a visible LR-only limitation. Its learner pair, budget, seeds, and
  estimand were frozen before RF/LightGBM outcomes; the later baseline-refit
  mismatch is disclosed and bounds those panels to corroborative contrasts.

## Final move

The strongest conclusion is methodological: a leakage intervention is credible
only when the semantic reference, oracle isolation, cost unit, matched control,
paired downstream response, and independent unit are explicit.
