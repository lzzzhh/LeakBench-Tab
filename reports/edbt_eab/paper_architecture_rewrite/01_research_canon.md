# Research canon

## Central method object

CDXR is a contract-grounded evaluation architecture with four non-interchangeable
objects:

- **C — construction validity:** whether a field violates the declared
  prediction-time availability or lineage contract.
- **D — detectability:** whether an oracle-blind training-side diagnostic ranks
  contaminated fields.
- **X — exploitability:** the paired strict-versus-permissive AUROC difference
  for a named learner.
- **R — repair response:** the downstream change induced by a removal policy,
  measured under matched cost by SDR, contaminated-field recall, and legitimate
  retention.

The initial strict--full distance is the repair opportunity. It conditions R;
it is not a fifth detector and does not make repairability causal or automatic.

Every CDXR instance exposes a minimum information contract: prediction event
and horizon; a semantic-to-encoded feature map; availability and lineage
labels; policy-visible metadata; the hidden oracle mask; intervention cost;
split and independent-unit definitions; learner; and evaluation metric. The
selector, evaluator, and verifier are role-separated. Only the verifier may
read the oracle mask after selection is frozen.

The verifier returns a claim certificate with scope, cost unit, effect,
interval, recall, retention, provenance, and status. Protocol violations are
`INVALID`; an unresolved prespecified direction is `UNSUPPORTED`; a supported
primary result whose declared sensitivity changes support is `NARROWED`; and a
scope-bounded result that passes all gates is `SUPPORTED`.

## Frozen experimental facts

- Measurement registry: 20 designed tasks, 11 mechanisms, five strengths, five
  injection seeds, five learners, 27,500 paired cells.
- Primary measurement contrast: simple mechanisms produce 0.164 more paired
  AUROC harm than structured mechanisms, 95% CI [0.150, 0.178].
- Governance: 5,500 keys and 20 P2 governance seeds at the primary 20% budget.
- Encoded-cost overall P3-minus-mean-P2 SDR:
  - LR +0.043, 95% CI [+0.004,+0.078].
  - RF +0.055, 95% CI [+0.025,+0.082].
  - LightGBM +0.056, 95% CI [+0.028,+0.082].
- Direct learner-effect contrasts cross zero; this is no detected interaction,
  not equivalence or learner invariance.
- Structured-family averages cross zero. M04/M05/M08 have little repair
  opportunity; M09 is a positive structured counterexample across learners.
- LR gap strata reverse direction: Q1/Q2 negative, Q3/Q4 positive. This is a
  sensitivity pattern, not proof that opportunity alone causes policy success.
- Sparse is a negative archetype; every leave-one-archetype-out point estimate
  remains positive.
- Post-hoc sparse failure anatomy verifies 1,100/1,100 P3 selection hashes.
  All four sparse tasks and nine of eleven mechanism means are negative. P3
  removes on average 1.918 of the three legitimate sparse-signal fields;
  x_000 and x_003 are removed in 89.3% and 84.9% of keys. Recall is 0.570 and
  retention is 0.848. This is construction-grounded diagnosis, not a randomized
  causal decomposition.
- Under semantic-group cost, M09 remains positive (+0.109, 95% CI
  [+0.054,+0.158]), but the recomposed overall interval crosses zero (+0.040,
  [-0.003,+0.077]).
- Natural governance is mixed: four of five fixed cases favor P3 and NYC311 is
  negative; the exact five-case sign-flip value is 0.1875.
- NYC311 is a low-opportunity failure: initial gap 0.019, repair advantage
  -0.108, recall 0.50, and retention 0.816. P3 removes one of two invalid
  fields and seven contract-valid fields at k=8/40. All three ledger selection
  hashes match deterministic reconstruction.
- B2 RF and LightGBM strict/full baselines were re-fitted under a disclosed
  protocol deviation. The extension was designed after the LR limitation was
  visible, but learner identities, budget, seeds, and estimand were frozen
  before RF/LightGBM outcomes.

## Terminology

- Use `construction-invalid`, not `statistically invalid`.
- Use `repair response` for the observed R vector and `repair opportunity` for
  the initial strict--full distance.
- Use `matched-cost removal`, not `automatic leakage correction`.
- Use `designed registry`, not `sample of datasets`.
- Use `registry-reweighting interval`, not `population confidence interval`.

## Forbidden claims

- MI solves, detects, or repairs tabular leakage in general.
- CDXR infers semantic validity from association.
- The governance effect is learner-invariant.
- MI fails on structured leakage.
- Initial gap alone determines repairability.
- Natural cases externally validate a population-level claim.
- The result is invariant to encoded-column versus semantic-group cost.
- The architecture is a production decision system or causal identification method.

## Unresolved items

- Metadata-aware repair policies are not implemented in this paper.
- Sparse signal concentration is a supported descriptive diagnosis but not a
  separately randomized causal explanation.
- Runtime and deployment utility are outside the frozen evidence.
- Author names, affiliations, email, and ORCID/CMT registration require author input.
