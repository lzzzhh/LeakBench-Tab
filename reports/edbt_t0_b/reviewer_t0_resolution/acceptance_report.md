# Reviewer T0 Acceptance Report

Status: **COMPLETE**

## Evidence boundary

The formal Full-B1 execution contains 5,500 canonical keys, 11,000 baseline
rows, 792,000 governed rows, 792,000 selection rows, and no failed run. It
crosses five policies (20 matched-random seeds for P2 and deterministic P3--P6),
two cost contracts, and three budgets. The 64 admitted shards and canonical
merge pass the strict validator. The canonical merge manifest SHA-256 is
`2c8ea06edb6f52e81f843bb1d94452636385bc7a51b926e5aecc8b972b3baf3d`.

The primary estimator averages P2 over its 20 governance seeds within each
canonical key, forms each deterministic-policy-minus-P2 contrast within the
same key, reduces to task means, and cluster-bootstraps the 20 tasks for 5,000
replicates. These intervals describe the designed registry, not a task
population.

## RC1: repair construct validity

**Resolved by narrowing the claim.** Score-distance recovery alone is not
accepted as semantic repair. Full-B1 reconstructs oracle-isolated semantic
metrics from the selection ledger after execution: directional repair,
full-group recall, deletion precision, legitimate-feature retention,
overcorrection, and introduced distortion. At the primary semantic-group 20%
contract, every learned policy receives `TRADEOFF`, not `SUPPORTED`:

| Policy | Legacy SDR vs P2 | Directional repair vs P2 | Full-group recall vs P2 | Overcorrection vs P2 |
|---|---:|---:|---:|---:|
| P3 | +0.043 [-0.005,+0.084] | +0.095 [+0.079,+0.110] | +0.483 [+0.462,+0.503] | +0.052 [+0.023,+0.085] |
| P4 | +0.013 [-0.033,+0.053] | +0.089 [+0.073,+0.103] | +0.520 [+0.509,+0.530] | +0.075 [+0.049,+0.107] |
| P5 | +0.019 [-0.025,+0.057] | +0.089 [+0.074,+0.104] | +0.511 [+0.491,+0.527] | +0.070 [+0.043,+0.099] |
| P6 | +0.032 [-0.009,+0.068] | +0.091 [+0.075,+0.106] | +0.583 [+0.565,+0.600] | +0.059 [+0.036,+0.086] |

Allowed conclusion: learned selectors localize more complete leakage groups and
reduce distortion in the correct direction more often than matched random
removal, but they also overcorrect more; the joint semantic-repair gate is not
passed. Forbidden conclusion: blind MI or any P3--P6 policy generally solves
tabular leakage.

## RC3: random-policy variance, policy breadth, and cost contracts

**Resolved for the LR controlled registry.** P2 is integrated over 20 matched
governance seeds for every key, budget, and cost contract. P3--P6 are compared
against the within-key P2 mean. Results cover 5%, 10%, and 20% budgets under
both encoded-column and semantic-group cost.

The cost contract changes what is removed. At 20%, semantic-group minus
encoded-column full-group recall is +0.144 for P3, while its legacy-SDR contrast
is -0.001 [-0.010,+0.007]. Semantic grouping therefore fixes the partial-block
construct problem without manufacturing a stronger score-only claim. P3 has a
positive legacy-SDR contrast against P4 and P5, but not reliably against P6;
policy ranking is consequently metric- and contract-dependent.

The earlier strongly negative sparse result is not reproduced by the formal
semantic-group protocol: P3 sparse legacy SDR is +0.016 [-0.070,+0.067], with
directional repair +0.077 [+0.047,+0.106] and overcorrection +0.061
[+0.022,+0.116]. This is reported as a tradeoff regime, not a repaired regime.

## RC2: learner dependence

**Resolved within a deliberately narrower contract.** The separately frozen R2
amendment evaluates LR, RF, and LightGBM for P3 versus matched random removal at
the encoded-column 20% contract. It reports `SCORE_RECOVERY_ONLY` for all three
learners and `NO_RELIABLE_INTERACTION_DETECTED` for the legacy-SDR contrast.
This supports cross-learner consistency of that one score estimand; it does not
establish learner invariance for all metrics, policies, budgets, or semantic
contracts. Directional repair shows a small LR disadvantage, and all three
learners fail the overcorrection gate.

Allowed conclusion: no reliable learner interaction was detected for the
encoded-column 20% P3-minus-random legacy-SDR contrast across LR, RF, and
LightGBM. Forbidden conclusion: leakage governance is learner-invariant.

## Authoritative artifacts

- Full-B1 validation: `results/edbt_t0_b_full_b1/validation_receipt.json`
- Canonical ledgers: `results/edbt_t0_b_full_b1/merged/`
- R10e analysis and claim gate: `results/edbt_t0_b_full_b1_analysis/`
- Cross-learner claim state: `results/edbt_t0_r2/claim_state_r2.json`
- Reproduction entrypoint: `scripts/analyze_full_b1_r10e.py`

Raw per-shard fragments and runner logs are recomputable execution cache and are
excluded from Git. Canonical merged ledgers, validation receipt, deterministic
analysis outputs, and their hash manifest are tracked.

## Manuscript integration

The EDBT manuscript is titled ``CDXR: A Contract-Grounded Evaluation
Architecture for Tabular Leakage Repair.'' CDXR is specified through an
information contract, policy-access set, encoded/semantic cost map, immutable
selector output, repair vector, and deterministic claim-admission gate. The
abstract, methods, results, discussion, limitations, and conclusion use the
Full-B1 `TRADEOFF` status and explicitly forbid general repair or learner
invariance claims. NYC311 is linked to its low-opportunity regime, and the
RF/LightGBM study is disclosed as a post-run amendment.

The official-template PDF compiles to 11 pages with no horizontal overflow,
undefined references, or missing figures. Visual inspection covered the title,
architecture, Full-B1 figure/table, conclusion, and references. The standalone
source ZIP was independently extracted and compiled. Author identity and
affiliation placeholders remain the only submission-administration blocker.
