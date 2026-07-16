# LeakBench-Tab paper writing readiness

**Target:** AAAI-27 Main Technical Track  
**Identity:** benchmark and measurement paper with critical empirical analysis  
**Current gate:** SP8 governance evidence is frozen; the submission PDF remains
blocked until the corrected-v2 paper claim release is rebuilt.

## Recommended paper scope

Keep C/D/X as the central contribution. Present the governance experiment as a
bounded intervention analysis that asks whether an oracle-blind diagnostic can
support useful removal decisions under a matched field budget. It is not a new
governance algorithm and does not establish operational or natural-task
transfer.

The strongest integration is one compact main-text result plus a complete
supplementary analysis:

1. Main text: one paragraph and one panel comparing blind MI with matched-cost
   random removal across budgets and mechanism categories.
2. Supplement: policy definitions, all four budgets, per-dataset effects,
   category intervals, retention/recall, and the complete provenance chain.
3. Limitations: the frozen governance panel uses LR only; lifecycle/provenance
   policies and natural-task governance were not evaluated.

## Frozen governance statements allowed in the manuscript

All numbers below must be generated from the SP8 machine-readable evidence,
not copied manually into LaTeX.

- At a 20% field budget, oracle-blind mutual-information removal exceeds
  matched-cost random removal by +0.05045 strict-distance reduction, with a
  paired dataset-cluster bootstrap 95% CI of [+0.008107, +0.086505] and
  P(diff > 0) = 0.9894.
- The advantage is mechanism-dependent: simple mechanisms show +0.1092 with a
  positive interval, structured mechanisms show -0.0032 with an interval that
  crosses zero, and boundary mechanisms show a smaller positive advantage.
- At 1% budget, blind MI attains more than 40% leak recall while retaining
  97.6% of legitimate fields. This is a controlled-panel result, not a
  deployment guarantee.
- Lifecycle/provenance metadata gains remain inconclusive because the required
  operational metadata is unavailable and the corresponding policies are not
  applicable in the frozen bundle protocol.

## Statements that remain prohibited

- Do not claim that governance works generally, on natural tasks, or across all
  model families.
- Do not describe the blind MI policy as a newly proposed algorithm.
- Do not turn the structured-mechanism null result into evidence that structured
  leakage cannot be governed.
- Do not treat P(diff > 0) as a frequentist p-value.
- Do not combine the 55,000 governance rows with model-training or diagnostic
  cells into one headline experiment count.
- Do not cite the legacy 77,000-row governance output as claim evidence.

## Canonical SP8 evidence chain

| Role | Path | SHA-256 prefix |
|---|---|---|
| Clean runner | `scripts/run_sp8_clean.py` | `6089aaca` |
| Clean result table | `artifacts/sp8/governance_clean.csv` | `6e3aa4c7` |
| Analysis script | `scripts/analyze_sp8_governance.py` | `daae7c39` |
| Bootstrap result | `artifacts/sp8/bootstrap_analysis.json` | `51c9e877` |
| Claim decisions (JSON) | `artifacts/sp8/claims/claim_evidence_matrix_sp8.json` | `48271050` |
| Claim decisions (CSV) | `artifacts/sp8/claims/claim_evidence_matrix_sp8.csv` | `1ed10523` |
| Freeze manifest | `artifacts/sp8/governance_clean_manifest.json` | authoritative entry point |

## Paper evidence gate before prose can be finalized

The current manuscript accepts numerical content only from
`results/corrected_v2/paper_claims.json`. That file is absent in the current
workspace, as are its required `canonical_cells.csv`, canonical manifest, and
TabM confirmatory input. Consequently, `generate_result_macros.py --check-only`
correctly fails and the current PDF is not submission-ready.

Before integrating SP8 numbers, rebuild or restore the corrected-v2 claim
release and extend its schema/generator with a hash-bound `governance` block.
The generator should verify the SP8 manifest and emit governance macros and
tables. Hand-edited governance macros are not acceptable.

## Writing order after the evidence gate clears

1. Freeze the one-sentence thesis and contribution list.
2. Rebuild the unified claim source and generated tables/macros.
3. Rewrite the abstract and introduction from the final claim set.
4. Tighten problem setup and experimental protocol around estimands.
5. Write results claim-by-claim, including the bounded governance result.
6. Write limitations before the conclusion so scope restrictions cannot be
   lost during compression.
7. Build and audit the anonymous PDF, reproducibility checklist, references,
   page count, fonts, and claim-to-source traceability.
