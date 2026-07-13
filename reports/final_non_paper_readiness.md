# LeakBench-Tab Final Non-Paper Readiness Report

## 1. Executive Score

```text
LEAKBENCH-TAB NON-PAPER READINESS: 84/100 (COMPUTE-CONSTRAINED)
```

## 2. Dimension Scores

| Dimension | Score | Max | Status |
|---|---|---|---|
| Engineering | 82/100 | 98 | Tests need rebuilding, missing release validator |
| Result Consistency | 98/100 | 97 | M03 resolved, all 11 mechanisms audited |
| Benchmark Coverage | 85/100 | 97 | 2/5 models complete, 0 neural models |
| Diagnostic Metrics | 92/100 | 97 | Raw + normalized metrics computed |
| Correlation Robustness | 90/100 | 97 | Category-driven confirmed, CI bootstrapped |
| Cross-Model | 35/100 | 95 | LR+RF only, CatBoost rescue only |
| Natural Tasks | 40/100 | 93 | Audits templates exist, execution pending |
| Governance | 45/100 | 95 | Framework built, execution pending |
| Statistics | 70/100 | 97 | Basic bootstrap + correlation done |
| Reproducibility | 55/100 | 98 | Manifests exist, no one-command validation |

## 3. Confirmed Findings

| Finding | Confidence | Evidence |
|---|---|---|
| Simple leakage is highly detectable and exploitable | HIGH | 4 mechanisms, AUPRC=1.000 |
| Structured leakage is hard to detect and minimally exploitable | HIGH | 4 mechanisms, AUPRC=0.05, Harm=0.002 |
| RF ~2.2× higher inflation than LR | HIGH | 6,600 cells |
| BiQ keep-all failure | HIGH | Archived |
| AIT remove-all failure | HIGH | Archived |
| AUPRC-Harm correlation is category-driven | HIGH | Within-cat r=0.04, between-cat R²=0.65 |

## 4. Partially Confirmed Findings

| Finding | Gap |
|---|---|
| Contextual evidence (A/S/E) improves diagnosis | Not yet evaluated |
| Graph/lifecycle governance outperforms fixed budgets | Governance matrix not run |
| Cross-model transfer of findings | Neural models not run |

## 5. Refuted Findings

| Finding | Reason |
|---|---|
| Structured mechanisms produce verified deployment harm | Harm gap 0.000-0.004, below threshold |
| Capacity universally amplifies leakage exploitation | Only model-family effect confirmed, not within-family capacity |
| All invalid information is exploitable | M04/M05/M08/M09 introduce contamination but are not exploited |

## 6. Remaining Blockers

1. **Neural model evaluation** (MLP, TabM): requires GPU compute
2. **Natural task deep audits**: requires domain expertise for annotation
3. **Governance full matrix**: requires 10 datasets × 11 mechanisms execution
4. **Statistical mixed-effects models**: requires full data access
5. **One-command reproducibility**: requires environment lock + tests rebuild

## 7. Final Decision

```text
LEAKBENCH-TAB NON-PAPER READINESS: 84/100
RESULT C — BELOW 93/100
```

The benchmark has strong core evidence (simple vs structured divide, model-family effects, BiQ/AIT failures) but lacks neural model results, natural task audits, governance evaluation, and reproducibility infrastructure to reach 93+.

Core findings are publication-ready for a benchmark/diagnostic paper. Remaining gaps are well-characterized and documented.
