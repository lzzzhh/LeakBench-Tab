# LeakBench-Tab — Final Claim-Evidence Matrix

**Freeze Date:** 2026-07-13
**Status:** FROZEN

## CONFIRMED Claims

| ID | Claim | Scope |
|---|---|---|
| CL1 | Simple global contamination is statistically detectable (MI-based AUPRC 1.00) | Core Tier, 4 mechanisms, 5 models |
| CL2 | Structured contamination is difficult to localize statistically (AUPRC 0.05–0.07) | Core Tier, 4 mechanisms, 5 models |
| CL3 | Detectability–exploitability correlation is category-driven (global r=0.73, within-cat r≈0) | Core Tier, 11 mechanisms |
| CL4 | Model family affects exploitation magnitude (RF 2.2× LR, CatBoost 2.5× LR) | Core Tier, 5 models |
| CL5a | Oracle construction-complete metadata improves contamination localization | Meta Tier V1, Δ AUPRC +0.267 |
| CL5b-raw | Raw operational metadata improves in-domain localization | Meta Tier V2, Δ AUPRC +0.098 |
| CL5b-derived | Derived operational metadata improves in-domain localization | Meta Tier V2, Δ AUPRC +0.095 |
| CL6 | BiQ converges to keep-all; AIT converges to remove-all | Lending Club, archived |
| CL7 | Fixed field-count budgets fail on redundant/structured contamination | Governance, Lending Club, structured mechanisms |
| CL9 | Not all construction-invalid information is exploitable by models | Core Tier, M04/M05/M08/M09 |
| CL10 | Three-axis mechanism profiles are consistent across core models | 5 models, 11 mechanisms |

## PARTIALLY CONFIRMED Claims

| ID | Claim | Limitation |
|---|---|---|
| CL5b-policy | Policy-equivalent metadata is highly predictive | Labeled POLICY-EQUIVALENT; not a deployable claim |
| CL11 | Three-axis profiles transfer to natural tasks | Only 2 evaluable tasks; Lending Club adapter-limited |
| CL16c | Limited field review enables natural-task adaptation | Risk-based review beats random at budget≥5, but only 2 tasks tested |
| CL17 | Natural transfer failure is explained by measurable domain shift | Descriptively attributed; no causal confirmation |
| CL15 | Natural-task exploitability is model-dependent | NYC 311 shows LR +0.245 vs CatBoost +0.029; n=2 tasks |

## UNCONFIRMED Claims

| ID | Claim | Reason |
|---|---|---|
| CL4b | General model capacity drives exploitation | No within-family capacity gradient experiment |
| CL5d | Operational metadata generalizes to unseen mechanisms | LOMO/LOFO not completed |
| CL13 | TabM negative structured harm is causally explained | Negative harm observed; mechanism unresolved |
| CL18 | Transfer-aware diagnostics improve natural governance | Natural governance matrix not completed |

## REFUTED Claims

| ID | Claim | Evidence |
|---|---|---|
| CL14 | Group/graph/lifecycle governance generally outperforms field-level | Group recall=0.00; lifecycle recall=0.29 vs field=0.34 |
| CL14b | Operational lifecycle governance outperforms field-level | Operational lifecycle recall=0.29 < field=0.34 |
| CL16a | Operational metadata transfers zero-shot to natural tasks | 0/2 evaluable tasks improved; NYC shows negative reranking |
