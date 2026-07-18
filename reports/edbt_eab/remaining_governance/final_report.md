# Remaining Governance Final Report

## Verdict

Both previously missing experiments are complete. Natural governance provides mixed descriptive external evidence. Semantic-group sensitivity preserves the positive M09 result but narrows the overall governance claim because the recomposed full-panel interval crosses zero.

## Natural governance

The formal table contains 315 successful cells: five fixed cases, three LR training seeds, one blind-MI policy, and 20 random-removal governance seeds at 20% matched retained-feature cost. There are no failed or duplicate cells.

The across-case descriptive mean P3-minus-mean-P2 SDR is +0.187 with case-bootstrap interval [+0.025,+0.323]. Four of five case effects are positive: Bank Marketing +0.102, Lending Club +0.266, BTS Flights +0.348, and Chicago Food +0.326. NYC311 is negative at -0.108. The exact two-sided sign-flip value over five fixed cases is 0.1875; this is not population-level confirmation.

## Semantic-group cost

M09 is the only frozen mechanism where one semantic field expands to multiple encoded columns. The formal v4 table contains 10,500 successful M09 cells with no duplicate run IDs: 500 keys, one P3 result and 20 P2 governance seeds per key.

Under encoded-column cost, M09 P3-minus-P2 is +0.149, CI [+0.125,+0.171]. Under semantic-group cost it remains positive at +0.109, CI [+0.054,+0.158]. The semantic-minus-encoded M09 contrast is -0.040, CI [-0.076,-0.008]. After replacing M09 in the complete 5,500-key panel, the overall effect is +0.040, CI [-0.003,+0.077], compared with encoded-cost +0.043, CI [+0.004,+0.078].

## Claim decisions

- C5 natural governance: `MIXED`.
- C6 semantic-group budget: `NARROWED`.
- Allowed conclusion: M09's positive governance result survives semantic grouping, but the full-panel governance advantage is sensitive to the cost definition.
- Forbidden conclusion: governance is generally validated on natural data or invariant to semantic-group cost.

## Canonical outputs

- `results/edbt_eab_revision/natural_governance_cells.csv` — 315 rows, SHA256 `26b86084`.
- `results/edbt_eab_revision/semantic_m09_cells.csv` — 10,500 rows, SHA256 `f2e98d3c`.
- `results/edbt_eab_revision/natural_governance_summary.csv` — five paper-facing case rows.
- `results/edbt_eab_revision/semantic_budget_summary.csv` — five paper-facing cost-sensitivity rows.
- `results/edbt_eab_revision/remaining_governance_summary.json` — formal statistical summary, SHA256 `b9e24e1e`.

## Validation

- Claim state: C1/C2/C4 `SUPPORTED`, C3/C6 `NARROWED`, C5 `MIXED`.
- Revision manifest: `COMPLETE_WITH_DISCLOSED_LIMITATIONS`, 35 bound artifacts.
- Tests: 233 passed, 0 failed.
- EDBT manuscript: Tectonic compile succeeds; 10 A4 pages.
