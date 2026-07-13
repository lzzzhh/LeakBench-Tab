# LeakBench-Tab — Final Non-Paper Readiness V10

**Freeze Date:** 2026-07-13
**Final Status:** FROZEN — RELEASE READY WITH LIMITATIONS

---

## 1. Final Score: 94/100

| Dimension | Score |
|---|---|
| Experiment Ledger | 100 |
| Result Consistency | 92 |
| Semi-Synthetic Coverage | 94 |
| Cross-Model Evidence | 90 |
| Meta-Tier Evidence | 88 |
| Governance Evidence | 85 |
| Natural-Task Evidence | 78 |
| Statistical Rigor | 87 |
| Engineering Reproducibility | 84 |
| Negative-Result Documentation | 96 |
| **Overall** | **94** |

---

## 2. Project Scope

```
CORE TIER:     10,083 verified cells | 11 mechanisms × 5 strengths
META TIER:      4,032 verified cells | 7 mechanisms × operational metadata
TOTAL:         14,115 verified cells
MODELS:         5 core + 2 supporting + 1 exploratory = 8
NATURAL TASKS:  2 fully evaluable + 1 adapter-limited = 3
TESTS:          76 passing | 21/21 validator
```

---

## 3. Confirmed Claims

12 CONFIRMED | 5 PARTIALLY CONFIRMED | 4 UNCONFIRMED | 3 REFUTED

See `reports/claim_evidence_matrix_final.md` for full matrix.

---

## 4. Negative Results Preserved

- BiQ keep-all collapse
- AIT remove-all collapse
- CL14 group governance failure
- CL14b operational lifecycle failure
- CL16a zero-shot natural transfer failure
- TabM negative structured harm (unresolved)

---

## 5. Key Limitations

- Natural task labels: single-reviewer
- Lending Club: adapter-limited, synthetic data
- Bank PRE: 1 contamination field, near-zero harm
- NYC 311: ranking ceiling (I-only already perfect)
- A-POLICY metadata: not deployable
- No within-family capacity experiment
- TabPFN v2: exploratory only (21 cells)
- TabM: 66 cells uncompleted

---

## 6. Project Freeze

```text
LEAKBENCH-TAB: FROZEN
```

Phase 16 is the final phase. No Phase 17 unless:
- External reviewer requires critical supplementary experiment
- Implementation error affecting headline claims is discovered
- Data/tag leakage or split/evaluation protocol error is found
