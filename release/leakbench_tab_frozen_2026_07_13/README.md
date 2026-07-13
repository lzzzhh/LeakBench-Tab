# LeakBench-Tab

**Version:** v1.0.0-frozen
**Status:** FROZEN — RELEASE READY WITH LIMITATIONS
**Non-Paper Readiness:** 94/100
**Freeze Date:** 2026-07-13

## Overview

LeakBench-Tab is a mechanism-centric benchmark for studying prediction-time feature leakage in tabular learning. It evaluates whether statistical diagnostics can distinguish valid predictive features from those that would be unavailable at deployment time.

## Scale

| Tier | Verified Cells | Purpose |
|---|---|---|
| Core Tier | 10,083 | Mechanism coverage, strength sweep, cross-model profiles |
| Meta Tier | 4,032 | Metadata-complete diagnostic + governance validation |
| **Total** | **14,115** | |

- **11 leakage mechanisms** across statistical, temporal, structural, and distributional categories
- **5 frozen strength levels** (S1–S5)
- **8 model families** (5 core, 2 supporting, 1 exploratory)
- **3 natural-task entries** (2 fully evaluable, 1 adapter-limited)
- **99 tests** passing, 21/21 release validator

## Key Findings

### Confirmed
- Simple contamination is easily detected and exploited (AUPRC 1.00)
- Structured contamination is hard to localize statistically (AUPRC 0.05–0.07)
- Detectability–exploitability correlation is category-driven
- Operational metadata improves in-domain diagnosis (Δ AUPRC +0.098)
- Fixed field-count budgets fail on redundant contamination clusters
- Not all construction-invalid information is exploitable

### Refuted
- Group/graph/lifecycle governance does not generally outperform field-level governance
- Operational lifecycle governance does not outperform field-level governance
- Operational metadata does not transfer zero-shot to natural tasks

### Negative Results Preserved
- BiQ converges to keep-all; AIT converges to remove-all
- TabM shows negative structured harm (causally unresolved)
- NYC 311 shows negative operational-metadata reranking

## Known Limitations
- Natural task labels are single-reviewer
- Lending Club uses synthetic adapter (not evaluable for audited contamination)
- A-POLICY metadata is policy-equivalent, not deployable
- No within-family capacity gradient experiment
- TabPFN v2 is exploratory only (21 cells)

## Reproduction

```bash
bash scripts/reproduce_core_reports.sh
bash scripts/reproduce_meta_reports.sh
bash scripts/validate_frozen_release.sh
```

## License

MIT
