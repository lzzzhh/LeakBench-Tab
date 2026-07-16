# LeakBench-Tab

A mechanism-centric benchmark for prediction-time feature contamination in
tabular learning. Separates three axes: **C** (construction/prediction-time
validity), **D** (statistical detectability), **X** (model exploitability).

## Frozen core evidence (SP5, current)

| Dimension | Value |
|-----------|-------|
| Models | 5 (LR, RF, LightGBM, CatBoost, TabM) |
| Mechanisms | 11 (M01–M11) |
| Datasets | 20 confirmatory panels |
| Strengths | 5 (S1–S5) |
| Seeds | 5 (13, 42, 2026, 3407, 7777) |
| **Formal cells** | **27,500** |
| Diagnostic cells | 22,000 (4 oracle-blind rankers) |
| Metric | `paired_harm = full_auc − strict_auc` |

- Canonical ledger: `artifacts/sp5/claim_ledger_v2.csv` (sha256 `ccb2549f…`).
- Claim status: `artifacts/sp5/claim_evidence_matrix_v2.md`.
- Manifest: `artifacts/sp5/sp5_manifest.json` (sha256 `ac7da179…`).

### Core claims (locked, SP5.5 — see `artifacts/sp5_5/paper_claim_lock.md`)
- **CL2** (DOWNGRADED_PARTIAL): simple contamination near-perfectly localizable
  (AUPRC ~0.93); structured markedly lower on average (~0.35) but heterogeneous
  — temporal M04/M05 hard (~0.13), entity/source M08/M09 moderate (0.43–0.69).
- **CL3** (PARTIALLY_CONFIRMED): detectability–exploitability positively
  associated (r~0.69, n=11 wide CI); category not the sole driver (incremental
  ΔR²~0.11); "within-category≈0" refuted.
- **CL4** (CONFIRMED_WITH_REVISED_MAGNITUDES): model family affects exploitation
  modestly (~0.02–0.04 AUROC; ratios 1.1–1.3×, not 2.2–2.5×).
- **CL10** (BROADLY_CONSISTENT_WITH_EXCEPTIONS): profiles broadly consistent
  (mean Spearman ~0.85, Kendall W ~0.88) with mechanism-specific exceptions
  (TabM negative on M04/M05; M02/M03 disagreement).

## Deferred (NOT yet evaluated)

**Model expansion (Phase SP7):** ModernNCA, TabR, TabPFNv2, TabICL are **not**
part of the frozen core and have no formal results. They are a future
external-validity extension and cannot alter frozen core claims.

**Governance experiments (Phase SP8, Track B):** UNDER_AUDIT. Clean runner
(`run_sp8_clean.py`) implemented with oracle isolation and matched-cost
protocol. Preliminary results show blind field-score beats random at matched
budget. Claims pending. Clean runner done (55,000 cells), bootstrap analysis committed. P3 beats P2 at all budgets (20% diff +0.051 CI[+0.008,+0.087]). Simple mechs governable; structured not.
Old 77,000 rows NON_CLAIM_ELIGIBLE.

**Natural-task transfer:** limited; core-benchmark completeness does not imply
natural-data external validity.

## Reproducibility

```bash
bash artifacts/sp5/commands.sh          # rebuild ledger + claims + figures
python3 scripts/paper/generate_sp5_paper_macros.py   # regenerate paper macros
python3 -m pytest tests/test_sp5_ledger.py tests/test_sp5_5_manuscript.py -q
```

Analysis seed 20260714; cluster bootstrap over datasets, 10000 reps.

## Repository layout
- `src/leakbench/` — mechanisms, models, diagnostics, governance
- `experiments/leakbench/` — frozen runners (structured_prior, m10 amendment, tabm)
- `results/` — evidence (large files gitignored; formal SP5 outputs tracked)
- `artifacts/sp5/` — SP5 claim ledger, claim matrix, per-claim analyses
- `artifacts/sp5_5/` — manuscript sync: claim lock, traceability, audits
- `paper/aaai27/` — manuscript (macro-driven; numbers from claim_ledger_v2)
- `archive/` — excluded evidence (code-drift, interim), audit-only
