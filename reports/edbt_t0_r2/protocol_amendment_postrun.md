# T0-R2.1 Protocol Amendment — POST_RUN_CORRECTIVE_AUDIT

**Status:** POST_RUN_CORRECTIVE_AUDIT
**Date:** 2026-07-20
**Branch:** t0/repair-construct-validity-r2
**Parent commits:** 76f123f (freeze), 8596073 (T0-R2)

This amendment documents corrections to the T0-R2 audit discovered during
post-run review. It is NOT a prospective freeze. The original protocol
(reports/edbt_t0_r2/protocol.md) remains the authoritative frozen document;
this amendment supplements it with corrective actions.

---

## A1. Mapping Corrections

### A1.1 Archetype Mapping

The T0-R2 analysis used an incorrect hand-coded archetype mapping (contiguous
blocks of 4 dataset indices per archetype). The canonical mapping from
`src/leakbench/datasets.py` uses modulo-5 cycling:

```
dataset_index % 5 == 0 → linear
dataset_index % 5 == 1 → interaction
dataset_index % 5 == 2 → nonlinear
dataset_index % 5 == 3 → sparse
dataset_index % 5 == 4 → drifting
```

All archetype-level statistics (by-archetype ΔSDR, LOAO, false-repair by archetype)
must be recomputed using the canonical mapping.

Reading from the canonical source (`results/corrected_v2/core_cpu_cells.csv`):
```python
core = pd.read_csv(ROOT / 'results/corrected_v2/core_cpu_cells.csv')
arch_map = core[['dataset_index', 'archetype']].drop_duplicates()
```

### A1.2 Mechanism-Family Mapping

The T0-R2 analysis used an incorrect mechanism-family mapping. The canonical
mapping from `artifacts/sp5/mechanism_registry.yaml` and confirmed by
`scripts/analyze_governance_revision.py` is:

```python
CATS = {
    "M01": "simple", "M02": "simple", "M06": "simple", "M10": "simple",
    "M04": "structured", "M05": "structured", "M08": "structured", "M09": "structured",
    "M03": "boundary", "M07": "boundary", "M11": "boundary",
}
```

The prior analysis misclassified M03 (simple→boundary), M06 (boundary→simple),
and M10 (boundary→simple).

---

## A2. Claim-Status Gate Correction

The T0-R2 analysis introduced a non-canonical claim status
`SEMANTICALLY_CORROBORATED_WITH_OVERCORRECTION_CAVEAT`. The frozen protocol
(§11 of protocol.md) specifies exactly five statuses:

1. SEMANTICALLY_CORROBORATED
2. SCORE_RECOVERY_ONLY
3. MIXED
4. NEGATIVE
5. NOT_EVALUABLE

Since Δovercorrection > 0 for all learners (violating the SEMANTICALLY_CORROBORATED
gate: "Δovercorrection exact mean <= 0"), the C1 claim cannot be
SEMANTICALLY_CORROBORATED.

The correct status depends on which gates pass:
- Legacy SDR gates pass → at least SCORE_RECOVERY_ONLY
- Residual reduction and leak recall gates pass → semantic evidence is positive
- Overcorrection gate fails → prevents SEMANTICALLY_CORROBORATED

The most appropriate single status is **SCORE_RECOVERY_ONLY** (legacy SDR positive)
with ancillary descriptive subclaims for the semantic evidence.

Additional semantic evidence (Δleak_recall > 0, Δdirectional_repair > 0) is reported
as independent descriptive subclaims, NOT folded into a custom status.

---

## A3. Selection Reconstruction Scope

The T0-A1 reconstruction only covered 346,500 rows (20% budget only). The full
reconstruction must cover all 709,500 rows across all budgets, including P0 empty
selections and all B1 budgets (0.0, 0.01, 0.05, 0.10, 0.20).

Each bundle load must verify `bundle_sha256` against the manifest.

---

## A4. Semantic-Group Audit Scope

The T0-R2 analysis did not compute semantic-group metrics for M09. This is
required by the frozen protocol (§8). The eight M09 one-hot indicator columns
form a single semantic group, and full-group recall must be computed.

---

## A5. False-Repair Audit Scope

The T0-R2 false-repair audit only computed per-key counts. The frozen protocol
requires:
- Per-category breakdowns (learner, mechanism, mechanism-family, archetype,
  opportunity quartile, M09)
- Per-category denominators
- FR1-FR6 (all six categories, including FR2 and FR6)
- `false_repair_examples.csv` with top 20 per category

FR2: P3 legacy SDR > 0 but P3 removed_leak_count == 0.
FR6: Semantic group partially removed; encoded-column recall high but full-group recall = 0.

---

## A6. Learner Interaction

The T0-R2 analysis did not compute direct paired contrasts between learners.
These must be computed for legacy_sdr, directional_repair, and overcorrection.

---

## A7. Evidence-Chain Closure

All claim_state fields must bind to actual SHA-256 hashes of the analysis
summary. No null or placeholder values allowed.

All planned outputs (from protocol §12) must be verified to exist.
