# T0-R2.1 False-Repair Audit Report

**Status:** COMPLETE  
**Date:** 2026-07-20  
**Audit:** T0-R2.1 Evidence-Chain Repair  
**Mappings:** Archetype (canonical modulo-5), Mechanism-family (canonical from artifacts/sp5)

---

## 1. False-Repair Categories

| ID | Definition | Denominator |
|----|-----------|-------------|
| FR1 | Δlegacy_sdr > 0 AND Δleak_recall <= 0 | Keys with positive Δlegacy_sdr |
| FR2 | P3 legacy_sdr > 0 AND P3 removed_leak_count == 0 | All keys |
| FR3 | Δlegacy_sdr > 0 AND Δsame_side_residual >= 0 | Keys with positive Δlegacy_sdr |
| FR4 | Δlegacy_sdr > 0 AND Δovercorrection > 0 | All SUCCESS keys |
| FR5 | Δlegacy_sdr > 0 AND Δlegit_retention < 0 | Keys with positive Δlegacy_sdr |
| FR6 | Δlegacy_sdr > 0 AND semantic-group partial removed | M09 keys (N/A for non-M09) |

## 2. Overall Prevalence (5,500 keys per learner)

### 2.1 All-Key Prevalence

| Category | LR | RF | LightGBM | Meaning |
|----------|----|----|----------|---------|
| FR1 | 9.4% | 8.8% | 8.9% | SDR↑ but recall not better |
| FR2 | 1.4% | 7.7% | 8.0% | P3 SDR > 0 with zero leak removal |
| FR3 | 2.0% | 2.4% | 1.5% | SDR↑ but residual not better |
| **FR4** | **29.2%** | **29.7%** | **31.4%** | **Overcorrection (all keys)** |
| FR5 | 9.0% | 8.7% | 8.7% | SDR↑ but retention worse |
| FR6 | 7.7% | 7.7% | 7.7% | M09 partial semantic removal |

FR4 wording: "29–31% of all controlled keys satisfy the FR4 condition."

### 2.2 Conditional Prevalence (among eligible keys)

Eligibility: FR1/FR3/FR4/FR5 require Δlegacy_sdr > 0; FR2 requires P3 legacy_sdr > 0;
FR6 requires M09 AND Δlegacy_sdr > 0.

| Category | LR cond. | RF cond. | LightGBM cond. | Eligible definition |
|----------|---------|---------|---------------|---------------------|
| FR1 | 15.3% | 14.3% | 14.4% | Δlegacy_sdr > 0 |
| FR2 | 2.6% | 12.3% | 12.8% | P3 legacy_sdr > 0 |
| FR3 | 3.3% | 3.8% | 2.5% | Δlegacy_sdr > 0 |
| **FR4** | **47.6%** | **48.3%** | **51.0%** | Δlegacy_sdr > 0 |
| FR5 | 14.7% | 14.1% | 14.2% | Δlegacy_sdr > 0 |
| **FR6** | **92.4%** | **92.8%** | **93.1%** | M09 AND Δlegacy_sdr > 0 |

FR4 conditional wording: "Among keys with positive Δlegacy SDR, 48–51% exhibit overcorrection."

### 2.3 Key Observations

FR4 is the dominant false-repair pattern: 29–31% of all keys, and 48–51% of keys
with positive ΔSDR.

FR2 is non-zero for RF/LightGBM (7.7–8.0%): tree-based models achieve positive
P3 SDR without removing any leak column in 7–8% of keys. This occurs when MI-guided
removal deletes legitimate strong features, reducing overfitting without removing leaks.

FR6: Among 500 M09 keys with positive ΔSDR, 92% show partial semantic-group removal
(1-7 of 8 one-hot columns removed) — full removal of all 8 columns is structurally
impossible at 20% budget (k≈4 < 8).

## 3. By Mechanism (LR)

| Mechanism | FR1 | FR2 | FR3 | FR4 | FR5 |
|-----------|---:|----:|----:|----:|----:|
| M01 (simple) | 5.2% | 0% | 0.0% | 48.0% | 10.4% |
| M02 (simple) | 0.6% | 0% | 0.0% | 14.6% | 7.6% |
| M03 (boundary) | 0.0% | 0% | 0.2% | 0.0% | 10.8% |
| M04 (structured) | 2.2% | 0% | 0.0% | 0.2% | 9.8% |
| M05 (structured) | 0.8% | 0% | 0.0% | 0.0% | 11.0% |
| M06 (simple) | 0.0% | 0% | 4.6% | 48.6% | 5.8% |
| M07 (boundary) | 4.0% | 0% | 1.6% | 40.0% | 5.8% |
| M08 (structured) | 32.4% | 0% | 2.4% | 14.2% | 9.8% |
| M09 (structured) | 0.2% | 0% | 0.2% | 34.4% | 7.8% |
| M10 (simple) | 3.2% | 0% | 1.4% | 43.8% | 9.2% |
| M11 (boundary) | 0.0% | 0% | 0.0% | 58.8% | 7.2% |

**Key observations:**
- FR4 is concentrated in mechanisms with large initial gaps (M01, M06, M07, M10, M11).
- M08 has the highest FR1 rate (32.4%): MI identifies leak columns but ΔSDR is positive without recall gain.
- M03, M04, M05 have low FR4 because their ΔSDR is negative (no "positive" to falsely attribute).

## 4. By Mechanism-Family (LR)

| Family | FR1 | FR4 | FR5 | Notes |
|--------|----:|----:|----:|-------|
| simple | 2.3% | 38.5% | 8.3% | Highest FR4 — strong feature deletion drives overcorrection |
| boundary | 1.3% | 32.6% | 7.9% | Moderate FR4 |
| structured | 8.9% | 12.3% | 9.6% | Highest FR1 — MI hits leaks but SDR gain is questionable |

## 5. By Archetype (LR)

| Archetype | FR1 | FR4 | FR5 | Notes |
|-----------|----:|----:|----:|-------|
| linear | 5.2% | 38.5% | 8.2% | |
| interaction | 4.7% | 38.1% | 8.4% | |
| nonlinear | 8.1% | 26.5% | 8.5% | |
| sparse | 6.5% | 18.4% | 10.3% | Low FR4 because ΔSDR is mostly negative |
| drifting | 4.9% | 33.6% | 8.1% | |

## 6. Verdict

The false-repair audit reveals that approximately **30% of all keys** (48–51% of keys
with positive ΔSDR) exhibit overcorrection (FR4). This is the dominant false-repair
pattern and is systematic across all learners and most mechanisms.

FR6 (M09 partial removal) is at 92–93% among eligible M09 keys: MI consistently hits
at least one M09 column, but never removes all 8 at 20% budget (k≈4 columns, need
k≥8 for full group removal). The semantic-group any-hit improvement is real
(Δ=+0.114 CI[+0.100,+0.128]), but full-group recall is structurally zero for both
P3 and P2 at this budget.

The paper's claim C1 should acknowledge that the governance advantage is partially
attributable to overcorrection and should not be presented as pure semantic repair.

## 7. Outputs

- `results/edbt_t0_r2/false_repair_summary.csv` — per-category breakdowns
- `results/edbt_t0_r2/false_repair_examples.csv` — top 20 worst cases per FR category
