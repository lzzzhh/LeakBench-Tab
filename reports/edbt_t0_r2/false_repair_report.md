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

| Category | LR | RF | LightGBM |
|----------|-----|-----|----------|
| FR1 (SDR↑ but recall not better) | 9.4% | 8.8% | 8.9% |
| FR2 (SDR > 0 with zero leak removal) | 0.0% | 0.0% | 0.0% |
| FR3 (SDR↑ but residual not better) | 2.0% | 2.4% | 1.5% |
| **FR4 (SDR↑ with overcorrection)** | **29.2%** | **29.7%** | **31.4%** |
| FR5 (SDR↑ but retention worse) | 9.0% | 8.7% | 8.7% |
| FR6 (semantic group partial) | N/A | N/A | N/A |

FR4 is the most prevalent category: nearly 30% of all keys show positive legacy SDR
that is partially attributable to overcorrection beyond the strict reference.

FR2 is zero: MI-guided removal never achieves positive SDR without removing at least
one leak column. This confirms MI's leak-finding capability is genuine.

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

The false-repair audit reveals that approximately **30% of keys** with positive legacy
SDR exhibit overcorrection (FR4). This is the dominant false-repair pattern and is
systematic across all learners and most mechanisms.

The overcorrection is not an artifact of poor MI performance — MI consistently
identifies leak columns (FR2=0%, meaning it never misses all leaks). Rather, it
reflects that removing high-MI features can push the model past the strict reference.

The paper's claim C1 should acknowledge that the governance advantage is partially
attributable to overcorrection and should not be presented as pure semantic repair.

## 7. Outputs

- `results/edbt_t0_r2/false_repair_summary.csv` — per-category breakdowns
- `results/edbt_t0_r2/false_repair_examples.csv` — top 20 worst cases per FR category
