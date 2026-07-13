# M08 Final Analysis — Structured Prior v1 (Frozen Protocol)

**Status: COMPLETE (5-model frozen-protocol evidence).**
**Date:** 2026-07-13
**Source:** `results/structured_prior_replacement_v1/model_cells.csv` (7500 cells, sha256 `e3af534b…`)
**Metric:** `paired_harm = full_auc − strict_auc` (higher = more exploitable leakage)
**M08 subset:** 2500 cells = 5 models × 5 strengths × 5 seeds × 20 datasets.

## Provenance

- Injector: `StructuredPriorV1Injector` (outcome-independent constant 0.5 prior).
- Runner: `run_structured_prior_v1_bundle.py --allow-run`, unmodified, byte-frozen,
  per-fit hash verification (bundle / task / strict-view / full-view / mask).
- CPU models (LR/RF/LightGBM/CatBoost): posix macOS, 6000 cells.
- TabM: **WSL2 Ubuntu + CUDA on RTX 4060 Laptop GPU**, 1500 cells, `device=cuda`
  (no CPU/MPS fallback). See execution-environment note below.
- 7500/7500 SUCCESS, 7500 integrity-verified, 0 failures, 0 duplicates.

## Headline

| Statistic | Value |
|-----------|-------|
| M08 overall mean paired_harm | **+0.0045** |
| 95% bootstrap CI (20000 resamples) | **[+0.0035, +0.0054]** |
| Overall median | +0.0047 |

The corrected M08 entity leak is **small but real**: the CI lower bound is
strictly positive. This is far below the excluded interim number (+0.028) — as
expected, because the frozen mechanism uses an outcome-independent constant-0.5
prior on real confirmatory panels rather than an ad-hoc entity-mean leak.

## Per-model

| Model | mean | median | 95% CI | pos% | neg% | ~0% |
|-------|-----:|-------:|--------|-----:|-----:|----:|
| lr | +0.0039 | +0.0038 | [+0.0027, +0.0051] | 0.45 | 0.22 | 0.33 |
| rf | +0.0057 | +0.0071 | [+0.0039, +0.0075] | 0.55 | 0.29 | 0.17 |
| lightgbm | +0.0023 | +0.0043 | **[−0.0003, +0.0048]** | 0.48 | 0.35 | 0.17 |
| catboost | +0.0080 | +0.0077 | [+0.0050, +0.0109] | 0.55 | 0.31 | 0.14 |
| tabm | +0.0026 | +0.0036 | [+0.0011, +0.0041] | 0.46 | 0.27 | 0.27 |

All five models show a positive mean. **CatBoost** exploits the leak most
(+0.0080); **LightGBM** is weakest and its CI includes zero
([−0.0003, +0.0048]) — for LightGBM the M08 effect is not distinguishable from
zero at this scope. This is genuine architecture-dependent heterogeneity.

## Strength dose-response (monotonic)

| Strength | S1 | S2 | S3 | S4 | S5 |
|----------|----|----|----|----|----|
| mean paired_harm | −0.0024 | +0.0026 | +0.0058 | +0.0072 | +0.0092 |

Exploitability rises monotonically with contamination strength. At the lowest
strength (S1) the mean is slightly negative (the weak leaked feature adds noise
that slightly hurts), turning positive from S2 onward. This is the expected
signature of a real, tunable leak.

### model × strength interaction

| model | S1 | S2 | S3 | S4 | S5 |
|-------|----|----|----|----|----|
| catboost | +0.0015 | +0.0043 | +0.0090 | +0.0124 | +0.0129 |
| lightgbm | −0.0049 | +0.0025 | +0.0033 | +0.0020 | +0.0084 |
| lr | −0.0018 | +0.0020 | +0.0051 | +0.0067 | +0.0076 |
| rf | −0.0008 | +0.0036 | +0.0067 | +0.0088 | +0.0103 |
| tabm | −0.0057 | +0.0007 | +0.0051 | +0.0063 | +0.0067 |

## Heterogeneity across datasets / seeds

- Per-dataset mean spread: **−0.0186 to +0.0236** (mean +0.0045). Some panels
  show net-negative harm; M08 exploitability is dataset-dependent.
- Per-seed mean: 13→+0.0033, 42→+0.0015, 2026→+0.0042, 3407→+0.0052,
  7777→+0.0084. No single seed dominates.

## Bounded conclusions (permitted)

- The corrected M08 mechanism yields a small positive mean inflation for all five
  evaluated models, monotone in strength, with a strictly-positive overall CI.
- Effect is architecture-dependent: strongest for CatBoost, not distinguishable
  from zero for LightGBM.

## NOT permitted (out of scope)

- No "eight-model" claim (protocol has 5 models).
- No cross-model immunity/robustness claims from a single mechanism.
- No comparison to Simple mechanisms until the unified ledger is rebuilt.

## Execution-environment note (not a protocol deviation)

TabM ran under WSL2 Ubuntu (kernel 6.18-microsoft-standard-WSL2) on the Windows
host, using the **unmodified** byte-frozen runner (14/14 frozen-file hashes
verified inside WSL). GPU: NVIDIA GeForce RTX 4060 Laptop, driver 560.94,
torch 2.5.1+cu121 (CUDA 12.1 runtime), Python 3.12.3, `device=cuda`. The frozen
config pins `device: cuda`; this run satisfied it. Host/guest/driver/torch
versions are recorded as execution environment, not protocol content.
