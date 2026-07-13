# LeakBench-Tab — Current Status

**Date:** 2026-07-13
**Status:** NOT FROZEN

## Known Issues (pending fix)

1. **M08 entity generator** — computes entity_rates but never uses them. Feature = random entity IDs.
2. **Governance scripts** — run_meta_tier.py still uses old implementation (int indices as strings, seed=42, M04 centered conv, S-axis reads leak_mask).
3. **Natural tasks** — Bank PRE field name is bm_19 (synthetic fallback indicator). Lending adapter has undefined `ext` variable.
4. **TabPFN v2** — 594/594 cells returned constant 0.000 (API failures).

## Currently Valid

- C/D/X three-axis framework
- Simple contamination is easily detectable and exploitable
- Old operational metadata gain (+0.098) is invalid (feature-name peeking)
- Corrected operational gain is small (+0.017, +0.031) — insufficient for reliable claim
- Structured mechanisms show low/inconsistent exploitability (M08 excluded)

## Withdrawn/Pending

- 8-model consistency claim
- CL3/CL4/CL10 CONFIRMED
- CL14-R/G/T governance claims
- CL16a natural transfer (not properly evaluated)
- REFROZEN status
- 91/100, 93/100, 94/100 readiness scores
