# SP6-A Model Feasibility Report

## Summary
| Model | Status | Official source | Train-only enforceable |
|-------|--------|-----------------|------------------------|
| ModernNCA | READY_FOR_ADAPTER | LAMDA-TALENT @08301d6 | YES (explicit candidates) |
| TabR | REQUIRES_PROTOCOL_REVIEW | tabular-dl-tabr @17baa90 | YES (explicit candidates) |
| TabPFNv2 | DEFERRED (Priority 2) | pip tabpfn | pending audit |
| TabICL | DEFERRED (Priority 3) | — | high leakage risk |

## Key findings

Both TabR and ModernNCA expose `forward(..., candidate_x, candidate_y, ...)` with
**explicit retrieval memory passed at call time**. This means train-only
retrieval/reference is fully enforceable by the adapter (we control candidates),
satisfying Section 5.1/5.2 — no test rows can enter memory.

Neither ships as a pip package; both are research repos (source-only, MIT).

### ModernNCA — LOW-MEDIUM risk, READY
- Single-file `nn.Module` (TALENT/model/models/modernNCA.py).
- Only non-standard dependency: `make_module` helper (small, vendorable verbatim).
- Standard torch; compatible with our WSL torch 2.5.1/cu121.
- **Recommendation:** vendor the official model file + helper verbatim (pinned
  commit + SHA), wrap in adapter. This is NOT reimplementation — it uses the
  authors' exact model code.

### TabR — MEDIUM-HIGH risk, NEEDS DECISION
- `Model` class importable but `lib` package is heavy (faiss, delu, gpytorch,
  tensorboard, loguru) and repo pins torch **1.13.1/cu117** vs our 2.5.1/cu121.
- Using it cleanly requires either (a) vendoring `bin/tabr.py` + the minimal
  `lib` submodules it imports (data/deep/neighbors) with a torch-2.x compat
  shim, or (b) a separate torch-1.13 environment.
- Risk: option (a) touches enough glue code that it approaches the
  "reimplementation" line (stop-condition #16); option (b) adds a divergent
  environment.

## Decision needed (Section 4.2 / stop-condition #16)
ModernNCA can proceed now as READY_FOR_ADAPTER (vendor official file).
TabR REQUIRES_PROTOCOL_REVIEW: choose (a) vendor model+minimal lib with compat
shim, or (b) dedicated torch-1.13 env, or (c) defer TabR to a later batch and
run ModernNCA first.
