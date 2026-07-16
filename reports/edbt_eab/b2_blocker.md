# B2 Cross-Learner Blocker

**Status:** BLOCKED
**Date:** 2026-07-17

LightGBM GPU (WSL2, RTX 4060, CUDA 12.6) crashes during serial training
at approximately cell 1766/242000. RF ran successfully for 1765 cells
before the crash. The issue occurs on `device="gpu"` path in LGBMClassifier.

Attempted resolutions:
- Serial nohup via scheduled task → silent crash at same cell
- `--resume` → crash repeats at same point
- Multiprocessing Pool → silent failure (no workers spawned)

Possible root cause: LightGBM GPU backend incompatibility with specific
feature shapes or data values. Not a protocol issue.

Recommendation: run B2 on CPU only (RF + LightGBM), accept longer runtime.
Or defer B2 to post-submission revision.
